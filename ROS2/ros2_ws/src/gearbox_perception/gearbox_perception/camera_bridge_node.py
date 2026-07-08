#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Int32MultiArray

import cv2
import numpy as np
import open3d as o3d
from ultralytics import YOLO
from collections import deque
import threading


class CameraBridgeNode(Node):
    def __init__(self):
        super().__init__('camera_bridge_node')

        self.bridge = CvBridge()

        # Depth buffer: store recent depth frames
        self.depth_buffer = deque(maxlen=50)
        self.depth_lock = threading.Lock()        
        
        self.get_logger().info('Camera Bridge Node has been initialized!')

        # QoS profile for sensor data (best effort)
        qos_profile = rclpy.qos.qos_profile_sensor_data
        
        self.color_subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.color_callback,
            qos_profile
        )

        self.depth_subscription = self.create_subscription(
            Image,
            '/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            qos_profile
        )

        self.cleaned_depth_publisher = self.create_publisher(
            Image,
            '/perception/depth_mask',
            10
        )

        self.bbox_publisher = self.create_publisher(
            Int32MultiArray,
            '/perception/bounding_box',
            10
        )


        self.depth_burst_count = 25
        self.PAD = 20  
        self.frame_processed = False  # Flag to indicate if a frame has been processed
        
        # Load YOLO model once during initialization
        self.YOLO_MODEL_PATH = "06-09-2026.pt"
        self.yolo_model = YOLO(self.YOLO_MODEL_PATH)
    
    def depth_callback(self, msg: Image):
        """
        Continuously append incoming aligned depth frames into a rolling buffer.
        """
        try:
            # 16UC1 encoding is standard for RealSense depth (16-bit unsigned integer)
            depth_cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
            with self.depth_lock:
                self.depth_buffer.append(depth_cv)
        except CvBridgeError as e:
            self.get_logger().error(f'Depth Conversion Failed: {e}')

    def color_callback(self, msg: Image):
        """
        Process color frame, run YOLO, display to user, and await keypress.
        """
        try:
            color_cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'Color Conversion Failed: {e}')
            return

        best_box = None

        # PARADIGM 1: Still Searching / Dynamic Tracking Mode
        if not self.frame_processed:
            # 1. Run YOLOv12 Inference
            results = self.yolo_model(color_cv, conf=0.60, verbose=False)[0]
            
            # Check if any bounding boxes exist
            if len(results.boxes) > 0:
                box = results.boxes[0].xyxy[0].cpu().numpy()
                best_box = [int(val) for val in box]
                
                # Draw the bounding box for active user confirmation
                cv2.rectangle(color_cv, (best_box[0], best_box[1]), (best_box[2], best_box[3]), (0, 255, 0), 2)
                cv2.putText(color_cv, "Target Tracking - Press 'C' to Lock", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # PARADIGM 2: Target Locked / Lean Stream Mode
        else:
            cv2.putText(color_cv, "Target LOCKED - Stream Passive (Press 'R' to Reset)", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Safely grab and broadcast only the most recent raw depth reading frame
            with self.depth_lock:
                raw_depth_frame = self.depth_buffer[-1] if len(self.depth_buffer) >= 1 else None
            
            if raw_depth_frame is not None:
                numpy_depth_msg = self.bridge.cv2_to_imgmsg(raw_depth_frame, encoding='16UC1')
                self.cleaned_depth_publisher.publish(numpy_depth_msg)

        
        cv2.imshow("YOLO Perception Stream", color_cv)
        key = cv2.waitKey(1) & 0xFF
        
        # Capture Command Hook
        if key == ord('c') and not self.frame_processed and best_box is not None:
            self.get_logger().info("Capture triggered! Dispatched static 3D target data.")
            
            numpy_depth = self.process_3d_registration(best_box)
            if numpy_depth is not None:
                numpy_depth_msg = self.bridge.cv2_to_imgmsg(numpy_depth, encoding='16UC1')
                
                bbox_msg = Int32MultiArray()
                bbox_msg.data = best_box

                # Set lock flag and dispatch to the 3D Registration node
                self.frame_processed = True  
                self.bbox_publisher.publish(bbox_msg)
                self.cleaned_depth_publisher.publish(numpy_depth_msg)
                self.get_logger().info("Dispatched target locked data packets.")

        # Reset Command Hook: Press 'r' to release lock and search again
        elif key == ord('r') and self.frame_processed:
            self.frame_processed = False
            self.get_logger().info("Target released. Re-initializing tracking mode.")

    def process_3d_registration(self, bbox):
        """
        Executes the burst median, bilateral filter, ROI cropping, and PC generation.
        """
        with self.depth_lock:
            if len(self.depth_buffer) < self.depth_burst_count:
                self.get_logger().warn("Not enough depth frames in buffer yet. Try again.")
                return None
            
            # 3. Take a burst of the 25 most recent depth frames
            burst_frames = list(self.depth_buffer)[-self.depth_burst_count:]
            
        # Stack the 25 frames into a 3D Numpy array (Depth x Height x Width)
        stacked_depth = np.stack(burst_frames, axis=0)
        
        # 4. Take the Median across the time axis to eliminate sensor noise
        median_depth = np.median(stacked_depth, axis=0)
        
        # 5. Apply Bilateral Filter (requires casting to float32, then back to uint16)
        median_depth_f32 = median_depth.astype(np.float32)
        filtered_depth_f32 = cv2.bilateralFilter(median_depth_f32, d=5, sigmaColor=50, sigmaSpace=50)
        filtered_depth = filtered_depth_f32.astype(np.uint16)
        
        # 6. Crop everything outside the bounding box
        x1, y1, x2, y2 = bbox
        
        # Apply padding safely within image boundaries
        h, w = filtered_depth.shape
        x1 = max(0, x1 - self.PAD)
        y1 = max(0, y1 - self.PAD)
        x2 = min(w, x2 + self.PAD)
        y2 = min(h, y2 + self.PAD)
        
        # Create a blank mask array and paste ONLY the region of interest
        masked_depth = np.zeros_like(filtered_depth)
        masked_depth[y1:y2, x1:x2] = filtered_depth[y1:y2, x1:x2]
        
        return masked_depth  # Return the point cloud for further processing if needed

def main(args=None):
    rclpy.init(args=args)
    node = CameraBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Camera Bridge Node gracefully.')
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()