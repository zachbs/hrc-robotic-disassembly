#!/usr/bin/env python3

import os
# Define YOLO config dir BEFORE importing Ultralytics
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

from rclpy.qos import qos_profile_sensor_data
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo 
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Int32MultiArray
from ament_index_python.packages import get_package_share_directory

import cv2
import numpy as np
import open3d as o3d
from ultralytics import YOLO
from collections import deque
import threading
import time

from gearbox_interfaces.srv import Get2DBoundingBox, EstimatePose
from gearbox_perception.utils import transformation # Import your registration function

class GearboxRegistrationNode(Node):
    def __init__(self):
        super().__init__('gearbox_registration_node')
        self.bridge = CvBridge()

        self.depth_lock = threading.Lock()        
        
        self.get_logger().info('Gearbox Registration Node has been initialized!')

        # QoS profile for sensor data (best effort)
        qos_profile = qos_profile_sensor_data
        
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

        self.camera_info_subscription = self.create_subscription(CameraInfo, '/camera/depth/camera_info', self.camera_info_callback, 10)
        

        # Service Server
        self.EstimatePoseSrv = self.create_service(EstimatePose, 'get_target_pose', self.handle_get_target_pose)

        self.Get2DBoundingBoxSrv = self.create_service(Get2DBoundingBox, 'get_2d_bounding_box', self.handle_get_2d_bounding_box)


        self.depth_burst_count = 25
        self.PAD = 20  
        self.frame_processed = False  # Flag to indicate if a frame has been processed
        self.latest_depth = None  # Store the latest depth frame
        self.latest_rgb = None  # Store the latest RGB frame
        self.camera_intrinsics = None  # Store camera intrinsics
        
        # Load YOLO model once during initialization
        # 1. Dynamically find the install share directory for your ROS 2 package
        package_share = get_package_share_directory('gearbox_perception')

        # 2. Join it with the relative path inside share
        self.get_logger().info("Loading YOLO model...")
        model_path = os.path.join(package_share, 'resources', '06-09-2026.pt')
        self.YOLO_MODEL_PATH = model_path
        self.yolo_model = YOLO(self.YOLO_MODEL_PATH)

    def camera_info_callback(self, msg: CameraInfo):
            if self.camera_intrinsics is None:
                # Extract pinhole camera parameters from K matrix
                fx = msg.k[0]
                fy = msg.k[4]
                cx = msg.k[2]
                cy = msg.k[5]
                self.camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
                    msg.width, msg.height, fx, fy, cx, cy
                )
                self.get_logger().info("Camera intrinsics stored.")


    def color_callback(self, msg):
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def depth_callback(self, msg):
        # Convert depth image (assuming 16-bit unsigned int in millimeters)
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')


    def handle_get_target_pose(self, request, response):
        self.get_logger().info(f"Received target request: '{request.object_target}' with burst_count={request.burst_count}")
        burst_count = request.burst_count

        
        # Basic state checks
        if self.latest_rgb is None or self.latest_depth is None or self.camera_intrinsics is None:
            self.get_logger().error("Camera streams or intrinsics not ready!")
            response.success = False
            return response

        # 1. Capture 1st RGB Frame and Burst of Depth Frames
        
        first_rgb = self.latest_rgb.copy()

        depth_burst = []

        while len(depth_burst) < burst_count:
            if self.latest_depth is not None:
                depth_burst.append(self.latest_depth.copy())
            time.sleep(0.03)  # Rate match ~30fps depth stream

        # -------------------------------------------------------------
        # STEP 2: YOLO Detection on 1st RGB Frame
        # -------------------------------------------------------------
        results = self.yolo_model(first_rgb, verbose=False)[0]
        bbox = None  # Format: [xmin, ymin, xmax, ymax]
        
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = self.yolo_model.names[class_id]
                    
            if class_name is not None and class_id is not None:
                bbox = box.xyxy[0].cpu().numpy().astype(int)
                self.get_logger().info(f"Found {class_name} at bounding box: {bbox}")
        
        if bbox is None:
            self.get_logger().warn(f"Target not detected by YOLO!")
            response.success = False
            return response

        # -------------------------------------------------------------
        # STEP 3: Median Depth Map Filtering & Bounding Box Cropping
        # -------------------------------------------------------------
         # Compute pixel-wise median across the Z-axis of collected depth frames
        depth_stack = np.array(depth_burst)
        median_depth = np.median(depth_stack, axis=0).astype(np.float32)
        
         # Zero out depth map everywhere outside the bounding box
        xmin, ymin, xmax, ymax = bbox
        mask = np.zeros_like(median_depth, dtype=bool)
        mask[ymin - self.PAD:ymax + self.PAD, xmin - self.PAD:xmax + self.PAD] = True
                
        cropped_depth = np.where(mask, median_depth, 0.0)



def main(args=None):
    rclpy.init(args=args)
    node = GearboxRegistrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Gearbox Registration Node gracefully.')
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()