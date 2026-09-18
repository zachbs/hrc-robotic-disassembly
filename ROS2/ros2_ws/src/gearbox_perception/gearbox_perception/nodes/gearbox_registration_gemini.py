#!/usr/bin/env python3
import time
import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge

# Import your custom interface (replace with your package name)
from gearbox_interfaces.srv import GetTargetPose 

# Import your YOLO model and registration function
from ultralytics import YOLO
# from gearbox_perception.registration import register_3d_point_cloud  <-- Your registration module


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        self.bridge = CvBridge()

        # Load YOLO model
        self.get_logger().info("Loading YOLO model...")
        self.yolo_model = YOLO("yolov8n.pt")  # Replace with path to your custom trained weights (.pt)

        # Storage for current frames and camera calibration
        self.latest_rgb = None
        self.latest_depth = None
        self.camera_intrinsics = None

        # Subscriptions
        self.create_subscription(Image, '/camera/color/image_raw', self.rgb_callback, 10)
        self.create_subscription(Image, '/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.create_subscription(CameraInfo, '/camera/depth/camera_info', self.camera_info_callback, 10)

        # Service Server
        self.srv = self.create_service(GetTargetPose, 'get_target_pose', self.handle_get_target_pose)
        
        self.get_logger().info("Perception Node initialized and ready for state machine calls.")

    def rgb_callback(self, msg: Image):
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def depth_callback(self, msg: Image):
        # Convert depth image (assuming 16-bit unsigned int in millimeters)
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

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

    def handle_get_target_pose(self, request, response):
        self.get_logger().info(f"Received target request: '{request.object_target}' with burst_count={request.burst_count}")

        # Basic state checks
        if self.latest_rgb is None or self.latest_depth is None or self.camera_intrinsics is None:
            self.get_logger().error("Camera streams or intrinsics not ready!")
            response.success = False
            return response

        # -------------------------------------------------------------
        # STEP 1: Capture 1st RGB Frame and Burst of Depth Frames
        # -------------------------------------------------------------
        first_rgb = self.latest_rgb.copy()
        depth_burst = []

        self.get_logger().info(f"Collecting {request.burst_count} depth frames...")
        while len(depth_burst) < request.burst_count:
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
            
            if class_name == request.object_target:
                bbox = box.xyxy[0].cpu().numpy().astype(int)
                self.get_logger().info(f"Found {request.object_target} at bounding box: {bbox}")
                break

        if bbox is None:
            self.get_logger().warn(f"Target '{request.object_target}' not detected by YOLO!")
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
        mask[ymin:ymax, xmin:xmax] = True
        
        cropped_depth = np.where(mask, median_depth, 0.0)

        # -------------------------------------------------------------
        # STEP 4: Convert Cropped Depth to Open3D Point Cloud
        # -------------------------------------------------------------
        # Depth maps are typically in mm; convert to meters for Open3D
        o3d_depth = o3d.geometry.Image((cropped_depth / 1000.0).astype(np.float32))
        
        cropped_pcd = o3d.geometry.PointCloud.create_from_depth_image(
            o3d_depth, 
            self.camera_intrinsics,
            depth_scale=1.0,
            depth_trunc=3.0,
            stride=1
        )

        if len(cropped_pcd.points) == 0:
            self.get_logger().error("Point cloud is empty after cropping!")
            response.success = False
            return response

        # -------------------------------------------------------------
        # STEP 5: 3D Registration Call (ICP / TEASER++)
        # -------------------------------------------------------------
        # Pass the cropped cloud to your custom registration logic
        pose_matrix_4x4, fitness_score = self.mock_registration_function(cropped_pcd, request.object_target)

        # -------------------------------------------------------------
        # STEP 6: Convert $4 \times 4$ Matrix to geometry_msgs/Pose
        # -------------------------------------------------------------
        response.pose = self.matrix_to_pose_msg(pose_matrix_4x4)
        response.fitness_score = float(fitness_score)
        response.success = True if fitness_score > 0.5 else False

        self.get_logger().info(f"Registration finished. Success: {response.success}, Fitness: {response.fitness_score:.3f}")
        return response

    def mock_registration_function(self, target_pcd, object_target):
        """
        Placeholder for your 3D registration algorithm (e.g., TEASER++ / Open3D ICP).
        Returns a 4x4 homogenouse transformation matrix and a fitness score.
        """
        # Replace this dummy return with your actual module call:
        # matrix_4x4, score = register_3d_point_cloud(target_pcd, object_target)
        
        dummy_matrix = np.eye(4)
        dummy_matrix[0, 3] = 0.05  # X translation in meters
        dummy_matrix[1, 3] = -0.1  # Y translation in meters
        dummy_matrix[2, 3] = 0.45  # Z translation in meters
        
        dummy_fitness = 0.92
        return dummy_matrix, dummy_fitness

    def matrix_to_pose_msg(self, matrix: np.ndarray) -> Pose:
        """Converts a 4x4 homogeneous transformation matrix to a geometry_msgs/Pose message."""
        pose = Pose()
        
        # Translation
        pose.position.x = float(matrix[0, 3])
        pose.position.y = float(matrix[1, 3])
        pose.position.z = float(matrix[2, 3])
        
        # Rotation Matrix to Quaternion (x, y, z, w)
        rotation_matrix = matrix[:3, :3]
        quat = R.from_matrix(rotation_matrix).as_quat()
        
        pose.orientation.x = float(quat[0])
        pose.orientation.y = float(quat[1])
        pose.orientation.z = float(quat[2])
        pose.orientation.w = float(quat[3])
        
        return pose


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()