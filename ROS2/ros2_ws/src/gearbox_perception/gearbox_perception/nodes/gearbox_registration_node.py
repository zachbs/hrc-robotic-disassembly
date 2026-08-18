#!/usr/bin/env python3

import os
# Define YOLO config dir BEFORE importing Ultralytics
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Int32MultiArray
from ament_index_python.packages import get_package_share_directory

import cv2
import numpy as np
import open3d as o3d
from ultralytics import YOLO
from collections import deque
import threading

class GearboxRegistrationNode(Node):
    def __init__(self):
        super().__init__('gearbox_registration_node')
        self.bridge = CvBridge()

        self.depth_lock = threading.Lock()        
        
        self.get_logger().info('Gearbox Registration Node has been initialized!')

        # QoS profile for sensor data (best effort)
        qos_profile = rclpy.qos.qos_profile_system_default
        
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
        # 1. Dynamically find the install share directory for your ROS 2 package
        package_share = get_package_share_directory('gearbox_perception')

        # 2. Join it with the relative path inside share
        model_path = os.path.join(package_share, 'resources', '06-09-2026.pt')
        self.YOLO_MODEL_PATH = model_path
        self.yolo_model = YOLO(self.YOLO_MODEL_PATH)


    def color_callback(self, msg):
            try:
                # Convert ROS Image message to OpenCV image
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                self.get_logger().info('Received color image frame.')
                
                # Process the color image (e.g., YOLO detection)
                results = self.yolo_model(cv_image)
                if results:
                    bbox_msg = Int32MultiArray()
                    for result in results:
                        for box in result.boxes.xyxy:
                            bbox_msg.data.extend(box.cpu().numpy().astype(int).tolist())
                    self.bbox_publisher.publish(bbox_msg)
                    self.get_logger().info('Published bounding boxes.')

            except CvBridgeError as e:
                self.get_logger().error(f'CvBridge Error: {e}')

    def depth_callback(self, msg):
            try:
                # Convert ROS Image message to OpenCV image
                depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
                self.get_logger().info('Received depth image frame.')
                
                # Process the depth image (e.g., filtering, masking)
                # For demonstration, we will just publish the received depth image
                self.cleaned_depth_publisher.publish(msg)
                self.get_logger().info('Published cleaned depth image.')

            except CvBridgeError as e:
                self.get_logger().error(f'CvBridge Error: {e}')




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