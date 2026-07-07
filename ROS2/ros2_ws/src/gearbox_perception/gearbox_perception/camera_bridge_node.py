#!/usr/bin/env python3

# 1. Import core ROS 2 client library for Python
from collections import deque
import threading

import cv2
from ultralytics import YOLO

import rclpy
from rclpy.node import Node

# 2. Import standard message types (e.g., String, Image, PointCloud2)
from sensor_msgs.msg import Image 

from cv_bridge import CvBridge, CvBridgeError

import pyrealsense2 as rs

class CameraBridgeNode(Node):
    """
    A template ROS 2 Node. It inherits from the base 'Node' class,
    which gives it access to logging, timers, publishers, and subscribers.
    """
    def __init__(self):
        # Initialize the base Node with the name of this node in the graph
        super().__init__('camera_bridge_node')

        self.bridge = CvBridge()

        # Depth buffer: store recent depth frames (cv images) with timestamps
        self.depth_buffer = deque(maxlen=50)
        self.depth_lock = threading.Lock()        
        
        # Log an informational message to the console upon startup
        self.get_logger().info('Camera Bridge Node has been initialized!')

        # -----------------------------------------------------------------
        # COMPONENT A: The Subscriber (Listening to the RealSense)
        # -----------------------------------------------------------------
        qos_profile = rclpy.qos.QoSProfile(depth=10)  # tune to sensor_data QoS
        # Arguments: (Message Type, Topic Name, Callback Function, Queue Size/QoS)
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
        
        # -----------------------------------------------------------------
        # COMPONENT B: The Publisher (Sending data forward)
        # -----------------------------------------------------------------
        # Arguments: (Message Type, Topic Name, Queue Size/QoS)
        self.pose_data_publisher = self.create_publisher(
            Image,
            '/perception/filtered_image',
            10
        )
        self.depth_burst_count = 25
        self.YOLO_MODEL_PATH = "06-09-2026.pt"
        self.PAD = 20  # YOLO bounding box 2D padding
        self.align = rs.align(rs.stream.color)
    
    def color_callback(self, msg: Image):
        """
        This function executes automatically every single time a new 
        message arrives on the subscribed topic.
        """
        
        self.get_logger().info('Received an image frame from the color camera.')
        
        # --- YOUR ALGORITHM GOES HERE ---
        # This is where your custom image processing or routing logic lives.
        try:
            color_cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'Color Conversion Failed: {e}')
        color_cv = cv2.cvtColor(color_cv, cv2.COLOR_BGR2RGB)
        color_cv = self.align.process(color_cv)
        bboxes = self.run_yolo_on_image(color_cv)
        
        
        processed_msg = msg 
        # ---------------------------------
        
        # Publish the data downstream to the next node
        self.pose_data_publisher.publish(processed_msg)


def run_yolo_on_image(self, image):
    # Placeholder for YOLO object detection logic
    yolo_model = YOLO(self.YOLO_MODEL_PATH)
    results = yolo_model(image, conf=0.60, verbose=False)[0]
    return results


def main(args=None):
    # 1. Initialize the ROS 2 communications context
    rclpy.init(args=args)
    
    # 2. Instantiate your custom node class
    node = CameraBridgeNode()
    
    try:
        # 3. Spin the node so its callbacks can execute continuously in the background
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Gracefully handle a Ctrl+C termination
        node.get_logger().info('Shutting down Camera Bridge Node gracefully.')
    finally:
        # 4. Destroy the node explicitly and clean up the context
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()