from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os




def generate_launch_description():

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                '/opt/ros/humble/share/realsense2_camera/launch/rs_launch.py'
            )
        )
    )

    foxglove_bridge = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        parameters=[{
            'address': '0.0.0.0',
            'port': 8765,
            'send_buffer_limit': 10000000,
            'use_sim_time': False
        }]
    )

    camera_bridge = Node(
        package='gearbox_perception',
        executable='camera_bridge',
        output='screen'
    )

    registration = Node(
        package='gearbox_perception',
        executable='gearbox_registration',
        output='screen'
    )

    return LaunchDescription([
        realsense_launch,
        foxglove_bridge,
        camera_bridge,
        registration
    ])