cmake_minimum_required(VERSION 3.8)
project(gearbox_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 1. Find build tools and interface dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# 2. Tell ROS2 which custom files to generate
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/EstimatePose.srv"
  "action/ExecuteTask.action"
  DEPENDENCIES geometry_msgs std_msgs
)

# 3. Export package
ament_package()