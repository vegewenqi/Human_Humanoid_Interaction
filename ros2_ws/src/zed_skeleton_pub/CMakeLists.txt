cmake_minimum_required(VERSION 3.8)
project(zed_skeleton_pub)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# You will likely need to add:
# - ZED SDK include/link
# - OpenCV include/link
# - CUDA if your project uses it
#
# For now we build a small "dummy publisher" to validate ROS2 build/run.
add_executable(dummy_skeleton_pub src/dummy_skeleton_pub.cpp)
ament_target_dependencies(dummy_skeleton_pub rclcpp sensor_msgs std_msgs)

install(TARGETS dummy_skeleton_pub
  DESTINATION lib/${PROJECT_NAME})

ament_package()
