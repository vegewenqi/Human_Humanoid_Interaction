cmake_minimum_required(VERSION 3.8)
project(real_g1)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(unitree_hg REQUIRED)
find_package(unitree_go REQUIRED)

include_directories(
  /repos/unitree_ros2/example/src/include
)

add_executable(g1_arm_sdk_bridge
  src/g1_arm_sdk_bridge.cpp
)

add_executable(inspire_hand_bridge
  src/inspire_hand_bridge.cpp
)

ament_target_dependencies(g1_arm_sdk_bridge
  rclcpp
  std_msgs
  sensor_msgs
  unitree_hg
)

ament_target_dependencies(inspire_hand_bridge
  rclcpp
  std_msgs
  unitree_go
)

install(TARGETS
  g1_arm_sdk_bridge
  inspire_hand_bridge
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
