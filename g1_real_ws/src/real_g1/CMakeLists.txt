cmake_minimum_required(VERSION 3.8)
project(real_g1)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

set(UNITREE_SDK2_DIR /repos/unitree_sdk2)
set(UNITREE_ARCH x86_64)

include_directories(
  ${UNITREE_SDK2_DIR}/include
  ${UNITREE_SDK2_DIR}/thirdparty/include
  ${UNITREE_SDK2_DIR}/thirdparty/include/ddscxx
)

link_directories(
  ${UNITREE_SDK2_DIR}/lib/${UNITREE_ARCH}
  ${UNITREE_SDK2_DIR}/thirdparty/lib/${UNITREE_ARCH}
)

add_executable(g1_arm_sdk_bridge
  src/g1_arm_sdk_bridge.cpp
)

ament_target_dependencies(g1_arm_sdk_bridge
  rclcpp
  std_msgs
)

target_link_libraries(g1_arm_sdk_bridge
  unitree_sdk2
  ddscxx
  ddsc
)

install(TARGETS
  g1_arm_sdk_bridge
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()