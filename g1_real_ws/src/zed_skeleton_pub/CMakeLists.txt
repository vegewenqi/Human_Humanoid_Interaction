cmake_minimum_required(VERSION 3.8)
project(zed_skeleton_pub)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ROS2
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# -----------------------------
# The dummy publisher (for quick sanity tests)
add_executable(dummy_skeleton_pub src/dummy_skeleton_pub.cpp)
ament_target_dependencies(dummy_skeleton_pub rclcpp sensor_msgs std_msgs)
install(TARGETS dummy_skeleton_pub DESTINATION lib/${PROJECT_NAME})

# -----------------------------
# Real ZED publisher node (builds the existing ZED C++ project sources)
# Reuse the original dependencies from zed_skeleton_pub/cpp/CMakeLists.txt
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(LINK_SHARED_ZED "Link with the ZED SDK shared executable" ON)

find_package(ZED REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} REQUIRED)
find_package(OpenCV REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)

# Source files from the existing ZED code
file(GLOB_RECURSE ZED_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cpp/src/*.c*)
file(GLOB_RECURSE ZED_HDR_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include/*.h*)

add_executable(zed_skeleton_pub_node
  ${ZED_SRC_FILES}
  ${ZED_HDR_FILES}
)

target_include_directories(zed_skeleton_pub_node PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${ZED_INCLUDE_DIRS}
  ${GLEW_INCLUDE_DIRS}
  ${GLUT_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)

# ZED libs (same logic as the original CMakeLists)
if (LINK_SHARED_ZED)
  set(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} ${CUDA_DEP_LIBRARIES_ZED})
else()
  set(ZED_LIBS ${ZED_STATIC_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_LIBRARY})
endif()

target_link_libraries(zed_skeleton_pub_node
  ${ZED_LIBS}
  ${OPENGL_LIBRARIES}
  ${GLUT_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${GLEW_LIBRARIES}
)

ament_target_dependencies(zed_skeleton_pub_node rclcpp sensor_msgs std_msgs)

install(TARGETS zed_skeleton_pub_node DESTINATION lib/${PROJECT_NAME})

ament_package()