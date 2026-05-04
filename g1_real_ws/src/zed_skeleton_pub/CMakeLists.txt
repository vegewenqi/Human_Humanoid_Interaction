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
find_package(geometry_msgs REQUIRED)

# -----------------------------
# The dummy publisher (for quick sanity tests)
add_executable(dummy_skeleton_pub src/dummy_skeleton_pub.cpp)
ament_target_dependencies(dummy_skeleton_pub rclcpp sensor_msgs std_msgs)
install(TARGETS dummy_skeleton_pub DESTINATION lib/${PROJECT_NAME})

# -----------------------------
# ZED / OpenCV / OpenGL deps
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
find_package(PkgConfig REQUIRED)
pkg_check_modules(APRILTAG REQUIRED apriltag)

# Headers from the existing ZED code
file(GLOB_RECURSE ZED_HDR_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include/*.h*)

# Shared implementation files only.
set(ZED_COMMON_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/src/GLViewer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/src/TrackingViewer.cpp
)

# ZED libs (same logic as the original CMakeLists)
if(LINK_SHARED_ZED)
  set(ZED_LIBS
    ${ZED_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
    ${CUDA_CUDART_LIBRARY}
    ${CUDA_DEP_LIBRARIES_ZED}
  )
else()
  set(ZED_LIBS
    ${ZED_STATIC_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
    ${CUDA_LIBRARY}
  )
endif()

# -----------------------------
# Main skeleton node
add_executable(zed_skeleton_pub_node
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/src/main.cpp
  ${ZED_COMMON_SRC}
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

target_link_libraries(zed_skeleton_pub_node
  ${ZED_LIBS}
  ${OPENGL_LIBRARIES}
  ${GLUT_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${GLEW_LIBRARIES}
)

ament_target_dependencies(zed_skeleton_pub_node
  rclcpp sensor_msgs std_msgs
)

install(TARGETS zed_skeleton_pub_node DESTINATION lib/${PROJECT_NAME})

# -----------------------------
# Fusion skeleton node: dual ZED X + ZED Fusion API
add_executable(fusion_zed_skeleton_pub_node
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/src/fusion_main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/src/FusionGLViewer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/src/TrackingViewer.cpp
  ${ZED_HDR_FILES}
)

target_include_directories(fusion_zed_skeleton_pub_node PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${ZED_INCLUDE_DIRS}
  ${GLEW_INCLUDE_DIRS}
  ${GLUT_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)

target_link_libraries(fusion_zed_skeleton_pub_node
  ${ZED_LIBS}
  ${OPENGL_LIBRARIES}
  ${GLUT_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${GLEW_LIBRARIES}
)

ament_target_dependencies(fusion_zed_skeleton_pub_node
  rclcpp sensor_msgs std_msgs
)

install(TARGETS fusion_zed_skeleton_pub_node DESTINATION lib/${PROJECT_NAME})

# -----------------------------
# Calibration node
add_executable(zed_calibration_node
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/src/calibration_main.cpp
  ${ZED_COMMON_SRC}
  ${ZED_HDR_FILES}
)

target_include_directories(zed_calibration_node PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${ZED_INCLUDE_DIRS}
  ${GLEW_INCLUDE_DIRS}
  ${GLUT_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
  ${APRILTAG_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)

target_link_libraries(zed_calibration_node
  ${ZED_LIBS}
  ${OPENGL_LIBRARIES}
  ${GLUT_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${GLEW_LIBRARIES}
  ${APRILTAG_LIBRARIES}
)

ament_target_dependencies(zed_calibration_node
  rclcpp sensor_msgs std_msgs geometry_msgs
)

install(TARGETS zed_calibration_node DESTINATION lib/${PROJECT_NAME})

ament_package()

# -----------------------------
# Calibration node (fusion version)
add_executable(fusion_zed_calibration_node
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/src/fusion_calibration_main.cpp
  ${ZED_COMMON_SRC}
  ${ZED_HDR_FILES}
)

target_include_directories(fusion_zed_calibration_node PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${ZED_INCLUDE_DIRS}
  ${GLEW_INCLUDE_DIRS}
  ${GLUT_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
  ${APRILTAG_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)

target_link_libraries(fusion_zed_calibration_node
  ${ZED_LIBS}
  ${OPENGL_LIBRARIES}
  ${GLUT_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${GLEW_LIBRARIES}
  ${APRILTAG_LIBRARIES}
)

ament_target_dependencies(fusion_zed_calibration_node
  rclcpp sensor_msgs std_msgs geometry_msgs
)

install(TARGETS fusion_zed_calibration_node DESTINATION lib/${PROJECT_NAME})