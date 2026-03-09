#!/usr/bin/env bash
set -e

# ROS2 env
source /opt/ros/humble/setup.bash
source ~/Projects/Human_Humanoid_Interaction/ros2_ws/install/setup.bash

# ROS2 network config
export ROS_DOMAIN_ID=30
export ROS_LOCALHOST_ONLY=0
export ROS_DISCOVERY_SERVER=10.224.34.217:11811
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Attach to Jetson display
export DISPLAY=:0
export XAUTHORITY=/run/user/1000/gdm/Xauthority

# (Optional but helps when switching discovery modes)
ros2 daemon stop || true

exec ros2 run zed_skeleton_pub zed_skeleton_pub_node
