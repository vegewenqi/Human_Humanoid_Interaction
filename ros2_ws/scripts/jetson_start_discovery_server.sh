#!/usr/bin/env bash
set -e
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=30
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
exec fastdds discovery -i 0 -p 11811
