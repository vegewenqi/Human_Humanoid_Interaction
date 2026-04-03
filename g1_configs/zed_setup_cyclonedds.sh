#!/bin/bash
# Use with:
# source ~/Projects/Human_Humanoid_Interaction/g1_configs/zed_setup_cyclonedds.sh

unset ROS_DISCOVERY_SERVER
unset FASTRTPS_DEFAULT_PROFILES_FILE

export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CYCLONEDDS_URI="file://${SCRIPT_DIR}/zed_cyclonedds.xml"

# Attach to Jetson display
export DISPLAY=:0
export XAUTHORITY=/run/user/1000/gdm/Xauthority

source /opt/ros/humble/setup.bash
source ~/Projects/Human_Humanoid_Interaction/g1_real_ws/install/setup.bash

echo "--------------------------------------------------------"
echo "CycloneDDS environment ready on ZED"
echo "RMW_IMPLEMENTATION : $RMW_IMPLEMENTATION"
echo "ROS_DOMAIN_ID      : $ROS_DOMAIN_ID"
echo "CYCLONEDDS_URI     : $CYCLONEDDS_URI"
echo "--------------------------------------------------------"