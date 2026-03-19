#!/bin/bash
# Use with:
# source ~/Projects/Human_Humanoid_Interaction/g1_configs/zed_setup_cyclonedds.sh

unset ROS_DISCOVERY_SERVER
unset FASTRTPS_DEFAULT_PROFILES_FILE

export ROS_DOMAIN_ID=30
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CYCLONEDDS_URI="file://${SCRIPT_DIR}/zed_cyclonedds.xml"

if [ -f /opt/ros/humble/setup.bash ]; then
  source /opt/ros/humble/setup.bash
fi

echo "--------------------------------------------------------"
echo "CycloneDDS environment ready on ZED"
echo "RMW_IMPLEMENTATION : $RMW_IMPLEMENTATION"
echo "ROS_DOMAIN_ID      : $ROS_DOMAIN_ID"
echo "CYCLONEDDS_URI     : $CYCLONEDDS_URI"
echo "--------------------------------------------------------"