#!/bin/bash
# 1. clean up ROS2 env variables to avoid conflicts
unset ROS_DISCOVERY_SERVER
unset RMW_IMPLEMENTATION

# 2. set CycloneDDS core variables
export ROS_DOMAIN_ID=30
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# 3. specify CycloneDDS configuration file path
export CYCLONEDDS_URI=file:///home/$USER/Projects/Human_Humanoid_Interaction/g1_configs/zed_cyclonedds.xml

# 4. print status information feedback
echo "--------------------------------------------------------"
echo "✅ Communication Environment Activated!"
echo "RMW_IMPLEMENTATION : $RMW_IMPLEMENTATION"
echo "ROS_DOMAIN_ID      : $ROS_DOMAIN_ID"
echo "CYCLONEDDS_URI     : $CYCLONEDDS_URI"
echo "--------------------------------------------------------"
