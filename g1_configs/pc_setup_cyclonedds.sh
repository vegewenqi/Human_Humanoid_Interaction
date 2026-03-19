#!/bin/bash
set -e

CONTAINER_NAME="g1_real"
ROS_SETUP="/opt/ros/humble/setup.bash"
CYCLONE_XML="/configs/pc_cyclonedds.xml"

if ! docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "[ERROR] Docker container '$CONTAINER_NAME' does not exist."
  echo "Create it first, or edit CONTAINER_NAME in this script."
  exit 1
fi

if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME")" != "true" ]; then
  echo "[INFO] Starting container: $CONTAINER_NAME"
  docker start "$CONTAINER_NAME" >/dev/null
fi

exec docker exec -it \
  -e ROS_DOMAIN_ID=30 \
  -e ROS_LOCALHOST_ONLY=0 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e CYCLONEDDS_URI="file://${CYCLONE_XML}" \
  "$CONTAINER_NAME" \
  bash -lc "source '${ROS_SETUP}' && \
            unset ROS_DISCOVERY_SERVER && \
            echo '--------------------------------------------------------' && \
            echo 'CycloneDDS environment ready inside Docker' && \
            echo 'RMW_IMPLEMENTATION : ' \"\$RMW_IMPLEMENTATION\" && \
            echo 'ROS_DOMAIN_ID      : ' \"\$ROS_DOMAIN_ID\" && \
            echo 'CYCLONEDDS_URI     : ' \"\$CYCLONEDDS_URI\" && \
            echo '--------------------------------------------------------' && \
            exec bash"