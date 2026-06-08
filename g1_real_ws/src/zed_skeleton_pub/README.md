# ZEDposeDetect

This project uses the **ZED SDK** and **CUDA** to detect and track human bodies in real time.  
It calculates distances between specific joints for two detected people and visualizes the results on a 2D video stream.

## Prerequisites

- [Ubuntu 22.04](https://releases.ubuntu.com/jammy/)
- [ZED SDK](https://www.stereolabs.com/developers/)
- [CUDA](https://developer.nvidia.com/cuda-toolkit) (ZED SDK installation will prompt to install CUDA for you if you don't have it already)
- [OpenCV](https://opencv.org/)

```bash
# Install dependencies and build tools for C++ implementation
sudo apt install -y libopencv-dev freeglut3-dev libglew-dev libgl1-mesa-dev libglu1-mesa-dev cmake build-essential
```

```bash
# Install all of the dependencies for Python implementation
pip install -r requirements.txt
```

## ROS 2 topics

`zed_skeleton_pub_node` publishes the body skeleton topics and, by default, a
compressed left camera image for MediaPipe hand perception:

- `/skeleton/points`
- `/skeleton/confidence`
- `/image/compressed`

Useful image parameters:

```bash
ros2 run zed_skeleton_pub zed_skeleton_pub_node --ros-args \
  -p publish_left_image:=true \
  -p image_topic:=/image/compressed \
  -p image_publish_every_n:=2 \
  -p image_publish_width:=640 \
  -p image_jpeg_quality:=80
```
