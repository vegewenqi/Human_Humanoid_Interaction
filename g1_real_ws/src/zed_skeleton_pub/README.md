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
