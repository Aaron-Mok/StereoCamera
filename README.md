# Stereo Camera System from Scratch

**Author:** Aaron Mok  
**Status:** 🚧 Ongoing Project  
**Keywords:** Stereo Vision, Raspberry Pi HQ Camera, Depth Estimation, ISP Pipeline, Distortion Correction, Python, OpenCV, 3D Printing

## Overview

This project aims to build a complete **stereo camera system from scratch** using two Raspberry Pi HQ cameras. The system captures synchronized raw Bayer images, processes them through a custom image signal processing (ISP) pipeline, and performs stereo calibration and depth estimation. It also includes the design and fabrication of a **custom 3D-printed mount** to ensure precise camera alignment.

The project emphasizes **understanding and implementing each step** of the stereo vision pipeline, including distortion correction, to enable accurate and consistent depth reconstruction.

## Features

- 📷 **Raw Image Capture** from dual Raspberry Pi HQ cameras  
- 🧰 **Custom ISP Pipeline** including:
  - Stride/padding correction
  - Black level subtraction
  - Lens shading correction
  - Distortion correction
  - Demosaicing
  - White balance
  - Color calibration
  - Gamma correction
- 🔧 **Stereo Calibration & Rectification**
- 🌄 **Disparity & Depth Map Generation**
- 🖨️ **Custom 3D-Printed Mount** for accurate and stable stereo alignment
- 🧪 Real-time visualization and debugging tools
- 📦 Modular and extensible Python codebase

## Motivation

Modern stereo vision systems often hide their internal processes. This project aims to **demystify** the full imaging and stereo pipeline—bridging camera hardware, optics, and software—from first principles. It serves both as a learning platform and as a foundation for future computer vision or robotics applications.

## Current Progress
- [x] Image signal processing for raw capture
  - [ ] Stride/padding correction
  - Black level subtraction
  - Lens shading correction
  - Distortion correction
  - Demosaicing
  - White balance
  - Color calibration
  - Gamma correction
- [ ] Initial stereo calibration
- [ ] Real-time disparity map viewer
- [ ] 3D-printed mount design and testing
- [ ] Benchmarking under different lighting conditions

## Demo
Coming soon: live video walkthrough, depth map overlay, and stereo reconstruction results.

## About Me
I am an imaging system architect at KLA with a PhD from Cornell University.

## License
MIT License
