# Stereo Camera System

**Author:** Aaron Mok  
**Status:** 🚧 Ongoing Project  
**Keywords:** Optics, Stereo Vision, Raspberry Pi HQ Camera, Depth Estimation, ISP Pipeline, Distortion Correction, Python, OpenCV, 3D Printing, CADing

## Overview

This project aims to build a complete **stereo camera system from scratch** using two Raspberry Pi HQ cameras. The system captures synchronized raw Bayer images, processes them through a custom image signal processing (ISP) pipeline, and performs stereo calibration and depth estimation. It also includes the design and fabrication of a **custom 3D-printed mount** to ensure precise camera alignment.

This project showcases the complete stereo vision pipeline, from lens and sensor selection to camera alignment and image processing, with the goal of achieving accurate and reliable depth reconstruction.

## Features

- 📷 **Raw Image Capture** from dual Raspberry Pi HQ cameras  
- 🧰 **Custom ISP Pipeline** including:
  - Stride/padding correction
  - Black level subtraction
  - Distortion correction
  - Lens shading correction
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
  - [x] Stride/padding correction
  - [ ] Black level subtraction
  - [ ] Distortion correction
  - [ ] Lens shading correction
  - [x] Demosaicing
  - [x] White balance
  - [ ] Color calibration
  - [x] Gamma correction
- [ ] 3D-printed mount design and testing
- [ ] Initial stereo calibration
- [ ] Real-time disparity map viewer

## Demo
Coming soon: live video walkthrough, depth map overlay, and stereo reconstruction results.

## About Me
I am an Imaging System Architect at KLA, where I lead the development of advanced optical and imaging subsystems for high-precision inspection tools. I hold a PhD in Biomedical Engineering from Cornell University, where my research focused on developing next-generation optical imaging systems for deep-tissue brain imaging.

🔗 See my work and blog at [https://aaron-mok.github.io/](https://aaron-mok.github.io/).

## License
This project is licensed under the **GNU General Public License v3.0**. 
