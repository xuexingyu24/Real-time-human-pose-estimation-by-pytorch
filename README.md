# Real-time-human-pose-estimation-by-pytorch

A reimplementation of real time human pose estimation using Pytorch. Both [CMU's original model](https://arxiv.org/abs/1611.08050) and [mobilenet version](https://arxiv.org/pdf/1811.12004.pdf) backbone are provided. 

Instead of resizing heat maps and PAF maps to orignial input image, this code upsamples the feature map with factor 8 and then performed the NMS for peak location, bigartite grahps, line integral and grouping. 

**You can reach ~ 15 frames per second using GPU**

## Demo

CPU: Intel® Xeon(R) W-2123 @ 3.60GHz × 8 
GPU: Nvidia Quadro P4000 8G
Input Image: 640x480

| CMU's Original Model</br> on CPU | Mobilenet </br> on CPU |
|:---------|:--------------------|
|<img src="images/CPU_CMU.gif"  width="300">|<img src="images/CPU_Mobilenet.gif"  width="300" >|
| **~4 FPS** | **~8 FPS** |

| CMU's Original Model</br> on GPU | Mobilenet </br> on GPU |
|:---------|:--------------------|
|<img src="images/GPU_CMU.gif"  width="300">|<img src="images/GPU_Mobilenet.gif"  width="300" >|
| **~10 FPS** | **~15 FPS** |
