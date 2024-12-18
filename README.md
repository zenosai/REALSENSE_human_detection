## 简介

使用 RealsenseD435i 相机, 设计自顶向下的识别方法，先通过 yolo 利用纯 RGB 信息识别行人，然后将行人提取框在深度图中的对应部分进行深度变化识别，可区分照片和真人。在检测到人形目标后，计算距离、移动速度，可区分目标行为，包括：站立、坐下、倒地、行走

算法分析位于Analysis.md中

## 配置

conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

pip install opencv-python pyrealsense2 deep-sort-realtime ultralytics

## 使用
