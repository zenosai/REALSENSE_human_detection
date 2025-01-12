## 简介
华科数字图像处理综合设计与实验课程深度相机实验代码。

实验要求：使用 RealsenseD455 相机，设计自顶向下的识别方法，先通过 yolo 利用纯 RGB 信息识别行人，然后将行人提取框在深度图中的对应部分进行深度变化识别，可区分照片和真人。在检测到人形目标后，计算距离、移动速度，可区分目标行为，包括：站立、坐下、倒地、行走

算法分析位于Analysis.md中

## 配置

创建conda环境
```
conda create -n human-detection python=3.8
conda activate human-detection
```
安装依赖
```
conda install pytorch torchvision==0.13.0 pytorch-cuda -c pytorch -c nvidia
pip install ultralytics shapely lap onnx>=1.12.0 onnxslim onnxruntime -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pytorchvideo -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install opencv-python pyrealsense2 pyro4 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 使用
运行main.py，通过直接修改源码中Detector的输入参数进行效果调节

可以选择令is_parallel变量为True，激活异步并行推理模式。并行化需要先运行async_server.py文件，然后将该文件返回的PYRO uri 填入main.py的对应位置，然后再启动main.py
