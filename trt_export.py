import cv2
import torch
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slowfast_r50_detection
from Detector import Detector
import numpy as np
import pyrealsense2 as rs
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# torch: 1254.3 ms

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    video_model = slowfast_r50_detection(True).eval().to(device)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    # 导出为 ONNX 格式
    onnx_path = "slowfast_r50_detection.onnx"
    torch.onnx.export(video_model,  # 模型
                      dummy_input,  # 模型输入
                      "models/slowfast.onnx",  # 输出文件路径
                      export_params=True,  # 导出模型参数
                      opset_version=13,  # ONNX 版本
                      do_constant_folding=True,  # 是否进行常量折叠优化
                      input_names=['input'],  # 输入名称
                      output_names=['output'],  # 输出名称
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})  # 支持动态 batch size

    print(f"模型已导出为 {onnx_path}")
