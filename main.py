import cv2
import torch
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slowfast_r50_detection
from Detector import Detector
import numpy as np
import pyrealsense2 as rs
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    video_model = slowfast_r50_detection(True).eval().to(device)

    ava_labelnames, _ = AvaLabeledVideoFramePaths.read_label_map("configs/ava_action_list.pbtxt")
    is_parallel = False # 并行化目前无法正常运行

    speed_obj = Detector(
        classid=0,
        model="models/yolo11n-seg.pt", # or yolo11n.onnx # yolo11n_openvino_model
        conf= 0.25, # yolo的检测置信度
        show=True,
        tracker="bytetrack.yaml",    # or botsort.yaml
        showmask=False,
        device=device,
        slowfast=video_model,   # slowfast模型 or None
        ava_labels=ava_labelnames,
        detect_interval=50,  # 设定slowfast触发频率（/fps）
        deque_length=1,  # slowfast的输入帧队列长度
        is_parallel=is_parallel,
    )

    pipeline = rs.pipeline()
    align_to = rs.stream.color
    align = rs.align(align_to)
    config = rs.config()
    D455_imgWidth, D455_imgHeight = 640, 480
    config.enable_stream(rs.stream.color, D455_imgWidth, D455_imgHeight, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, D455_imgWidth, D455_imgHeight, rs.format.z16, 30)

    pipeline.start(config)
    profile = pipeline.get_active_profile()
    # 获取深度相机的内参
    depth_stream_profile = profile.get_stream(rs.stream.depth)
    intrinsics = depth_stream_profile.as_video_stream_profile().get_intrinsics()
    fx, fy, cx, cy = intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
    print("相机内参：",f"fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")

    if is_parallel:
        speed_obj.parallel_run(pipeline, align, fx, fy, cx, cy)
    else:
        while True:
            frames = pipeline.wait_for_frames()
            # RGB-D 对齐
            aligned_frames = align.process(frames)
            aligned_color_frame = aligned_frames.get_color_frame()
            aligned_depth_frame = aligned_frames.get_depth_frame()
            if not aligned_depth_frame or not aligned_color_frame:
                raise Exception("[info] No D455 data.")

            rgb = np.asanyarray(aligned_color_frame.get_data())
            d = np.asanyarray(aligned_depth_frame.get_data())

            # 在 Detector 的 estimate 中生成点云并完成速度计算
            annotated_frame = speed_obj.estimate(rgb, d, fx, fy, cx, cy)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break