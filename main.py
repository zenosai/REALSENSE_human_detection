import cv2
from Detector import Detector
import numpy as np
import pyrealsense2 as rs
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    speed_obj = Detector(
        classid=0,
        model="models/yolo11n-seg.pt", # or yolo11n.onnx # yolo11n_openvino_model
        show=True,
        tracker="bytetrack.yaml",    # or botsort.yaml
        # device='xpu'
        # slowfast=video_model,
        # ava_labels=ava_labelnames,
        # device=device,
        # detect_interval=25  # 设定slowfast触发频率（/fps）
    )

    pipeline = rs.pipeline()
    align_to = rs.stream.color
    align = rs.align(align_to)
    config = rs.config()
    D455_imgWidth, D455_imgHeight = 640, 480
    config.enable_stream(rs.stream.color, D455_imgWidth, D455_imgHeight, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, D455_imgWidth, D455_imgHeight, rs.format.z16, 30)

    profile = pipeline.start(config)

    # 相机内参示例
    fx, fy, cx, cy = 600, 600, 320, 240

    while True:
        frames = pipeline.wait_for_frames()
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