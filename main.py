import cv2
import torch
from Detector import Detector
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slowfast_r50_detection
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cap = cv2.VideoCapture("test2.mp4")
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter("test2.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    video_model = slowfast_r50_detection(True).eval().to(device)
    ava_labelnames, _ = AvaLabeledVideoFramePaths.read_label_map("configs/ava_action_list.pbtxt")

    speed_obj = Detector(
        classid=0,
        model="models/yolo11n.pt", # or yolo11n.onnx
        show=True,
        tracker="bytetrack.yaml",    # or botsort.yaml
        slowfast=video_model,
        ava_labels=ava_labelnames,
        device=device,
        detect_interval=25  # 设定slowfast触发频率（/fps）
    )

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            break
        im0 = speed_obj.estimate(im0)
        video_writer.write(im0)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()