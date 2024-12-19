import cv2
from Detector import Detector

if __name__ == '__main__':
    cap = cv2.VideoCapture("test.mp4")
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter("test.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Initialize SpeedEstimator
    speed_obj = Detector(
        classid=0,
        model="yolo11n.pt",
        show=True,
        tracker="bytetrack.yaml"    # or botsort.yaml
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