from collections import deque
from time import time
import cv2
import numpy as np
import torch
from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors
from AvaUtils import ava_inference_transform
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor


class Detector(BaseSolution):
    """
    目标识别+轨迹追踪，继承自 BaseSolution 类

    属性:
        spd (Dict[int, float]): 存储被追踪物体的速度数据。
        trkd_ids (List[int]): 存储已经估算过速度的被追踪物体 ID。
        trk_pt (Dict[int, float]): 存储已追踪物体的上一个时间戳。
        trk_pp (Dict[int, Tuple[float, float]]): 存储已追踪物体的上一个位置。
        annotator (Annotator): 用于在图像上绘制标注的对象。
        track_line (List[Tuple[float, float]]): 存储物体轨迹的点列表。

    方法:
        extract_tracks: 提取当前帧中的轨迹。
        store_tracking_history: 存储物体的轨迹历史。
        display_output: 显示带有标注的输出图像。
    """

    def __init__(self, ava_labels, detect_interval,pyro_model=None,deque_length=25, slowfast=None, is_parallel=False,
                 device="cpu", classid=0, showmask=False, **kwargs):
        super().__init__(**kwargs)

        self.classid = classid  # 要识别的类别（本课设识别人，默认填 0 即可）
        self.showmask = showmask
        self.center_buffer = []       # 用于存储最近 5 帧中心位置
        self.avg_position_old = None  # 用于存储上一帧的平均位置
        self.last_time = None         # 上一次计算平均位置的时刻
        self.spd = {}  # 存储速度数据
        self.trkd_ids = []  # 存储已经估算速度的物体 ID 列表
        self.trk_pt = {}  # 存储物体上一个时间戳
        self.trk_pp = {}  # 存储物体上一个位置
        # 每个 track_id 对应一个中心点队列、旧平均位置、最后记录时间
        self.track_centers = {}       # { track_id: [pos1, pos2, ...] }
        self.track_avg_position_old = {}  
        self.track_last_time = {}
        self.img_stack = deque(maxlen=deque_length)
        self.device=device
        self.action_labels = {}
        self.frame_count = detect_interval//2
        self.detect_interval = detect_interval
        self.slowfast = slowfast
        self.ava_labels = ava_labels
        self.slowfast_flag = 0
        self.normal_flag = 0
        self.is_parallel = is_parallel
        self.slowfast_session = None
        self.normal_session = None
        self.pyro_model = pyro_model


    def find_indices(self, lst, target):
        """
        返回列表中对应数的索引

        参数:
            lst (list): 输入列表
            target (int ): 要索引的数字
        返回:
            (list): 要索引的数字在lst中的对应下标
        """
        indices = [index for index, value in enumerate(lst) if value == target]
        return indices

    def get_clips(self):
        """
        返回tensor后的clip stack
        """
        clips = [torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).unsqueeze(0) for img in self.img_stack]
        clips = torch.cat(clips).permute(-1, 0, 1, 2)
        return clips

    def slowfast_inference(self, frame_count, track_ids, boxes, get_clips):
        # pyro4版本:
        boxes = np.array(boxes).tolist()
        get_clips = np.array(get_clips).tolist()
        return self.pyro_model.slowfast_inference(frame_count, track_ids, boxes, get_clips)

        # 线程管理版本:
        # self.slowfast_flag = 1
        # if frame_count % self.detect_interval == 0:
        #     inputs, inp_boxes, _ = ava_inference_transform(get_clips, boxes)
        #     inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0], 1), inp_boxes], dim=1)
        #     if isinstance(inputs, list):
        #         inputs = [inp.unsqueeze(0).to(self.device) for inp in inputs]
        #     else:
        #         inputs = inputs.unsqueeze(0).to(self.device)
        #     with torch.no_grad():
        #         slowfaster_preds = self.slowfast(inputs, inp_boxes.to(self.device))
        #         slowfaster_preds = slowfaster_preds.cpu()
        #
        #     for id, avalabel in zip(track_ids, np.argmax(slowfaster_preds, axis=1).tolist()):
        #         self.action_labels[id] = self.ava_labels[avalabel + 1]
        # self.slowfast_flag = 0

        # fastapi调用版本:
        # headers = {
        #     "Content-Type": "application/json"
        # }
        # # FastAPI 服务的 URL
        # url = "http://127.0.0.1:8000/predict/"
        #
        # data = {
        #     "frame_count": frame_count,
        #     "track_ids": track_ids,
        #     "boxes": boxes.cpu().numpy().tolist(),
        #     "get_clips": get_clips.numpy().tolist()
        # }
        #
        # # 发送 POST 请求到 FastAPI 服务器
        # response = requests.post(url, data=data, headers=headers)
        # # 输出响应内容
        # if response.status_code == 200:
        #     # 如果请求成功，解析 JSON 响应
        #     result = response.json()
        #     self.action_labels = result['prediction']
        #     print(f"Prediction: {result['prediction']}")
        # return self.action_labels


    def speed_pos_estimate(self, roi_cloud, box, track_id):
        label = None
        center_3d = self.get_3d_center(roi_cloud)
        if center_3d is not None:
            # 初始化当前 track_id 的中心队列
            if track_id not in self.track_centers:
                self.track_centers[track_id] = []
                self.track_avg_position_old[track_id] = None
                self.track_last_time[track_id] = None

            # 历史中心队列中追加新位置
            self.track_centers[track_id].append(center_3d)
            if len(self.track_centers[track_id]) > 5:
                self.track_centers[track_id].pop(0)

            # 收集到 5 帧中心后进行速度计算
            if len(self.track_centers[track_id]) == 5:
                new_avg = np.mean(self.track_centers[track_id], axis=0)
                current_time = cv2.getTickCount() / cv2.getTickFrequency()
                old_avg = self.track_avg_position_old[track_id]
                last_time = self.track_last_time[track_id]
                if old_avg is not None and last_time is not None:
                    dt = current_time - last_time
                    delta_pos = new_avg - old_avg
                    speed_3d = delta_pos / dt
                    speed_mag = np.linalg.norm(speed_3d)
                    label = f"ID:{track_id}, Pos:[{new_avg[0]:.2f},{new_avg[1]:.2f},{new_avg[2]:.2f}], Speed:{speed_mag:.2f}m/s"
                self.track_avg_position_old[track_id] = new_avg
                self.track_last_time[track_id] = current_time
        return label

    def extract_tracks(self, im0):
        """
        Applies object tracking and extracts tracks from an input image or frame.

        Args:
            im0 (ndarray): The input image or frame.

        Examples:
            >>> solution = BaseSolution()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> solution.extract_tracks(frame)
        """
        self.tracks = self.model.track(source=im0, persist=True, classes=self.CFG["classes"], **self.track_add_args)

        # Extract tracks for OBB or object detection
        self.track_data = self.tracks[0].obb or self.tracks[0].boxes

        if self.track_data and self.track_data.id is not None:
            self.boxes = self.track_data.xyxy.cpu()
            self.clss = self.track_data.cls.cpu().tolist()
            self.track_ids = self.track_data.id.int().cpu().tolist()
        else:
            self.boxes, self.clss, self.track_ids = [], [], []
        return self.tracks, self.track_data, self.boxes, self.clss, self.track_ids

    def estimate(self, rgb, depth, fx, fy, cx, cy, executor=None):
        self.normal_flag = 1
        self.annotator = Annotator(rgb, line_width=self.line_width)

        cloud = self.create_point_cloud_from_depth_image(depth, fx, fy, cx, cy)
        if not hasattr(self, 'last_center_3d'):
            self.last_center_3d = None
        if not hasattr(self, 'last_time'):
            self.last_time = None

        # yolo检测，并记录历史RGB信息，供给action检测
        # if self.is_parallel:
        #     if self.normal_session is None or self.normal_session.done():
        #         self.normal_session = executor.submit(self.extract_tracks, rgb)
        #         while not self.normal_session.done(): pass
        #         self.tracks, self.track_data, self.boxes, self.clss, self.track_ids = self.normal_session.result()
        # else:
        #     self.extract_tracks(rgb)

        self.extract_tracks(rgb)

        self.img_stack.append(rgb) # 入栈
        self.frame_count += 1

        # 检测类别过滤
        indices = self.find_indices(self.clss, self.classid)
        if len(indices) == 0:   # 没有检测到clss类别
            self.display_output(rgb)
            return rgb

        self.masks = [mask for mask in self.tracks[0].masks[indices].data]
        self.roi_clouds = [cloud[mask.cpu().numpy().astype(bool).flatten()] for mask in self.tracks[0].masks[indices].data]
        self.boxes = self.boxes[indices]
        self.track_ids = [self.track_ids[i] for i in indices]
        self.clss = [self.clss[i] for i in indices]

        # 活体检测
        if self.real_person_detect() == []:
            self.display_output(rgb)
            return rgb

        # 行为检测
        if self.is_parallel:
            if self.slowfast_session is not None and self.slowfast_session.done():
                self.action_labels = self.slowfast_session.result()
                # print(self.action_labels)
            if (self.slowfast_session is None or self.slowfast_session.done()) and self.frame_count % self.detect_interval == 0:
                self.frame_count=0
                self.slowfast_session = executor.submit(self.slowfast_inference, self.frame_count, self.track_ids, self.boxes,
                                self.get_clips())
        else:
            if self.frame_count % self.detect_interval == 0:
                self.frame_count=0
                start_time = time()
                self.slowfast_inference(self.frame_count, self.track_ids, self.boxes, self.get_clips())
                print(f"行为检测用时: {(time() - start_time)*1000:.1f} ms")

        # 绘制label
        for mask, roi_cloud, box, track_id, cls in zip(self.masks, self.roi_clouds, self.boxes, self.track_ids, self.clss):
            self.store_tracking_history(track_id, box)  # 存储物体的轨迹历史
            # 速度及位置检测
            label = self.speed_pos_estimate(roi_cloud, box, track_id)
            if label is None:
                label = self.names[int(cls)]

            # 如果该 track_id 还没有记录时间戳或位置，则初始化
            if track_id not in self.trk_pt:
                self.trk_pt[track_id] = 0
            if track_id not in self.trk_pp:
                self.trk_pp[track_id] = self.track_line[-1]

            if track_id in self.action_labels:
                label += " Action:"+self.action_labels[track_id]

            self.annotator.box_label(box, label=label, color=colors(track_id, True))  # 绘制边界框

            # 绘制物体的轨迹
            self.annotator.draw_centroid_and_tracks(
                self.track_line, color=colors(int(track_id), True), track_thickness=self.line_width)

            self.trk_pt[track_id] = time()
            self.trk_pp[track_id] = self.track_line[-1]

        # 绘制mask
        if self.showmask:
            self.annotator.masks(torch.stack(self.masks).to(self.device).squeeze(dim=1),
                                 colors=[colors(idx, True) for idx in self.track_ids],
                                 im_gpu=torch.tensor(rgb, device=self.device).permute(2, 0, 1), alpha=0.1)

        self.display_output(rgb)  # 使用基类方法显示输出图像
        self.normal_flag = 0
        return rgb


    def create_point_cloud_from_depth_image(self, depth, fx, fy, cx, cy, scale=1000.0):
        h, w = depth.shape
        xmap = np.arange(w)
        ymap = np.arange(h)
        xmap, ymap = np.meshgrid(xmap, ymap)
        z = depth / scale
        x = (xmap - cx) * z / fx
        y = (ymap - cy) * z / fy
        return np.stack([x, y, z], axis=-1).reshape(-1, 3)

    def real_person_detect(self):
        indices = []
        for idx, roi_cloud in enumerate(self.roi_clouds):
            if self.is_real_person_by_cloud(roi_cloud):
                indices.append(idx)

        self.masks = [self.masks[i] for i in indices]
        self.roi_clouds = [self.roi_clouds[i] for i in indices]
        self.boxes = self.boxes[indices]
        self.track_ids = [self.track_ids[i] for i in indices]
        self.clss = [self.clss[i] for i in indices]
        return indices

    def is_real_person_by_cloud(self, roi_cloud, std_threshold=0.2):
        if len(roi_cloud) == 0:
            return False
        z_std = np.std(roi_cloud[:, 2])
        return z_std > std_threshold

    def get_3d_center(self, roi_cloud):
        if len(roi_cloud) == 0:
            return None
        return np.mean(roi_cloud, axis=0)

    def parallel_run(self, pipeline, align, fx, fy, cx, cy):
        # normal_session = None
        # slowfast_session = None
        with ThreadPoolExecutor() as executor:
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

                self.estimate(rgb, d, fx, fy, cx, cy, executor)

                # if (slowfast_session is None or slowfast_session.done()) and self.frame_count % self.detect_interval == 0:
                #     slowfast_session = executor.submit(self.slowfast_inference, self.frame_count, self.track_ids, self.boxes,
                #                     self.get_clips())
                #
                # if normal_session is None or normal_session.done():
                #     normal_session = executor.submit(self.estimate, rgb, d, fx, fy, cx, cy)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return