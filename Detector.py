from time import time
import cv2
import numpy as np
import torch
from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors
# from collections import deque
# from AvaUtils import ava_inference_transform


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

    def __init__(self, classid=0, showmask=False, **kwargs):
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
        # self.img_stack = deque(maxlen=25)
        # self.device=device
        # self.action_labels = {}
        # self.frame_count = detect_interval//2
        # self.detect_interval = detect_interval


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

    def slowfast_inference(self,im0):
        inputs, inp_boxes, _ = ava_inference_transform(self.get_clips(), self.boxes)
        inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0], 1), inp_boxes], dim=1)
        if isinstance(inputs, list):
            inputs = [inp.unsqueeze(0).to(self.device) for inp in inputs]
        else:
            inputs = inputs.unsqueeze(0).to(self.device)
        with torch.no_grad():
            slowfaster_preds = self.slowfast(inputs, inp_boxes.to(self.device))
            slowfaster_preds = slowfaster_preds.cpu()

        for id, avalabel in zip(self.track_ids, np.argmax(slowfaster_preds, axis=1).tolist()):
            self.action_labels[id] = self.ava_labels[avalabel + 1]

    def estimate(self, rgb, depth, fx, fy, cx, cy):
        """生成点云并识别前景，再计算人体运动速度。
        返回: (results, annotated_frame)
        """
        # 原先的模型推理
        # results = self.model(rgb, conf=0.25)
        self.annotator = Annotator(rgb, line_width=self.line_width)

        # 在这里直接生成点云等逻辑
        cloud = self.create_point_cloud_from_depth_image(depth, fx, fy, cx, cy)
        if not hasattr(self, 'last_center_3d'):
            self.last_center_3d = None
        if not hasattr(self, 'last_time'):
            self.last_time = None
        
        self.extract_tracks(rgb)
        indices = self.find_indices(self.clss, self.classid)
        self.masks = self.tracks[0].masks[indices]
        self.boxes = [self.boxes[i] for i in indices]
        self.track_ids = [self.track_ids[i] for i in indices]
        self.clss = [self.clss[i] for i in indices]

        # 对每个人体的掩码做前景提取并计算速度
        if self.masks is not None:
            valid_idx = []
            for idx, (box, track_id) in enumerate(zip(self.boxes, self.track_ids)):
                mask2d = self.masks.data[idx].cpu().numpy().astype(bool)
                roi_cloud = cloud[mask2d.flatten()]
                if self.is_real_person_by_cloud(roi_cloud):
                    self.store_tracking_history(track_id, box)
                    valid_idx.append(idx)
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
                                self.annotator.box_label(
                                    box,
                                    label=f"ID:{track_id}, Pos:[{new_avg[0]:.2f},{new_avg[1]:.2f},{new_avg[2]:.2f}], Speed:{speed_mag:.2f}m/s",
                                    color=colors(idx, True)
                                )
                            self.track_avg_position_old[track_id] = new_avg
                            self.track_last_time[track_id] = current_time
                    self.annotator.draw_centroid_and_tracks(
                        self.track_line, color=colors(int(track_id), True), track_thickness=self.line_width)
            if self.showmask and len(valid_idx) > 0:
                self.annotator.masks(torch.stack([self.masks[i].data for i in valid_idx]).to('xpu').squeeze(dim=1),
                                     colors=[colors(idx, True) for idx in valid_idx],
                                     im_gpu=torch.tensor(rgb, device='xpu').permute(2,0,1), alpha=0.1)
            
            self.display_output(rgb)

        annotated_frame = self.annotator.result()
        return annotated_frame

    def create_point_cloud_from_depth_image(self, depth, fx, fy, cx, cy, scale=1000.0):
        h, w = depth.shape
        xmap = np.arange(w)
        ymap = np.arange(h)
        xmap, ymap = np.meshgrid(xmap, ymap)
        z = depth / scale
        x = (xmap - cx) * z / fx
        y = (ymap - cy) * z / fy
        return np.stack([x, y, z], axis=-1).reshape(-1, 3)

    def is_real_person_by_cloud(self, roi_cloud, std_threshold=0.2):
        if len(roi_cloud) == 0:
            return False
        z_std = np.std(roi_cloud[:, 2])
        return z_std > std_threshold

    def get_3d_center(self, roi_cloud):
        if len(roi_cloud) == 0:
            return None
        return np.mean(roi_cloud, axis=0)
