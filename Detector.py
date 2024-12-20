from time import time
import cv2
import numpy as np
import torch
from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors
from collections import deque
from AvaUtils import ava_inference_transform


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

    def __init__(self, slowfast,ava_labels,detect_interval,device="cpu",classid=0, **kwargs):
        super().__init__(**kwargs)

        self.classid = classid  # 要识别的类别（本课设识别人，默认填 0 即可）
        self.slowfast = slowfast
        self.ava_labels = ava_labels
        self.spd = {}  # 存储速度数据
        self.trkd_ids = []  # 存储已经估算速度的物体 ID 列表
        self.trk_pt = {}  # 存储物体上一个时间戳
        self.trk_pp = {}  # 存储物体上一个位置
        self.img_stack = deque(maxlen=25)
        self.device=device
        self.action_labels = {}
        self.frame_count = detect_interval//2
        self.detect_interval = detect_interval


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


    def estimate(self, im0):
        """
        基于追踪数据估算物体的速度。

        参数:
            im0 (np.ndarray): 输入图像，通常为形状 (H, W, C) 的 RGB 图像。

        返回:
            (np.ndarray): 带有标注的处理后图像。
        """
        self.annotator = Annotator(im0, line_width=self.line_width)  # 初始化标注工具

        self.extract_tracks(im0)  # 提取物体轨迹
        self.img_stack.append(im0) # 入栈
        self.frame_count += 1

        indices = self.find_indices(self.clss, self.classid)

        if len(indices) != 0:
            self.boxes = self.boxes[indices]
            self.track_ids = [self.track_ids[i] for i in indices]
            self.clss = [self.clss[i] for i in indices]

            if self.frame_count % self.detect_interval == 0:
                self.slowfast_inference(im0)
                self.frame_count = 0

            for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
                self.store_tracking_history(track_id, box)  # 存储物体的轨迹历史
                # 如果该 track_id 还没有记录时间戳或位置，则初始化
                if track_id not in self.trk_pt:
                    self.trk_pt[track_id] = 0
                if track_id not in self.trk_pp:
                    self.trk_pp[track_id] = self.track_line[-1]

                # label = f"{int(self.spd[track_id])} km/h" if track_id in self.spd else self.names[int(cls)]
                if track_id in self.action_labels:
                    label = self.action_labels[track_id]
                else:
                    label = self.names[int(cls)]

                self.annotator.box_label(box, label=label, color=colors(track_id, True))  # 绘制边界框

                # 绘制物体的轨迹
                self.annotator.draw_centroid_and_tracks(
                    self.track_line, color=colors(int(track_id), True), track_thickness=self.line_width
                )

                # 进行速度估算
                # if direction == "known" and track_id not in self.trkd_ids:
                #     self.trkd_ids.append(track_id)
                #     time_difference = time() - self.trk_pt[track_id]
                #     if time_difference > 0:
                #         self.spd[track_id] = np.abs(self.track_line[-1][1] - self.trk_pp[track_id][1]) / time_difference

                self.trk_pt[track_id] = time()
                self.trk_pp[track_id] = self.track_line[-1]

            self.display_output(im0)  # 使用基类方法显示输出图像

        return im0  # 返回处理后的图像，供后续使用
