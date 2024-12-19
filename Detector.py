from time import time
from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors


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

    def __init__(self, classid=0, **kwargs):
        super().__init__(**kwargs)

        self.classid = classid  # 要识别的类别（本课设识别人，默认填 0 即可）
        self.spd = {}  # 存储速度数据
        self.trkd_ids = []  # 存储已经估算速度的物体 ID 列表
        self.trk_pt = {}  # 存储物体上一个时间戳
        self.trk_pp = {}  # 存储物体上一个位置

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

        indices = self.find_indices(self.clss, self.classid)

        if len(indices) != 0:
            self.boxes = self.boxes[indices]
            self.track_ids = [self.track_ids[i] for i in indices]
            self.clss = [self.clss[i] for i in indices]

            print(self.boxes, self.track_ids, self.clss)

            for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
                if cls != self.classid:
                    continue

                self.store_tracking_history(track_id, box)  # 存储物体的轨迹历史

                # 如果该 track_id 还没有记录时间戳或位置，则初始化
                if track_id not in self.trk_pt:
                    self.trk_pt[track_id] = 0
                if track_id not in self.trk_pp:
                    self.trk_pp[track_id] = self.track_line[-1]

                speed_label = f"{int(self.spd[track_id])} km/h" if track_id in self.spd else self.names[int(cls)]
                self.annotator.box_label(box, label=speed_label, color=colors(track_id, True))  # 绘制边界框

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
