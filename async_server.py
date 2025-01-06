from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slowfast_r50_detection
from AvaUtils import ava_inference_transform
import torch
import numpy as np
import Pyro4

@Pyro4.expose
class Model:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.slowfast = slowfast_r50_detection(True).eval().to(self.device)
        self.ava_labels, _ = AvaLabeledVideoFramePaths.read_label_map("configs/ava_action_list.pbtxt")
        self.detect_interval = 50  # 设定slowfast触发频率（/fps）
        self.action_labels = {}

    def slowfast_inference(self, frame_count, track_ids, boxes, get_clips):
        boxes = torch.tensor(np.array(boxes)).float()
        get_clips = torch.tensor(np.array(get_clips)).float()
        if frame_count % self.detect_interval == 0:
            inputs, inp_boxes, _ = ava_inference_transform(get_clips, boxes)
            inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0], 1), inp_boxes], dim=1)
            if isinstance(inputs, list):
                inputs = [inp.unsqueeze(0).to(self.device) for inp in inputs]
            else:
                inputs = inputs.unsqueeze(0).to(self.device)
            with torch.no_grad():
                slowfaster_preds = self.slowfast(inputs, inp_boxes.to(self.device))
                slowfaster_preds = slowfaster_preds.cpu()

            for id, avalabel in zip(track_ids, np.argmax(slowfaster_preds, axis=1).tolist()):
                self.action_labels[id] = self.ava_labels[avalabel + 1]
        return self.action_labels

# 创建守护进程并绑定对象
def main():
    daemon = Pyro4.Daemon()  # 创建Pyro4守护进程
    uri = daemon.register(Model)  # 将Model类注册到守护进程中

    print(f"Model service is running. URI: {uri}")  # 打印对象的URI，远程调用时使用
    daemon.requestLoop()  # 启动守护进程并监听远程请求

if __name__ == '__main__':
    main()
