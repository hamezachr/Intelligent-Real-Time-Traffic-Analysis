import numpy as np
from ultralytics import YOLO
try:
    import torch
except Exception:
    torch = None

COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

TARGETS = {"person","bicycle","car","motorcycle","bus","truck","train"}

class YoloDetector:
    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.25, iou: float = 0.45, device: str = "auto"):
        self.model = YOLO(model_name)
        self.conf = conf
        self.iou = iou
        if device == "auto":
            self.device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        else:
            self.device = device
        self.half = True if self.device == "cuda" else False

    def detect(self, image):
        h, w = image.shape[:2]
        results = self.model.predict(image, conf=self.conf, iou=self.iou, verbose=False, device=self.device, half=self.half)
        dets = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            for i in range(len(cls)):
                c = COCO_CLASSES[cls[i]]
                dets.append((int(xyxy[i][0]), int(xyxy[i][1]), int(xyxy[i][2]), int(xyxy[i][3]), c, float(conf[i])))
        return dets
