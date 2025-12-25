import argparse
import time
import csv
from datetime import datetime, timezone
import cv2
import numpy as np
from src.yolo_detector import YoloDetector, TARGETS
from src.tracker import Tracker
FR_LABELS = {
    "person":"person",
    "bicycle":"bicycle",
    "car":"car",
    "motorcycle":"motorcycle",
    "bus":"bus",
    "truck":"truck",
    "train":"train"
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, default="0")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--meters_per_pixel", type=float, default=0.05)
    p.add_argument("--logfile", type=str, default="traffic_log.csv")
    p.add_argument("--model", type=str, default="yolov8n.pt")
    p.add_argument("--no_show", action="store_true")
    p.add_argument("--out_video", type=str, default=None)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--classes", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()

def draw_text(img, text, x, y, color=(0, 255, 0)):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

def open_source(src):
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
    else:
        cap = cv2.VideoCapture(src)
    return cap

def main():
    args = parse_args()
    det = YoloDetector(model_name=args.model, conf=args.conf, iou=args.iou, device=args.device)
    trk = Tracker()
    cap = open_source(args.source)
    if not cap.isOpened():
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-3:
        fps = 30.0
    writer = open(args.logfile, "w", newline="")
    log = csv.writer(writer)
    log.writerow(["time","frame","track_id","class","class_fr","speed_mps","speed_kmh","x1","y1","x2","y2"])
    selected = list(TARGETS) if args.classes is None else [c.strip() for c in args.classes.split(",") if c.strip()]
    counts = {c:0 for c in selected}
    speeds = {c:[] for c in selected}
    frame_idx = 0
    last_t = time.time()
    display = not args.no_show
    if display:
        try:
            cv2.namedWindow("Traffic", cv2.WINDOW_NORMAL)
        except cv2.error:
            display = False
    out_writer = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = time.time()
        dets = det.detect(frame)
        filtered = [d for d in dets if d[4] in selected]
        tracks = trk.update(filtered, t, fps, args.meters_per_pixel)
        counts = {c:0 for c in selected}
        speeds = {c:[] for c in selected}
        for tr in tracks:
            if tr.cls in counts:
                counts[tr.cls] += 1
                if tr.speed_mps > 0:
                    speeds[tr.cls].append(tr.speed_mps * 3.6)
            x1, y1, x2, y2 = tr.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            lbl = FR_LABELS.get(tr.cls, tr.cls)
            draw_text(frame, f"{lbl} #{tr.id}", x1, max(0, y1-10))
            log.writerow([datetime.now(timezone.utc).isoformat(), frame_idx, tr.id, tr.cls, lbl, f"{tr.speed_mps:.3f}", f"{tr.speed_mps*3.6:.3f}", x1, y1, x2, y2])
        y = 30
        for c in selected:
            lbl = FR_LABELS.get(c, c)
            draw_text(frame, f"{lbl}: {counts[c]}", 10, y)
            y += 30
        if args.out_video and out_writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_writer = cv2.VideoWriter(args.out_video, fourcc, fps, (w, h))
        if out_writer is not None:
            out_writer.write(frame)
        if display:
            try:
                cv2.imshow("Traffic", frame)
            except cv2.error:
                display = False
        key = -1
        if display:
            try:
                key = cv2.waitKey(1) & 0xFF
            except cv2.error:
                key = -1
        frame_idx += 1
        if args.max_frames is not None and frame_idx >= args.max_frames:
            break
        if key == 27 or key == ord('q'):
            break
    writer.close()
    cap.release()
    if out_writer is not None:
        out_writer.release()
    if display:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

if __name__ == "__main__":
    main()
