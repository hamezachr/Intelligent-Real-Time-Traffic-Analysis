import numpy as np

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    aw = max(0, ax2 - ax1)
    ah = max(0, ay2 - ay1)
    bw = max(0, bx2 - bx1)
    bh = max(0, by2 - by1)
    union = aw * ah + bw * bh - inter
    if union == 0:
        return 0.0
    return inter / union

class Track:
    def __init__(self, tid, bbox, cls, conf, center, t):
        self.id = tid
        self.bbox = bbox
        self.cls = cls
        self.conf = conf
        self.center = center
        self.last_center = center
        self.last_time = t
        self.speed_mps = 0.0
        self.age = 0
        self.time_since_update = 0

class Tracker:
    def __init__(self, iou_thresh=0.3, max_age=30, speed_alpha=0.6):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.speed_alpha = speed_alpha
        self.tracks = []
        self.next_id = 1

    def update(self, detections, t, fps, meters_per_pixel):
        det_bboxes = [d[:4] for d in detections]
        det_meta = [(d[4], d[5]) for d in detections]
        det_centers = [((b[0]+b[2])//2, (b[1]+b[3])//2) for b in det_bboxes]
        assigned = set()
        for tr in self.tracks:
            tr.age += 1
            tr.time_since_update += 1
            best = -1
            best_iou = self.iou_thresh
            for i, b in enumerate(det_bboxes):
                if i in assigned:
                    continue
                s = iou(tr.bbox, b)
                if s > best_iou:
                    best_iou = s
                    best = i
            if best >= 0:
                assigned.add(best)
                tr.bbox = det_bboxes[best]
                tr.cls = det_meta[best][0]
                tr.conf = det_meta[best][1]
                tr.last_center = tr.center
                tr.center = det_centers[best]
                dt = max(1.0 / max(fps, 1e-6), 1e-3) if tr.last_time is None else (t - tr.last_time)
                tr.last_time = t
                dx = tr.center[0] - tr.last_center[0]
                dy = tr.center[1] - tr.last_center[1]
                dist_pixels = float(np.hypot(dx, dy))
                inst = (dist_pixels * meters_per_pixel) / max(dt, 1e-6)
                tr.speed_mps = self.speed_alpha * inst + (1.0 - self.speed_alpha) * tr.speed_mps
                tr.time_since_update = 0
        for i, b in enumerate(det_bboxes):
            if i in assigned:
                continue
            c = det_centers[i]
            cls, conf = det_meta[i]
            tr = Track(self.next_id, b, cls, conf, c, t)
            self.next_id += 1
            self.tracks.append(tr)
        self.tracks = [tr for tr in self.tracks if tr.time_since_update <= self.max_age]
        return self.tracks
