import cv2
import numpy as np
try:
    import easyocr
except Exception:
    easyocr = None

CANONICAL_SPEEDS = {10,20,30,40,50,60,70,80,90,100,110,120,130}

class SpeedLimitDetector:
    def __init__(self, sign_model: str = None):
        self.reader = None
        if easyocr is not None:
            try:
                self.reader = easyocr.Reader(['en'], gpu=False)
            except Exception:
                self.reader = None
        self.current_limit = None
        self.last_limit = None
        self.stable_count = 0

    def _red_mask(self, hsv):
        lower1 = np.array([0, 70, 50])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([170, 70, 50])
        upper2 = np.array([180, 255, 255])
        m1 = cv2.inRange(hsv, lower1, upper1)
        m2 = cv2.inRange(hsv, lower2, upper2)
        return cv2.bitwise_or(m1, m2)

    def _ocr_speed(self, roi):
        if self.reader is None:
            return None
        h, w = roi.shape[:2]
        if h < 20 or w < 20:
            return None
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_LINEAR)
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
        res = self.reader.readtext(th, detail=0)
        nums = []
        for s in res:
            s = "".join([c for c in s if c.isdigit()])
            if len(s) >= 2:
                try:
                    val = int(s[:3])
                    nums.append(val)
                except Exception:
                    pass
        if not nums:
            return None
        nums = [n for n in nums if n in CANONICAL_SPEEDS]
        if not nums:
            return None
        return max(nums)

    def detect(self, frame):
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red = self._red_mask(hsv)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 50, param1=80, param2=30, minRadius=12, maxRadius=int(min(h, w) * 0.15))
        dets = []
        limit = None
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (cx, cy, r) in circles:
                x1 = max(cx - r, 0)
                y1 = max(cy - r, 0)
                x2 = min(cx + r, w - 1)
                y2 = min(cy + r, h - 1)
                roi = frame[y1:y2, x1:x2]
                mask_roi = red[y1:y2, x1:x2]
                ring = cv2.Canny(mask_roi, 50, 150)
                red_ratio = float(np.count_nonzero(ring)) / max(1, ring.size)
                if red_ratio < 0.02:
                    continue
                sp = self._ocr_speed(roi)
                dets.append((x1, y1, x2, y2, "speed_sign", float(red_ratio), sp))
                if sp is not None:
                    limit = sp
        if limit is not None:
            if self.last_limit == limit:
                self.stable_count += 1
            else:
                self.stable_count = 1
            self.last_limit = limit
            if self.stable_count >= 2 or self.current_limit is None:
                self.current_limit = limit
        return self.current_limit, dets
