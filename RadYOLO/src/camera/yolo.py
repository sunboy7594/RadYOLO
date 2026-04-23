"""
camera/yolo.py
==============
YOLOv8-seg 객체 감지 모듈
"""

import numpy as np
import cv2
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import List

# ── 설정 ──────────────────────────────────────────────────────────
MODEL_PATH  = "yolov8n-seg.pt"   # 없으면 자동 다운로드
CONF_THRES  = 0.5
IOU_THRES   = 0.45
IMG_SIZE    = 640

# 클래스별 색상 (BGR)
COLORS = [
    (0, 255, 0),    # 초록
    (255, 0, 0),    # 파랑
    (0, 0, 255),    # 빨강
    (255, 255, 0),  # 노랑
    (0, 255, 255),  # 시안
    (255, 0, 255),  # 마젠타
]


@dataclass
class Detection:
    """
    YOLO 감지 결과 하나
    """
    class_id:   int
    class_name: str
    conf:       float
    bbox:       np.ndarray          # (x1, y1, x2, y2) int
    mask:       np.ndarray | None   # (H, W) bool, 픽셀 마스크
    center:     tuple               # (cx, cy) int, bbox 중심

    # 레이더 데이터 (매칭 후 채워짐)
    radar_range:    float | None = None
    radar_azimuth:  float | None = None
    radar_elevation: float | None = None
    radar_velocity: float | None = None


class YOLODetector:
    """
    YOLOv8-seg 객체 감지 클래스

    사용법:
        detector = YOLODetector()
        detections = detector.detect(frame)
        result_frame = detector.draw(frame, detections)
    """

    def __init__(self, model_path: str = MODEL_PATH):
        self.model = YOLO(model_path)
        print(f"[YOLO] 모델 로드 완료: {model_path}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        프레임에서 객체 감지

        입력: (H, W, 3) BGR
        출력: Detection 리스트
        """
        results = self.model(
            frame,
            conf=CONF_THRES,
            iou=IOU_THRES,
            imgsz=IMG_SIZE,
            verbose=False,
        )[0]

        detections = []
        H, W = frame.shape[:2]

        if results.boxes is None:
            return detections

        for i, box in enumerate(results.boxes):
            class_id   = int(box.cls[0])
            class_name = self.model.names[class_id]
            conf       = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # 세그멘테이션 마스크
            mask = None
            if results.masks is not None and i < len(results.masks):
                mask_data = results.masks[i].data[0].cpu().numpy()
                mask = cv2.resize(
                    mask_data.astype(np.float32), (W, H)
                ).astype(bool)

            detections.append(Detection(
                class_id   = class_id,
                class_name = class_name,
                conf       = conf,
                bbox       = np.array([x1, y1, x2, y2]),
                mask       = mask,
                center     = (cx, cy),
            ))

        return detections

    def draw(
        self,
        frame: np.ndarray,
        detections: List[Detection],
    ) -> np.ndarray:
        """
        감지 결과 + 레이더 데이터를 프레임에 그리기

        입력: (H, W, 3) BGR
        출력: (H, W, 3) BGR (오버레이 포함)
        """
        out = frame.copy()

        for det in detections:
            color = COLORS[det.class_id % len(COLORS)]
            x1, y1, x2, y2 = det.bbox
            cx, cy = det.center

            # 세그멘테이션 마스크 오버레이
            if det.mask is not None:
                overlay = out.copy()
                overlay[det.mask] = (
                    np.array(color) * 0.4 + out[det.mask] * 0.6
                ).astype(np.uint8)
                out = overlay

            # bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # 레이더 데이터 텍스트
            lines = [f"{det.class_name} {det.conf:.2f}"]

            if det.radar_range is not None:
                lines.append(f"dist: {det.radar_range:.1f}m")
            if det.radar_velocity is not None:
                lines.append(f"vel:  {det.radar_velocity:+.2f}m/s")
            if det.radar_azimuth is not None:
                lines.append(f"az:   {det.radar_azimuth:.1f}deg")
            if det.radar_elevation is not None:
                lines.append(f"el:   {det.radar_elevation:.1f}deg")

            # 텍스트 배경 + 출력
            font       = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness  = 1
            line_h     = 18
            text_x     = x1 + 4
            text_y     = y1 + 16

            for j, line in enumerate(lines):
                ty = text_y + j * line_h
                (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
                cv2.rectangle(out, (text_x - 2, ty - th - 2),
                              (text_x + tw + 2, ty + 2),
                              (0, 0, 0), -1)
                cv2.putText(out, line, (text_x, ty),
                            font, font_scale, color, thickness)

        return out