"""
main.py
=======
카메라 + 레이더 + 외부 캘리브레이션 통합
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
import threading

from src.camera.capture        import CameraCapture
from src.camera.yolo           import YOLODetector, Detection
from src.radar.udp             import RadarUDP
from src.calibration.extrinsic import ExtrinsicCalibration

# ── 설정 ──────────────────────────────────────────────────────────
RADAR_START  = True
CONF_CALIB   = 0.8

# 레이더 포인트 표시 색상
COLOR_MATCHED   = (0, 255, 0)    # 초록: bbox에 매칭된 포인트
COLOR_UNMATCHED = (0, 0, 255)    # 빨강: bbox 밖 포인트
POINT_RADIUS    = 5


def project_radar_points(
    points: np.ndarray,
    calib: ExtrinsicCalibration,
    frame_w: int,
    frame_h: int,
) -> list[tuple[int, int]]:
    """레이더 포인트 전체를 픽셀 좌표로 투영"""
    projected = []
    for pt in points:
        px, py = calib.project(pt)
        px = int(np.clip(px, 0, frame_w - 1))
        py = int(np.clip(py, 0, frame_h - 1))
        projected.append((px, py))
    return projected


def match_radar_to_detections(
    detections: list[Detection],
    points: np.ndarray,
    projected: list[tuple[int, int]],
) -> tuple[list[Detection], set[int]]:
    """
    투영된 레이더 포인트 → bbox 매칭
    반환: (매칭된 Detection 리스트, 매칭된 포인트 인덱스 집합)
    """
    matched_indices = set()

    for det in detections:
        x1, y1, x2, y2 = det.bbox

        candidates = [
            (i, px, py, points[i])
            for i, (px, py) in enumerate(projected)
            if x1 <= px <= x2 and y1 <= py <= y2
        ]

        if not candidates:
            continue

        # 가장 가까운 포인트 (range 최솟값)
        best = min(candidates, key=lambda c: c[3][0])
        i, px, py, pt = best
        matched_indices.add(i)

        det.radar_range     = float(pt[0])
        det.radar_azimuth   = float(pt[1])
        det.radar_elevation = float(pt[2])
        det.radar_velocity  = float(pt[3])

    return detections, matched_indices


def draw_radar_points(
    frame: np.ndarray,
    projected: list[tuple[int, int]],
    matched_indices: set[int],
) -> np.ndarray:
    """
    레이더 포인트를 화면에 점으로 표시
    - 초록: bbox에 매칭된 포인트
    - 빨강: bbox 밖 포인트
    """
    for i, (px, py) in enumerate(projected):
        color = COLOR_MATCHED if i in matched_indices else COLOR_UNMATCHED
        cv2.circle(frame, (px, py), POINT_RADIUS, color, -1)
        cv2.circle(frame, (px, py), POINT_RADIUS + 1, (0, 0, 0), 1)  # 테두리
    return frame


def collect_calib_points(
    detections: list[Detection],
    points: np.ndarray,
    projected: list[tuple[int, int]],
    calib: ExtrinsicCalibration,
):
    """캘리브레이션 대응점 수집"""
    if calib._converged:
        return

    for det in detections:
        if det.conf < CONF_CALIB:
            continue

        x1, y1, x2, y2 = det.bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # bbox 안에 투영된 포인트 정확히 1개일 때만 수집
        inside = [
            points[i] for i, (px, py) in enumerate(projected)
            if x1 <= px <= x2 and y1 <= py <= y2
        ]

        if len(inside) != 1:
            continue

        calib.add_point(
            px       = cx,
            py       = cy,
            radar_pt = inside[0],
            class_id = det.class_id,
            conf     = det.conf,
        )


def draw_calib_status(frame: np.ndarray, calib: ExtrinsicCalibration) -> np.ndarray:
    """캘리브레이션 상태 표시"""
    acc   = calib._accuracy
    color = (0, 255, 0) if acc >= 85 else (0, 165, 255) if acc >= 50 else (0, 0, 255)

    n   = len(calib._points)
    err = calib._error_px
    status = "DONE" if calib._converged else "RUNNING"
    text = f"CALIB[{status}]: {n}pts | err {err:.1f}px | acc {acc:.0f}%"

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (8, 8), (tw + 16, th + 20), (0, 0, 0), -1)
    cv2.putText(frame, text, (12, th + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    # 정확도 바
    bar_x, bar_y, bar_w, bar_h = 12, th + 24, 200, 8
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    fill = int(bar_w * min(acc, 100) / 100)
    if fill > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), color, -1)

    return frame


def main():
    cam      = CameraCapture()
    detector = YOLODetector()
    calib    = ExtrinsicCalibration()
    radar    = RadarUDP() if RADAR_START else None

    calib.load()
    cam.start()

    if radar:
        radar.start()
        print("[radar] mmWave Studio -> Trigger Frame")

    print("[main] running. press q to quit")

    latest_points = None
    points_lock   = threading.Lock()

    def radar_loop():
        nonlocal latest_points
        while True:
            if radar is None:
                break
            pts = radar.get_points(timeout=1.0)
            if pts is not None:
                with points_lock:
                    latest_points = pts

    if radar:
        t = threading.Thread(target=radar_loop, daemon=True)
        t.start()

    while True:
        frame = cam.get_frame(timeout=1.0)
        if frame is None:
            continue

        H, W = frame.shape[:2]

        # YOLO 감지
        detections = detector.detect(frame)

        # 레이더 포인트 가져오기
        with points_lock:
            pts = latest_points.copy() if latest_points is not None else None

        projected     = []
        matched_idxs  = set()

        if pts is not None and len(pts) > 0:
            # 레이더 포인트 → 픽셀 투영
            projected = project_radar_points(pts, calib, W, H)

            # bbox 매칭
            detections, matched_idxs = match_radar_to_detections(
                detections, pts, projected
            )

            # 캘리브레이션 대응점 수집
            if not calib._converged:
                collect_calib_points(detections, pts, projected, calib)

        # 그리기
        out = detector.draw(frame, detections)
        out = draw_radar_points(out, projected, matched_idxs)
        out = draw_calib_status(out, calib)

        cv2.imshow("RadYOLO", out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop()
    if radar:
        radar.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()