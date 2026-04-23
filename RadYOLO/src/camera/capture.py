"""
camera/capture.py
=================
웹캠 캡처 모듈
"""

import cv2
import threading
import queue
import numpy as np

# ── 설정 ──────────────────────────────────────────────────────────
CAM_INDEX   = 0
CAM_BACKEND = cv2.CAP_DSHOW
CAM_WIDTH   = 1920
CAM_HEIGHT  = 1080
CAM_FPS     = 30


class CameraCapture:
    """
    웹캠 실시간 캡처 클래스

    사용법:
        cam = CameraCapture()
        cam.start()
        frame = cam.get_frame()  # (H, W, 3) BGR or None
        cam.stop()
    """

    def __init__(self):
        self.cap = cv2.VideoCapture(CAM_INDEX, CAM_BACKEND)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS,          CAM_FPS)

        assert self.cap.isOpened(), f"카메라 {CAM_INDEX} 열기 실패"

        self._queue  = queue.Queue(maxsize=2)
        self._stop   = threading.Event()
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="cam_capture"
        )

    def start(self):
        self._thread.start()
        print(f"[카메라] 캡처 시작 — 인덱스 {CAM_INDEX}, {CAM_WIDTH}x{CAM_HEIGHT}")

    def stop(self):
        self._stop.set()
        self.cap.release()
        print("[카메라] 캡처 종료")

    def get_frame(self, timeout: float = 1.0) -> np.ndarray | None:
        """
        최신 프레임 반환
        반환: (H, W, 3) BGR uint8 또는 None
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _capture_loop(self):
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if not ret:
                continue

            # 최신 프레임만 유지
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            self._queue.put_nowait(frame)