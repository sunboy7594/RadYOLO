import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import open3d as o3d
import threading
import time

from radar.old.capture_udp import UDPReceiver
from radar.old.signal import process

# ── 시각화 파라미터 ────────────────────────────────────────────
POINT_SIZE   = 4.0
BG_COLOR     = [0.05, 0.05, 0.05]      # 거의 검정
MAX_RANGE    = 21.3                     # m

# 포인트 색상: intensity 기반 (낮음=파랑, 높음=빨강)
def intensity_to_color(points: np.ndarray) -> np.ndarray:
    """
    intensity 값을 0~1로 정규화해서 파랑→빨강 컬러맵 적용
    """
    if len(points) == 0:
        return np.zeros((0, 3))
    intensity = points[:, 4]
    norm = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-6)
    colors = np.zeros((len(points), 3))
    colors[:, 0] = norm           # R
    colors[:, 2] = 1.0 - norm     # B
    return colors


def points_to_xyz(points: np.ndarray) -> np.ndarray:
    """
    (range, azimuth, elevation, velocity, intensity) → (X, Y, Z)

    좌표계:
      X = 오른쪽
      Y = 앞 (레이더 방향)
      Z = 위
    """
    if len(points) == 0:
        return np.zeros((0, 3))

    r   = points[:, 0]
    az  = np.radians(points[:, 1])
    el  = np.radians(points[:, 2])

    X = r * np.cos(el) * np.sin(az)
    Y = r * np.cos(el) * np.cos(az)
    Z = r * np.sin(el)

    return np.stack([X, Y, Z], axis=1)


class RadarVisualizer:
    def __init__(self):
        self.receiver   = UDPReceiver()
        self.pcd        = o3d.geometry.PointCloud()
        self._latest    = None
        self._lock      = threading.Lock()
        self._running   = True

    def _process_loop(self):
        """UDP 수신 + 신호처리 스레드"""
        while self._running:
            frame = self.receiver.get_frame(timeout=2.0)
            if frame is None:
                continue

            raw    = frame[np.newaxis, ...]         # (1, loops, tx, rx, samples)
            points = process(raw)

            if len(points) == 0:
                continue

            xyz    = points_to_xyz(points)
            colors = intensity_to_color(points)

            with self._lock:
                self._latest = (xyz, colors)

    def run(self):
        # Open3D 뷰어 초기화
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="RadYOLO — 레이더 포인트클라우드", width=1280, height=720)

        # 렌더 옵션
        opt = vis.get_render_option()
        opt.background_color = np.array(BG_COLOR)
        opt.point_size       = POINT_SIZE

        # 초기 포인트클라우드 추가
        self.pcd.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))
        self.pcd.colors = o3d.utility.Vector3dVector(np.zeros((1, 3)))
        vis.add_geometry(self.pcd)

        # 좌표축 표시
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        vis.add_geometry(axis)

        # 카메라 초기 위치 설정 (위에서 내려다보는 각도)
        ctr = vis.get_view_control()
        ctr.set_zoom(0.5)
        ctr.set_front([0, -1, 0.5])
        ctr.set_up([0, 0, 1])

        # ESC 키로 종료
        vis.register_key_callback(256, lambda v: self._quit(v))

        # UDP 수신 시작
        self.receiver.start()

        # 처리 스레드 시작
        t = threading.Thread(target=self._process_loop, daemon=True)
        t.start()

        print("레이더 시각화 시작. 종료: ESC 또는 창 닫기")
        print("mmWave Studio에서 Trigger Frame을 눌러주세요.")

        # 메인 루프
        while self._running:
            with self._lock:
                data = self._latest
                self._latest = None

            if data is not None:
                xyz, colors = data
                self.pcd.points = o3d.utility.Vector3dVector(xyz)
                self.pcd.colors = o3d.utility.Vector3dVector(colors)
                vis.update_geometry(self.pcd)

            vis.poll_events()
            vis.update_renderer()

            if not vis.poll_events():
                break

            time.sleep(0.01)

        vis.destroy_window()
        self.receiver.stop()

    def _quit(self, vis):
        self._running = False
        return False


if __name__ == "__main__":
    viz = RadarVisualizer()
    viz.run()