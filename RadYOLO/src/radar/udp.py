"""
udp.py
======
DCA1000 UDP 실시간 스트리밍 수신 + 신호처리 통합

구조:
  수신 스레드: UDP 패킷 → 프레임 조립 → raw_queue
  처리 스레드: raw_queue → capture.parse_bytes → signal.process → point_queue
  외부:        point_queue.get() → 카메라 파이프라인
"""

import socket
import threading
import queue
import numpy as np

from src.radar.capture import parse_bytes, FRAME_BYTES
from src.radar.signal  import process

# ── 설정 ──────────────────────────────────────────────────────────
UDP_IP   = "192.168.33.30"
UDP_PORT = 4098


class RadarUDP:
    """
    DCA1000 UDP 스트림 수신 + 신호처리 통합 클래스

    사용법:
        radar = RadarUDP()
        radar.start()
        # mmWave Studio → Trigger Frame
        points = radar.get_points()  # (N, 5) or None
        radar.stop()
    """

    def __init__(self):
        # 소켓 설정
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF,
                             64 * 1024 * 1024)   # 64MB 수신 버퍼
        self.sock.bind((UDP_IP, UDP_PORT))
        self.sock.settimeout(2.0)

        # 큐: 수신 → 처리 (raw bytes), 처리 → 외부 (points)
        self._raw_queue   = queue.Queue(maxsize=8)
        self.point_queue  = queue.Queue(maxsize=4)

        self._stop        = threading.Event()
        self._recv_thread = threading.Thread(
            target=self._recv_loop, daemon=True, name="radar_recv"
        )
        self._proc_thread = threading.Thread(
            target=self._proc_loop, daemon=True, name="radar_proc"
        )

    # ── 공개 API ──────────────────────────────────────────────────
    def start(self):
        self._recv_thread.start()
        self._proc_thread.start()
        print(f"[UDP] 수신 시작 — {UDP_IP}:{UDP_PORT}")

    def stop(self):
        self._stop.set()
        self.sock.close()
        print("[UDP] 수신 종료")

    def get_points(self, timeout: float = 5.0) -> np.ndarray | None:
        """
        처리된 포인트클라우드 반환
        반환: (N, 5) float32 [range, azimuth, elevation, velocity, intensity]
              또는 None (타임아웃)
        """
        try:
            return self.point_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ── 내부 스레드 ───────────────────────────────────────────────
    def _recv_loop(self):
        """
        수신 전용 스레드: 패킷 수신 → 프레임 조립
        신호처리 없이 최대한 빠르게 수신 → DCA1000 타임아웃 방지
        """
        buf          = bytearray()
        expected_seq = None

        while not self._stop.is_set():
            try:
                data, _ = self.sock.recvfrom(1466 + 64)
            except socket.timeout:
                continue
            except OSError:
                break

            if len(data) < 10:
                continue

            # DCA1000 패킷 헤더 파싱
            seq_num  = int.from_bytes(data[0:4], 'little')
            adc_data = data[10:]

            # 패킷 드롭 감지 → 버퍼 리셋
            if expected_seq is not None and seq_num != expected_seq:
                print(f"[UDP] 패킷 드롭: {seq_num - expected_seq}개")
                buf.clear()

            expected_seq = seq_num + 1
            buf.extend(adc_data)

            # 프레임 1개 크기 쌓이면 처리 큐에 넣기
            while len(buf) >= FRAME_BYTES:
                frame_bytes = bytes(buf[:FRAME_BYTES])
                buf = bytearray(buf[FRAME_BYTES:])

                # 큐 가득 차면 오래된 것 버리고 최신 유지
                if self._raw_queue.full():
                    try:
                        self._raw_queue.get_nowait()
                    except queue.Empty:
                        pass
                try:
                    self._raw_queue.put_nowait(frame_bytes)
                except queue.Full:
                    pass

    def _proc_loop(self):
        """
        처리 스레드: 프레임 파싱 → 신호처리 → 포인트 큐
        """
        while not self._stop.is_set():
            try:
                frame_bytes = self._raw_queue.get(timeout=2.0)
            except queue.Empty:
                continue

            try:
                # capture.py: 바이트 → (384, 4, 256)
                frame = parse_bytes(frame_bytes)

                # signal.py: (1, 384, 4, 256) → (N, 5)
                raw    = frame[np.newaxis, ...]
                points = process(raw)

                if len(points) == 0:
                    continue

                # 큐 가득 차면 오래된 것 버리고 최신 유지
                if self.point_queue.full():
                    try:
                        self.point_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.point_queue.put_nowait(points)

            except Exception as e:
                print(f"[처리 오류] {e}")


# ── 단독 실행 (검증용) ─────────────────────────────────────────────
if __name__ == "__main__":
    radar = RadarUDP()
    radar.start()

    print("mmWave Studio에서 Trigger Frame 누르세요. 종료: Ctrl+C")

    frame_count = 0
    try:
        while True:
            points = radar.get_points(timeout=5.0)
            if points is None:
                print("[UDP] 타임아웃")
                continue

            frame_count += 1
            print(f"[프레임 {frame_count}] 포인트 수: {len(points)}")
            if len(points) > 0:
                print(points[:3])

    except KeyboardInterrupt:
        print("\n종료 중...")
    finally:
        radar.stop()