import socket
import numpy as np
import queue
import threading

# ── UDP 설정 ───────────────────────────────────────────────────
UDP_IP       = "192.168.33.30"
UDP_PORT     = 4098
PACKET_SIZE  = 1466

# ── 프레임 파라미터 ────────────────────────────────────────────
NUM_LOOPS    = 128
NUM_TX       = 3
NUM_RX       = 4
NUM_SAMPLES  = 256
FRAME_BYTES  = NUM_LOOPS * NUM_TX * NUM_RX * NUM_SAMPLES * 2 * 2  # 6,291,456


def parse_frame(buf: bytes) -> np.ndarray:
    raw = np.frombuffer(buf, dtype=np.int16)
    raw = raw.reshape(-1, 8)
    iq  = raw[:, 0::2].astype(np.float32) + 1j * raw[:, 1::2].astype(np.float32)
    iq  = iq.reshape(NUM_LOOPS, NUM_TX, NUM_SAMPLES, NUM_RX)
    iq  = iq.transpose(0, 1, 3, 2)
    return iq


class UDPReceiver:
    """
    수신 스레드: UDP 패킷만 최대한 빠르게 받아서 raw_queue에 넣음.
    신호처리는 별도 스레드에서 처리 → DCA1000 타임아웃 방지.
    """

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024 * 64)  # 64MB
        self.sock.bind((UDP_IP, UDP_PORT))
        self.sock.settimeout(2.0)

        self.raw_queue = queue.Queue(maxsize=8)
        self._stop     = threading.Event()
        self._thread   = threading.Thread(target=self._recv_loop, daemon=True)

    def start(self):
        self._thread.start()
        print(f"[UDP] 수신 시작 — {UDP_IP}:{UDP_PORT}")

    def stop(self):
        self._stop.set()
        self.sock.close()
        print("[UDP] 수신 종료")

    def get_frame(self, timeout=5.0) -> np.ndarray | None:
        try:
            return self.raw_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _recv_loop(self):
        buf          = bytearray()
        expected_seq = None

        while not self._stop.is_set():
            try:
                data, _ = self.sock.recvfrom(PACKET_SIZE + 64)
            except socket.timeout:
                continue
            except OSError:
                break

            if len(data) < 10:
                continue

            seq_num  = int.from_bytes(data[0:4], 'little')
            adc_data = data[10:]

            if expected_seq is not None and seq_num != expected_seq:
                print(f"[UDP] 패킷 드롭: {seq_num - expected_seq}개")
                buf.clear()

            expected_seq = seq_num + 1
            buf.extend(adc_data)

            while len(buf) >= FRAME_BYTES:
                frame_bytes = bytes(buf[:FRAME_BYTES])
                buf = bytearray(buf[FRAME_BYTES:])
                try:
                    frame = parse_frame(frame_bytes)
                    # 가득 차면 오래된 것 버리고 최신 것 유지
                    if self.raw_queue.full():
                        try:
                            self.raw_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.raw_queue.put_nowait(frame)
                except Exception as e:
                    print(f"[UDP] 파싱 오류: {e}")


if __name__ == "__main__":
    from radar.old.signal import process

    receiver = UDPReceiver()
    receiver.start()

    print("mmWave Studio에서 Trigger Frame 누르세요. 종료: Ctrl+C")

    frame_count = 0
    try:
        while True:
            frame = receiver.get_frame(timeout=5.0)
            if frame is None:
                print("[UDP] 타임아웃")
                continue

            raw    = frame[np.newaxis, ...]
            points = process(raw)
            frame_count += 1
            print(f"[프레임 {frame_count}] 포인트 수: {len(points)}")
            if len(points) > 0:
                print(points[:3])

    except KeyboardInterrupt:
        print("\n종료 중...")
    finally:
        receiver.stop()