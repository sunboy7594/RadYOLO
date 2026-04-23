"""
capture.py
==========
rawDataReader.m → dp_reshape4LaneLVDS() 포팅

TI 공식 DCA1000 4-lane LVDS 데이터 파싱.
bin 파일 또는 UDP 바이트 버퍼 → numpy 복소수 배열로 변환.
"""

import numpy as np

# ── 파라미터 ──────────────────────────────────────────────────────
NUM_TX      = 3
NUM_RX      = 4
NUM_LOOPS   = 128
NUM_SAMPLES = 256
NUM_FRAMES  = 8          # bin 파일 기준

# 프레임 1개 int16 개수: loops × tx × rx × samples × 2(IQ)
INT16_PER_FRAME = NUM_LOOPS * NUM_TX * NUM_SAMPLES * NUM_RX * 2  # 786,432

# 프레임 1개 바이트: int16 × 2bytes
FRAME_BYTES = INT16_PER_FRAME * 2  # 1,572,864 bytes


def parse_frame(raw: np.ndarray) -> np.ndarray:
    """
    rawDataReader.m → dp_reshape4LaneLVDS() 포팅

    ── TI 공식 4-lane LVDS 포맷 ──────────────────────────────────
    샘플 하나(8개 int16) = [RX0_I, RX1_I, RX2_I, RX3_I,
                           RX0_Q, RX1_Q, RX2_Q, RX3_Q]

    MATLAB 원본:
        rawData8 = reshape(rawData, [8, length(rawData)/8]);
        rawDataI = reshape(rawData8(1:4,:), [], 1);   ← 앞 4개 = I
        rawDataQ = reshape(rawData8(5:8,:), [], 1);   ← 뒤 4개 = Q
        frameCplx = rawDataI + 1i*rawDataQ;

    이전 코드(오류): raw[0::2], raw[1::2] 방식 → 채널 섞임
    수정 후:        raw[:, 0:4], raw[:, 4:8] 방식 → TI 공식 일치

    ── Reshape 순서 ──────────────────────────────────────────────
    raw(int16) → 8개씩 묶기 → I/Q 분리 → 복소수
                            → (chirps_total, samples, rx) reshape
                            → transpose → (chirps_total, rx, samples)

    입력: int16 1D array, 한 프레임 (INT16_PER_FRAME 개)
    출력: (chirps_total=384, rx=4, samples=256) complex64
    """
    # 8개씩 묶기: 각 행 = [RX0_I, RX1_I, RX2_I, RX3_I, RX0_Q, RX1_Q, RX2_Q, RX3_Q]
    raw8 = raw.reshape(-1, 8)                           # (98304, 8)

    # 앞 4개 = I (RX0~3), 뒤 4개 = Q (RX0~3)
    i_vals = raw8[:, 0:4].astype(np.float32)
    q_vals = raw8[:, 4:8].astype(np.float32)
    iq     = i_vals + 1j * q_vals                       # (98304, 4)

    # 98304 = chirps_total(384) × samples(256)
    chirps_total = NUM_LOOPS * NUM_TX  # 384

    # (chirps_total, samples, rx) — row-major reshape
    iq = iq.reshape(chirps_total, NUM_SAMPLES, NUM_RX)  # (384, 256, 4)

    # → (chirps_total, rx, samples)
    iq = iq.transpose(0, 2, 1).astype(np.complex64)     # (384, 4, 256)

    return iq


def load_bin(path: str) -> np.ndarray:
    """
    bin 파일 전체 로드 (rawDataReader.m dp_loadOneFrameData 포팅)

    출력: (frames, chirps_total, rx, samples) = (8, 384, 4, 256)
    """
    raw = np.fromfile(path, dtype=np.int16)

    expected = NUM_FRAMES * INT16_PER_FRAME
    assert raw.size == expected, \
        f"파일 크기 불일치: 예상 {expected}, 실제 {raw.size}"

    frames = []
    for f in range(NUM_FRAMES):
        start = f * INT16_PER_FRAME
        end   = start + INT16_PER_FRAME
        frames.append(parse_frame(raw[start:end]))

    return np.stack(frames, axis=0)  # (8, 384, 4, 256)


def parse_bytes(data: bytes) -> np.ndarray:
    """
    UDP 바이트 버퍼 파싱 (단일 프레임)

    출력: (chirps_total, rx, samples) = (384, 4, 256)
    """
    raw = np.frombuffer(data, dtype=np.int16)
    return parse_frame(raw)