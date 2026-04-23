import numpy as np

# ── 데이터 구조 파라미터 ───────────────────────────────────────
NUM_FRAMES  = 8
NUM_LOOPS   = 128   # TX당 chirp 반복 수
NUM_TX      = 3
NUM_RX      = 4
NUM_SAMPLES = 256


def load_bin(path: str) -> np.ndarray:
    """
    TDM-MIMO .bin → (frames, loops, tx, rx, samples) 복소수 배열

    DCA1000 포맷: 샘플마다 [RX0_I, RX0_Q, RX1_I, RX1_Q, RX2_I, RX2_Q, RX3_I, RX3_Q]
    TDM 순서: chirp 0(TX0), chirp 1(TX1), chirp 2(TX2), chirp 3(TX0), ...
    총 chirp 수: 8 × 128 × 3 = 3072
    """
    raw = np.fromfile(path, dtype=np.int16)

    # 샘플 하나 = int16 × 8 (RX 4개 × IQ 2)
    raw = raw.reshape(-1, 8)                                        # (786432, 8)

    # RX별 IQ 분리
    iq = raw[:, 0::2].astype(np.float32) + 1j * raw[:, 1::2].astype(np.float32)
    # shape: (786432, 4)

    total_chirps = NUM_FRAMES * NUM_LOOPS * NUM_TX                  # 3072
    assert iq.shape[0] == total_chirps * NUM_SAMPLES, \
        f"예상 {total_chirps * NUM_SAMPLES}개, 실제 {iq.shape[0]}개"

    # (frames, loops, tx, samples, rx)
    iq = iq.reshape(NUM_FRAMES, NUM_LOOPS, NUM_TX, NUM_SAMPLES, NUM_RX)

    # → (frames, loops, tx, rx, samples)
    iq = iq.transpose(0, 1, 2, 4, 3)

    return iq                                                        # (8, 128, 3, 4, 256)