import numpy as np
from scipy.signal import windows

# ── 레이더 파라미터 ────────────────────────────────────────────
C           = 3e8
FC          = 77e9
BW          = 1798.92e6
NUM_SAMPLES = 256
NUM_LOOPS   = 128
NUM_TX      = 3
NUM_RX      = 4
NUM_FRAMES  = 8
NUM_VIRT    = NUM_TX * NUM_RX                   # 가상 안테나 12개

# 파생 파라미터
LAMBDA      = C / FC                            # ~3.9 mm
RANGE_RES   = C / (2 * BW)                     # ~0.083 m/bin
MAX_RANGE   = RANGE_RES * NUM_SAMPLES           # ~21.3 m

# TC = Idle Time + Ramp End Time (실측값)
TC          = (99.94 + 60.0) * 1e-6            # 159.94 μs
VEL_RES     = LAMBDA / (2 * TC * NUM_LOOPS)
MAX_VEL     = LAMBDA / (4 * TC)

# CFAR 파라미터
CFAR_GUARD  = 4
CFAR_TRAIN  = 16
CFAR_THRESH = 8.0           # 실험으로 튜닝 (낮을수록 포인트 많아짐)

# Angle FFT zero-padding
ANGLE_NFFT  = 64


# ── 1. Range FFT ───────────────────────────────────────────────
def range_fft(frame: np.ndarray) -> np.ndarray:
    """
    입력 : (loops, tx, rx, samples)
    출력 : (loops, tx, rx, range_bins)
    """
    win = windows.hann(NUM_SAMPLES)
    fw  = frame * win[np.newaxis, np.newaxis, np.newaxis, :]
    return np.fft.fft(fw, n=NUM_SAMPLES, axis=-1)


# ── 2. 가상 안테나 배열 구성 + Doppler FFT ────────────────────
def doppler_fft(rfft: np.ndarray) -> np.ndarray:
    """
    입력 : (loops, tx, rx, range_bins)
    출력 : (doppler_bins, virt_ant, range_bins)  — fftshift로 0속도 중앙

    TDM 위상 보정:
      TX_i 는 매 loop마다 i*TC 만큼 늦게 송신 → 도플러 위상 오차 누적
      보정: data[:, i, :, :] *= exp(-j * 2π * i * TC / (NUM_TX*TC) * doppler_bin)
      → Doppler FFT 후 bin별로 보정하면 닭-달걀 문제 발생
      → 근사: 속도가 MAX_VEL 이내면 오차가 작으므로 1차 근사 생략 (추후 정밀화)
    """
    # virtual array 구성: (loops, tx*rx, range_bins)
    loops, tx, rx, rbins = rfft.shape
    virt = rfft.reshape(loops, tx * rx, rbins)          # (128, 12, 256)

    # Doppler FFT — loops 축
    win  = windows.hann(NUM_LOOPS)
    virt_w = virt * win[:, np.newaxis, np.newaxis]
    dfft = np.fft.fft(virt_w, n=NUM_LOOPS, axis=0)
    dfft = np.fft.fftshift(dfft, axes=0)                # (128, 12, 256)

    return dfft


# ── 3. CFAR (CA-CFAR 2D) ──────────────────────────────────────
def cfar_2d(rd_mag: np.ndarray) -> list[tuple[int, int]]:
    """
    입력 : (doppler_bins, range_bins) — 가상 안테나 평균 magnitude
    출력 : 검출된 (doppler_idx, range_idx) 리스트
    """
    g, t = CFAR_GUARD, CFAR_TRAIN
    n_dop, n_rng = rd_mag.shape
    dets = []

    for d in range(g + t, n_dop - g - t):
        for r in range(g + t, n_rng - g - t):
            cut   = rd_mag[d, r]
            train = np.concatenate([
                rd_mag[d-g-t : d-g,     r-g-t : r+g+t+1].ravel(),
                rd_mag[d+g+1 : d+g+t+1, r-g-t : r+g+t+1].ravel(),
                rd_mag[d-g-t : d+g+t+1, r-g-t : r-g    ].ravel(),
                rd_mag[d-g-t : d+g+t+1, r+g+1 : r+g+t+1].ravel(),
            ])
            if cut > CFAR_THRESH * np.mean(train):
                dets.append((d, r))

    return dets


# ── 4. Angle FFT (Azimuth) ────────────────────────────────────
def angle_fft(dfft: np.ndarray, det: tuple[int, int]) -> float:
    """
    입력 : dfft (doppler_bins, virt_ant, range_bins), 검출 포인트
    출력 : azimuth (degree)

    NOTE: AWR2243 안테나 배열 geometry 적용 시 elevation 추정 가능
          현재는 12개 가상 안테나를 선형 배열로 근사 → azimuth만 추정
          정밀 구현은 안테나 배열 geometry 확인 후 추가 예정
    """
    d_idx, r_idx = det
    steering = dfft[d_idx, :, r_idx]                   # (12,) 복소수

    padded = np.zeros(ANGLE_NFFT, dtype=np.complex64)
    padded[:NUM_VIRT] = steering

    spec     = np.fft.fftshift(np.fft.fft(padded))
    peak_idx = int(np.argmax(np.abs(spec)))

    sin_theta = (peak_idx - ANGLE_NFFT // 2) / ANGLE_NFFT
    sin_theta = np.clip(sin_theta, -1.0, 1.0)
    return float(np.degrees(np.arcsin(sin_theta)))


# ── 5. 메인 처리 함수 ──────────────────────────────────────────
def process(raw: np.ndarray) -> np.ndarray:
    """
    입력 : (frames, loops, tx, rx, samples)
    출력 : (N, 5)  [range(m), azimuth(deg), elevation(deg), velocity(m/s), intensity]

    NOTE: elevation 현재 0.0 고정 — 안테나 geometry 기반 2D angle 추정은 추후 구현
    NOTE: TDM 위상 보정 미적용 — 고속 물체 감지 시 추가 필요
    """
    all_points = []

    for f in range(raw.shape[0]):
        frame = raw[f]                                  # (128, 3, 4, 256)

        rfft  = range_fft(frame)                        # (128, 3, 4, 256)
        dfft  = doppler_fft(rfft)                       # (128, 12, 256)

        # CFAR용 가상 안테나 평균 magnitude
        rd_mag = np.mean(np.abs(dfft), axis=1)          # (128, 256)
        dets   = cfar_2d(rd_mag)

        for d_idx, r_idx in dets:
            rng       = r_idx * RANGE_RES
            vel_idx   = d_idx - NUM_LOOPS // 2
            velocity  = vel_idx * VEL_RES
            azimuth   = angle_fft(dfft, (d_idx, r_idx))
            elevation = 0.0
            intensity = float(rd_mag[d_idx, r_idx])

            all_points.append([rng, azimuth, elevation, velocity, intensity])

    if not all_points:
        return np.zeros((0, 5), dtype=np.float32)

    return np.array(all_points, dtype=np.float32)