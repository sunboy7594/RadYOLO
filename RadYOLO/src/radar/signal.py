"""
signal.py
=========
TI MATLAB 예제 포팅:
  rangeProcCascade.m         → Range FFT
  DopplerProcClutterRemove.m → Doppler FFT
  CFAR_CASO_Range.m          → Range 방향 CFAR
  CFAR_CASO_Doppler_overlap.m + datapath.m → Doppler CFAR + TDM 위상 보정
  DOA_BF_PeakDet_loc.m       → 피크 검출
  DOA_beamformingFFT_2D_joint.m → 2D DOA (azimuth + elevation)

AWR2243BOOST 안테나 geometry 적용:
  TX1(0,0) TX2(2,1) TX3(4,0) × RX0~3(0~3, 0)
  단위: λ/2
"""

import numpy as np
from typing import List, Tuple

from src.radar.cfar_vectorized import cfar_range_vec, cfar_doppler_phase_vec

# ── 레이더 파라미터 ────────────────────────────────────────────────
C           = 3e8
FC          = 77e9
BW          = 1798.92e6
TC          = (99.94 + 60.0) * 1e-6    # Idle Time + Ramp End Time (s)

NUM_TX      = 3
NUM_RX      = 4
NUM_VIRT    = NUM_TX * NUM_RX           # 가상 안테나 12개
NUM_LOOPS   = 128
NUM_SAMPLES = 256

RANGE_FFT_SIZE   = 256
DOPPLER_FFT_SIZE = 128

LAMBDA    = C / FC                      # ~3.9mm
RANGE_RES = C / (2 * BW)               # ~0.083 m/bin
VEL_RES   = LAMBDA / (2 * TC * NUM_LOOPS)
MAX_VEL   = LAMBDA / (4 * TC)

# ── AWR2243BOOST 가상 안테나 배열 (λ/2 단위) ─────────────────────
# 안테나 그림(Figure 2-9) 기반:
#   TX1 위치: 수평 0, 수직 0
#   TX2 위치: 수평 2(=λ), 수직 1(=λ/2)
#   TX3 위치: 수평 4(=2λ), 수직 0
#   RX0~3: 수평 0,1,2,3, 수직 0
#
# 가상 안테나 = TX 위치 + RX 위치
# TX1×RX0~3: (0,0)(1,0)(2,0)(3,0)
# TX2×RX0~3: (2,1)(3,1)(4,1)(5,1)
# TX3×RX0~3: (4,0)(5,0)(6,0)(7,0)
ANT_D = np.array([
    [0, 0], [1, 0], [2, 0], [3, 0],    # TX1×RX0~3 → virt[0:4]
    [2, 1], [3, 1], [4, 1], [5, 1],    # TX2×RX0~3 → virt[4:8]
    [4, 0], [5, 0], [6, 0], [7, 0],    # TX3×RX0~3 → virt[8:12]
], dtype=np.int32)                       # (12, 2): [수평, 수직]

ANT_DIS = 0.5  # λ/2 단위

# ── CFAR 파라미터 ──────────────────────────────────────────────────
CFAR_REF_WIN       = [8, 4]     # [range, doppler] reference window 크기
CFAR_GUARD_WIN     = [4, 2]     # [range, doppler] guard window 크기
CFAR_K0            = [15.0, 8.0] # [range, doppler] 임계값 배수 (실험으로 튜닝)
CFAR_DISCARD_LEFT  = 5          # range 양쪽 버릴 bin 수
CFAR_DISCARD_RIGHT = 2
CFAR_POWER_THRE    = 0.0        # 최소 파워 임계값

# ── DOA 파라미터 ───────────────────────────────────────────────────
DOA_FFT_SIZE    = 128
DOA_AZ_RANGE    = [-80.0, 80.0]
DOA_EL_RANGE    = [-30.0, 30.0]
DOA_GAMMA       = 1.5
DOA_SIDELOBE_AZ = 15.0
DOA_SIDELOBE_EL = 15.0


# ─────────────────────────────────────────────────────────────────────
# STEP 1. Range FFT  (rangeProcCascade.m 포팅)
# ─────────────────────────────────────────────────────────────────────
def range_fft(frame: np.ndarray) -> np.ndarray:
    """
    rangeProcCascade.m 포팅

    ── MATLAB 원본 ────────────────────────────────────────────────
    입력: (samples, chirps, antennas)
    처리:
      inputMat = bsxfun(@minus, inputMat, mean(inputMat))
        → 각 chirp의 sample 평균 제거 (DC offset 제거)
          MATLAB mean(matrix) = 각 column 평균 = sample 축 평균
      inputMat = bsxfun(@times, inputMat, rangeWindowCoeffVec)
        → Hann 윈도우 (rangeWindowCoeffVec = hann(numSamples))
      fftOutput = fft(inputMat, rangeFFTSize)
        → sample 축 방향 FFT
    출력: (range_bins, chirps, antennas)

    ── Python 변환 ────────────────────────────────────────────────
    입력: (chirps_total=384, rx=4, samples=256)  ← capture.py 출력
    출력: (range_bins=256, chirps_total=384, rx=4)
    """
    # capture.py 출력 (384, 4, 256) → MATLAB 관례 (256, 384, 4)
    data = frame.transpose(2, 0, 1).astype(np.complex64)  # (samples, chirps, rx)

    # DC offset 제거: 각 chirp(column)의 sample 평균 제거
    # MATLAB: mean(inputMat) → axis=0(sample 축) 평균
    data = data - data.mean(axis=0, keepdims=True)

    # Hann 윈도우 (sample 축 적용)
    win  = np.hanning(NUM_SAMPLES).astype(np.float32).reshape(-1, 1, 1)
    data = data * win

    # Range FFT (sample 축 = axis=0)
    rfft = np.fft.fft(data, n=RANGE_FFT_SIZE, axis=0).astype(np.complex64)
    # 출력: (range_bins=256, chirps_total=384, rx=4)

    return rfft


# ─────────────────────────────────────────────────────────────────────
# STEP 2. TX 분리 + 가상 안테나 배열 구성
# ─────────────────────────────────────────────────────────────────────
def form_virtual_array(rfft: np.ndarray) -> np.ndarray:
    """
    TDM TX 분리 후 12개 가상 안테나 배열 구성

    ── TDM 순서 ──────────────────────────────────────────────────
    chirp 0 = TX0, chirp 1 = TX1, chirp 2 = TX2,
    chirp 3 = TX0, chirp 4 = TX1, ...
    → 3개 건너뛰며 추출

    ── 가상 안테나 순서 (ANT_D 행 순서와 일치) ───────────────────
    virt[..., 0:4]  = TX0×RX0~3  ← D[(0,0)(1,0)(2,0)(3,0)]
    virt[..., 4:8]  = TX1×RX0~3  ← D[(2,1)(3,1)(4,1)(5,1)]
    virt[..., 8:12] = TX2×RX0~3  ← D[(4,0)(5,0)(6,0)(7,0)]

    입력: (range_bins=256, chirps_total=384, rx=4)
    출력: (range_bins=256, loops=128, virt_ant=12)
    """
    tx0 = rfft[:, 0::NUM_TX, :]  # (256, 128, 4)  TX0
    tx1 = rfft[:, 1::NUM_TX, :]  # (256, 128, 4)  TX1
    tx2 = rfft[:, 2::NUM_TX, :]  # (256, 128, 4)  TX2

    # 12개 가상 안테나로 연결
    virt = np.concatenate([tx0, tx1, tx2], axis=2)  # (256, 128, 12)

    return virt


# ─────────────────────────────────────────────────────────────────────
# STEP 3. Doppler FFT  (DopplerProcClutterRemove.m 포팅)
# ─────────────────────────────────────────────────────────────────────
def doppler_fft(virt: np.ndarray, clutter_remove: bool = True) -> np.ndarray:
    """
    DopplerProcClutterRemove.m 포팅

    ── MATLAB 원본 ────────────────────────────────────────────────
    입력: (range_bins, chirps, antennas)
    처리:
      inputMat = bsxfun(@times, inputMat, dopplerWindowCoeffVec.')
        → Hann 윈도우 (chirp 축 적용)
      inputMat = inputMat - repmat(mean(inputMat'), size,1)'
        → clutter 제거: chirp 축 평균 제거 (정지 물체 신호 제거)
          정지 물체는 모든 chirp에서 동일한 위상 → 평균 빼면 사라짐
      fftOutput = fft(inputMat, dopplerFFTSize, 2)
        → chirp 축 FFT
      fftOutput = fftshift(fftOutput, 2)
        → 0속도를 중앙으로 이동
    출력: (range_bins, doppler_bins, antennas)

    ── Python 변환 ────────────────────────────────────────────────
    입력: (range_bins=256, loops=128, virt_ant=12)
    출력: (range_bins=256, doppler_bins=128, virt_ant=12)
    """
    data = virt.copy()  # (256, 128, 12)

    # Hann 윈도우 (loops 축 = axis=1)
    win  = np.hanning(NUM_LOOPS).astype(np.float32).reshape(1, -1, 1)
    data = data * win

    # Clutter 제거: 각 range bin의 loop 평균 제거
    # MATLAB: mean(inputMat') → loop 축 평균 → (range_bins, 1, antennas)
    if clutter_remove:
        data = data - data.mean(axis=1, keepdims=True)

    # Doppler FFT (loops 축 = axis=1)
    dfft = np.fft.fft(data, n=DOPPLER_FFT_SIZE, axis=1).astype(np.complex64)

    # fftshift: 0속도 중앙 정렬
    dfft = np.fft.fftshift(dfft, axes=1)
    # 출력: (256, 128, 12)

    return dfft


# ─────────────────────────────────────────────────────────────────────
# STEP 4-1. Range CFAR  (CFAR_CASO_Range.m 포팅)
# ─────────────────────────────────────────────────────────────────────
def cfar_range(sig_integrate: np.ndarray) -> np.ndarray:
    """
    CFAR_CASO_Range.m 포팅

    ── MATLAB 원본 ────────────────────────────────────────────────
    입력: sig (range_bins × doppler_bins) 실수
    처리:
      각 Doppler bin에 대해 range 방향 슬라이딩 윈도우:
        vecLeft/vecRight: 경계값 복사로 패딩 (비순환)
        cellave1a = 왼쪽 cellNum개 평균
        cellave1b = 오른쪽 cellNum개 평균
        CASO: cellave1 = min(cellave1a, cellave1b)
        maxEnable=1: 윈도우 내 최댓값일 때만 검출
    출력: Ind_obj (range_idx, doppler_idx), 1-indexed

    ── Python 변환 ────────────────────────────────────────────────
    입력: (range_bins=256, doppler_bins=128)
    출력: (N, 2) int32 [range_idx, doppler_idx], 0-indexed
    """
    cell_num = CFAR_REF_WIN[0]
    gap_num  = CFAR_GUARD_WIN[0]
    K0       = CFAR_K0[0]
    gap_tot  = cell_num + gap_num  # 12

    M_samp = sig_integrate.shape[0]  # range_bins = 256
    N_pul  = sig_integrate.shape[1]  # doppler_bins = 128

    dl = CFAR_DISCARD_LEFT   # 5
    dr = CFAR_DISCARD_RIGHT  # 2

    detections = []

    for k in range(N_pul):  # 각 Doppler bin 순회
        sigv = sig_integrate[:, k]  # (256,)

        # 유효 구간: 양 끝 dl, dr 버리기
        vec_core  = sigv[dl : M_samp - dr]          # (249,)
        n_core    = len(vec_core)

        # 경계 패딩: 끝값 복사 (MATLAB과 동일)
        vec_left  = vec_core[:gap_tot]               # 첫 12개
        vec_right = vec_core[-gap_tot:]              # 마지막 12개
        vec       = np.concatenate([vec_left, vec_core, vec_right])  # (273,)

        for j in range(n_core):
            j_vec = j + gap_tot  # vec 내 현재 CUT 인덱스

            # Training cells (guard 제외)
            left_cells  = vec[j_vec - gap_tot : j_vec - gap_num]
            right_cells = vec[j_vec + gap_num + 1 : j_vec + gap_tot + 1]

            if len(left_cells) == 0 or len(right_cells) == 0:
                continue

            # CASO: 두 평균 중 작은 것
            left_avg  = np.mean(left_cells)
            right_avg = np.mean(right_cells)
            cellave   = min(left_avg, right_avg)

            cut = vec[j_vec]  # Cell Under Test

            # maxEnable=1: 윈도우 내 최댓값이어야 함
            all_train  = np.concatenate([left_cells, right_cells])
            max_in_win = np.max(all_train)

            if cut > K0 * cellave and cut >= max_in_win:
                range_idx = j + dl  # 0-indexed 실제 range bin
                detections.append([range_idx, k])

    if len(detections) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    return np.array(detections, dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────
# STEP 4-2. Doppler CFAR + TDM 위상 보정
#           (CFAR_CASO_Doppler_overlap.m + datapath.m 포팅)
# ─────────────────────────────────────────────────────────────────────
def cfar_doppler_phase(
    ind_obj_range: np.ndarray,
    sig_cpml: np.ndarray,
    sig_integrate: np.ndarray
) -> List[dict]:
    """
    CFAR_CASO_Doppler_overlap.m + datapath.m 포팅

    ── MATLAB 원본 ────────────────────────────────────────────────
    1단계 결과(range 검출)에서 각 range bin에 대해 Doppler CFAR:
      - Doppler 방향 순환 경계 (wrap-around)
      - Range와 Doppler 둘 다 검출 시에만 최종 확정 (overlap)

    TDM MIMO 위상 보정 (datapath.m):
      TDM에서 각 TX는 서로 다른 chirp 시점에 송신
      → TX1보다 TX2는 1 chirp 늦게, TX3은 2 chirp 늦게 송신
      → 같은 속도라도 TX마다 누적 위상이 다름
      → 보정: sig_bin[TX_i] *= exp(-j * i * deltaPhi)
      deltaPhi = 2π × doppler_shifted / (NUM_TX × DOPPLER_FFT_SIZE)

    입력:
      ind_obj_range: (N, 2) Range CFAR 결과 [range_idx, doppler_idx]
      sig_cpml:     (range_bins, doppler_bins, 12) 복소수
      sig_integrate: (range_bins, doppler_bins) 실수
    출력:
      list of dict: {range, doppler, range_idx, doppler_idx, bin_val(12,)}
    """
    cell_num = CFAR_REF_WIN[1]
    gap_num  = CFAR_GUARD_WIN[1]
    K0       = CFAR_K0[1]
    gap_tot  = cell_num + gap_num  # 6

    N_pul = sig_integrate.shape[1]  # doppler_bins = 128

    detected_range_cells = np.unique(ind_obj_range[:, 0])
    final_detections = []
    seen = set()

    for rng_cell in detected_range_cells:
        sigv = sig_integrate[rng_cell, :]  # (128,)

        # 순환 경계 패딩 (wrap-around)
        vec = np.concatenate([sigv[-gap_tot:], sigv, sigv[:gap_tot]])

        # 이 range bin에서 range CFAR가 찾은 doppler 인덱스 집합
        mask      = ind_obj_range[:, 0] == rng_cell
        dop_cands = set(ind_obj_range[mask, 1].tolist())

        for j in range(N_pul):
            j_vec = j + gap_tot

            left_cells  = vec[j_vec - gap_tot : j_vec - gap_num]
            right_cells = vec[j_vec + gap_num + 1 : j_vec + gap_tot + 1]

            if len(left_cells) == 0 or len(right_cells) == 0:
                continue

            left_avg  = np.mean(left_cells)
            right_avg = np.mean(right_cells)
            cellave   = min(left_avg, right_avg)
            cut       = vec[j_vec]

            if cut > K0 * cellave:
                # Overlap 확인: range와 doppler 둘 다 검출된 경우만
                if j not in dop_cands:
                    continue

                # 파워 임계값
                if np.min(np.abs(sig_cpml[rng_cell, j, :]) ** 2) < CFAR_POWER_THRE:
                    continue

                # 중복 제거
                key = (int(rng_cell), int(j))
                if key in seen:
                    continue
                seen.add(key)

                # ── TDM MIMO 위상 보정 (datapath.m) ────────────────
                # fftshift 기준 0속도 위치 = DOPPLER_FFT_SIZE//2
                doppler_shifted = j - DOPPLER_FFT_SIZE // 2
                delta_phi = (2 * np.pi * doppler_shifted
                             / (NUM_TX * DOPPLER_FFT_SIZE))

                sig_bin_org = sig_cpml[rng_cell, j, :].copy()  # (12,)
                sig_bin     = sig_bin_org.copy()

                for tx_idx in range(NUM_TX):
                    rx_s = slice(tx_idx * NUM_RX, (tx_idx + 1) * NUM_RX)
                    sig_bin[rx_s] = (sig_bin_org[rx_s]
                                     * np.exp(-1j * tx_idx * delta_phi))

                # 물리 단위 변환
                range_m   = rng_cell * RANGE_RES
                doppler_v = doppler_shifted * VEL_RES

                final_detections.append({
                    'range_idx':   int(rng_cell),
                    'doppler_idx': int(j),
                    'range':       float(range_m),
                    'doppler':     float(doppler_v),
                    'bin_val':     sig_bin,   # (12,) 위상 보정된 복소수
                    'noise_var':   float(cellave),
                })

    return final_detections


# ─────────────────────────────────────────────────────────────────────
# STEP 5-1. DOA 피크 검출  (DOA_BF_PeakDet_loc.m 포팅)
# ─────────────────────────────────────────────────────────────────────
def doa_peak_detect(
    in_data: np.ndarray,
    sidelobe_db: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    DOA_BF_PeakDet_loc.m 포팅

    ── MATLAB 원본 ────────────────────────────────────────────────
    gamma 기반 피크 검출:
      - 신호가 minVal × gamma 이상으로 오르면 피크 탐색 시작
      - maxVal / gamma 이하로 내려가면 피크 확정
      - 피크 중 absMaxValue × 10^(-sidelobeLevel/10) 미만 제거
        (사이드로브 억제)
      - 순환: 스펙트럼이 wrap-around 되므로 extendLoc만큼 추가 탐색

    입력: in_data (1D 각도 스펙트럼)
    출력: (peakVal, peakLoc)
    """
    gamma    = DOA_GAMMA
    in_data  = np.asarray(in_data, dtype=np.float64).ravel()
    N        = len(in_data)

    min_val    = np.inf
    max_val    = 0.0
    max_loc    = 0
    max_loc_r  = 0
    locate_max = False
    init_stage = True
    abs_max    = 0.0
    extend_loc = 0
    max_data   = []

    i = 0
    while i < (N + extend_loc - 1):
        i_loc   = i % N
        cur_val = in_data[i_loc]

        if cur_val > abs_max:
            abs_max = cur_val
        if cur_val > max_val:
            max_val   = cur_val
            max_loc   = i_loc
            max_loc_r = i
        if cur_val < min_val:
            min_val = cur_val

        if locate_max:
            if cur_val < max_val / gamma:
                max_data.append([max_loc, max_val, i - max_loc_r, max_loc_r])
                min_val    = cur_val
                locate_max = False
        else:
            if cur_val > min_val * gamma:
                locate_max = True
                max_val    = cur_val
                if init_stage:
                    extend_loc = i
                    init_stage = False
        i += 1

    # 사이드로브 제거: absMaxValue × 10^(-sidelobe_db/10) 미만 제거
    threshold = abs_max * (10 ** (-sidelobe_db / 10))
    max_data  = [m for m in max_data if m[1] >= threshold]

    if not max_data:
        return np.array([]), np.array([], dtype=np.int32)

    peak_val = np.array([m[1] for m in max_data])
    peak_loc = np.array([m[0] % N for m in max_data], dtype=np.int32)

    return peak_val, peak_loc


# ─────────────────────────────────────────────────────────────────────
# STEP 5-2. 2D DOA  (DOA_beamformingFFT_2D_joint.m 포팅)
# ─────────────────────────────────────────────────────────────────────
def doa_2d_joint(sig: np.ndarray) -> List[Tuple[float, float]]:
    """
    DOA_beamformingFFT_2D_joint.m 포팅
    AWR2243BOOST 안테나 geometry 적용

    ── MATLAB 원본 ────────────────────────────────────────────────
    입력: sig (12,) — CFAR 검출 포인트의 12개 가상 안테나 복소수 신호
    처리:
      1. ANT_D 행렬로 sig를 2D 격자에 배치 (수평 × 수직)
         AWR2243BOOST: (8, 2) 격자
           수직=0: TX1×RX0~3(pos 1~4), TX3×RX0~3(pos 5~8)
           수직=1: TX2×RX0~3(pos 3~6)
         → 수평 pos 3,4,5,6에서 수직 0,1 둘 다 있음 → elevation 추정 가능

      2. 수평 FFT (azimuth): axis=0 → azimuth 스펙트럼
      3. 수직 FFT (elevation): axis=1 → elevation 스펙트럼

      4. Joint 방식:
         [spec_azim, peak_ele_ind] = max(abs(spec_2d'))
         → 각 azimuth 위치에서 elevation 최댓값 추출 (가장 강한 elevation)
         azimuth 방향으로 여러 피크 검출 (복수 물체)
         elevation은 azimuth당 1개

      5. 각도 변환:
         azimuth: sin(az) = wx / (2π × antDis) / cos(el)  (elevation 보정)
         elevation: sin(el) = wz / (2π × antDis)

    출력: list of (azimuth_deg, elevation_deg)
    """
    d        = ANT_DIS              # 0.5
    D_idx    = ANT_D + 1            # 1-indexed로 변환 (MATLAB 원본)
    fft_size = DOA_FFT_SIZE         # 128

    ap_az = int(D_idx[:, 0].max())  # 수평 최대 인덱스 = 8
    ap_el = int(D_idx[:, 1].max())  # 수직 최대 인덱스 = 2

    # 2D 격자 구성: sig_2d (ap_az=8, ap_el=2)
    sig_2d = np.zeros((ap_az, ap_el), dtype=np.complex64)

    for i_line in range(1, ap_el + 1):
        idx   = np.where(D_idx[:, 1] == i_line)[0]
        d_sel = D_idx[idx, 0]
        s_sel = sig[idx]
        _, ind_u = np.unique(d_sel, return_index=True)
        for ii in ind_u:
            sig_2d[d_sel[ii] - 1, i_line - 1] = s_sel[ii]

    # 수평 FFT → azimuth 스펙트럼 (axis=0)
    spec_1d = np.fft.fftshift(
        np.fft.fft(sig_2d, n=fft_size, axis=0), axes=0
    )  # (128, 2)

    # 수직 FFT → elevation 스펙트럼 (axis=1)
    spec_2d = np.fft.fftshift(
        np.fft.fft(spec_1d, n=fft_size, axis=1), axes=1
    )  # (128, 128)

    # 각도 벡터
    wx_vec = np.linspace(-np.pi, np.pi, fft_size, endpoint=False)
    wz_vec = np.linspace(-np.pi, np.pi, fft_size, endpoint=False)
    el_vec = np.degrees(np.arcsin(np.clip(wz_vec / (2 * np.pi * d), -1, 1)))

    results = []

    if ap_el == 1:
        # 수직 배열 없음: azimuth만 추정
        spec_az = np.abs(spec_1d[:, 0])
        _, peak_locs = doa_peak_detect(spec_az, DOA_SIDELOBE_AZ)
        for idx in peak_locs:
            sin_az = wx_vec[idx] / (2 * np.pi * d)
            if abs(sin_az) < 1:
                az = float(np.degrees(np.arcsin(sin_az)))
                if DOA_AZ_RANGE[0] <= az <= DOA_AZ_RANGE[1]:
                    results.append((az, 0.0))
    else:
        # Joint: 각 azimuth 위치에서 elevation 최댓값 추출
        spec_2d_abs = np.abs(spec_2d)                   # (128, 128)
        peak_el_ind = np.argmax(spec_2d_abs, axis=1)    # (128,) elevation 최강 위치
        spec_az     = np.max(spec_2d_abs, axis=1)       # (128,) azimuth 스펙트럼

        _, peak_locs_az = doa_peak_detect(spec_az, DOA_SIDELOBE_AZ)

        for az_idx in peak_locs_az:
            el_idx  = int(peak_el_ind[az_idx])
            el_est  = el_vec[el_idx]

            # Elevation 보정 azimuth 계산 (MATLAB 원본)
            az_sind = ((wx_vec[az_idx] / (2 * np.pi * d))
                       / np.cos(np.radians(el_est)))

            if abs(az_sind) < 1:
                az_est = float(np.degrees(np.arcsin(az_sind)))
                if (DOA_AZ_RANGE[0] <= az_est <= DOA_AZ_RANGE[1]
                        and DOA_EL_RANGE[0] <= el_est <= DOA_EL_RANGE[1]):
                    results.append((az_est, float(el_est)))

    return results


# ─────────────────────────────────────────────────────────────────────
# 메인 처리 함수
# ─────────────────────────────────────────────────────────────────────
def process(raw: np.ndarray) -> np.ndarray:
    """
    전체 신호처리 파이프라인

    입력: (frames, chirps_total=384, rx=4, samples=256)
    출력: (N, 5) float32
          컬럼: [range(m), azimuth(deg), elevation(deg), velocity(m/s), intensity]
    """
    all_points = []

    for f in range(raw.shape[0]):
        frame = raw[f]  # (384, 4, 256)

        # 1. Range FFT
        rfft = range_fft(frame)                              # (256, 384, 4)

        # 2. TX 분리 + 가상 안테나
        virt = form_virtual_array(rfft)                      # (256, 128, 12)

        # 3. Doppler FFT
        dfft = doppler_fft(virt, clutter_remove=True)        # (256, 128, 12)

        # 4. CFAR
        # 안테나 전체 파워 합산 (datapath.m: sum(abs(input).^2, 3) + 1)
        sig_integrate = np.sum(np.abs(dfft) ** 2, axis=2) + 1.0  # (256, 128)

        ind_range = cfar_range_vec(sig_integrate)                # (N, 2)
        if len(ind_range) == 0:
            continue

        detections = cfar_doppler_phase_vec(ind_range, dfft, sig_integrate, RANGE_RES, VEL_RES)

        # 5. 2D DOA
        for det in detections:
            angles = doa_2d_joint(det['bin_val'])
            for az, el in angles:
                intensity = float(np.sum(np.abs(det['bin_val']) ** 2))
                all_points.append([
                    det['range'],
                    az,
                    el,
                    det['doppler'],
                    intensity,
                ])

    if not all_points:
        return np.zeros((0, 5), dtype=np.float32)

    return np.array(all_points, dtype=np.float32)