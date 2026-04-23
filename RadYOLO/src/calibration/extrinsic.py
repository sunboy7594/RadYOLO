"""
extrinsic.py
============
레이더-카메라 외부 캘리브레이션 (온라인 자동)

변환 구조:
  레이더 구면 좌표 (range, azimuth, elevation)
    → 레이더 직교 좌표 (Xr, Yr, Zr)
    → 카메라 직교 좌표 (Xc, Yc, Zc)  ← 4×4 변환 행렬
    → 카메라 픽셀 좌표 (px, py)       ← 내부 파라미터

물리 배치:
  카메라가 레이더보다 약 13cm 위, 약간 뒤, 약간 옆
  초기값: ty=-0.13, 나머지 0

수렴 조건:
  정확도 ≥ 85% AND 수집 ≥ 30개 AND 최근 3회 변화 < 2%

이상 감지:
  새 행렬 오차 > 이전 오차 × 2배, 연속 3회 → 초기값 재시작
"""

import numpy as np
import json
import os
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

# ── 경로 ──────────────────────────────────────────────────────────
CALIB_PATH = "config/calibration/extrinsic.json"

# ── 카메라 내부 파라미터 ───────────────────────────────────────────
# Lenovo Performance FHD 웹캠, 1920×1080, FOV 95°
CAM_W  = 1920
CAM_H  = 1080
CAM_FX = CAM_W / (2 * np.tan(np.radians(95.0 / 2)))   # ~1003
CAM_FY = CAM_FX
CAM_CX = CAM_W / 2.0
CAM_CY = CAM_H / 2.0

# ── 캘리브레이션 파라미터 ──────────────────────────────────────────
MIN_POINTS_CALC   = 10    # 첫 계산 최소 대응점
CONVERGE_POINTS   = 30    # 수렴 판단 최소 대응점
MAX_POINTS        = 200   # 최대 누적 대응점
UPDATE_EVERY      = 10    # N개 추가마다 행렬 재계산
CONVERGE_ACC      = 85.0  # 수렴 정확도 기준 (%)
CONVERGE_DELTA    = 2.0   # 최근 3회 정확도 변화 기준 (%)
RESET_CONSECUTIVE = 3     # 연속 이상 감지 횟수 → 재시작

# 다양성 기준
DIV_PX_STD    = 80.0   # 위치 다양성: 픽셀 표준편차 (px)
DIV_RANGE_STD = 0.5    # 거리 다양성: range 표준편차 (m)
DIV_MIN_SCORE = 2      # 3점 중 최소 점수


@dataclass
class CorrespondencePoint:
    """대응점 하나"""
    px:         float   # 카메라 픽셀 x
    py:         float   # 카메라 픽셀 y
    xr:         float   # 레이더 직교 x (m)
    yr:         float   # 레이더 직교 y (m)
    zr:         float   # 레이더 직교 z (m)
    class_id:   int     # YOLO 클래스 ID
    conf:       float   # YOLO confidence


class ExtrinsicCalibration:
    """
    레이더-카메라 외부 캘리브레이션 클래스

    사용법:
        calib = ExtrinsicCalibration()
        calib.load()

        # 매 프레임
        calib.add_point(px, py, radar_point, class_id, conf)
        px_proj, py_proj = calib.project(radar_point)

        # 화면 표시
        text = calib.status_text()
    """

    def __init__(self):
        # 변환 행렬 초기값 (레이더 → 카메라)
        # 카메라가 레이더보다 13cm 위 → ty = -0.13
        self.T = np.eye(4, dtype=np.float64)
        self.T[1, 3] = -0.13   # ty: 카메라가 위에 있으니 레이더 기준 -y

        self._init_T    = self.T.copy()   # 초기값 보존

        self._points:   List[CorrespondencePoint] = []
        self._accuracy  = 0.0
        self._error_px  = 999.0
        self._acc_hist: List[float] = []  # 최근 3회 정확도 이력
        self._bad_cnt   = 0               # 연속 이상 감지 횟수
        self._converged = False
        self._n_since_update = 0          # 마지막 업데이트 이후 추가 수

    # ── 공개 API ──────────────────────────────────────────────────
    def load(self):
        """저장된 캘리브레이션 로드"""
        if not os.path.exists(CALIB_PATH):
            print("[캘리브레이션] 저장 파일 없음 → 초기값 사용")
            return

        try:
            with open(CALIB_PATH, 'r') as f:
                data = json.load(f)
            self.T         = np.array(data['T'])
            self._accuracy = data.get('accuracy', 0.0)
            self._error_px = data.get('error_px', 999.0)
            self._converged = data.get('converged', False)
            print(f"[캘리브레이션] 로드 완료 — 정확도 {self._accuracy:.1f}%, 오차 {self._error_px:.1f}px")
        except Exception as e:
            print(f"[캘리브레이션] 로드 실패 ({e}) → 초기값 사용")

    def save(self):
        """현재 캘리브레이션 저장"""
        os.makedirs(os.path.dirname(CALIB_PATH), exist_ok=True)
        data = {
            'T':          self.T.tolist(),
            'accuracy':   self._accuracy,
            'error_px':   self._error_px,
            'converged':  self._converged,
            'n_points':   len(self._points),
        }
        with open(CALIB_PATH, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[캘리브레이션] 저장 완료 — {CALIB_PATH}")

    def add_point(
        self,
        px: float, py: float,
        radar_pt: np.ndarray,
        class_id: int,
        conf: float,
    ):
        """
        대응점 추가

        입력:
          px, py:    카메라 픽셀 좌표
          radar_pt:  (5,) [range, azimuth, elevation, velocity, intensity]
          class_id:  YOLO 클래스 ID
          conf:      YOLO confidence
        """
        if self._converged:
            return

        # 레이더 구면 → 직교 좌표
        r, az, el = float(radar_pt[0]), float(radar_pt[1]), float(radar_pt[2])
        xr = r * np.cos(np.radians(el)) * np.sin(np.radians(az))
        yr = r * np.sin(np.radians(el))
        zr = r * np.cos(np.radians(el)) * np.cos(np.radians(az))

        self._points.append(CorrespondencePoint(
            px=px, py=py,
            xr=xr, yr=yr, zr=zr,
            class_id=class_id, conf=conf,
        ))

        # 최대 개수 초과 시 오래된 것 제거
        if len(self._points) > MAX_POINTS:
            self._points = self._points[-MAX_POINTS:]

        self._n_since_update += 1

        # UPDATE_EVERY개 추가마다 행렬 재계산
        if (self._n_since_update >= UPDATE_EVERY
                and len(self._points) >= MIN_POINTS_CALC):
            self._n_since_update = 0
            self._update()

    def project(self, radar_pt: np.ndarray) -> Tuple[float, float]:
        """
        레이더 포인트 → 카메라 픽셀 좌표 투영

        입력: (5,) [range, azimuth, elevation, ...]
        출력: (px, py)
        """
        r, az, el = float(radar_pt[0]), float(radar_pt[1]), float(radar_pt[2])
        xr = r * np.cos(np.radians(el)) * np.sin(np.radians(az))
        yr = r * np.sin(np.radians(el))
        zr = r * np.cos(np.radians(el)) * np.cos(np.radians(az))

        # 레이더 직교 → 카메라 직교
        p_r = np.array([xr, yr, zr, 1.0])
        p_c = self.T @ p_r
        xc, yc, zc = p_c[0], p_c[1], p_c[2]

        if zc <= 0:
            return CAM_CX, CAM_CY

        # 카메라 직교 → 픽셀
        px = CAM_FX * xc / zc + CAM_CX
        py = CAM_FY * yc / zc + CAM_CY

        return float(px), float(py)

    def status_text(self) -> str:
        """화면 표시용 상태 텍스트"""
        if self._converged:
            return f"CALIB: 수렴 완료 | 오차 {self._error_px:.1f}px | 정확도 {self._accuracy:.0f}%"
        return (f"CALIB: 수집 {len(self._points)}개 | "
                f"오차 {self._error_px:.1f}px | "
                f"정확도 {self._accuracy:.0f}%")

    # ── 내부 메서드 ───────────────────────────────────────────────
    def _check_diversity(self) -> bool:
        """다양성 점수 계산 (3점 중 2점 이상)"""
        pts = self._points
        score = 0

        # 1. 위치 다양성
        px_arr = np.array([p.px for p in pts])
        py_arr = np.array([p.py for p in pts])
        if np.std(px_arr) >= DIV_PX_STD or np.std(py_arr) >= DIV_PX_STD:
            score += 1

        # 2. 거리 다양성
        rng_arr = np.array([np.sqrt(p.xr**2 + p.yr**2 + p.zr**2) for p in pts])
        if np.std(rng_arr) >= DIV_RANGE_STD:
            score += 1

        # 3. 객체 다양성
        unique_classes = len(set(p.class_id for p in pts))
        if unique_classes >= 2:
            score += 1

        return score >= DIV_MIN_SCORE

    def _compute_transform(self) -> Optional[np.ndarray]:
        """
        대응점으로 4×4 변환 행렬 계산 (SVD 최소자승법)

        레이더 직교 좌표 → 카메라 픽셀 좌표 역투영해서
        3D-3D 대응으로 변환 행렬 추정
        """
        pts = self._points

        # 카메라 픽셀 → 카메라 직교 (깊이는 레이더 range 사용)
        src = []  # 레이더 직교
        dst = []  # 카메라 직교 (추정)

        for p in pts:
            r = np.sqrt(p.xr**2 + p.yr**2 + p.zr**2)
            if r < 0.1:
                continue
            # 픽셀 → 카메라 직교 (range로 스케일)
            xc = (p.px - CAM_CX) / CAM_FX * r
            yc = (p.py - CAM_CY) / CAM_FY * r
            zc = r
            src.append([p.xr, p.yr, p.zr])
            dst.append([xc, yc, zc])

        if len(src) < MIN_POINTS_CALC:
            return None

        src = np.array(src)  # (N, 3) 레이더 직교
        dst = np.array(dst)  # (N, 3) 카메라 직교

        # 중심화
        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)
        src_c    = src - src_mean
        dst_c    = dst - dst_mean

        # SVD로 회전 행렬 계산
        H  = src_c.T @ dst_c
        U, S, Vt = np.linalg.svd(H)
        R  = Vt.T @ U.T

        # 반사 보정
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # 이동 벡터
        t = dst_mean - R @ src_mean

        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = t

        return T

    def _compute_error(self, T: np.ndarray) -> float:
        """재투영 오차 계산 (픽셀 거리 평균)"""
        errors = []
        for p in self._points:
            p_r = np.array([p.xr, p.yr, p.zr, 1.0])
            p_c = T @ p_r
            xc, yc, zc = p_c[0], p_c[1], p_c[2]
            if zc <= 0:
                continue
            px_proj = CAM_FX * xc / zc + CAM_CX
            py_proj = CAM_FY * yc / zc + CAM_CY
            err = np.sqrt((px_proj - p.px)**2 + (py_proj - p.py)**2)
            errors.append(err)

        return float(np.mean(errors)) if errors else 999.0

    def _error_to_accuracy(self, error_px: float) -> float:
        """오차 → 정확도 변환"""
        return float(max(0.0, 100.0 - error_px * 2.0))

    def _update(self):
        """행렬 재계산 + 수렴/이상 판단"""
        # 다양성 체크 (50개 이상이면 완화)
        if len(self._points) < 50 and not self._check_diversity():
            print(f"[캘리브레이션] 다양성 부족 ({len(self._points)}개) — 대기")
            return

        T_new = self._compute_transform()
        if T_new is None:
            return

        err_new = self._compute_error(T_new)
        acc_new = self._error_to_accuracy(err_new)

        # ── 이상 감지 ────────────────────────────────────────────
        prev_err = self._error_px
        if prev_err < 900 and err_new > prev_err * 2.0:
            self._bad_cnt += 1
            print(f"[캘리브레이션] 이상 감지 {self._bad_cnt}회 "
                  f"(오차 {prev_err:.1f} → {err_new:.1f}px)")
            if self._bad_cnt >= RESET_CONSECUTIVE:
                print("[캘리브레이션] 초기값으로 재시작")
                self.T          = self._init_T.copy()
                self._points    = []
                self._error_px  = 999.0
                self._accuracy  = 0.0
                self._acc_hist  = []
                self._bad_cnt   = 0
            return
        else:
            self._bad_cnt = 0

        # ── 행렬 업데이트 ────────────────────────────────────────
        self.T         = T_new
        self._error_px = err_new
        self._accuracy = acc_new
        self._acc_hist.append(acc_new)
        if len(self._acc_hist) > 3:
            self._acc_hist = self._acc_hist[-3:]

        print(f"[캘리브레이션] 업데이트 — "
              f"수집 {len(self._points)}개 | "
              f"오차 {err_new:.1f}px | 정확도 {acc_new:.1f}%")

        # ── 수렴 판단 ────────────────────────────────────────────
        if (acc_new >= CONVERGE_ACC
                and len(self._points) >= CONVERGE_POINTS
                and len(self._acc_hist) == 3
                and max(self._acc_hist) - min(self._acc_hist) < CONVERGE_DELTA):
            self._converged = True
            print(f"[캘리브레이션] 수렴 완료! 정확도 {acc_new:.1f}%")
            self.save()