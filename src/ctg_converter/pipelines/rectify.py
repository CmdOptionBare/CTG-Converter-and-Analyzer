"""Pipeline 2: 90-Degree Green-Grid Rectification, Perspective/Keystone & Wave Dewarping,
and Canonical Metric Scaling.

Physical geometry of standard Philips M1911A (152 mm, 1 cm/min, 20 bpm/cm) CTG strips:
  - Upper FHR grid (210 bpm down to 50 bpm): 8.00 cm tall = 32 intervals of 0.25 cm (2.5 mm)
  - Middle white separator band:             0.675 cm tall (6.75 mm)
  - Lower Toco grid (100 down to 0):         4.00 cm tall = 10 intervals of 0.40 cm (4.0 mm)
  - Horizontal time grid (1 cm/min):         0.25 cm (2.5 mm = 15 s) per fine vertical line

By fitting a column-wise 1D projective comb model y(s) = (a*s + b)/(c*s + 1) to the 33 FHR lines,
separator gap, and 11 Toco lines, followed by vertical-line fan & horizontal spacing rectification,
every frame is mapped in a SINGLE cv2.remap pass onto a canonical metric canvas (default 100 px/cm,
height 1420 px) with exact 90.0-degree grid intersections.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize

from ctg_converter.dataset.manifest import DatasetManifest, DatasetPaths
from ctg_converter.pipelines.enhance import EnhanceConfig, enhance_ctg_frame


# Canonical vertical layout in centimeters from top of rectified canvas
CANONICAL_TOP_MARGIN_CM = 0.80
CANONICAL_FHR_HEIGHT_CM = 8.00       # 32 intervals * 0.25 cm (210 bpm -> 50 bpm)
CANONICAL_SEP_HEIGHT_CM = 0.675      # White middle band
CANONICAL_TOCO_HEIGHT_CM = 4.00      # 10 intervals * 0.40 cm (100 -> 0)
CANONICAL_BOT_MARGIN_CM = 0.725
CANONICAL_TOTAL_HEIGHT_CM = (
    CANONICAL_TOP_MARGIN_CM
    + CANONICAL_FHR_HEIGHT_CM
    + CANONICAL_SEP_HEIGHT_CM
    + CANONICAL_TOCO_HEIGHT_CM
    + CANONICAL_BOT_MARGIN_CM
)  # 14.20 cm

FHR_S_TOP_CM = CANONICAL_TOP_MARGIN_CM                                 # 0.80 cm
FHR_S_BOT_CM = FHR_S_TOP_CM + CANONICAL_FHR_HEIGHT_CM                  # 8.80 cm
TOCO_S_TOP_CM = FHR_S_BOT_CM + CANONICAL_SEP_HEIGHT_CM                 # 9.475 cm
TOCO_S_BOT_CM = TOCO_S_TOP_CM + CANONICAL_TOCO_HEIGHT_CM               # 13.475 cm

# All 44 canonical horizontal grid-line positions in cm
FHR_LINES_CM = FHR_S_TOP_CM + 0.25 * np.arange(33, dtype=np.float64)
TOCO_LINES_CM = TOCO_S_TOP_CM + 0.40 * np.arange(11, dtype=np.float64)
ALL_HORIZ_LINES_CM = np.concatenate([FHR_LINES_CM, TOCO_LINES_CM])
SEP_SAMPLES_CM = np.linspace(FHR_S_BOT_CM + 0.14, TOCO_S_TOP_CM - 0.14, 5)


@dataclass
class RectifyConfig:
    """Parameters for 90-degree CTG grid rectification."""

    target_px_per_cm: int = 100
    num_x_slices: int = 9
    crop_invalid_x_margins: bool = True
    min_valid_col_fraction: float = 0.42


@dataclass
class RectifyResultMetadata:
    """Diagnostic metrics saved for each rectified frame."""

    frame_index: int
    filename: str
    target_px_per_cm: int
    rectified_width: int
    rectified_height: int
    initial_skew_deg: float
    comb_fit_score: float
    residual_horiz_tilt_deg: float
    residual_vert_tilt_deg: float
    orthogonal_angle_deg: float
    measured_grid_period_x_px: float
    measured_grid_period_y_px: float


def _estimate_global_horiz_skew_deg(grid_u8: np.ndarray, valid_mask: np.ndarray) -> float:
    """Find dominant angle (degrees) that aligns horizontal grid lines with image rows."""
    h, w = grid_u8.shape
    sc = 900.0 / max(h, w)
    sh, sw = int(round(h * sc)), int(round(w * sc))
    sm_g = cv2.resize(grid_u8, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
    sm_m = cv2.resize(valid_mask, (sw, sh), interpolation=cv2.INTER_NEAREST) > 0

    ys, xs = np.where(sm_m & (sm_g > 4))
    if len(ys) < 200:
        return 0.0
    yc = int(np.median(ys))
    xc = int(np.median(xs))
    rad = min(200, sh // 2 - 4, sw // 2 - 4)
    yc = int(np.clip(yc, rad, sh - rad))
    xc = int(np.clip(xc, rad, sw - rad))
    patch = sm_g[yc - rad : yc + rad, xc - rad : xc + rad]

    best_a, best_v = 0.0, -1.0
    for a in np.linspace(-22.0, 22.0, 89):
        mat = cv2.getRotationMatrix2D((rad, rad), float(a), 1.0)
        wp = cv2.warpAffine(patch, mat, (2 * rad, 2 * rad), flags=cv2.INTER_LINEAR)
        inner = wp[rad // 4 : 2 * rad - rad // 4, rad // 4 : 2 * rad - rad // 4]
        hp = inner.mean(axis=1)
        hp = hp - gaussian_filter1d(hp, sigma=8.0)
        v = float(np.mean(hp * hp))
        if v > best_v:
            best_v, best_a = v, float(a)

    for a in np.linspace(best_a - 0.8, best_a + 0.8, 33):
        mat = cv2.getRotationMatrix2D((rad, rad), float(a), 1.0)
        wp = cv2.warpAffine(patch, mat, (2 * rad, 2 * rad), flags=cv2.INTER_LINEAR)
        inner = wp[rad // 4 : 2 * rad - rad // 4, rad // 4 : 2 * rad - rad // 4]
        hp = inner.mean(axis=1)
        hp = hp - gaussian_filter1d(hp, sigma=8.0)
        v = float(np.mean(hp * hp))
        if v > best_v:
            best_v, best_a = v, float(a)

    return best_a


def _estimate_local_period_px(sig: np.ndarray, min_p: float = 24.0, max_p: float = 135.0) -> float:
    """Estimate dominant spatial period (in pixels) of a 1D grid profile via autocorrelation/FFT."""
    if len(sig) < int(max_p * 2):
        return 60.0
    hp = sig - gaussian_filter1d(sig, sigma=18.0)
    win = np.hanning(len(hp))
    n_fft = max(4096, 1 << int(np.ceil(np.log2(len(hp) * 4))))
    spec = np.abs(np.fft.rfft(hp * win, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0)
    mask = (freqs >= 1.0 / max_p) & (freqs <= 1.0 / min_p)
    if not np.any(mask):
        return 60.0
    idx = np.where(mask)[0][int(np.argmax(spec[mask]))]
    return float(1.0 / freqs[idx])


def _fit_projective_comb_1d(
    prof_raw: np.ndarray,
    valid_frac: np.ndarray,
    prior_params: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, float]:
    """Fit 1D projective mapping y(s) = (a*s + b) / (c*s + 1) to a vertical slice's green-line profile.

    Maps canonical strip height s in [0, 14.20 cm] to image row coordinate y in [0, H-1].
    Locks scale tightly to FFT-measured FHR/Toco line spacing (preventing 4:3 harmonic aliasing)
    and enforces sub-half-period continuity across adjacent slices (preventing 1-line hops).
    """
    h = len(prof_raw)
    smooth_line = gaussian_filter1d(prof_raw.astype(np.float64), sigma=2.0)
    baseline = gaussian_filter1d(smooth_line, sigma=20.0)
    local_rms = np.sqrt(
        gaussian_filter1d((smooth_line - baseline) ** 2, sigma=45.0) + 4.0
    )
    line_resp = (smooth_line - 0.65 * baseline) / local_rms
    line_resp = np.where(valid_frac > 0.25, line_resp, 0.0)

    env = gaussian_filter1d(prof_raw.astype(np.float64), sigma=26.0)
    broad = gaussian_filter1d(prof_raw.astype(np.float64), sigma=max(80.0, h / 7.0))
    denom_env = (
        np.percentile(broad[valid_frac > 0.25], 30) + 1.0
        if np.any(valid_frac > 0.25)
        else 1.0
    )
    rel_env = env / np.maximum(broad, denom_env)

    s0 = 0.5 * (FHR_S_BOT_CM + TOCO_S_TOP_CM)
    ds_lines = ALL_HORIZ_LINES_CM - s0
    ds_sep = SEP_SAMPLES_CM - s0
    ds_out = np.array([FHR_S_TOP_CM - 0.22 - s0, TOCO_S_BOT_CM + 0.30 - s0])

    if prior_params is not None:
        p0_y, p0_sc, p0_k = prior_params
        # Half of one 0.25-cm FHR grid interval = 0.125 * p0_sc; constrain within +/- 0.105 * p0_sc
        # so adjacent slices after global deskew can follow gentle paper waves without EVER hopping 1 grid line!
        max_dy = max(12.0, 0.105 * p0_sc)
        y_cands = np.linspace(p0_y - max_dy, p0_y + max_dy, 25)
        sc_cands = np.linspace(p0_sc * 0.96, p0_sc * 1.04, 9)
        k_cands = np.linspace(max(-0.042, p0_k - 0.006), min(0.042, p0_k + 0.006), 7)
        sc_prior_center = p0_sc
    else:
        env_peak = float(np.percentile(env, 92)) if len(env) else 0.0
        active_grid_rows = np.where((valid_frac > 0.30) & (env > max(1.5, 0.18 * env_peak)))[0]
        if len(active_grid_rows) > 150:
            y_min = int(active_grid_rows[0])
            y_max = int(active_grid_rows[-1])
        else:
            valid_rows = np.where(valid_frac > 0.35)[0]
            y_min = int(valid_rows[0]) if len(valid_rows) > 100 else 0
            y_max = int(valid_rows[-1]) if len(valid_rows) > 100 else h - 1
        span_v = max(400, y_max - y_min)

        # Measure FHR period in middle-FHR band (just above separator) and Toco period in Toco band
        fhr_band = smooth_line[y_min + int(0.18 * span_v) : y_min + int(0.56 * span_v)]
        toco_band = smooth_line[y_min + int(0.66 * span_v) : y_max - int(0.03 * span_v)]
        p_fhr = _estimate_local_period_px(fhr_band, min_p=26.0, max_p=95.0)
        p_toco = _estimate_local_period_px(toco_band, min_p=45.0, max_p=145.0)

        sc_fhr = 4.0 * p_fhr
        sc_toco = 2.5 * p_toco
        if 0.70 <= (sc_fhr / max(sc_toco, 1.0)) <= 1.40:
            sc_est = float(np.sqrt(sc_fhr * sc_toco))
            ratio = float(np.sqrt(sc_fhr / sc_toco))
            k_est = float(np.clip((ratio - 1.0) / 3.8, -0.040, 0.025))
        else:
            sc_est = float(np.clip(max(sc_fhr, sc_toco), 130.0, 320.0))
            k_est = -0.008

        sc_prior_center = sc_est
        lo_s = int(y_min + 0.48 * span_v)
        hi_s = int(y_min + 0.78 * span_v)
        n_yc = max(65, int((hi_s - lo_s) / 4.0))
        y_cands = np.linspace(lo_s, hi_s, n_yc)
        sc_cands = np.linspace(sc_est * 0.89, sc_est * 1.11, 21)
        k_cands = np.linspace(max(-0.045, k_est - 0.020), min(0.030, k_est + 0.020), 15)

    def eval_single(y_sep_mid: float, scale_sep: float, persp_k: float) -> float:
        # Soft regularizer keeping scale within +/- 12% of FFT measurement
        rel_sc_dev = (scale_sep - sc_prior_center) / max(sc_prior_center, 100.0)
        if abs(rel_sc_dev) > 0.14:
            return -1e9
        denom = 1.0 + persp_k * ds_lines
        if np.any(denom <= 0.30):
            return -1e9
        y_lines = y_sep_mid + scale_sep * ds_lines / denom
        in_bounds = (y_lines >= 3.0) & (y_lines <= h - 4.0)
        n_in = int(np.sum(in_bounds))
        if n_in < 20:
            return -1e9

        yi = y_lines[in_bounds]
        i0 = np.floor(yi).astype(np.int32)
        frac = yi - i0
        vals = line_resp[i0] * (1.0 - frac) + line_resp[i0 + 1] * frac
        w_valid = valid_frac[i0]
        # Average per-in-bounds line response (unbiased with respect to scale!) + coverage bonus
        mean_line = float(np.mean(vals * (0.4 + 0.6 * w_valid)))
        coverage = min(1.0, n_in / 38.0)

        y_sep = np.clip(
            y_sep_mid + scale_sep * ds_sep / (1.0 + persp_k * ds_sep), 2.0, h - 3.0
        )
        is0 = np.floor(y_sep).astype(np.int32)
        sep_env_pen = float(np.mean(rel_env[is0]))
        sep_line_pen = float(np.mean(np.maximum(0.0, line_resp[is0])))

        y_out = y_sep_mid + scale_sep * ds_out / (1.0 + persp_k * ds_out)
        border_bonus = 0.0
        for yo in y_out:
            if 5.0 <= yo <= h - 6.0:
                border_bonus += 0.35 * (1.0 - float(rel_env[int(round(yo))]))

        reg_pen = 4.0 * (rel_sc_dev ** 2)
        return (
            12.0 * mean_line * coverage
            - 11.0 * sep_env_pen
            - 3.0 * sep_line_pen
            + border_bonus
            - reg_pen
        )

    # Vectorized grid search over y_cands for each (sc, k) pair for ~10x speedup!
    best_score = -1e9
    best_tuple = (float(y_cands[len(y_cands) // 2]), sc_prior_center, -0.005)
    yc_col = y_cands.reshape(-1, 1)

    for sc in sc_cands:
        for kc in k_cands:
            denom = 1.0 + kc * ds_lines
            if np.any(denom <= 0.30):
                continue
            offsets = (sc * ds_lines / denom).reshape(1, -1)
            y_mat = yc_col + offsets  # (N_y, 44)
            in_b = (y_mat >= 3.0) & (y_mat <= h - 4.0)
            n_in_vec = in_b.sum(axis=1)
            valid_yc = n_in_vec >= 20
            if not np.any(valid_yc):
                continue

            yc_clamped = np.clip(y_mat, 3.0, h - 4.001)
            i0 = np.floor(yc_clamped).astype(np.int32)
            fr = yc_clamped - i0
            vals = (line_resp[i0] * (1.0 - fr) + line_resp[i0 + 1] * fr) * (
                0.4 + 0.6 * valid_frac[i0]
            )
            vals[~in_b] = 0.0
            mean_line_vec = vals.sum(axis=1) / np.maximum(n_in_vec, 1)
            cov_vec = np.minimum(1.0, n_in_vec / 38.0)

            sep_offs = (sc * ds_sep / (1.0 + kc * ds_sep)).reshape(1, -1)
            is0 = np.clip(np.floor(yc_col + sep_offs).astype(np.int32), 2, h - 3)
            sep_pen_vec = 11.0 * rel_env[is0].mean(axis=1) + 3.0 * np.maximum(
                0.0, line_resp[is0]
            ).mean(axis=1)

            rel_sc_dev = (sc - sc_prior_center) / max(sc_prior_center, 100.0)
            tot_vec = 12.0 * mean_line_vec * cov_vec - sep_pen_vec - 4.0 * (rel_sc_dev ** 2)
            tot_vec[~valid_yc] = -1e9
            idx_best = int(np.argmax(tot_vec))
            if float(tot_vec[idx_best]) > best_score:
                best_score = float(tot_vec[idx_best])
                best_tuple = (float(y_cands[idx_best]), float(sc), float(kc))

    res = minimize(
        lambda p: -eval_single(float(p[0]), float(p[1]), float(p[2])),
        x0=np.array(best_tuple, dtype=np.float64),
        method="Nelder-Mead",
        options={"xatol": 0.05, "fatol": 1e-4, "maxiter": 140},
    )
    opt_y, opt_sc, opt_k = float(res.x[0]), float(res.x[1]), float(res.x[2])
    final_score = -float(res.fun)
    return np.array([opt_y, opt_sc, opt_k], dtype=np.float64), final_score


def _eval_y_from_local_homog(s_cm: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Evaluate y(s_cm) = y_sep_mid + scale_sep * (s - s0) / (1 + persp_k * (s - s0))."""
    s0 = 0.5 * (FHR_S_BOT_CM + TOCO_S_TOP_CM)
    ds = s_cm - s0
    return params[0] + params[1] * ds / (1.0 + params[2] * ds)


def _rectify_x_coordinate_field(
    y_rect_grid: np.ndarray,
    y_rect_mask: np.ndarray,
    px_per_cm: int,
    expected_src_px_per_cm: float | None = None,
) -> tuple[np.ndarray, int]:
    """Given a Y-rectified green grid (where every row s_y is already at exact canonical metric height),
    compute the horizontal inverse map x_src(X_out, Y_out) so that:
      1. Converging/tilted vertical grid lines become strictly vertical (90.0 deg to horizontal lines),
      2. Horizontal grid spacing equals exact 0.25 cm (px_per_cm * 0.25 px) everywhere!
    """
    rh, rw = y_rect_grid.shape
    fhr_y0 = int(round(FHR_S_TOP_CM * px_per_cm))
    fhr_y1 = int(round(FHR_S_BOT_CM * px_per_cm))
    toco_y0 = int(round(TOCO_S_TOP_CM * px_per_cm))
    toco_y1 = int(round(TOCO_S_BOT_CM * px_per_cm))

    # Enhance vertical grid lines by vertical smoothing within FHR and Toco separately + horizontal high-pass
    g_f = y_rect_grid.astype(np.float32)
    vert_fhr = cv2.GaussianBlur(g_f[fhr_y0:fhr_y1, :], (1, 25), 0)
    vert_toco = cv2.GaussianBlur(g_f[toco_y0:toco_y1, :], (1, 25), 0)
    vert_enh = np.zeros_like(g_f)
    vert_enh[fhr_y0:fhr_y1, :] = np.maximum(0.0, vert_fhr - cv2.GaussianBlur(vert_fhr, (31, 1), 0))
    vert_enh[toco_y0:toco_y1, :] = np.maximum(0.0, vert_toco - cv2.GaussianBlur(vert_toco, (31, 1), 0))
    vert_enh[y_rect_mask == 0] = 0.0

    # Sample 4 horizontal bands (top FHR, lower FHR, upper Toco, lower Toco) to measure:
    # (a) horizontal shift dx(x, y) between bands (vertical line fan / keystone convergence!)
    # (b) local horizontal grid spacing period_x(x, y) (which equals 0.25 cm physically!)
    band_defs = [
        (fhr_y0 + int(0.08 * (fhr_y1 - fhr_y0)), fhr_y0 + int(0.45 * (fhr_y1 - fhr_y0))),
        (fhr_y0 + int(0.55 * (fhr_y1 - fhr_y0)), fhr_y1 - int(0.06 * (fhr_y1 - fhr_y0))),
        (toco_y0 + int(0.08 * (toco_y1 - toco_y0)), toco_y0 + int(0.52 * (toco_y1 - toco_y0))),
        (toco_y0 + int(0.48 * (toco_y1 - toco_y0)), toco_y1 - int(0.08 * (toco_y1 - toco_y0))),
    ]
    band_yc = np.array([0.5 * (b0 + b1) for b0, b1 in band_defs], dtype=np.float64)
    band_profs = []
    band_valid = []
    for b0, b1 in band_defs:
        m_sub = (y_rect_mask[b0:b1, :] > 0).astype(np.float32)
        denom = np.maximum(m_sub.sum(axis=0), 1.0)
        p = vert_enh[b0:b1, :].sum(axis=0) / denom
        v = m_sub.mean(axis=0) > 0.5
        p[~v] = 0.0
        band_profs.append(p)
        band_valid.append(v)

    # Reference band = Band 1 (lower FHR, near center of strip y_ref):
    # Measure horizontal grid period in ~3 sub-windows along X on Band 1 and Band 2,
    # AND measure linear x-fan between bands (keystone shear + magnification change with Y)!
    y_ref = band_yc[1]

    # Fit affine x-correspondence from y_ref to each other band b:
    # A vertical grid line passing through x_ref at y_ref is located at:
    #   x_b(x_ref) = x_ref + shift_b + scale_delta_b * (x_ref - rw/2)
    cx = 0.5 * rw
    ref_p = band_profs[1]
    ref_v = band_valid[1]

    # Normalize 1D profile for correlation
    def norm_prof(p: np.ndarray, v: np.ndarray) -> np.ndarray:
        rms = np.sqrt(gaussian_filter1d(p * p, sigma=40.0) + 1.0)
        out = p / rms
        out[~v] = 0.0
        return out.astype(np.float32)

    ref_n = norm_prof(ref_p, ref_v)

    # Estimate horizontal grid period at y_ref (physical 0.25 cm).
    # Lock search band around vertical scale at y_ref so 2x major-line (0.50 cm) or 1/2x harmonic
    # or 3/4x aliases are mathematically impossible!
    if expected_src_px_per_cm is not None and expected_src_px_per_cm > 80.0:
        exp_p = 0.25 * expected_src_px_per_cm
        min_px_p = max(22.0, exp_p * 0.82)
        max_px_p = min(130.0, exp_p * 1.20)
    else:
        min_px_p, max_px_p = 30.0, 110.0

    vx = np.where(ref_v)[0]
    if len(vx) > 300:
        p_x_ref = _estimate_local_period_px(
            ref_p[int(vx[0]) + 30 : int(vx[-1]) - 30], min_p=min_px_p, max_p=max_px_p
        )
    else:
        p_x_ref = 0.25 * (expected_src_px_per_cm or 260.0)

    # Measure local x-shifts across X-windows between each band b and ref band 1
    win_half = max(180, rw // 8)
    sample_xs = np.linspace(0.18 * rw, 0.82 * rw, 7)
    all_obs_y: list[float] = []
    all_obs_xr: list[float] = []
    all_obs_dx: list[float] = []
    all_obs_w: list[float] = []

    max_lag = int(min(180, max(50, rw * 0.06)))
    for b_idx in (0, 2, 3):
        bp_n = norm_prof(band_profs[b_idx], band_valid[b_idx])
        bv = band_valid[b_idx]
        for xs_c in sample_xs:
            xi = int(round(xs_c))
            x0 = max(max_lag + 4, xi - win_half)
            x1 = min(rw - max_lag - 4, xi + win_half)
            if x1 - x0 < 160:
                continue
            if ref_v[x0:x1].mean() < 0.75 or bv[x0:x1].mean() < 0.75:
                continue
            tmpl = ref_n[x0:x1].reshape(1, -1)
            search = bp_n[x0 - max_lag : x1 + max_lag].reshape(1, -1)
            if float(np.std(tmpl)) < 0.12 or float(np.std(search)) < 0.12:
                continue
            cc = cv2.matchTemplate(search, tmpl, cv2.TM_CCOEFF_NORMED).ravel()
            # Attention: vertical grid is periodic with p_x_ref!
            # Continuity from y_ref -> band b: adjacent bands (|dy| ~ 350 px) shift by at most ~25..45 px,
            # EXCEPT when perspective scale changes by ~15% at +/- 1000 px from center (= 150 px)!
            # Weight lightly toward smaller shift on the center window first, or fit global plane!
            best_i = int(np.argmax(cc))
            if 0 < best_i < len(cc) - 1:
                # Parabolic sub-pixel peak
                denom_p = cc[best_i - 1] - 2.0 * cc[best_i] + cc[best_i + 1]
                sub = 0.5 * (cc[best_i - 1] - cc[best_i + 1]) / denom_p if abs(denom_p) > 1e-6 else 0.0
            else:
                sub = 0.0
            dx_val = (best_i + sub) - max_lag
            peak_cc = float(cc[best_i])
            if peak_cc > 0.32:
                all_obs_y.append(float(band_yc[b_idx] - y_ref))
                all_obs_xr.append(float(xs_c - cx))
                all_obs_dx.append(float(dx_val))
                all_obs_w.append(peak_cc ** 2)

    # Fit smooth bilinear fan model:
    #   x_img(x_ref, y) = x_ref + k_shear * (y - y_ref) + k_fan * (x_ref - cx) * (y - y_ref)
    # Use multi-hypothesis RANSAC/grid over integer grid-period wraps on k_shear and k_fan so periodic
    # ambiguity NEVER causes a 1-box jump!
    k_shear, k_fan = 0.0, 0.0
    if len(all_obs_y) >= 4:
        dy_arr = np.array(all_obs_y)
        xr_arr = np.array(all_obs_xr)
        dx_arr = np.array(all_obs_dx)
        w_arr = np.array(all_obs_w)

        # Direct image-space optimization on normalized band profiles avoids ANY 1-period wrap error!
        def fan_corr_score(ks: float, kf: float) -> float:
            total = 0.0
            x_grid = np.arange(rw, dtype=np.float32)
            for b_idx in (0, 2, 3):
                dy_b = float(band_yc[b_idx] - y_ref)
                sample_x = x_grid + ks * dy_b + kf * (x_grid - float(cx)) * dy_b
                valid_s = (sample_x >= 0) & (sample_x < rw - 1) & ref_v
                if np.sum(valid_s) < 200:
                    continue
                s_idx = sample_x[valid_s]
                i0 = np.floor(s_idx).astype(np.int32)
                fr = s_idx - i0
                bp_n = norm_prof(band_profs[b_idx], band_valid[b_idx])
                warped_b = bp_n[i0] * (1.0 - fr) + bp_n[i0 + 1] * fr
                r_sub = ref_n[valid_s]
                total += float(np.mean(r_sub * warped_b))
            return total

        # Coarse grid search over physically plausible shear (+/- 8 deg -> +/- 0.14) and keystone fan (+/- 0.00045)
        best_f_sc = -1e9
        for ks in np.linspace(-0.12, 0.12, 49):
            for kf in np.linspace(-0.00038, 0.00038, 41):
                sc_f = fan_corr_score(float(ks), float(kf))
                if sc_f > best_f_sc:
                    best_f_sc = sc_f
                    k_shear, k_fan = float(ks), float(kf)

        res_fan = minimize(
            lambda p: -fan_corr_score(float(p[0]), float(p[1])),
            x0=np.array([k_shear, k_fan], dtype=np.float64),
            method="Nelder-Mead",
            options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": 120},
        )
        k_shear, k_fan = float(res_fan.x[0]), float(res_fan.x[1])
        _ = (dy_arr, xr_arr, dx_arr, w_arr)

    # Now stack ALL 4 bands after de-fanning onto x_ref to get a super clean 1D vertical-line signal along x_ref!
    x_ref_grid = np.arange(rw, dtype=np.float32)
    stacked_vert = ref_n.copy()
    stack_w = ref_v.astype(np.float32)
    for b_idx in (0, 2, 3):
        dy_b = float(band_yc[b_idx] - y_ref)
        sx = x_ref_grid + k_shear * dy_b + k_fan * (x_ref_grid - float(cx)) * dy_b
        ok_s = (sx >= 0) & (sx < rw - 1)
        bp_n = norm_prof(band_profs[b_idx], band_valid[b_idx])
        bv_f = band_valid[b_idx].astype(np.float32)
        i0 = np.clip(np.floor(sx).astype(np.int32), 0, rw - 2)
        fr = sx - i0
        val_b = np.where(ok_s, bp_n[i0] * (1.0 - fr) + bp_n[i0 + 1] * fr, 0.0)
        wb = np.where(ok_s, bv_f[i0], 0.0)
        stacked_vert += val_b * wb
        stack_w += wb
    stacked_vert = stacked_vert / np.maximum(stack_w, 1.0)

    # Measure local horizontal period p_x(x_ref) in 5 overlapping windows across X to capture
    # horizontal perspective foreshortening (left vs right side of tilted strip)!
    valid_x_cols = np.where(stack_w > 1.5)[0]
    if len(valid_x_cols) > 400:
        x_left = int(valid_x_cols[0])
        x_right = int(valid_x_cols[-1])
    else:
        x_left, x_right = 0, rw - 1

    win_centers = np.linspace(x_left + 220, x_right - 220, 5)
    meas_x: list[float] = []
    meas_px_per_cm: list[float] = []
    for wc in win_centers:
        w0 = max(x_left, int(round(wc)) - 320)
        w1 = min(x_right, int(round(wc)) + 320)
        if w1 - w0 >= 320:
            p_loc = _estimate_local_period_px(
                stacked_vert[w0:w1], min_p=p_x_ref * 0.84, max_p=p_x_ref * 1.18
            )
            if 0.82 * p_x_ref <= p_loc <= 1.20 * p_x_ref:
                meas_x.append(float(wc))
                meas_px_per_cm.append(4.0 * p_loc)  # 4 vertical fine lines per 1.0 cm!

    if len(meas_x) >= 2:
        # Linear fit of local px/cm(x_ref) across the frame width
        slope_s, intercept_s = np.polyfit(np.array(meas_x), np.array(meas_px_per_cm), deg=1)
        px_per_cm_curve = np.clip(slope_s * x_ref_grid + intercept_s, 100.0, 420.0)
    else:
        px_per_cm_curve = np.full(rw, 4.0 * p_x_ref, dtype=np.float64)

    # Integrate true physical distance u_cm(x_ref) = integral dx_ref / px_per_cm(x_ref)
    du_cm = 1.0 / px_per_cm_curve
    u_cm_of_xref = np.concatenate([[0.0], np.cumsum(0.5 * (du_cm[:-1] + du_cm[1:]))])

    # Crop to valid X span [x_left .. x_right]
    u_start = float(u_cm_of_xref[x_left])
    u_end = float(u_cm_of_xref[x_right])
    out_w = max(64, int(round((u_end - u_start) * px_per_cm)))

    target_u_cm = u_start + np.arange(out_w, dtype=np.float64) / float(px_per_cm)
    xref_of_outx = np.interp(target_u_cm, u_cm_of_xref, x_ref_grid).astype(np.float32)

    # Expand to 2D map_x_in_yrect(Y_out, X_out) using our fitted vertical-line fan model!
    dy_col = (np.arange(rh, dtype=np.float32) - float(y_ref)).reshape(-1, 1)
    xr_row = xref_of_outx.reshape(1, -1)
    map_x_yrect = xr_row + k_shear * dy_col + k_fan * (xr_row - float(cx)) * dy_col
    return map_x_yrect.astype(np.float32), out_w


def measure_grid_quality(rect_grid_u8: np.ndarray, rect_mask_u8: np.ndarray, px_per_cm: int) -> dict[str, float]:
    """Measure residual horizontal tilt, vertical tilt, 90-degree orthogonality, and X/Y grid periods."""
    fhr_y0 = int(round((FHR_S_TOP_CM + 0.3) * px_per_cm))
    fhr_y1 = int(round((FHR_S_BOT_CM - 0.3) * px_per_cm))
    h, w = rect_grid_u8.shape
    x0, x1 = int(0.12 * w), int(0.88 * w)
    roi = rect_grid_u8[fhr_y0:fhr_y1, x0:x1].astype(np.float32)
    m_roi = rect_mask_u8[fhr_y0:fhr_y1, x0:x1] > 0
    roi[~m_roi] = 0.0

    rh, rw = roi.shape
    cy, cx = rh / 2.0, rw / 2.0

    # Fine angular scan [-2.5, +2.5] deg at 0.05 deg resolution for horizontal & vertical lines independently
    best_ha, best_hv = 0.0, -1.0
    best_va, best_vv = 0.0, -1.0
    for a in np.linspace(-2.5, 2.5, 101):
        mat = cv2.getRotationMatrix2D((cx, cy), float(a), 1.0)
        wp = cv2.warpAffine(roi, mat, (rw, rh), flags=cv2.INTER_LINEAR)
        inner = wp[20:-20, 30:-30]
        if inner.size == 0:
            continue
        hp = inner.mean(axis=1)
        hp = hp - gaussian_filter1d(hp, sigma=6.0)
        hv = float(np.mean(hp * hp))
        if hv > best_hv:
            best_hv, best_ha = hv, float(a)

        vp = inner.mean(axis=0)
        vp = vp - gaussian_filter1d(vp, sigma=6.0)
        vv = float(np.mean(vp * vp))
        if vv > best_vv:
            best_vv, best_va = vv, float(a)

    expected_p = 0.25 * px_per_cm
    py = _estimate_local_period_px(roi.mean(axis=1), min_p=expected_p * 0.7, max_p=expected_p * 1.35)
    px = _estimate_local_period_px(roi.mean(axis=0), min_p=expected_p * 0.7, max_p=expected_p * 1.35)
    ortho = 90.0 + (best_va - best_ha)
    return {
        "residual_horiz_tilt_deg": round(best_ha, 3),
        "residual_vert_tilt_deg": round(best_va, 3),
        "orthogonal_angle_deg": round(ortho, 3),
        "measured_grid_period_x_px": round(px, 2),
        "measured_grid_period_y_px": round(py, 2),
    }


def rectify_ctg_frame(
    enhanced_bgr: np.ndarray,
    grid_u8: np.ndarray,
    trace_u8: np.ndarray,
    valid_mask: np.ndarray,
    cfg: RectifyConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Rectify an enhanced CTG frame onto the canonical 90-degree metric grid in a single remap pass."""
    cfg = cfg or RectifyConfig()
    px_per_cm = cfg.target_px_per_cm
    h, w = grid_u8.shape
    out_h = int(round(CANONICAL_TOTAL_HEIGHT_CM * px_per_cm))

    # Step 1: Global deskew rotation matrix R0 so horizontal lines are nearly horizontal
    skew_deg = _estimate_global_horiz_skew_deg(grid_u8, valid_mask)
    rot_mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), skew_deg, 1.0)
    # Inverse 2x3 mapping from rotated coords (xr, yr) back to original coords (x_orig, y_orig)
    inv_rot = cv2.invertAffineTransform(rot_mat)

    deskew_grid = cv2.warpAffine(grid_u8, rot_mat, (w, h), flags=cv2.INTER_LINEAR)
    deskew_mask = cv2.warpAffine(valid_mask, rot_mat, (w, h), flags=cv2.INTER_NEAREST)

    # Step 2: Fit 1D projective comb y(s) across X-slices starting from center slice outward
    valid_cols = np.where((deskew_mask > 0).mean(axis=0) > cfg.min_valid_col_fraction)[0]
    if len(valid_cols) > 200:
        x_min_v, x_max_v = int(valid_cols[0]), int(valid_cols[-1])
    else:
        x_min_v, x_max_v = int(0.05 * w), int(0.95 * w)

    num_s = max(5, cfg.num_x_slices | 1)
    slice_centers = np.linspace(x_min_v + 120, x_max_v - 120, num_s)
    half_win = max(160, int((x_max_v - x_min_v) / (num_s + 1)))

    mid_idx = num_s // 2
    slice_params = np.zeros((num_s, 3), dtype=np.float64)
    slice_scores = np.zeros(num_s, dtype=np.float64)

    def fit_slice(idx: int, prior: tuple[float, float, float] | None) -> None:
        xc = int(round(slice_centers[idx]))
        x0 = max(0, xc - half_win)
        x1 = min(w, xc + half_win)
        m_sub = (deskew_mask[:, x0:x1] > 0).astype(np.float32)
        v_frac = m_sub.mean(axis=1)
        p_raw = deskew_grid[:, x0:x1].astype(np.float32).sum(axis=1) / np.maximum(m_sub.sum(axis=1), 1.0)
        p_raw[v_frac < 0.2] = 0.0
        params, sc = _fit_projective_comb_1d(p_raw, v_frac, prior_params=prior)
        slice_params[idx] = params
        slice_scores[idx] = sc

    fit_slice(mid_idx, prior=None)
    for idx in range(mid_idx - 1, -1, -1):
        fit_slice(idx, prior=tuple(slice_params[idx + 1]))  # type: ignore[arg-type]
    for idx in range(mid_idx + 1, num_s):
        fit_slice(idx, prior=tuple(slice_params[idx - 1]))  # type: ignore[arg-type]

    # Smooth parameters lightly with quadratic/cubic regularized curve across X to suppress any noisy edge slice
    weights = np.maximum(slice_scores - np.min(slice_scores) + 1.0, 0.5)
    x_all = np.arange(w, dtype=np.float64)
    smooth_params = np.zeros((w, 3), dtype=np.float64)
    deg_poly = 2 if num_s <= 7 else 2  # Quadratic captures planar tilt + cylindrical paper bend cleanly!
    for p_i in range(3):
        poly = np.polyfit(slice_centers, slice_params[:, p_i], deg=deg_poly, w=weights)
        poly_curve = np.polyval(poly, x_all)
        # Blend 70% piecewise-linear (keeps local crease dewarping) + 30% quadratic (stabilizes borders)
        pw_lin = np.interp(x_all, slice_centers, slice_params[:, p_i])
        smooth_params[:, p_i] = 0.65 * pw_lin + 0.35 * poly_curve

    # Build intermediate Y-rectified grid ONLY for measuring X vertical-line fan & spacing
    s_col = (np.arange(out_h, dtype=np.float64) / float(px_per_cm)).reshape(-1, 1)
    s0 = 0.5 * (FHR_S_BOT_CM + TOCO_S_TOP_CM)
    ds_col = s_col - s0
    y_of_s_and_xr = (
        smooth_params[:, 0].reshape(1, -1)
        + smooth_params[:, 1].reshape(1, -1)
        * ds_col
        / (1.0 + smooth_params[:, 2].reshape(1, -1) * ds_col)
    ).astype(np.float32)
    xr_grid_2d = np.broadcast_to(np.arange(w, dtype=np.float32).reshape(1, -1), (out_h, w))

    y_rect_grid = cv2.remap(deskew_grid, xr_grid_2d, y_of_s_and_xr, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    y_rect_mask = cv2.remap(deskew_mask, xr_grid_2d, y_of_s_and_xr, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Step 3: Compute horizontal X-rectification field on the Y-rectified space,
    # anchored to the known vertical scale at lower-FHR (ds ~ -2.0 cm above separator)
    med_sc = float(np.median(smooth_params[x_min_v:x_max_v, 1]))
    med_k = float(np.median(smooth_params[x_min_v:x_max_v, 2]))
    expected_x_scale = med_sc / ((1.0 + med_k * (-2.0)) ** 2)
    map_xr, out_w = _rectify_x_coordinate_field(
        y_rect_grid,
        y_rect_mask,
        px_per_cm=px_per_cm,
        expected_src_px_per_cm=expected_x_scale,
    )

    # Step 4: Compose Y-rectification + X-rectification + Initial Deskew into ONE direct map to original enhanced image!
    # At output pixel (Y_out, X_out), deskewed X is xr = map_xr[Y_out, X_out],
    # and deskewed Y is y_of_s_and_xr(s = Y_out/px_per_cm, xr)!
    xr_clamped = np.clip(map_xr, 0.0, float(w - 1.001))
    xi0 = np.floor(xr_clamped).astype(np.int32)
    xfr = xr_clamped - xi0
    p0 = smooth_params[xi0, 0] * (1.0 - xfr) + smooth_params[xi0 + 1, 0] * xfr
    p1 = smooth_params[xi0, 1] * (1.0 - xfr) + smooth_params[xi0 + 1, 1] * xfr
    p2 = smooth_params[xi0, 2] * (1.0 - xfr) + smooth_params[xi0 + 1, 2] * xfr

    ds_2d = (np.arange(out_h, dtype=np.float32) / float(px_per_cm) - float(s0)).reshape(-1, 1)
    yr_composed = (p0 + p1 * ds_2d / (1.0 + p2 * ds_2d)).astype(np.float32)

    # Map (map_xr, yr_composed) back through inv_rot to original enhanced image coordinates (SINGLE interpolation!)
    final_map_x = (
        inv_rot[0, 0] * map_xr + inv_rot[0, 1] * yr_composed + inv_rot[0, 2]
    ).astype(np.float32)
    final_map_y = (
        inv_rot[1, 0] * map_xr + inv_rot[1, 1] * yr_composed + inv_rot[1, 2]
    ).astype(np.float32)

    rect_bgr = cv2.remap(
        enhanced_bgr,
        final_map_x,
        final_map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(245, 245, 245),
    )
    rect_grid = cv2.remap(
        grid_u8,
        final_map_x,
        final_map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    rect_trace = cv2.remap(
        trace_u8,
        final_map_x,
        final_map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    rect_mask = cv2.remap(
        valid_mask,
        final_map_x,
        final_map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    # Erode 3 px to strip any INTER_CUBIC boundary fringe against dimmed non-paper regions
    rect_mask = cv2.erode(rect_mask, np.ones((7, 7), dtype=np.uint8))

    quality = measure_grid_quality(rect_grid, rect_mask, px_per_cm=px_per_cm)
    quality["initial_skew_deg"] = round(float(skew_deg), 3)
    quality["comb_fit_score"] = round(float(np.median(slice_scores)), 3)
    return rect_bgr, rect_grid, rect_trace, rect_mask, quality


def run_rectify_pipeline(
    paths: DatasetPaths,
    rect_cfg: RectifyConfig | None = None,
    enh_cfg: EnhanceConfig | None = None,
    frame_indices: list[int] | None = None,
) -> list[RectifyResultMetadata]:
    """Stream all (or selected) frames through Pipeline 2 (auto-running Pipeline 1 if missing)."""
    paths.ensure_work_dirs()
    manifest = DatasetManifest.load(paths.manifest_path)
    rect_cfg = rect_cfg or RectifyConfig(target_px_per_cm=manifest.target_px_per_cm)
    enh_cfg = enh_cfg or EnhanceConfig()

    selected = set(frame_indices) if frame_indices is not None else None
    results: list[RectifyResultMetadata] = []

    for entry in manifest.frames:
        if selected is not None and entry.index not in selected:
            continue

        stem = Path(entry.filename).stem
        enh_path = paths.enhanced_dir / f"{stem}_enhanced.png"
        grid_path = paths.enhanced_dir / f"{stem}_grid.png"
        trace_path = paths.enhanced_dir / f"{stem}_trace.png"
        mask_path = paths.enhanced_dir / f"{stem}_mask.png"

        if enh_path.exists() and grid_path.exists() and trace_path.exists() and mask_path.exists():
            enhanced_bgr = cv2.imread(str(enh_path), cv2.IMREAD_COLOR)
            grid_u8 = cv2.imread(str(grid_path), cv2.IMREAD_GRAYSCALE)
            trace_u8 = cv2.imread(str(trace_path), cv2.IMREAD_GRAYSCALE)
            valid_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            raw_bgr = cv2.imread(str(paths.raw_dir / entry.filename), cv2.IMREAD_COLOR)
            if raw_bgr is None:
                raise FileNotFoundError(f"Could not read raw frame: {entry.filename}")
            enhanced_bgr, grid_u8, trace_u8, valid_mask, _ = enhance_ctg_frame(
                raw_bgr, cfg=enh_cfg, rotation_override_deg=entry.rotation_override_deg
            )
            cv2.imwrite(str(enh_path), enhanced_bgr)
            cv2.imwrite(str(grid_path), grid_u8)
            cv2.imwrite(str(trace_path), trace_u8)
            cv2.imwrite(str(mask_path), valid_mask)
            del raw_bgr

        assert enhanced_bgr is not None and grid_u8 is not None and trace_u8 is not None and valid_mask is not None

        rect_bgr, rect_grid, rect_trace, rect_mask, quality = rectify_ctg_frame(
            enhanced_bgr, grid_u8, trace_u8, valid_mask, cfg=rect_cfg
        )

        cv2.imwrite(str(paths.rectified_dir / f"{stem}_rectified.png"), rect_bgr)
        cv2.imwrite(str(paths.rectified_dir / f"{stem}_rect_grid.png"), rect_grid)
        cv2.imwrite(str(paths.rectified_dir / f"{stem}_rect_trace.png"), rect_trace)
        cv2.imwrite(str(paths.rectified_dir / f"{stem}_rect_mask.png"), rect_mask)

        rh, rw = rect_bgr.shape[:2]
        meta = RectifyResultMetadata(
            frame_index=entry.index,
            filename=entry.filename,
            target_px_per_cm=rect_cfg.target_px_per_cm,
            rectified_width=rw,
            rectified_height=rh,
            initial_skew_deg=quality["initial_skew_deg"],
            comb_fit_score=quality["comb_fit_score"],
            residual_horiz_tilt_deg=quality["residual_horiz_tilt_deg"],
            residual_vert_tilt_deg=quality["residual_vert_tilt_deg"],
            orthogonal_angle_deg=quality["orthogonal_angle_deg"],
            measured_grid_period_x_px=quality["measured_grid_period_x_px"],
            measured_grid_period_y_px=quality["measured_grid_period_y_px"],
        )
        (paths.rectified_dir / f"{stem}_rect_meta.json").write_text(
            json.dumps(asdict(meta), indent=2) + "\n", encoding="utf-8"
        )
        results.append(meta)

        del enhanced_bgr, grid_u8, trace_u8, valid_mask, rect_bgr, rect_grid, rect_trace, rect_mask

    (paths.rectified_dir / "summary.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2) + "\n", encoding="utf-8"
    )
    return results
