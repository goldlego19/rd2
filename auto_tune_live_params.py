import json
import os
import random

import cv2
import numpy as np
from scipy.interpolate import interp1d

from vision_speed_live_debug import (
    SEGMENT_PATH,
    VIDEO_PATH,
    PARAMS_PATH,
    extract_gps_speed_kmh,
    infer_flow_proxy,
    robust_affine_calibration,
)


def load_data():
    can_speed = np.load(os.path.join(SEGMENT_PATH, "processed_log/CAN/speed/value")).flatten() * 3.6
    can_time = np.load(os.path.join(SEGMENT_PATH, "processed_log/CAN/speed/t")).flatten()
    gnss_val = np.load(os.path.join(SEGMENT_PATH, "processed_log/GNSS/live_gnss_qcom/value"))
    gnss_time = np.load(os.path.join(SEGMENT_PATH, "processed_log/GNSS/live_gnss_qcom/t")).flatten()
    frame_times = np.load(os.path.join(SEGMENT_PATH, "global_pose/frame_times")).flatten()

    gps_speed, _ = extract_gps_speed_kmh(gnss_val)

    can_interp = interp1d(
        can_time,
        can_speed,
        bounds_error=False,
        fill_value=(can_speed[0], can_speed[-1]),
    )
    gps_interp = interp1d(
        gnss_time,
        gps_speed,
        bounds_error=False,
        fill_value=(gps_speed[0], gps_speed[-1]),
    )

    return can_interp(frame_times), gps_interp(frame_times)


def compute_proxy_series(flow_params):
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, prev_frame = cap.read()
    if not ret:
        raise RuntimeError("Could not load video")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    h, w = prev_gray.shape
    y0, y1 = int(h * 0.40), int(h * 0.76)
    x0, x1 = int(w * 0.15), int(w * 0.85)

    proxies = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray[y0:y1, x0:x1],
            gray[y0:y1, x0:x1],
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        proxy = infer_flow_proxy(
            flow,
            p_low=flow_params["p_low"],
            p_high=flow_params["p_high"],
            lateral_ratio=flow_params["lat_ratio"],
            min_count=flow_params["min_count"],
            trim_low=flow_params["trim_low"],
            trim_high=flow_params["trim_high"],
        )
        proxies.append(proxy)
        prev_gray = gray

    cap.release()
    return np.array(proxies, dtype=float)


def simulate(proxy_series, can_series, params):
    n = min(len(proxy_series), len(can_series))
    proxy_series = proxy_series[:n]
    can_series = can_series[:n]

    proxy_hist = []
    can_hist = []
    vis = []

    a_prev = 0.0
    b_prev = 0.0
    p05 = 0.0
    p95 = 1.0
    vis_prev = 0.0

    for i in range(n):
        proxy_hist.append(float(proxy_series[i]))
        can_hist.append(float(can_series[i]))

        med_win = int(params["proxy_median_window"])
        if med_win < 1:
            med_win = 1
        if med_win % 2 == 0:
            med_win += 1
        tail = np.array(proxy_hist[-med_win:], dtype=float)
        proxy_smooth = float(np.median(tail)) if tail.size else 0.0

        calib = robust_affine_calibration(proxy_hist, can_hist, int(params["calib_len"]))
        if calib is not None:
            a_new, b_new, p05_new, p95_new = calib
            a_new = float(np.clip(a_new, 0.0, 250.0))
            b_new = float(np.clip(b_new, 0.0, 220.0))

            sa = float(params["scale_alpha"])
            if a_prev <= 1e-6:
                a = a_new
                b = b_new
            else:
                a = sa * a_new + (1.0 - sa) * a_prev
                b = sa * b_new + (1.0 - sa) * b_prev

            a_prev, b_prev = a, b
            p05, p95 = p05_new, p95_new

        if p95 - p05 > 1e-6:
            pn = float(np.clip((proxy_smooth - p05) / (p95 - p05), 0.0, 1.0))
        else:
            pn = 0.0

        vis_raw = pn * a_prev + b_prev

        alpha = float(params["alpha"])
        vis_smooth = alpha * vis_raw + (1.0 - alpha) * vis_prev
        step = float(params["rate_limit"])
        vis_now = vis_prev + np.clip(vis_smooth - vis_prev, -step, step)
        vis_now = float(np.clip(vis_now, 0.0, float(params["max_speed"])))
        vis_prev = vis_now
        vis.append(vis_now)

    vis = np.array(vis, dtype=float)
    err = can_series - vis
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    # Penalize high-frequency jitter.
    jitter = float(np.std(np.diff(vis))) if vis.size > 2 else 0.0
    score = mae + 0.35 * jitter
    return score, mae, rmse, jitter


def main():
    with open(PARAMS_PATH, "r", encoding="utf-8") as f:
        base = json.load(f)

    print("Loading aligned CAN/GPS...")
    can_aligned, _ = load_data()
    print("Computing proxy series once from video...")
    proxy_series = compute_proxy_series(base)

    n = min(len(proxy_series), len(can_aligned))
    proxy_series = proxy_series[:n]
    can_aligned = can_aligned[:n]

    base_score, base_mae, base_rmse, base_jitter = simulate(proxy_series, can_aligned, base)
    print("Baseline:", {"score": round(base_score, 3), "mae": round(base_mae, 3), "rmse": round(base_rmse, 3), "jitter": round(base_jitter, 3)})

    random.seed(42)
    candidates = []
    for _ in range(80):
        cand = dict(base)
        cand["calib_len"] = random.choice([300, 450, 600, 750, 900])
        cand["alpha"] = random.choice([0.20, 0.24, 0.28, 0.32, 0.36, 0.40])
        cand["rate_limit"] = random.choice([1.0, 1.4, 1.8, 2.2, 2.6, 3.0])
        cand["proxy_median_window"] = random.choice([7, 9, 11, 13, 15, 17, 21])
        cand["scale_alpha"] = random.choice([0.03, 0.05, 0.07, 0.10, 0.14])
        cand["max_speed"] = random.choice([130, 140, 150, 160])
        candidates.append(cand)

    best = (base_score, base_mae, base_rmse, base_jitter, dict(base))
    for i, cand in enumerate(candidates, 1):
        score, mae, rmse, jitter = simulate(proxy_series, can_aligned, cand)
        if score < best[0]:
            best = (score, mae, rmse, jitter, cand)
        if i % 20 == 0:
            print(f"Checked {i}/{len(candidates)} candidates...")

    best_score, best_mae, best_rmse, best_jitter, best_params = best
    print("Best:", {"score": round(best_score, 3), "mae": round(best_mae, 3), "rmse": round(best_rmse, 3), "jitter": round(best_jitter, 3)})

    # Preserve flow-selection params and only update post-processing defaults.
    out = dict(base)
    for k in ["calib_len", "alpha", "rate_limit", "proxy_median_window", "scale_alpha", "max_speed"]:
        out[k] = best_params[k]

    with open(PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Updated {PARAMS_PATH} with tuned post-processing params")


if __name__ == "__main__":
    main()
