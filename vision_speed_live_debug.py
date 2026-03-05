import json
import os

import cv2
import numpy as np
from scipy.interpolate import interp1d


SEGMENT_PATH = "comma2k19/Chunk_3/99c94dc769b5d96e_2018-05-01--08-13-53/30"
VIDEO_PATH = os.path.join(SEGMENT_PATH, "video.mp4")
PARAMS_PATH = "vision_live_debug_params.json"


def extract_gps_speed_kmh(gnss_values):
    if gnss_values.ndim != 2 or gnss_values.shape[1] == 0:
        raise ValueError(f"Unexpected GNSS shape: {gnss_values.shape}")

    preferred_idx = 2
    if gnss_values.shape[1] > preferred_idx:
        candidate = gnss_values[:, preferred_idx].astype(float)
        if np.nanmax(np.abs(candidate)) < 120:
            return candidate * 3.6, preferred_idx

    for idx in range(gnss_values.shape[1]):
        candidate = gnss_values[:, idx].astype(float)
        if np.nanmax(np.abs(candidate)) < 120:
            return candidate * 3.6, idx

    raise ValueError("Could not find plausible GNSS speed column")


def safe_percentile(arr, q, fallback=0.0):
    if arr.size == 0:
        return fallback
    return float(np.percentile(arr, q))


def robust_scale(proxy_hist, can_hist, calib_len, fit_low, fit_high):
    if len(proxy_hist) < 30:
        return 0.0

    x = np.array(proxy_hist, dtype=float)
    y = np.array(can_hist, dtype=float)

    n = min(calib_len, x.size, y.size)
    x = x[:n]
    y = y[:n]

    valid = x > 1e-6
    x = x[valid]
    y = y[valid]

    if x.size < 20:
        return 0.0

    ql = np.percentile(x, fit_low)
    qh = np.percentile(x, fit_high)
    inlier = (x >= ql) & (x <= qh)
    x_fit = x[inlier]
    y_fit = y[inlier]

    if x_fit.size < 20:
        return 0.0

    denom = float(np.dot(x_fit, x_fit))
    if denom < 1e-9:
        return 0.0

    return float(np.dot(y_fit, x_fit) / denom)


def robust_affine_calibration(proxy_hist, can_hist, calib_len):
    """Fit speed = a * norm_proxy + b on recent history for better stability."""
    if len(proxy_hist) < 40:
        return None

    x = np.array(proxy_hist, dtype=float)
    y = np.array(can_hist, dtype=float)

    n = min(calib_len, x.size, y.size)
    x = x[-n:]
    y = y[-n:]

    valid = x > 1e-6
    x = x[valid]
    y = y[valid]

    if x.size < 30:
        return None

    p05 = float(np.percentile(x, 5))
    p95 = float(np.percentile(x, 95))
    if p95 - p05 < 1e-6:
        return None

    x_norm = np.clip((x - p05) / (p95 - p05), 0.0, 1.0)

    # Lightweight outlier suppression before fit.
    ql, qh = np.percentile(x_norm, [10, 90])
    inlier = (x_norm >= ql) & (x_norm <= qh)
    x_fit = x_norm[inlier]
    y_fit = y[inlier]

    if x_fit.size < 20:
        return None

    a, b = np.polyfit(x_fit, y_fit, 1)
    return float(a), float(b), p05, p95


def infer_flow_proxy(flow, p_low, p_high, lateral_ratio, min_count, trim_low, trim_high):
    fx = flow[..., 0]
    fy = flow[..., 1]
    mag = np.sqrt(fx * fx + fy * fy)

    lo = np.percentile(mag, p_low)
    hi = np.percentile(mag, p_high)
    moving = (mag > lo) & (mag < hi)

    forward = fy[moving]
    lateral = fx[moving]

    if forward.size == 0:
        return 0.0

    dominant = np.abs(forward) > (lateral_ratio * np.abs(lateral) + 1e-6)
    forward = forward[dominant]

    if forward.size == 0:
        return 0.0

    sign = 1.0 if np.median(forward) >= 0 else -1.0
    forward = forward * sign
    forward = forward[forward > 0]

    if forward.size < min_count:
        return 0.0

    ql = np.percentile(forward, trim_low)
    qh = np.percentile(forward, trim_high)
    trimmed = forward[(forward >= ql) & (forward <= qh)]

    if trimmed.size == 0:
        return float(np.median(forward))

    return float(np.mean(trimmed))


def draw_series_panel(width, can_hist, gps_hist, vis_hist, max_points=240):
    panel_h = 220
    panel = np.full((panel_h, width, 3), 24, dtype=np.uint8)

    can_arr = np.array(can_hist[-max_points:], dtype=float)
    gps_arr = np.array(gps_hist[-max_points:], dtype=float)
    vis_arr = np.array(vis_hist[-max_points:], dtype=float)

    if can_arr.size < 2:
        return panel

    y_min = min(np.min(can_arr), np.min(gps_arr), np.min(vis_arr)) - 5.0
    y_max = max(np.max(can_arr), np.max(gps_arr), np.max(vis_arr)) + 5.0
    if y_max - y_min < 1e-6:
        y_max = y_min + 1.0

    def to_xy(vals):
        xs = np.linspace(0, width - 1, vals.size).astype(np.int32)
        ys = (panel_h - 1 - (vals - y_min) / (y_max - y_min) * (panel_h - 1)).astype(np.int32)
        return np.stack([xs, ys], axis=1).reshape(-1, 1, 2)

    cv2.polylines(panel, [to_xy(can_arr)], False, (255, 180, 60), 2)
    cv2.polylines(panel, [to_xy(gps_arr)], False, (80, 170, 255), 2)
    cv2.polylines(panel, [to_xy(vis_arr)], False, (60, 255, 90), 2)

    cv2.putText(panel, "CAN", (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 60), 2)
    cv2.putText(panel, "GPS", (70, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 170, 255), 2)
    cv2.putText(panel, "Vision", (128, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 255, 90), 2)

    return panel


def load_saved_params(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as e:
        print(f"Could not load saved params: {e}")
    return {}


def set_if_present(win, name, value):
    try:
        cv2.setTrackbarPos(name, win, int(value))
    except Exception:
        pass


def main():
    print("Loading data...")

    can_speed = np.load(os.path.join(SEGMENT_PATH, "processed_log/CAN/speed/value")).flatten() * 3.6
    can_time = np.load(os.path.join(SEGMENT_PATH, "processed_log/CAN/speed/t")).flatten()

    gnss_val = np.load(os.path.join(SEGMENT_PATH, "processed_log/GNSS/live_gnss_qcom/value"))
    gnss_time = np.load(os.path.join(SEGMENT_PATH, "processed_log/GNSS/live_gnss_qcom/t")).flatten()

    frame_times = np.load(os.path.join(SEGMENT_PATH, "global_pose/frame_times")).flatten()

    gps_speed, gps_col = extract_gps_speed_kmh(gnss_val)
    print("GNSS speed column:", gps_col)

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

    can_aligned = can_interp(frame_times)
    gps_aligned = gps_interp(frame_times)

    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, prev_frame = cap.read()
    if not ret:
        raise RuntimeError("Could not load video")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    h, w = prev_gray.shape

    y0, y1 = int(h * 0.40), int(h * 0.76)
    x0, x1 = int(w * 0.15), int(w * 0.85)

    win = "Vision Speed Live Debug"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    cv2.createTrackbar("p_low", win, 60, 95, lambda _: None)
    cv2.createTrackbar("p_high", win, 95, 99, lambda _: None)
    cv2.createTrackbar("lat_ratio_x10", win, 15, 40, lambda _: None)
    cv2.createTrackbar("min_count", win, 80, 300, lambda _: None)
    cv2.createTrackbar("trim_low", win, 30, 49, lambda _: None)
    cv2.createTrackbar("trim_high", win, 70, 99, lambda _: None)
    cv2.createTrackbar("calib_len", win, 600, 1200, lambda _: None)
    cv2.createTrackbar("fit_low", win, 10, 49, lambda _: None)
    cv2.createTrackbar("fit_high", win, 90, 99, lambda _: None)
    cv2.createTrackbar("alpha_x100", win, 18, 80, lambda _: None)
    cv2.createTrackbar("rate_x10", win, 12, 60, lambda _: None)
    cv2.createTrackbar("proxy_med_win", win, 9, 41, lambda _: None)
    cv2.createTrackbar("scale_alpha_x100", win, 8, 60, lambda _: None)
    cv2.createTrackbar("max_speed", win, 180, 300, lambda _: None)

    # Load previously saved slider settings if available.
    saved = load_saved_params(PARAMS_PATH)
    if saved:
        set_if_present(win, "p_low", saved.get("p_low", 60))
        set_if_present(win, "p_high", saved.get("p_high", 95))
        set_if_present(win, "lat_ratio_x10", int(round(saved.get("lat_ratio", 1.5) * 10)))
        set_if_present(win, "min_count", saved.get("min_count", 80))
        set_if_present(win, "trim_low", saved.get("trim_low", 30))
        set_if_present(win, "trim_high", saved.get("trim_high", 70))
        set_if_present(win, "calib_len", saved.get("calib_len", 600))
        set_if_present(win, "fit_low", saved.get("fit_low", 10))
        set_if_present(win, "fit_high", saved.get("fit_high", 90))
        set_if_present(win, "alpha_x100", int(round(saved.get("alpha", 0.18) * 100)))
        set_if_present(win, "rate_x10", int(round(saved.get("rate_limit", 1.2) * 10)))
        set_if_present(win, "proxy_med_win", saved.get("proxy_median_window", 9))
        set_if_present(win, "scale_alpha_x100", int(round(saved.get("scale_alpha", 0.08) * 100)))
        set_if_present(win, "max_speed", saved.get("max_speed", 180))
        print(f"Loaded params from {PARAMS_PATH}")

    frame_idx = 0
    paused = False

    proxy_hist = []
    vis_hist = []
    can_hist = []
    gps_hist = []
    scale_k = 0.0
    scale_prev = 0.0
    bias_b = 0.0
    bias_prev = 0.0
    proxy_p05 = 0.0
    proxy_p95 = 1.0
    vis_prev = 0.0

    print("Keys: q=quit, space=pause/resume, s=save params, r=reset history")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi_prev = prev_gray[y0:y1, x0:x1]
            roi_curr = gray[y0:y1, x0:x1]

            flow = cv2.calcOpticalFlowFarneback(
                roi_prev,
                roi_curr,
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )

            p_low = max(1, cv2.getTrackbarPos("p_low", win))
            p_high = max(p_low + 1, cv2.getTrackbarPos("p_high", win))
            lat_ratio = max(0.1, cv2.getTrackbarPos("lat_ratio_x10", win) / 10.0)
            min_count = max(10, cv2.getTrackbarPos("min_count", win))
            trim_low = cv2.getTrackbarPos("trim_low", win)
            trim_high = max(trim_low + 1, cv2.getTrackbarPos("trim_high", win))
            calib_len = max(50, cv2.getTrackbarPos("calib_len", win))
            fit_low = cv2.getTrackbarPos("fit_low", win)
            fit_high = max(fit_low + 1, cv2.getTrackbarPos("fit_high", win))
            alpha = max(0.01, cv2.getTrackbarPos("alpha_x100", win) / 100.0)
            max_step = max(0.1, cv2.getTrackbarPos("rate_x10", win) / 10.0)
            proxy_med_win = max(1, cv2.getTrackbarPos("proxy_med_win", win))
            scale_alpha = max(0.01, cv2.getTrackbarPos("scale_alpha_x100", win) / 100.0)
            max_speed = max(20, cv2.getTrackbarPos("max_speed", win))

            if proxy_med_win % 2 == 0:
                proxy_med_win += 1

            proxy = infer_flow_proxy(
                flow,
                p_low=p_low,
                p_high=p_high,
                lateral_ratio=lat_ratio,
                min_count=min_count,
                trim_low=trim_low,
                trim_high=trim_high,
            )

            can_now = float(can_aligned[min(frame_idx, len(can_aligned) - 1)])
            gps_now = float(gps_aligned[min(frame_idx, len(gps_aligned) - 1)])

            proxy_hist.append(proxy)
            can_hist.append(can_now)
            gps_hist.append(gps_now)

            # Robust short-window median to suppress proxy spikes before scaling.
            tail = np.array(proxy_hist[-proxy_med_win:], dtype=float)
            proxy_smooth = float(np.median(tail)) if tail.size > 0 else 0.0

            calib = robust_affine_calibration(proxy_hist, can_hist, calib_len)
            if calib is not None:
                a_new, b_new, p05_new, p95_new = calib
                a_new = max(0.0, min(a_new, 250.0))
                b_new = max(0.0, min(b_new, 220.0))

                if scale_prev <= 1e-6:
                    scale_k = a_new
                    bias_b = b_new
                else:
                    scale_k = scale_alpha * a_new + (1.0 - scale_alpha) * scale_prev
                    bias_b = scale_alpha * b_new + (1.0 - scale_alpha) * bias_prev

                proxy_p05 = p05_new
                proxy_p95 = p95_new

            scale_prev = scale_k
            bias_prev = bias_b

            if proxy_p95 - proxy_p05 > 1e-6:
                proxy_norm = float(np.clip((proxy_smooth - proxy_p05) / (proxy_p95 - proxy_p05), 0.0, 1.0))
            else:
                proxy_norm = 0.0

            vis_raw = proxy_norm * scale_k + bias_b

            vis_smooth = alpha * vis_raw + (1.0 - alpha) * vis_prev
            vis_now = vis_prev + np.clip(vis_smooth - vis_prev, -max_step, max_step)
            vis_now = np.clip(vis_now, 0.0, float(max_speed))
            vis_prev = vis_now
            vis_hist.append(vis_now)

            prev_gray = gray

            vis_err = abs(vis_now - can_now)

            vis_frame = frame.copy()
            cv2.rectangle(vis_frame, (x0, y0), (x1, y1), (90, 255, 90), 2)

            # Draw sparse flow vectors in ROI for quick visual inspection.
            step = 28
            for yy in range(0, flow.shape[0], step):
                for xx in range(0, flow.shape[1], step):
                    dx, dy = flow[yy, xx]
                    p1 = (x0 + xx, y0 + yy)
                    p2 = (int(p1[0] + dx * 4.0), int(p1[1] + dy * 4.0))
                    cv2.arrowedLine(vis_frame, p1, p2, (0, 180, 255), 1, tipLength=0.3)

            cv2.putText(vis_frame, f"Frame: {frame_idx}", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_frame, f"Vision: {vis_now:.2f} km/h", (14, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (60, 255, 90), 2)
            cv2.putText(vis_frame, f"CAN: {can_now:.2f}  GPS: {gps_now:.2f}", (14, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 80), 2)
            cv2.putText(vis_frame, f"|Vision-CAN|: {vis_err:.2f}", (14, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 200, 255), 2)
            cv2.putText(
                vis_frame,
                f"Proxy: {proxy:.4f} ({proxy_smooth:.4f}) N:{proxy_norm:.3f}  a:{scale_k:.2f} b:{bias_b:.2f}",
                (14, 138),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (220, 220, 220),
                2,
            )

            panel = draw_series_panel(vis_frame.shape[1], can_hist, gps_hist, vis_hist)
            combined = np.vstack([vis_frame, panel])

            cv2.imshow(win, combined)
            frame_idx += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused
        if key == ord("r"):
            proxy_hist.clear()
            vis_hist.clear()
            can_hist.clear()
            gps_hist.clear()
            scale_k = 0.0
            scale_prev = 0.0
            bias_b = 0.0
            bias_prev = 0.0
            proxy_p05 = 0.0
            proxy_p95 = 1.0
            vis_prev = 0.0
            print("Reset history buffers")
        if key == ord("s"):
            params = {
                "p_low": cv2.getTrackbarPos("p_low", win),
                "p_high": cv2.getTrackbarPos("p_high", win),
                "lat_ratio": cv2.getTrackbarPos("lat_ratio_x10", win) / 10.0,
                "min_count": cv2.getTrackbarPos("min_count", win),
                "trim_low": cv2.getTrackbarPos("trim_low", win),
                "trim_high": cv2.getTrackbarPos("trim_high", win),
                "calib_len": cv2.getTrackbarPos("calib_len", win),
                "fit_low": cv2.getTrackbarPos("fit_low", win),
                "fit_high": cv2.getTrackbarPos("fit_high", win),
                "alpha": cv2.getTrackbarPos("alpha_x100", win) / 100.0,
                "rate_limit": cv2.getTrackbarPos("rate_x10", win) / 10.0,
                "proxy_median_window": proxy_med_win,
                "scale_alpha": cv2.getTrackbarPos("scale_alpha_x100", win) / 100.0,
                "max_speed": cv2.getTrackbarPos("max_speed", win),
                "scale_k": scale_k,
                "bias_b": bias_b,
            }
            with open(PARAMS_PATH, "w", encoding="utf-8") as f:
                json.dump(params, f, indent=2)
            print(f"Saved {PARAMS_PATH}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
