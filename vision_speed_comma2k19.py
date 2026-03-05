import cv2
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------------------------------
# SEGMENT PATH
# ---------------------------------------------------

segment_path = "comma2k19/Chunk_3/99c94dc769b5d96e_2018-05-01--08-13-53/30"

video_path = os.path.join(segment_path, "video.mp4")

segment_name = os.path.basename(os.path.normpath(segment_path))
output_dir = "csvs"
os.makedirs(output_dir, exist_ok=True)
output_csv_path = os.path.join(output_dir, f"segment_{segment_name}_fast_vision.csv")
SHOW_PLOT = False

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

print("Loading dataset...")

can_speed = np.load(os.path.join(segment_path, "processed_log/CAN/speed/value")).flatten()
can_time = np.load(os.path.join(segment_path, "processed_log/CAN/speed/t")).flatten()

gnss_val = np.load(os.path.join(segment_path, "processed_log/GNSS/live_gnss_qcom/value"))
gnss_time = np.load(os.path.join(segment_path, "processed_log/GNSS/live_gnss_qcom/t")).flatten()

frame_times = np.load(os.path.join(segment_path, "global_pose/frame_times")).flatten()


def extract_gps_speed_kmh(gnss_values):
    """Pick the GNSS speed column and return speed in km/h."""
    if gnss_values.ndim != 2 or gnss_values.shape[1] == 0:
        raise ValueError(f"Unexpected GNSS shape: {gnss_values.shape}")

    # In comma2k19, column 2 is typically GNSS speed in m/s.
    preferred_idx = 2
    if gnss_values.shape[1] > preferred_idx:
        candidate = gnss_values[:, preferred_idx].astype(float)
        if np.nanmax(np.abs(candidate)) < 120:
            return candidate * 3.6, preferred_idx

    # Fallback: choose a column with plausible speed magnitude in m/s.
    for idx in range(gnss_values.shape[1]):
        candidate = gnss_values[:, idx].astype(float)
        if np.nanmax(np.abs(candidate)) < 120:
            return candidate * 3.6, idx

    raise ValueError("Could not find a plausible GNSS speed column.")


def moving_average(x, window):
    if x.size == 0 or window <= 1:
        return x
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(x, kernel, mode="same")


def ewma(x, alpha=0.12):
    if x.size == 0:
        return x
    out = np.empty_like(x, dtype=float)
    out[0] = float(x[0])
    for i in range(1, x.size):
        out[i] = alpha * float(x[i]) + (1.0 - alpha) * out[i - 1]
    return out


def clip_outliers(x, z=3.0):
    if x.size == 0:
        return x
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad < 1e-9:
        return x
    sigma = 1.4826 * mad
    lo = med - z * sigma
    hi = med + z * sigma
    return np.clip(x, lo, hi)


def rate_limit(x, max_step=2.0):
    if x.size == 0:
        return x
    out = np.empty_like(x, dtype=float)
    out[0] = float(x[0])
    for i in range(1, x.size):
        delta = float(x[i]) - out[i - 1]
        delta = np.clip(delta, -max_step, max_step)
        out[i] = out[i - 1] + delta
    return out


def hampel_filter(x, window=12, n_sigma=3.0):
    if x.size == 0:
        return x
    out = x.copy().astype(float)
    n = x.size
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        local = out[lo:hi]
        med = np.median(local)
        mad = np.median(np.abs(local - med))
        if mad < 1e-9:
            continue
        sigma = 1.4826 * mad
        if np.abs(out[i] - med) > n_sigma * sigma:
            out[i] = med
    return out


def robust_affine_calibration(proxy_hist, can_hist, calib_len):
    """Fit speed = a * norm_proxy + b on recent history for stability."""
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
    ql, qh = np.percentile(x_norm, [10, 90])
    inlier = (x_norm >= ql) & (x_norm <= qh)
    x_fit = x_norm[inlier]
    y_fit = y[inlier]

    if x_fit.size < 20:
        return None

    a, b = np.polyfit(x_fit, y_fit, 1)
    return float(a), float(b), p05, p95

# convert CAN speed to km/h
can_speed_kmh = can_speed * 3.6

# Detect and convert GNSS speed to km/h.
gps_speed, gps_col = extract_gps_speed_kmh(gnss_val)

print("CAN samples:", len(can_speed_kmh))
print("GPS samples:", len(gps_speed))
print("Frame count:", len(frame_times))
print("GNSS speed column:", gps_col)

# ---------------------------------------------------
# INTERPOLATE CAN AND GPS TO VIDEO FRAMES
# ---------------------------------------------------

print("Aligning timestamps...")

interp_can = interp1d(
    can_time,
    can_speed_kmh,
    bounds_error=False,
    fill_value=(can_speed_kmh[0], can_speed_kmh[-1]),
)
interp_gps = interp1d(
    gnss_time,
    gps_speed,
    bounds_error=False,
    fill_value=(gps_speed[0], gps_speed[-1]),
)

can_aligned = interp_can(frame_times)
gps_aligned = interp_gps(frame_times)

# ---------------------------------------------------
# OPEN VIDEO
# ---------------------------------------------------

print("Opening video...")

cap = cv2.VideoCapture(video_path)

ret, prev_frame = cap.read()

if not ret:
    raise RuntimeError("Video failed to load")

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

vision_flow_proxy = []

frame_index = 0

print("Processing video frames...")

# ---------------------------------------------------
# COMPUTE OPTICAL FLOW SPEED
# ---------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape

    y0, y1 = int(h * 0.42), int(h * 0.78)
    x0, x1 = int(w * 0.15), int(w * 0.85)
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
        0
    )

    fx = flow[..., 0]
    fy = flow[..., 1]
    mag = np.sqrt(fx * fx + fy * fy)

    # Keep moderate motion; reject tiny noise and extreme movers (often other vehicles).
    p_low = np.percentile(mag, 45)
    p_high = np.percentile(mag, 97)
    moving_mask = (mag > p_low) & (mag < p_high)
    forward_candidates = fy[moving_mask]
    lateral_candidates = fx[moving_mask]

    # Prefer pixels whose motion is mostly forward/backward rather than lateral.
    if forward_candidates.size > 0:
        forward_dominant = np.abs(forward_candidates) > (1.0 * np.abs(lateral_candidates) + 1e-6)
        forward_candidates = forward_candidates[forward_dominant]

    # Determine dominant vertical-flow direction from data (camera setup may invert sign).
    if forward_candidates.size > 0:
        direction = 1.0 if np.median(forward_candidates) >= 0 else -1.0
        forward_candidates = forward_candidates * direction

    forward_candidates = forward_candidates[forward_candidates > 0]

    if forward_candidates.size > 120:
        q30, q70 = np.percentile(forward_candidates, [25, 75])
        trimmed = forward_candidates[(forward_candidates >= q30) & (forward_candidates <= q70)]
        flow_proxy = float(np.mean(trimmed)) if trimmed.size > 0 else float(np.median(forward_candidates))
    else:
        flow_proxy = 0.0

    vision_flow_proxy.append(flow_proxy)

    prev_gray = gray
    frame_index += 1

cap.release()

vision_flow_proxy = np.array(vision_flow_proxy)

print("Video processing complete")

# ---------------------------------------------------
# ALIGN LENGTHS
# ---------------------------------------------------

n = min(len(vision_flow_proxy), len(can_aligned), len(gps_aligned))

vision_flow_proxy = vision_flow_proxy[:n]
can_aligned = can_aligned[:n]
gps_aligned = gps_aligned[:n]

# Use tuned live-debug style online affine calibration.
proxy_hist = []
can_hist = []
vision_speed = []

a_prev = 0.0
b_prev = 0.0
p05 = 0.0
p95 = 1.0
vis_prev = 0.0
first_calibrated_idx = None

calib_len = 750
proxy_med_window = 11
scale_alpha = 0.14
alpha = 0.36
max_step = 3.0
max_speed = 130.0

for i in range(n):
    proxy_hist.append(float(vision_flow_proxy[i]))
    can_hist.append(float(can_aligned[i]))

    tail = np.array(proxy_hist[-proxy_med_window:], dtype=float)
    proxy_smooth = float(np.median(tail)) if tail.size > 0 else 0.0

    calib = robust_affine_calibration(proxy_hist, can_hist, calib_len)
    if calib is not None:
        a_new, b_new, p05_new, p95_new = calib
        a_new = float(np.clip(a_new, 0.0, 250.0))
        b_new = float(np.clip(b_new, 0.0, 220.0))

        if a_prev <= 1e-6:
            a_prev = a_new
            b_prev = b_new
        else:
            a_prev = scale_alpha * a_new + (1.0 - scale_alpha) * a_prev
            b_prev = scale_alpha * b_new + (1.0 - scale_alpha) * b_prev

        p05, p95 = p05_new, p95_new
        if first_calibrated_idx is None:
            first_calibrated_idx = i

    if p95 - p05 > 1e-6:
        proxy_norm = float(np.clip((proxy_smooth - p05) / (p95 - p05), 0.0, 1.0))
    else:
        proxy_norm = 0.0

    vis_raw = proxy_norm * a_prev + b_prev
    vis_smooth = alpha * vis_raw + (1.0 - alpha) * vis_prev
    vis_now = vis_prev + np.clip(vis_smooth - vis_prev, -max_step, max_step)
    vis_now = float(np.clip(vis_now, 0.0, max_speed))

    vision_speed.append(vis_now)
    vis_prev = vis_now

vision_speed = np.array(vision_speed, dtype=float)

# Ignore early frames before calibration is established.
if first_calibrated_idx is None:
    warmup_frames = min(calib_len, n)
else:
    warmup_frames = first_calibrated_idx

print("Calibration warm-up frames ignored:", warmup_frames)

print("Vision affine params:", f"a={a_prev:.3f}", f"b={b_prev:.3f}")

# ---------------------------------------------------
# ERROR METRICS
# ---------------------------------------------------

print("\nError Metrics")

valid_mask = np.arange(n) >= warmup_frames

mae_vision = mean_absolute_error(can_aligned[valid_mask], vision_speed[valid_mask])
rmse_vision = np.sqrt(mean_squared_error(can_aligned[valid_mask], vision_speed[valid_mask]))

mae_gps = mean_absolute_error(can_aligned[valid_mask], gps_aligned[valid_mask])
rmse_gps = np.sqrt(mean_squared_error(can_aligned[valid_mask], gps_aligned[valid_mask]))

print("Vision vs CAN MAE:", round(mae_vision,2))
print("Vision vs CAN RMSE:", round(rmse_vision,2))

print("GPS vs CAN MAE:", round(mae_gps,2))
print("GPS vs CAN RMSE:", round(rmse_gps,2))

# ---------------------------------------------------
# SAVE CSV (same schema as previous fast-vision exports)
# ---------------------------------------------------

if frame_times.size >= (n + 1):
    time_aligned = frame_times[1:n + 1]
else:
    time_aligned = frame_times[:n]

with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["time", "pixel_shift", "can_v_kmh", "gps_v_kmh", "Vision_v_kmh"])
    for i in range(n):
        writer.writerow([
            float(time_aligned[i]),
            float(vision_flow_proxy[i]),
            float(can_aligned[i]),
            float(gps_aligned[i]),
            float(vision_speed[i]),
        ])

print("Saved CSV:", output_csv_path)

# ---------------------------------------------------
# PLOT RESULTS
# ---------------------------------------------------

if SHOW_PLOT:
    plt.figure(figsize=(12,6))

    vision_plot = vision_speed.copy()
    vision_plot[:warmup_frames] = np.nan

    plt.plot(can_aligned, label="CAN Speed (Ground Truth)", linewidth=2)
    plt.plot(gps_aligned, label="GPS Speed", alpha=0.8)
    plt.plot(vision_plot, label="Vision Speed", alpha=0.8)

    plt.xlabel("Frame")
    plt.ylabel("Speed (km/h)")
    plt.title("Speed Comparison: Vision vs GPS vs CAN")
    plt.legend()
    plt.grid()

    plt.show()