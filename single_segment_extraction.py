import cv2
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

# --- CONFIGURATION ---
segment_path = "comma2k19/Chunk_3/99c94dc769b5d96e_2018-05-01--08-13-53/30/"
output_file = "csvs/segment_30_bev_vision.csv"
fps = 20.0
frame_skip = 2  

def get_bev_matrix(h, w):
    """
    Creates a Perspective Transform matrix to 'unwarp' the dashcam view.
    Adjust these src points if your lane lines aren't perfectly parallel in BEV.
    """
    src = np.float32([
        [w * 0.42, h * 0.55], # Top Left (Horizon)
        [w * 0.58, h * 0.55], # Top Right (Horizon)
        [w * 0.95, h * 0.95], # Bottom Right (Near Hood)
        [w * 0.05, h * 0.95]  # Bottom Left (Near Hood)
    ])
    
    dst = np.float32([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ])
    return cv2.getPerspectiveTransform(src, dst)

def extract_single_segment_vision(segment_path, output_filename, is_nighttime=False):
    print(f"Extracting BEV Vision: {segment_path} | Night Mode: {is_nighttime}")
    results = []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if is_nighttime else None
    
    try:
        # Load Data Arrays
        can_speed = np.load(os.path.join(segment_path, "processed_log/CAN/speed/value")).flatten()
        can_time = np.load(os.path.join(segment_path, "processed_log/CAN/speed/t")).flatten()
        gnss_val = np.load(os.path.join(segment_path, "processed_log/GNSS/live_gnss_qcom/value"))
        gnss_time = np.load(os.path.join(segment_path, "processed_log/GNSS/live_gnss_qcom/t")).flatten()
        frame_times = np.load(os.path.join(segment_path, "global_pose/frame_times")).flatten()
        
        cap = cv2.VideoCapture(os.path.join(segment_path, "video.hevc"))
        ret, first_frame = cap.read()
        if not ret: return
        
        h, w = first_frame.shape[:2]
        M = get_bev_matrix(h, w)

        def preprocess(frame):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Remove high-frequency sensor noise (crucial for nighttime grain)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            # Warp to Bird's Eye View
            warped = cv2.warpPerspective(gray, M, (w, h))
            # Focus on a central strip to avoid noisy roadside objects
            roi = warped[int(h*0.2):int(h*0.8), int(w*0.3):int(w*0.7)]
            if is_nighttime:
                roi = clahe.apply(roi)
            return roi

        prev_roi = preprocess(first_frame)

        # GPS Calculation
        lats, lons = np.radians(gnss_val[:, 0]), np.radians(gnss_val[:, 1])
        gps_dist = 6371000.0 * 2 * np.arcsin(np.sqrt(np.sin(np.diff(lats)/2)**2 + 
                   np.cos(lats[:-1]) * np.cos(lats[1:]) * np.sin(np.diff(lons)/2)**2))
        gps_v_ms = gps_dist / np.diff(gnss_time)

        for i in tqdm(range(1, len(frame_times), frame_skip)):
            for _ in range(frame_skip - 1): cap.grab()
            ret, frame = cap.read()
            if not ret: break
            
            roi = preprocess(frame)
            flow = cv2.calcOpticalFlowFarneback(prev_roi, roi, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            # Use vertical flow component (y-axis) for forward motion
            v_movement = flow[..., 1]
            valid = v_movement[(v_movement > 0.1) & (v_movement < 30.0)]
            shift = np.median(valid) if len(valid) > 0 else 0.0
            
            results.append({
                'time': frame_times[i],
                'pixel_shift': shift, 
                'can_v_kmh': np.interp(frame_times[i], can_time, can_speed) * 3.6,
                'gps_v_kmh': np.interp(frame_times[i], gnss_time[1:], gps_v_ms) * 3.6
            })
            prev_roi = roi
        cap.release()
        
    except Exception as e:
        print(f"Error: {e}")
        return

    df = pd.DataFrame(results)
    
    # Calibration: Use fit_intercept=False to force 0 shift = 0 speed
    model = LinearRegression(fit_intercept=False).fit(df[['pixel_shift']], df['can_v_kmh'])
    df['Vision_v_kmh'] = model.predict(df[['pixel_shift']])
    
    # Temporal Smoothing
    smooth_win = 40 if is_nighttime else 20
    df['Vision_v_kmh'] = df['Vision_v_kmh'].rolling(window=smooth_win, min_periods=1).mean()
    
    df.to_csv(output_filename, index=False)
    mae = (df['Vision_v_kmh'] - df['can_v_kmh']).abs().mean()
    print(f"\nBEV Success! MAE: {mae:.2f} km/h")

if __name__ == "__main__":
    extract_single_segment_vision(segment_path, output_file, is_nighttime=False)