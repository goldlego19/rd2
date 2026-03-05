import cv2
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

# --- CONFIGURATION ---
segment_path = "comma2k19/Chunk_3/99c94dc769b5d96e_2018-05-01--08-13-53/30/"
output_file = "segment_30_fast_vision.csv"
fps = 20.0
frame_skip = 2  

# UPDATED: Added is_nighttime parameter
def extract_single_segment_vision(segment_path, output_filename, is_nighttime=False):
    print(f"Loading arrays for segment: {segment_path} | Night Mode: {is_nighttime}")
    results = []
    
    # Initialise CLAHE for night mode
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if is_nighttime else None
    
    try:
        # 1. Load Data Arrays
        can_speed = np.load(os.path.join(segment_path, "processed_log/CAN/speed/value")).flatten()
        can_time = np.load(os.path.join(segment_path, "processed_log/CAN/speed/t")).flatten()
        gnss_val = np.load(os.path.join(segment_path, "processed_log/GNSS/live_gnss_qcom/value"))
        gnss_time = np.load(os.path.join(segment_path, "processed_log/GNSS/live_gnss_qcom/t")).flatten()
        frame_times = np.load(os.path.join(segment_path, "global_pose/frame_times")).flatten()
        
        # 2. Setup Video and ROI
        cap = cv2.VideoCapture(os.path.join(segment_path, "video.hevc"))
        ret, prev_frame = cap.read()
        if not ret: 
            print("Error: Could not read video file.")
            return
            
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        h, w = prev_gray.shape
        
        # At night, you might eventually want to lower this ROI slightly to focus purely on the headlight beam
        r_t, r_b, r_l, r_r = int(h*0.55), int(h*0.75), int(w*0.15), int(w*0.85)
        prev_roi = prev_gray[r_t:r_b, r_l:r_r]
        
        # Apply CLAHE to the first frame if in night mode
        if is_nighttime:
            prev_roi = clahe.apply(prev_roi)

        # 3. Haversine GPS Calculation
        lats, lons = np.radians(gnss_val[:, 0]), np.radians(gnss_val[:, 1])
        gps_dist = 6371000.0 * 2 * np.arcsin(np.sqrt(np.sin(np.diff(lats)/2)**2 + np.cos(lats[:-1]) * np.cos(lats[1:]) * np.sin(np.diff(lons)/2)**2))
        gps_v_ms = gps_dist / np.diff(gnss_time)

        # 4. Frame-by-Frame Optical Flow with TQDM Progress Bar
        total_frames_to_process = len(range(1, len(frame_times), frame_skip))
        
        # Set filtering thresholds based on the time of day
        upper_flow_limit = 25.0 if is_nighttime else 45.0
        
        print("\nProcessing Video...")
        for i in tqdm(range(1, len(frame_times), frame_skip), total=total_frames_to_process, desc="Optical Flow", unit="frames"):
            for _ in range(frame_skip - 1): cap.grab()
            ret, frame = cap.read()
            if not ret: break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi = gray[r_t:r_b, r_l:r_r]
            
            # Apply CLAHE to current frame if in night mode
            if is_nighttime:
                roi = clahe.apply(roi)
            
            # Optical Flow Calculation
            flow = cv2.calcOpticalFlowFarneback(prev_roi, roi, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            # Filter out extreme shifts (headlight glare at night, shadows during the day)
            valid = flow[..., 1][(flow[..., 1] > 0.5) & (flow[..., 1] < upper_flow_limit)]
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
        print(f"An error occurred: {e}")
        return

    # 5. Finalising and Calibrating
    print("\nApplying Linear Regression Calibration...")
    df = pd.DataFrame(results)
    
    # Train the model on this specific segment's data
    model = LinearRegression().fit(df[['pixel_shift']], df['can_v_kmh'])
    
    # Apply the formula to convert pixel shift into km/h
    df['Vision_v_kmh'] = model.predict(df[['pixel_shift']])
    
    # Apply temporal smoothing (more aggressive smoothing for night mode)
    smoothing_window = 40 if is_nighttime else 20
    df['Vision_v_kmh'] = df['Vision_v_kmh'].rolling(window=smoothing_window, min_periods=1).mean()
    
    # 6. Save and print metrics
    df.to_csv(output_filename, index=False)
    
    # Quick error check
    mae = (df['Vision_v_kmh'] - df['can_v_kmh']).abs().mean()
    print(f"\nSuccess! Data exported to {output_filename}")
    print(f"Self-Calibrated Vision MAE for this segment: {mae:.2f} km/h")

# Run the script
if __name__ == "__main__":
    # Simply flip this to True when running your night segments!
    extract_single_segment_vision(segment_path, output_file, is_nighttime=False)