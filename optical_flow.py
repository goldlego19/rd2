import cv2
import numpy as np
import os

# 1. Path to your specific segment's video
segment_path = "comma2k19/Chunk_3/99c94dc769b5d96e_2018-05-01--08-13-53/30/"
video_path = os.path.join(segment_path, "video.hevc")

def visualise_smoothed_optical_flow(video_path):
    print(f"Opening video player for: {video_path}")
    print("Press 'q' on your keyboard to close the video window.")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    fps = 20.0 
    ret, prev_frame = cap.read()
    if not ret:
        return
        
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # 2. Define Region of Interest (ROI)
    # 2. Define Region of Interest (ROI)
    height, width = prev_gray.shape
    roi_top = int(height * 0.55)    # Lowered slightly to avoid oncoming headlights
    roi_bottom = int(height * 0.75) # Lifted significantly to crop out the car's bonnet
    roi_left = int(width * 0.15)    
    roi_right = int(width * 0.85)  
    
    prev_roi = prev_gray[roi_top:roi_bottom, roi_left:roi_right]
    
    calibration_scalar = 2.81 
    
    # Initialize the smoothing variables
    speed_history = []
    smoothing_window = 40
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached.")
            break 
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = gray[roi_top:roi_bottom, roi_left:roi_right]
        
        # 3. Calculate Flow
        flow = cv2.calcOpticalFlowFarneback(prev_roi, roi, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # Extract the vertical flow
  # Extract the vertical flow
        vertical_movement = flow[..., 1]
        
        # FIX 1: Ignore the featureless dark asphalt AND insane reflection spikes.
        # Only look at pixels moving downwards between 0.5 and 15.0 pixels per frame
        valid_pixels = vertical_movement[(vertical_movement > 0.5) & (vertical_movement < 45.0)]
        
        # Calculate the median movement of ONLY the valid features
        if len(valid_pixels) > 0:
            median_pixel_shift = np.median(valid_pixels)
        else:
            median_pixel_shift = 0.0
            
        raw_speed_kmh = median_pixel_shift * fps * calibration_scalar
        
        # FIX 2: Increase the Temporal Smoothing Window (Update the variable before the loop to 40)
        speed_history.append(raw_speed_kmh)
        if len(speed_history) > 40:  # Increased from 15 to 40
            speed_history.pop(0)
            
        smoothed_speed_kmh = sum(speed_history) / len(speed_history)
        
        # 4. VISUALISATION
        cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 2)
        
        # Display both raw and smoothed speeds to see the difference
        text_smoothed = f"Smoothed Speed: {smoothed_speed_kmh:.2f} km/h"
        text_raw = f"Raw Speed: {raw_speed_kmh:.2f} km/h"
        
        # Draw the text onto the video frame
        cv2.putText(frame, text_smoothed, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.putText(frame, text_raw, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("Optical Flow Debug View", frame)
        
        if cv2.waitKey(50) & 0xFF == ord('q'):
            print("Video playback terminated by user.")
            break
            
        prev_roi = roi

    cap.release()
    cv2.destroyAllWindows()

# Run the debug visualizer
visualise_smoothed_optical_flow(video_path)