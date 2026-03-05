import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the data
df = pd.read_csv('csvs/segment_30_bev_vision.csv')

# 2. Calculate Standard Metrics for BOTH sources (vs CAN)
mae_vis = mean_absolute_error(df['can_v_kmh'], df['Vision_v_kmh'])
mae_gps = mean_absolute_error(df['can_v_kmh'], df['gps_v_kmh'])

rmse_vis = np.sqrt(mean_squared_error(df['can_v_kmh'], df['Vision_v_kmh']))
rmse_gps = np.sqrt(mean_squared_error(df['can_v_kmh'], df['gps_v_kmh']))

mbe_vis = np.mean(df['Vision_v_kmh'] - df['can_v_kmh'])
mbe_gps = np.mean(df['gps_v_kmh'] - df['can_v_kmh'])

# Print Results for the terminal
print("--- RESEARCH METRICS (Baseline = CAN) ---")
print(f"MAE  | Vision: {mae_vis:.2f} km/h | GPS: {mae_gps:.2f} km/h")
print(f"RMSE | Vision: {rmse_vis:.2f} km/h | GPS: {rmse_gps:.2f} km/h")
print(f"MBE  | Vision: {mbe_vis:.2f} km/h | GPS: {mbe_gps:.2f} km/h")

# 3. Setup the Research Dashboard (2x3 Grid)
fig, axes = plt.subplots(2, 3, figsize=(22, 12))
fig.suptitle('3-Way Speed Analysis: Vision vs GPS vs CAN Ground Truth', fontsize=18, fontweight='bold')

# --- ROW 1: STANDARD VISUALISATIONS ---

# Plot A: Time Series (All 3)
axes[0, 0].plot(df['time'], df['can_v_kmh'], label='CAN Speed (Truth)', color='blue', linewidth=2)
axes[0, 0].plot(df['time'], df['gps_v_kmh'], label='GPS Speed', color='green', linewidth=1.5, linestyle='--')
axes[0, 0].plot(df['time'], df['Vision_v_kmh'], label='Vision Prediction', color='orange', linewidth=1.5)
axes[0, 0].set_title('Speed vs Time')
axes[0, 0].set_xlabel('Time (seconds)')
axes[0, 0].set_ylabel('Speed (km/h)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot B: Scatter Plot (Vision & GPS vs CAN)
axes[0, 1].scatter(df['can_v_kmh'], df['Vision_v_kmh'], label='Vision vs CAN', color='orange', alpha=0.6)
axes[0, 1].scatter(df['can_v_kmh'], df['gps_v_kmh'], label='GPS vs CAN', color='green', alpha=0.6, marker='x')

min_val = min(df['can_v_kmh'].min(), df['Vision_v_kmh'].min(), df['gps_v_kmh'].min())
max_val = max(df['can_v_kmh'].max(), df['Vision_v_kmh'].max(), df['gps_v_kmh'].max())
axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect 1:1 Correlation')

axes[0, 1].set_title('Measurement vs Actual Speed')
axes[0, 1].set_xlabel('CAN Speed (km/h)')
axes[0, 1].set_ylabel('Measured Speed (km/h)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot C: Error Distribution (Vision & GPS)
error_vis = df['Vision_v_kmh'] - df['can_v_kmh']
error_gps = df['gps_v_kmh'] - df['can_v_kmh']

axes[0, 2].hist(error_vis, bins=30, color='orange', alpha=0.6, edgecolor='black', label='Vision Error')
axes[0, 2].hist(error_gps, bins=30, color='green', alpha=0.6, edgecolor='black', label='GPS Error')
axes[0, 2].axvline(0, color='k', linestyle='dashed', linewidth=2, label='Zero Error')
axes[0, 2].set_title('Distribution of Errors (vs CAN)')
axes[0, 2].set_xlabel('Error (km/h) [Negative = Underestimated]')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].legend()

# --- ROW 2: ADVANCED RESEARCH VISUALISATIONS ---

# Plot D: Correlation Heatmap (All relevant columns)
cols_to_correlate = ['pixel_shift', 'Vision_v_kmh', 'gps_v_kmh', 'can_v_kmh']
corr_matrix = df[cols_to_correlate].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".3f", ax=axes[1, 0], vmin=0.5, vmax=1.0)
axes[1, 0].set_title('Pearson Correlation Matrix')

# Plot E: Multi-Bland-Altman Plot
# Vision Calculations
mean_vis_can = (df['Vision_v_kmh'] + df['can_v_kmh']) / 2
diff_vis_can = df['Vision_v_kmh'] - df['can_v_kmh']
md_vis = np.mean(diff_vis_can)
sd_vis = np.std(diff_vis_can)

# GPS Calculations
mean_gps_can = (df['gps_v_kmh'] + df['can_v_kmh']) / 2
diff_gps_can = df['gps_v_kmh'] - df['can_v_kmh']
md_gps = np.mean(diff_gps_can)
sd_gps = np.std(diff_gps_can)

axes[1, 1].scatter(mean_vis_can, diff_vis_can, alpha=0.5, color='orange', label='Vision vs CAN')
axes[1, 1].scatter(mean_gps_can, diff_gps_can, alpha=0.5, color='green', marker='x', label='GPS vs CAN')

axes[1, 1].axhline(md_vis, color='orange', linestyle='-', linewidth=2)
axes[1, 1].axhline(md_gps, color='green', linestyle='-', linewidth=2)
axes[1, 1].axhline(0, color='black', linestyle='--', linewidth=1)

axes[1, 1].set_title('Bland-Altman Plot (Comparing Both to CAN)')
axes[1, 1].set_xlabel('Average Speed (km/h)')
axes[1, 1].set_ylabel('Difference from CAN (km/h)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Plot F: Residuals vs True Speed Plot (Heteroscedasticity Check for Both)
axes[1, 2].scatter(df['can_v_kmh'], error_vis, alpha=0.5, color='orange', label='Vision Residuals')
axes[1, 2].scatter(df['can_v_kmh'], error_gps, alpha=0.5, color='green', marker='x', label='GPS Residuals')

axes[1, 2].axhline(0, color='black', linestyle='--', linewidth=1.5)
axes[1, 2].set_title('Residuals vs True Speed (CAN)')
axes[1, 2].set_xlabel('True Speed (CAN) (km/h)')
axes[1, 2].set_ylabel('Residual Error (km/h)')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

# Final formatting adjustments
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
plt.savefig('advanced_research_dashboard_3way.png', dpi=300) 
plt.show()