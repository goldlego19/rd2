import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


REQUIRED_COLUMNS = [
    "time",
    "pixel_shift",
    "can_v_kmh",
    "gps_v_kmh",
    "Vision_v_kmh",
]


def validate_columns(field_names) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in field_names]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")


def load_csv_as_arrays(csv_path: str) -> dict:
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float, encoding="utf-8")
    if data.size == 0:
        raise ValueError("CSV has no data rows.")

    field_names = data.dtype.names or []
    validate_columns(field_names)

    arrays = {name: np.asarray(data[name], dtype=float) for name in REQUIRED_COLUMNS}

    # Build validity mask so all compared arrays are aligned and finite.
    valid = np.ones_like(arrays["time"], dtype=bool)
    for name in REQUIRED_COLUMNS:
        valid &= np.isfinite(arrays[name])

    filtered = {name: arrays[name][valid] for name in REQUIRED_COLUMNS}
    if filtered["time"].size == 0:
        raise ValueError("No valid rows left after filtering non-finite values.")
    return filtered


def calc_metrics(true_speed: np.ndarray, test_speed: np.ndarray) -> dict:
    err = test_speed - true_speed
    mae = mean_absolute_error(true_speed, test_speed)
    rmse = np.sqrt(mean_squared_error(true_speed, test_speed))
    mbe = float(np.mean(err))
    std_err = float(np.std(err))
    corr = float(np.corrcoef(true_speed, test_speed)[0, 1])
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MBE": mbe,
        "STD_ERR": std_err,
        "CORR": corr,
    }


def rolling_mae(true_speed: np.ndarray, test_speed: np.ndarray, window: int) -> np.ndarray:
    abs_err = np.abs(test_speed - true_speed)
    if window <= 1:
        return abs_err
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(abs_err, kernel, mode="same")


def evaluate_segment(csv_path: str, output_dir: str, window: int = 25, show_plot: bool = False) -> None:
    arrays = load_csv_as_arrays(csv_path)

    t = arrays["time"]
    can = arrays["can_v_kmh"]
    gps = arrays["gps_v_kmh"]
    vis = arrays["Vision_v_kmh"]

    vis_metrics = calc_metrics(can, vis)
    gps_metrics = calc_metrics(can, gps)

    print("--- Segment Evaluation (Baseline: CAN) ---")
    print(f"Rows used: {len(t)}")
    print("Vision vs CAN")
    print(
        f"  MAE={vis_metrics['MAE']:.3f}  RMSE={vis_metrics['RMSE']:.3f}  "
        f"MBE={vis_metrics['MBE']:.3f}  STD={vis_metrics['STD_ERR']:.3f}  "
        f"Corr={vis_metrics['CORR']:.4f}"
    )
    print("GPS vs CAN")
    print(
        f"  MAE={gps_metrics['MAE']:.3f}  RMSE={gps_metrics['RMSE']:.3f}  "
        f"MBE={gps_metrics['MBE']:.3f}  STD={gps_metrics['STD_ERR']:.3f}  "
        f"Corr={gps_metrics['CORR']:.4f}"
    )

    vis_err = vis - can
    gps_err = gps - can

    vis_roll = rolling_mae(can, vis, window)
    gps_roll = rolling_mae(can, gps, window)

    stem = os.path.splitext(os.path.basename(csv_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, f"{stem}_evaluation.png")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Segment Evaluation: {stem}", fontsize=16, fontweight="bold")

    axes[0, 0].plot(t, can, label="CAN (Ground Truth)", linewidth=2.0, color="tab:blue")
    axes[0, 0].plot(t, gps, label="GPS", linewidth=1.6, linestyle="--", color="tab:green")
    axes[0, 0].plot(t, vis, label="Vision", linewidth=1.6, color="tab:orange")
    axes[0, 0].set_title("Speed vs Time (CAN / GPS / Vision)")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Speed (km/h)")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    min_v = float(np.nanmin([can.min(), gps.min(), vis.min()]))
    max_v = float(np.nanmax([can.max(), gps.max(), vis.max()]))
    axes[0, 1].scatter(can, vis, s=10, alpha=0.5, label="Vision vs CAN", color="tab:orange")
    axes[0, 1].scatter(can, gps, s=10, alpha=0.5, label="GPS vs CAN", color="tab:green")
    axes[0, 1].plot([min_v, max_v], [min_v, max_v], "k--", linewidth=1.2, label="Ideal 1:1")
    axes[0, 1].set_title("Measured vs CAN")
    axes[0, 1].set_xlabel("CAN speed (km/h)")
    axes[0, 1].set_ylabel("Measured speed (km/h)")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(t, vis_err, label="Vision - CAN", color="tab:orange", alpha=0.85)
    axes[1, 0].plot(t, gps_err, label="GPS - CAN", color="tab:green", alpha=0.85)
    axes[1, 0].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1, 0].set_title("Signed Error vs Time")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Error (km/h)")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(t, vis_roll, label=f"Vision rolling MAE ({window})", color="tab:orange")
    axes[1, 1].plot(t, gps_roll, label=f"GPS rolling MAE ({window})", color="tab:green")
    axes[1, 1].set_title("Local Error Trend")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("MAE (km/h)")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(fig_path, dpi=220)
    print(f"Saved plot: {fig_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CAN/GPS/Vision speed CSVs.")
    parser.add_argument(
        "--csv",
        type=str,
        default="csvs/segment_30_fast_vision.csv",
        help="Path to input CSV containing CAN/GPS/Vision speeds.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="csvs",
        help="Directory where evaluation image will be saved.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=25,
        help="Rolling MAE window (in rows).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure window in addition to saving the image.",
    )
    args = parser.parse_args()

    evaluate_segment(
        csv_path=args.csv,
        output_dir=args.output_dir,
        window=max(1, args.window),
        show_plot=args.show,
    )


if __name__ == "__main__":
    main()
