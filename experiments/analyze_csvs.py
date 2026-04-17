#!/usr/bin/env python3
"""
Post-hoc analyzer for metrics CSVs produced by dinov2_servo.py.

For each CSV it computes:
- final EKF-reported position error (cm)
- iterations to reach a tolerance (default 1 cm)
- trajectory length (sum of per-step FK deltas, mm)
- straight-line FK distance from first to last logged robot pose (mm)
- path efficiency = straight_line / trajectory_length
- median per-iteration wall-clock time (ms)
- run duration (s), total frames
- perception pipeline tag (ekf / ibvs)

Usage:
    python analyze_csvs.py runs/
    python analyze_csvs.py runs/ --tol-cm 1.0 --csv summary.csv
    python analyze_csvs.py runs/ --group-by pipeline,run_tag
"""
import argparse
import csv
import glob
import os
from pathlib import Path
from statistics import median


def _f(row, key, default=None):
    """Parse a float cell; return default if empty / non-numeric."""
    val = row.get(key, "")
    if val is None or val == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


def analyze_one(path: Path, tol_m: float) -> dict:
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {"file": path.name, "error": "empty"}

    frames = len(rows)
    pipeline = rows[0].get("pipeline") or ""
    run_tag = rows[0].get("run_tag") or ""

    # Error at convergence (last row with non-empty err_m)
    err_rows = [r for r in rows if _f(r, "err_m") is not None]
    final_err_m = _f(err_rows[-1], "err_m") if err_rows else None

    # First frame where err <= tol
    iters_to_tol = None
    for idx, r in enumerate(rows, 1):
        e = _f(r, "err_m")
        if e is not None and e <= tol_m:
            iters_to_tol = idx
            break

    # FK trajectory
    fk = []
    for r in rows:
        x = _f(r, "robot_x_mm")
        y = _f(r, "robot_y_mm")
        z = _f(r, "robot_z_mm")
        if x is not None and y is not None and z is not None:
            fk.append((x, y, z))
    traj_len_mm = 0.0
    for i in range(1, len(fk)):
        dx = fk[i][0] - fk[i - 1][0]
        dy = fk[i][1] - fk[i - 1][1]
        dz = fk[i][2] - fk[i - 1][2]
        traj_len_mm += (dx * dx + dy * dy + dz * dz) ** 0.5
    straight_mm = None
    path_eff = None
    if len(fk) >= 2:
        sx = fk[-1][0] - fk[0][0]
        sy = fk[-1][1] - fk[0][1]
        sz = fk[-1][2] - fk[0][2]
        straight_mm = (sx * sx + sy * sy + sz * sz) ** 0.5
        if traj_len_mm > 1e-6:
            path_eff = straight_mm / traj_len_mm

    # Timing
    iter_times = [_f(r, "iter_time_ms") for r in rows]
    iter_times = [t for t in iter_times if t is not None]
    median_iter_ms = median(iter_times) if iter_times else None

    # Duration
    ts = [_f(r, "timestamp") for r in rows]
    ts = [t for t in ts if t is not None]
    duration_s = (ts[-1] - ts[0]) if len(ts) >= 2 else None

    # FoundationPose jitter vs EKF smoothness: compare consecutive raw
    # fp_raw_z values to the filtered ekf_z. Only meaningful in FP mode.
    fp_raw_std_cm = None
    if pipeline == "foundationpose":
        fp_raw = []
        ekf_z = []
        for r in rows:
            rz = _f(r, "fp_raw_z")
            ez = _f(r, "ekf_z")
            if rz is not None and ez is not None:
                fp_raw.append(rz)
                ekf_z.append(ez)
        if len(fp_raw) >= 3:
            import statistics as st
            diffs = [fp_raw[i] - fp_raw[i - 1] for i in range(1, len(fp_raw))]
            fp_raw_std_cm = st.pstdev(diffs) * 100.0

    return {
        "file": path.name,
        "pipeline": pipeline,
        "run_tag": run_tag,
        "frames": frames,
        "duration_s": duration_s,
        "final_err_cm": (final_err_m * 100.0) if final_err_m is not None else None,
        "iters_to_tol": iters_to_tol,
        "traj_len_mm": traj_len_mm if fk else None,
        "straight_mm": straight_mm,
        "path_eff": path_eff,
        "median_iter_ms": median_iter_ms,
        "fp_raw_step_std_cm": fp_raw_std_cm,
    }


def fmt(val, spec):
    return f"{val:{spec}}" if val is not None else "-"


def print_table(rows):
    headers = [
        ("run_tag", 28), ("pipeline", 14), ("frames", 6), ("dur_s", 7),
        ("err_cm", 7), ("iter@tol", 8), ("traj_mm", 8),
        ("straight_mm", 11), ("path_eff", 8), ("iter_ms", 8),
        ("fp_jit_cm", 9),
    ]
    line = " ".join(f"{h:<{w}}" for h, w in headers)
    print(line)
    print("-" * len(line))
    for r in rows:
        print(" ".join([
            f"{r.get('run_tag', '')[:28]:<28}",
            f"{r.get('pipeline', ''):<14}",
            f"{r.get('frames', 0):<6}",
            f"{fmt(r.get('duration_s'), '.1f'):<7}",
            f"{fmt(r.get('final_err_cm'), '.2f'):<7}",
            f"{fmt(r.get('iters_to_tol'), 'd'):<8}",
            f"{fmt(r.get('traj_len_mm'), '.1f'):<8}",
            f"{fmt(r.get('straight_mm'), '.1f'):<11}",
            f"{fmt(r.get('path_eff'), '.3f'):<8}",
            f"{fmt(r.get('median_iter_ms'), '.1f'):<8}",
            f"{fmt(r.get('fp_raw_step_std_cm'), '.2f'):<9}",
        ]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path,
                    help="Directory of metrics CSVs or a single CSV file")
    ap.add_argument("--tol-cm", type=float, default=1.0,
                    help="Convergence tolerance in cm (default: 1.0)")
    ap.add_argument("--csv", type=Path,
                    help="Also write the summary rows to this CSV")
    args = ap.parse_args()

    if args.path.is_file():
        files = [args.path]
    else:
        files = sorted(args.path.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files found under {args.path}")

    tol_m = args.tol_cm / 100.0
    rows = [analyze_one(p, tol_m) for p in files]

    print_table(rows)

    if args.csv:
        with args.csv.open("w", newline="") as f:
            fieldnames = [
                "file", "pipeline", "run_tag", "frames", "duration_s",
                "final_err_cm", "iters_to_tol", "traj_len_mm",
                "straight_mm", "path_eff", "median_iter_ms",
                "fp_raw_step_std_cm",
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in fieldnames})
        print(f"\nSummary written to {args.csv}")


if __name__ == "__main__":
    main()
