#!/usr/bin/env python3
"""
Post-hoc analyzer for metrics CSVs produced by dinov2_servo.py.

For each CSV it computes:
- final EKF-reported position error (cm)
- iterations to reach each tolerance in --tols-cm (default: 0.5, 1.0, 2.0 cm)
- trajectory length (sum of per-step FK deltas, mm)
- straight-line FK distance from first to last logged robot pose (mm)
- path efficiency = straight_line / trajectory_length
- median per-iteration wall-clock time (ms)
- run duration (s), total frames
- perception pipeline tag (ekf / ibvs / foundationpose)
- FoundationPose frame-to-frame raw-pose jitter (cm stddev)

Convergence is reported at multiple thresholds so the paper can say
something like "100% of EKF-FP trials reached 2 cm, 85% reached 1 cm,
20% reached 0.5 cm" rather than collapsing everything into a single
pass/fail. Strict tolerances discriminate between good and excellent
pipelines; loose tolerances discriminate against catastrophic failure.

Usage:
    python analyze_csvs.py runs/
    python analyze_csvs.py runs/ --tols-cm 0.5,1.0,2.0 --csv summary.csv
    python analyze_csvs.py runs/ --tols-cm 1.0 --csv summary.csv  # single tol
"""
import argparse
import csv
import os
from pathlib import Path
from statistics import median


DEFAULT_TOLS_CM = "0.5,1.0,2.0"


def _f(row, key, default=None):
    """Parse a float cell; return default if empty / non-numeric."""
    val = row.get(key, "")
    if val is None or val == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _parse_tols(spec: str) -> list:
    """Parse a comma-separated list of cm tolerances, sorted ascending."""
    try:
        vals = sorted({float(t.strip()) for t in spec.split(",") if t.strip()})
    except ValueError as exc:
        raise SystemExit(f"Invalid --tols-cm value {spec!r}: {exc}")
    if not vals:
        raise SystemExit("--tols-cm produced an empty list")
    if any(v <= 0 for v in vals):
        raise SystemExit("--tols-cm values must all be positive (cm)")
    return vals


def analyze_one(path: Path, tols_m: list) -> dict:
    """
    Parse one metrics CSV and compute per-run metrics.

    Parameters
    ----------
    path : Path
        CSV file produced by dinov2_servo.py.
    tols_m : list[float]
        Convergence tolerances in METRES, sorted ascending. For each tol,
        the output dict gets an ``iter@{tol_cm}cm`` entry: either the
        1-indexed frame at which err_m first dropped below tol, or None
        if the run never reached that tolerance.
    """
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

    # First frame where err <= each tolerance. Single pass over rows:
    # track the smallest unsatisfied tolerance index, check, advance.
    iters_to_tol = {t: None for t in tols_m}
    for idx, r in enumerate(rows, 1):
        e = _f(r, "err_m")
        if e is None:
            continue
        for t in tols_m:
            if iters_to_tol[t] is None and e <= t:
                iters_to_tol[t] = idx

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

    # FoundationPose jitter (frame-to-frame stddev of raw Z, cm).
    # Quantifies how much noise the EKF is responsible for damping.
    fp_raw_std_cm = None
    if pipeline == "foundationpose":
        fp_raw = []
        for r in rows:
            rz = _f(r, "fp_raw_z")
            if rz is not None:
                fp_raw.append(rz)
        if len(fp_raw) >= 3:
            import statistics as st
            diffs = [fp_raw[i] - fp_raw[i - 1] for i in range(1, len(fp_raw))]
            fp_raw_std_cm = st.pstdev(diffs) * 100.0

    out = {
        "file": path.name,
        "pipeline": pipeline,
        "run_tag": run_tag,
        "frames": frames,
        "duration_s": duration_s,
        "final_err_cm": (final_err_m * 100.0) if final_err_m is not None else None,
        "traj_len_mm": traj_len_mm if fk else None,
        "straight_mm": straight_mm,
        "path_eff": path_eff,
        "median_iter_ms": median_iter_ms,
        "fp_raw_step_std_cm": fp_raw_std_cm,
    }
    # One column per tolerance, keyed by the cm value as a float string
    for t_m in tols_m:
        t_cm = t_m * 100.0
        key = f"iter_at_{_tol_key(t_cm)}cm"
        out[key] = iters_to_tol[t_m]
    return out


def _tol_key(t_cm: float) -> str:
    """Format a cm tolerance as a compact string for column names."""
    if abs(t_cm - round(t_cm)) < 1e-6:
        return f"{int(round(t_cm))}"
    return f"{t_cm:g}".replace(".", "p")


def fmt(val, spec):
    return f"{val:{spec}}" if val is not None else "-"


def print_table(rows, tols_cm):
    """
    Print a table whose fixed columns describe the run and whose
    trailing columns show "iter to reach X cm" for each tolerance.
    """
    fixed_headers = [
        ("run_tag", 28), ("pipeline", 14), ("frames", 6), ("dur_s", 7),
        ("err_cm", 7),
    ]
    tol_headers = [(f"it@{_tol_key(t)}cm", 9) for t in tols_cm]
    trailing_headers = [
        ("traj_mm", 8), ("straight_mm", 11), ("path_eff", 8),
        ("iter_ms", 8), ("fp_jit_cm", 9),
    ]
    headers = fixed_headers + tol_headers + trailing_headers

    line = " ".join(f"{h:<{w}}" for h, w in headers)
    print(line)
    print("-" * len(line))

    for r in rows:
        cells = [
            f"{r.get('run_tag', '')[:28]:<28}",
            f"{r.get('pipeline', ''):<14}",
            f"{r.get('frames', 0):<6}",
            f"{fmt(r.get('duration_s'), '.1f'):<7}",
            f"{fmt(r.get('final_err_cm'), '.2f'):<7}",
        ]
        for t_cm in tols_cm:
            key = f"iter_at_{_tol_key(t_cm)}cm"
            cells.append(f"{fmt(r.get(key), 'd'):<9}")
        cells.extend([
            f"{fmt(r.get('traj_len_mm'), '.1f'):<8}",
            f"{fmt(r.get('straight_mm'), '.1f'):<11}",
            f"{fmt(r.get('path_eff'), '.3f'):<8}",
            f"{fmt(r.get('median_iter_ms'), '.1f'):<8}",
            f"{fmt(r.get('fp_raw_step_std_cm'), '.2f'):<9}",
        ])
        print(" ".join(cells))


def print_aggregate(rows, tols_cm):
    """
    Print a pipeline-grouped aggregate that collapses individual runs
    into pass-rates at each tolerance. Useful for the paper's headline:
    "what fraction of trials converged within X cm, grouped by pipeline?"
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[r.get("pipeline") or "(unknown)"].append(r)

    print()
    print("Aggregate pass-rate by pipeline:")
    headers = ["pipeline", "n"]
    for t_cm in tols_cm:
        headers.append(f"<={t_cm}cm")
    headers += ["med_err_cm", "med_path_eff"]
    print("  " + "  ".join(f"{h:<12}" for h in headers))
    print("  " + "-" * (14 * len(headers)))
    for pipe, grp in sorted(groups.items()):
        n = len(grp)
        row = [f"{pipe:<12}", f"{n:<12}"]
        for t_cm in tols_cm:
            key = f"iter_at_{_tol_key(t_cm)}cm"
            passed = sum(1 for r in grp if r.get(key) is not None)
            row.append(f"{passed/n*100:5.1f}%      " if n else "-")
        # Median final error and path efficiency across the group
        errs = [r.get("final_err_cm") for r in grp
                if r.get("final_err_cm") is not None]
        effs = [r.get("path_eff") for r in grp
                if r.get("path_eff") is not None]
        row.append(f"{median(errs):<12.2f}" if errs else f"{'-':<12}")
        row.append(f"{median(effs):<12.3f}" if effs else f"{'-':<12}")
        print("  " + "  ".join(row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path,
                    help="Directory of metrics CSVs or a single CSV file")
    ap.add_argument("--tols-cm", type=str, default=DEFAULT_TOLS_CM,
                    help="Comma-separated convergence tolerances in cm "
                         f"(default: {DEFAULT_TOLS_CM})")
    ap.add_argument("--tol-cm", type=float, default=None,
                    help="Legacy single-tolerance flag; equivalent to "
                         "--tols-cm VALUE. Kept for backward compatibility.")
    ap.add_argument("--csv", type=Path,
                    help="Also write the summary rows to this CSV")
    ap.add_argument("--no-aggregate", action="store_true",
                    help="Skip the pipeline-level aggregate at the bottom")
    args = ap.parse_args()

    if args.path.is_file():
        files = [args.path]
    else:
        files = sorted(args.path.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files found under {args.path}")

    # --tol-cm (singular, legacy) overrides --tols-cm if both given
    tols_spec = (str(args.tol_cm) if args.tol_cm is not None
                 else args.tols_cm)
    tols_cm = _parse_tols(tols_spec)
    tols_m = [t / 100.0 for t in tols_cm]

    rows = [analyze_one(p, tols_m) for p in files]

    print_table(rows, tols_cm)

    if not args.no_aggregate and len(rows) > 1:
        print_aggregate(rows, tols_cm)

    if args.csv:
        # Dynamic fieldnames: core columns plus one iter_at_*cm per tolerance
        base_fields = [
            "file", "pipeline", "run_tag", "frames", "duration_s",
            "final_err_cm",
        ]
        tol_fields = [f"iter_at_{_tol_key(t)}cm" for t in tols_cm]
        trail_fields = [
            "traj_len_mm", "straight_mm", "path_eff",
            "median_iter_ms", "fp_raw_step_std_cm",
        ]
        fieldnames = base_fields + tol_fields + trail_fields
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in fieldnames})
        print(f"\nSummary written to {args.csv}")


if __name__ == "__main__":
    main()
