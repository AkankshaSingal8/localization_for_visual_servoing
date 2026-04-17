#!/usr/bin/env python3
"""
Batch runner for on-arm visual servoing trials.

Reads a trials file (YAML or JSON), prompts the operator between trials,
and launches dinov2_servo.py for each trial with the appropriate --mode,
--reference, and --run-tag. The operator stops each trial by pressing 'q'
in the OpenCV window.

Example trials.yaml:
    common:
        cam_to_robot: zed_forward
        depth_cal_m: 0.30
    trials:
        - tag: cheezit_pose1_ekf_t1
          mode: ekf
          reference: ../masked_objects/cheez_it_box.png
        - tag: cheezit_pose1_ibvs_t1
          mode: ibvs
          reference: ../masked_objects/cheez_it_box.png

Usage:
    python run_batch.py --trials trials.yaml
    python run_batch.py --trials trials.json --csv-dir runs/
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SERVO_SCRIPT = Path(__file__).resolve().parent.parent / "FoundationModel" / "dinov2_servo.py"


def load_trials(path: Path) -> dict:
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError:
            sys.exit("PyYAML not installed. Install it or use JSON: "
                     "pip install pyyaml")
        return yaml.safe_load(text)
    return json.loads(text)


def build_cmd(trial: dict, common: dict) -> list:
    cmd = [sys.executable, str(SERVO_SCRIPT)]
    ref = trial.get("reference") or common.get("reference")
    if not ref:
        sys.exit(f"Trial {trial.get('tag')} has no reference image")
    cmd += ["--reference", ref]
    mode = trial.get("mode", common.get("mode", "ekf"))
    cmd += ["--mode", mode]
    cmd += ["--run-tag", trial.get("tag", "")]
    cmd += ["--cam-to-robot",
            trial.get("cam_to_robot", common.get("cam_to_robot", "identity"))]
    cmd += ["--depth-cal-m",
            str(trial.get("depth_cal_m", common.get("depth_cal_m", 0.30)))]

    # Persisted depth calibration (skip per-trial 'c' keypress)
    for key, flag in [
        ("depth_scale",         "--depth-scale"),
        ("depth_offset",        "--depth-offset"),
        ("process_noise_pos",   "--process-noise-pos"),
        ("process_noise_vel",   "--process-noise-vel"),
        ("meas_noise_uv",       "--meas-noise-uv"),
        ("meas_noise_z",        "--meas-noise-z"),
        ("pbvs_gain",           "--pbvs-gain"),
        ("target_depth",        "--target-depth"),
        ("pbvs_max_vel",        "--pbvs-max-vel"),
        ("pbvs_dead_zone",      "--pbvs-dead-zone"),
    ]:
        val = trial.get(key, common.get(key))
        if val is not None:
            cmd += [flag, str(val)]

    # FoundationPose-specific flags (only forwarded when --mode foundationpose)
    if mode == "foundationpose":
        fp_mesh = trial.get("fp_mesh") or common.get("fp_mesh")
        fp_box = trial.get("fp_box") or common.get("fp_box")
        if fp_mesh:
            cmd += ["--fp-mesh", fp_mesh]
        elif fp_box:
            cmd += ["--fp-box", *[str(v) for v in fp_box]]
        else:
            sys.exit(f"Trial {trial.get('tag')} is foundationpose "
                     "but has no fp_mesh or fp_box")
        for key, flag in [
            ("fp_repo_dir",       "--fp-repo-dir"),
            ("fp_weights_dir",    "--fp-weights-dir"),
            ("fp_est_iter",       "--fp-est-iter"),
            ("fp_track_iter",     "--fp-track-iter"),
            ("fp_redetect_interval", "--fp-redetect-interval"),
            ("fp_meas_noise",     "--fp-meas-noise"),
        ]:
            val = trial.get(key, common.get(key))
            if val is not None:
                cmd += [flag, str(val)]

    extra = trial.get("extra_args") or common.get("extra_args") or []
    cmd += list(extra)
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", required=True, type=Path,
                    help="Path to YAML/JSON trials file")
    ap.add_argument("--csv-dir", type=Path, default=Path("runs"),
                    help="Directory to move each trial's metrics CSV into")
    ap.add_argument("--skip-confirm", action="store_true",
                    help="Don't wait for ENTER between trials (dangerous)")
    args = ap.parse_args()

    spec = load_trials(args.trials)
    common = spec.get("common", {}) or {}
    trials = spec.get("trials", []) or []
    if not trials:
        sys.exit("No trials found in spec")

    args.csv_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoaded {len(trials)} trials from {args.trials}")
    print(f"CSVs will be collected into: {args.csv_dir}\n")

    for idx, trial in enumerate(trials, 1):
        tag = trial.get("tag", f"trial_{idx}")
        print("=" * 70)
        print(f"Trial {idx}/{len(trials)}: {tag}")
        print(f"  mode:      {trial.get('mode', 'ekf')}")
        print(f"  reference: {trial.get('reference')}")
        if trial.get("notes"):
            print(f"  notes:     {trial['notes']}")
        print("=" * 70)
        if not args.skip_confirm:
            input("Reset object / robot pose, then press ENTER to start. "
                  "Press 'q' in the OpenCV window to end the trial.")

        # Snapshot existing metrics_*.csv so we can identify this run's file
        before = set(Path.cwd().glob("metrics_*.csv"))
        t_start = time.time()

        cmd = build_cmd(trial, common)
        print(f"$ {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=False)
        except KeyboardInterrupt:
            print("\nInterrupted; stopping batch.")
            break

        # Identify the new CSV
        after = set(Path.cwd().glob("metrics_*.csv"))
        new_csvs = after - before
        if new_csvs:
            src = max(new_csvs, key=lambda p: p.stat().st_mtime)
            dst = args.csv_dir / f"{tag}.csv"
            shutil.move(str(src), str(dst))
            print(f"  -> CSV: {dst}  (trial took {time.time() - t_start:.1f}s)")
        else:
            print("  WARNING: no new metrics_*.csv produced for this trial")

    print("\nBatch complete.")


if __name__ == "__main__":
    main()
