#!/usr/bin/env python3
"""
Sign sanity checks for the PBVS visual-servo loop.

Why this exists
---------------
The PBVS control law lives in ``EKF/ekf_servo.py::PBVSController``. A
single sign error there (bug history: the minus in ``v_cam = -gain*err``)
silently inverts the servo on every axis and the arm drifts *away from*
the target. This script provides three independent ways to verify the
sign chain before enabling the real arm:

  --software         Pure-Python test of PBVSController.compute_velocity
                     on canonical object positions. Needs nothing.

  --hardware-robot   Connects to the xArm, commands small +/- Y, +/- Z,
                     +/- X steps from the current pose, and reports the
                     FK deltas the arm actually executed. Confirms the
                     arm interprets +Y/+Z/+X the same way the code does.

  --hardware-visual  Also opens the ZED (via pyzed), measures median
                     optical flow across each small step, and checks
                     the sign of (dpx, dpy) against the mapping the
                     ``R_cam_to_robot`` preset advertises. Catches a
                     mis-oriented camera mount.

Run the software check first, every time. Only run the hardware checks
in a clear workspace with the arm at a pose with >= 2 cm of clearance
on every axis.

Usage
-----
    # Pure software, no hardware needed
    python experiments/check_servo_signs.py --software

    # Full end-to-end check; defaults to zed_forward mount, +/- 10 mm
    python experiments/check_servo_signs.py --hardware-visual --confirm

    # Robot only (no camera required)
    python experiments/check_servo_signs.py --hardware-robot --confirm
"""
import argparse
import os
import sys
import time

import numpy as np

# Make the repo's EKF and FoundationModel packages importable regardless
# of where the script is launched from.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "EKF"))
sys.path.insert(0, os.path.join(REPO_ROOT, "FoundationModel"))

from ekf_servo import PBVSController, CameraIntrinsics  # noqa: E402


# ── Camera-to-robot mount presets (must match dinov2_servo.py) ───────
CAM_ROT_PRESETS = {
    "identity": np.eye(3),
    "zed_forward": np.array([
        [0.0,  0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]),
}


# ══════════════════════════════════════════════════════════════════════
#  1. Software-only sign check
# ══════════════════════════════════════════════════════════════════════

def _make_pbvs(mount: str, gain: float, target_depth: float,
               max_vel: float) -> PBVSController:
    return PBVSController(
        gain=gain,
        target_depth=target_depth,
        max_vel=max_vel,
        dead_zone_m=0.001,  # tiny so our 2 cm probes always produce a command
        R_cam_to_robot=CAM_ROT_PRESETS[mount],
    )


def _sign(x: float, tol: float = 1e-9) -> int:
    if x > tol:
        return +1
    if x < -tol:
        return -1
    return 0


def software_check(mount: str = "zed_forward",
                   target_depth: float = 0.30,
                   gain: float = 0.5,
                   max_vel: float = 0.02) -> int:
    """
    Run ``PBVSController.compute_velocity`` on six canonical object
    positions. Return 0 if every case produces the physically-correct
    dominant axis sign, else non-zero.
    """
    pbvs = _make_pbvs(mount, gain, target_depth, max_vel)

    # For each case:
    #   name, obj_in_cam, check_axis ('dx_mm'|'dy_mm'|'dz_mm'), expected sign
    # Expected signs are derived from the physics of an eye-in-hand rig
    # with a world-static object and the 'zed_forward' mount preset:
    #   object LEFT  (cam_x < 0)  ->  camera should swing LEFT  ->  dy_mm > 0
    #   object RIGHT (cam_x > 0)  ->  camera should swing RIGHT ->  dy_mm < 0
    #   object ABOVE (cam_y < 0)  ->  camera should rise         ->  dz_mm > 0
    #   object BELOW (cam_y > 0)  ->  camera should drop         ->  dz_mm < 0
    #   object FAR   (cam_z > Z*) ->  camera should advance      ->  dx_mm > 0
    #   object NEAR  (cam_z < Z*) ->  camera should retreat      ->  dx_mm < 0
    cases = [
        ("LEFT  (obj at x=-0.10, z=0.30)",  (-0.10,  0.00,  target_depth),
         "dy_mm", +1),
        ("RIGHT (obj at x=+0.10, z=0.30)",  (+0.10,  0.00,  target_depth),
         "dy_mm", -1),
        ("ABOVE (obj at y=-0.10, z=0.30)",  ( 0.00, -0.10,  target_depth),
         "dz_mm", +1),
        ("BELOW (obj at y=+0.10, z=0.30)",  ( 0.00, +0.10,  target_depth),
         "dz_mm", -1),
        ("FAR   (obj at z=0.50, Z*=0.30)",  ( 0.00,  0.00,  target_depth + 0.20),
         "dx_mm", +1),
        ("NEAR  (obj at z=0.15, Z*=0.30)",  ( 0.00,  0.00,  target_depth - 0.15),
         "dx_mm", -1),
    ]

    # These match FoundationModel/negative_weighing.py so a failure here
    # corresponds directly to what the real servo would command.
    VS_RATE_S = 0.3
    VS_APPROACH_MM = 3.0

    print(f"\n=== Software check (mount={mount}, gain={gain}, Z*={target_depth}) ===")
    print(f"{'case':<36} {'dx_mm':>9} {'dy_mm':>9} {'dz_mm':>9}  check          result")
    print("-" * 92)

    passes = 0
    for name, obj, axis, expected_sign in cases:
        v_robot, v_cam, err_m = pbvs.compute_velocity(np.array(obj))
        dx_mm = float(v_robot[0]) * VS_RATE_S * 1000.0 + VS_APPROACH_MM
        dy_mm = float(v_robot[1]) * VS_RATE_S * 1000.0
        dz_mm = float(v_robot[2]) * VS_RATE_S * 1000.0

        # For the FAR/NEAR cases, dx_mm is the axis of interest; subtract
        # VS_APPROACH to check the controller's own contribution.
        dx_ctrl = dx_mm - VS_APPROACH_MM

        deltas = {"dx_mm": dx_ctrl, "dy_mm": dy_mm, "dz_mm": dz_mm}
        got_sign = _sign(deltas[axis])

        symbol = "+" if expected_sign > 0 else "-"
        ok = (got_sign == expected_sign)
        if ok:
            passes += 1
        status = "PASS" if ok else "**FAIL**"
        print(f"{name:<36} {dx_mm:+9.2f} {dy_mm:+9.2f} {dz_mm:+9.2f}  "
              f"{axis}{symbol}  {status}")

    ok_all = passes == len(cases)
    print("-" * 92)
    print(f"Software check: {passes}/{len(cases)} passed  "
          f"({'OK' if ok_all else 'CHECK SIGNS'})\n")
    return 0 if ok_all else 1


# ══════════════════════════════════════════════════════════════════════
#  2. Hardware-robot check (arm only)
# ══════════════════════════════════════════════════════════════════════

def _connect_arm(ip: str, speed: int, mvacc: int):
    """Return an xArm API handle in Cartesian mode, or raise."""
    from xarm.wrapper import XArmAPI  # local import, keeps --software deps minimal
    arm = XArmAPI(ip, baud_checkset=False)
    time.sleep(0.5)
    if not arm.connected:
        raise RuntimeError(f"xArm not connected at {ip}")
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.5)
    print(f"Arm connected at {ip}  (speed={speed} mvacc={mvacc})")
    return arm


def _get_pos(arm):
    ret = arm.get_position()
    if not (isinstance(ret, (list, tuple)) and len(ret) >= 2 and ret[0] == 0):
        raise RuntimeError(f"get_position failed: {ret}")
    return list(ret[1])  # [x, y, z, roll, pitch, yaw]


def _move_abs(arm, pos, speed, mvacc, wait=True):
    code = arm.set_position(
        x=pos[0], y=pos[1], z=pos[2],
        roll=pos[3], pitch=pos[4], yaw=pos[5],
        speed=speed, mvacc=mvacc, wait=wait)
    if code != 0:
        raise RuntimeError(f"set_position failed: code={code}")


def _axis_probe(arm, home, axis_idx, axis_name, delta_mm,
                speed, mvacc, settle_s, get_frame=None):
    """
    Move the arm by ``delta_mm`` on axis ``axis_idx`` from ``home``,
    grab before/after frames if ``get_frame`` is provided, and return
    to home. Returns a dict with the measured deltas.
    """
    before_pose = _get_pos(arm)
    before_frame = get_frame() if get_frame is not None else None

    target = list(home)
    target[axis_idx] = home[axis_idx] + delta_mm
    print(f"  [{axis_name}{delta_mm:+.0f}] moving...")
    _move_abs(arm, target, speed=speed, mvacc=mvacc, wait=True)
    time.sleep(settle_s)

    after_pose = _get_pos(arm)
    after_frame = get_frame() if get_frame is not None else None

    # Return to home before computing so even an exception later leaves
    # the arm safe.
    _move_abs(arm, home, speed=speed, mvacc=mvacc, wait=True)
    time.sleep(settle_s)

    fk_delta = [after_pose[i] - before_pose[i] for i in range(3)]

    flow = None
    if before_frame is not None and after_frame is not None:
        flow = _measure_flow_local(before_frame, after_frame)

    return {
        "axis": axis_name,
        "commanded_mm": delta_mm,
        "fk_delta_mm": fk_delta,
        "flow": flow,
    }


def _measure_flow_local(frame_a: np.ndarray, frame_b: np.ndarray):
    """Median Lucas-Kanade flow (dpx, dpy) across good feature points."""
    import cv2
    ga = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(ga, maxCorners=300, qualityLevel=0.01,
                                  minDistance=8, blockSize=7)
    if pts is None or len(pts) < 10:
        return None
    pts_next, status, _ = cv2.calcOpticalFlowPyrLK(ga, gb, pts, None)
    ok = status.ravel() == 1
    if ok.sum() < 10:
        return None
    flow = pts_next[ok] - pts[ok]
    return float(np.median(flow[:, 0, 0])), float(np.median(flow[:, 0, 1]))


# Expected optical flow direction per robot axis step under the
# 'zed_forward' preset:
#   robot +Y  ->  camera -X  ->  world-static features appear +X in image
#                                -> median dpx > 0
#   robot +Z  ->  camera -Y  ->  features appear +Y (downward) -> dpy > 0
#   robot +X  ->  camera +Z  ->  features expand outward;
#                                median (dpx, dpy) near 0 (unreliable check)
EXPECTED_FLOW_ZED_FORWARD = {
    ("Y", +1): ("dpx", +1),
    ("Y", -1): ("dpx", -1),
    ("Z", +1): ("dpy", +1),
    ("Z", -1): ("dpy", -1),
}


def hardware_robot_check(ip: str, step_mm: float, speed: int, mvacc: int,
                         settle_s: float, confirm: bool,
                         probes=(("Y", 1), ("Z", 2), ("X", 0))) -> int:
    if not confirm:
        print("Refusing to move the arm without --confirm.")
        return 2

    arm = _connect_arm(ip, speed, mvacc)
    home = _get_pos(arm)
    print(f"Home: [{home[0]:.1f}, {home[1]:.1f}, {home[2]:.1f}] mm "
          f"(roll/pitch/yaw preserved)\n")

    results = []
    try:
        for name, idx in probes:
            for sign in (+1, -1):
                res = _axis_probe(
                    arm, home, axis_idx=idx, axis_name=name,
                    delta_mm=sign * step_mm,
                    speed=speed, mvacc=mvacc, settle_s=settle_s,
                    get_frame=None)
                results.append(res)
    finally:
        # Extra safety: ensure home pose regardless of what happened.
        try:
            _move_abs(arm, home, speed=speed, mvacc=mvacc, wait=True)
        except Exception as e:
            print(f"warning: failed to restore home pose: {e}")

    print("\n=== Hardware-robot results ===")
    print(f"{'axis':<6} {'cmd_mm':>8} {'fk_dx':>8} {'fk_dy':>8} {'fk_dz':>8}")
    print("-" * 42)
    ok = True
    for r in results:
        print(f"{r['axis']:<6} {r['commanded_mm']:+8.1f} "
              f"{r['fk_delta_mm'][0]:+8.2f} "
              f"{r['fk_delta_mm'][1]:+8.2f} "
              f"{r['fk_delta_mm'][2]:+8.2f}")
        idx = {"X": 0, "Y": 1, "Z": 2}[r["axis"]]
        fk = r["fk_delta_mm"][idx]
        if _sign(fk) != _sign(r["commanded_mm"]):
            ok = False
            print(f"  ** axis {r['axis']} FK delta has wrong sign! "
                  f"(expected same sign as {r['commanded_mm']:+.0f})")
    print(f"\nHardware-robot: {'OK' if ok else 'CHECK xArm coordinate frame'}\n")
    return 0 if ok else 1


# ══════════════════════════════════════════════════════════════════════
#  3. Hardware-visual check (arm + ZED)
# ══════════════════════════════════════════════════════════════════════

def hardware_visual_check(ip: str, step_mm: float, speed: int, mvacc: int,
                          settle_s: float, confirm: bool, mount: str) -> int:
    if not confirm:
        print("Refusing to move the arm without --confirm.")
        return 2
    if mount != "zed_forward":
        # The expected-flow table is written for zed_forward. The code
        # below still reports observed flow for other mounts, it just
        # cannot pass/fail them automatically.
        print(f"(note: only mount=zed_forward has an automatic "
              f"flow-direction table; mount={mount} will only print "
              f"observed flow.)")

    try:
        import pyzed.sl as sl
    except ImportError:
        print("pyzed not available — use --hardware-robot for an arm-only check.")
        return 3
    import cv2  # noqa: F401  (used by _measure_flow_local)

    cam = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 30
    init.depth_mode = sl.DEPTH_MODE.NONE
    err = cam.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"ZED open failed: {err}")
        return 3
    mat = sl.Mat()
    runtime = sl.RuntimeParameters()

    def _grab():
        # Try a few times in case the first grab returns stale data.
        for _ in range(5):
            if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                cam.retrieve_image(mat, sl.VIEW.LEFT)
                frame = mat.get_data()[:, :, :3].copy()
                return frame
            time.sleep(0.03)
        return None

    # Warm up the camera (first few frames are often dark / auto-exposing).
    for _ in range(10):
        _grab()

    arm = _connect_arm(ip, speed, mvacc)
    home = _get_pos(arm)
    print(f"Home: [{home[0]:.1f}, {home[1]:.1f}, {home[2]:.1f}] mm\n")

    # Y and Z are the informative axes; X is forward motion (features
    # expand outward from the FOE, median flow is uninformative).
    probes = (("Y", 1), ("Z", 2), ("X", 0))
    results = []
    try:
        for name, idx in probes:
            for sign in (+1, -1):
                res = _axis_probe(
                    arm, home, axis_idx=idx, axis_name=name,
                    delta_mm=sign * step_mm,
                    speed=speed, mvacc=mvacc, settle_s=settle_s,
                    get_frame=_grab)
                results.append(res)
    finally:
        try:
            _move_abs(arm, home, speed=speed, mvacc=mvacc, wait=True)
        except Exception as e:
            print(f"warning: failed to restore home pose: {e}")
        cam.close()

    print("\n=== Hardware-visual results ===")
    print(f"{'axis':<6} {'cmd_mm':>8} {'fk_delta_mm (x,y,z)':>26}  "
          f"{'flow (dpx, dpy)':>20}  check")
    print("-" * 86)
    ok = True
    for r in results:
        fk = r["fk_delta_mm"]
        fk_str = f"({fk[0]:+6.2f},{fk[1]:+6.2f},{fk[2]:+6.2f})"
        if r["flow"] is None:
            flow_str = "n/a"
            check_str = "no-flow"
        else:
            dpx, dpy = r["flow"]
            flow_str = f"({dpx:+6.1f},{dpy:+6.1f})"
            key = (r["axis"], +1 if r["commanded_mm"] > 0 else -1)
            if mount == "zed_forward" and key in EXPECTED_FLOW_ZED_FORWARD:
                comp, exp_sign = EXPECTED_FLOW_ZED_FORWARD[key]
                observed = dpx if comp == "dpx" else dpy
                if _sign(observed) == exp_sign:
                    check_str = f"{comp}{'+' if exp_sign > 0 else '-'} PASS"
                else:
                    check_str = f"{comp}{'+' if exp_sign > 0 else '-'} **FAIL**"
                    ok = False
            else:
                check_str = "forward axis (expand, not checked)"
        print(f"{r['axis']:<6} {r['commanded_mm']:+8.1f} {fk_str:>26}  "
              f"{flow_str:>20}  {check_str}")
    print(f"\nHardware-visual: {'OK' if ok else 'CHECK camera mount / preset'}\n")
    return 0 if ok else 1


# ══════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--software", action="store_true",
                    help="Run pure-software sign test (no hardware).")
    ap.add_argument("--hardware-robot", action="store_true",
                    help="Arm-only FK check. Requires --confirm.")
    ap.add_argument("--hardware-visual", action="store_true",
                    help="Arm + ZED optical-flow sign check. "
                         "Requires --confirm.")
    ap.add_argument("--confirm", action="store_true",
                    help="Opt-in required for any hardware motion.")

    ap.add_argument("--ip", default="192.168.1.241",
                    help="xArm IP (default: %(default)s).")
    ap.add_argument("--mount", default="zed_forward",
                    choices=list(CAM_ROT_PRESETS.keys()),
                    help="Camera-to-robot mount preset for the software "
                         "check and the expected-flow table.")
    ap.add_argument("--step-mm", type=float, default=10.0,
                    help="Per-axis step size for hardware probes (mm).")
    ap.add_argument("--speed", type=int, default=80,
                    help="xArm set_position speed (mm/s).")
    ap.add_argument("--mvacc", type=int, default=500,
                    help="xArm set_position acceleration (mm/s^2).")
    ap.add_argument("--settle-s", type=float, default=0.3,
                    help="Post-move pause before reading pose/frame.")
    ap.add_argument("--gain", type=float, default=0.5,
                    help="PBVS proportional gain.")
    ap.add_argument("--target-depth", type=float, default=0.30,
                    help="PBVS target depth Z* (m).")
    ap.add_argument("--max-vel", type=float, default=0.02,
                    help="PBVS max per-axis velocity (m/s).")

    args = ap.parse_args()

    if not any([args.software, args.hardware_robot, args.hardware_visual]):
        ap.error("pick at least one of --software / --hardware-robot / "
                 "--hardware-visual")

    rc = 0
    if args.software:
        rc |= software_check(
            mount=args.mount,
            gain=args.gain,
            target_depth=args.target_depth,
            max_vel=args.max_vel,
        )

    if args.hardware_robot:
        rc |= hardware_robot_check(
            ip=args.ip, step_mm=args.step_mm,
            speed=args.speed, mvacc=args.mvacc,
            settle_s=args.settle_s, confirm=args.confirm,
        )

    if args.hardware_visual:
        rc |= hardware_visual_check(
            ip=args.ip, step_mm=args.step_mm,
            speed=args.speed, mvacc=args.mvacc,
            settle_s=args.settle_s, confirm=args.confirm,
            mount=args.mount,
        )

    sys.exit(rc)


if __name__ == "__main__":
    main()
