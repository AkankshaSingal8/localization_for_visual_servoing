#!/usr/bin/env python3
"""
Send the xArm back to a fixed home pose. Run this between trials so
every trial starts from the same configuration.

Edit HOME_POSE below to match your lab's preferred start pose.
"""
from xarm.wrapper import XArmAPI

ROBOT_IP = "192.168.1.241"

# x, y, z in mm; roll, pitch, yaw in degrees.
# Default: hand pointing forward (x+) at chest height, zero tilt.
HOME_POSE = dict(
    x=300, y=0, z=300,
    roll=180, pitch=0, yaw=0,
    speed=80, mvacc=500, wait=True,
)


def main():
    arm = XArmAPI(ROBOT_IP, baud_checkset=False)
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    print(f"Sending arm to home: {HOME_POSE}")
    ret = arm.set_position(**HOME_POSE)
    if ret == 0:
        print("Home pose reached.")
    else:
        print(f"set_position returned {ret} (see xArm docs for codes)")


if __name__ == "__main__":
    main()
