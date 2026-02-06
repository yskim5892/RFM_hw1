#!/usr/bin/env python3
"""
dancer.py (TEMPLATE)

- Subscribes to a String topic (default: /hw/motion).
- When a message arrives (e.g. "rookie"), executes a predefined joint-waypoint routine using RTDE moveJ.

Parameters:
  robot_ip     (string) : UR IP
  speed        (float)  : moveJ speed
  accel        (float)  : moveJ acceleration
  dwell_sec    (float)  : sleep between waypoints
  num_waypoints(int)    : how many waypoints to run (-1 = all)

NOTE: Do NOT run this together with other RTDE control nodes that command the robot.
"""

import os
import json

import threading
import time
from typing import Dict, List
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

class Dancer(Node):
    def __init__(self):
        super().__init__("dancer")

        self.robot_ip = self.declare_parameter("robot_ip", "192.168.0.43").value
        self.speed = float(self.declare_parameter("speed", 0.2).value)
        self.accel = float(self.declare_parameter("accel", 0.2).value)

        # 각 waypoint 사이에 몇 초동안 쉴지 정의하는 parameter입니다. 
        # 각 waypoint가 끝날 때마다 time.sleep(self.dwell_sec) 를 쓰세요. 
        self.dwell_sec = float(self.declare_parameter("dwell_sec", 0.2).value)

        self._status = "UNKNOWN"
        self._status_cv = threading.Condition()

        # ur5_rtde_bridge에게 Joint 기반 이동 명령을 내릴 때 사용할 topic입니다. 
        # 다른 종류의 명령도 사용하고 싶다면 같은 방식으로 publisher를 추가하고 사용하면 됩니다.
        self.pub_joint_rel =  self.create_publisher(JointState, "/hw/goal_joint_r", 10)
        self.pub_tcp_rel = self.create_publisher(PoseStamped, "/hw/goal_tcp_pose_r", 10)
        self.pub_cmd = self.create_publisher(String, "/hw/cmd", 10)

        # ur5_rtde_bridge가 내보내는 status(IDLE/MOVING) topic을 받을 때마다 _on_status 함수가 불립니다.
        self.sub_status = self.create_subscription(String, "/hw/status", self._on_status, 10)

        self.num_waypoints = int(self.declare_parameter("num_waypoints", -1).value)

        self._busy = False
        self._lock = threading.Lock()

        # json 파일에 저장된 pose들을 불러옵니다.
        self.pose_db_path = Path(__file__).resolve().parent / "ur5_saved_poses.json"
        self.pose_db = {}
        self._load_pose_db()

        self.sub = self.create_subscription(String, "/hw/motion", self._on_cmd_motion, 10)

        self.get_logger().info(f"Dancer ready. Subscribing /hw/motion. RTDE -> {self.robot_ip}")

    def _on_status(self, msg: String):
        # main thread에서는 ur5_rtde_bridge가 보내는 status를 받아서
        # worker thread에게 알립니다.
        s = (msg.data or "").strip().upper()
        with self._status_cv:
            self._status = s
            self._status_cv.notify_all()

    def _wait_status(self, target: str, timeout_sec: float) -> bool:
        # worker thread에서는 status가 target으로 바뀔때까지 기다리다가,
        # main thread에서 _on_status를 통해 status가 업데이트되면 True를 리턴합니다.
        # timeout_sec 초 동안 status가 target으로 바뀌지 않으면 False를 리턴합니다.
        end = time.time() + timeout_sec
        with self._status_cv:
            while time.time() < end:
                if self._status == target:
                    return True
                self._status_cv.wait(timeout=max(0.0, end - time.time()))
        return False

    def _on_cmd_motion(self, msg: String):
        name = (msg.data or "").strip().lower() # motion 이름
        if not name:
            return

        self.get_logger().info(f"Received motion : {name}")
        with self._lock:
            if self._busy:
                self.get_logger().warn("Busy. Ignore new motion command.")
                return
            self._busy = True

        # TODO: motion마다 동작을 정의하면 됩니다. 
        # 예시 :
        def worker():
            try:
                # 명령을 내리기 전에 로봇의 status가 IDLE로 바뀔 때까지 대기
                if not self._wait_status("IDLE", timeout_sec=5.0):
                    self.get_logger().warn("Robot is not IDLE -> skip this step (or retry later)")
                    return
                
                if name == "test1":
                    msg_j = JointState()
                    msg_j.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.3] # Wrist3 관절을 0.3(rad) 회전
                    self.pub_joint_rel.publish(msg_j)

                    # 로봇의 status가 MOVING으로 바뀔 때까지 대기(최대 1초)
                    self._wait_status("MOVING", timeout_sec=1.0)

                    # 다시 로봇의 status가 IDLE로 바뀔 때까지 대기(최대 20초)
                    if not self._wait_status("IDLE", timeout_sec=20.0):
                        self.get_logger().warn("Timeout waiting IDLE (motion may be stuck or motion is too long.)")

                    msg_j = JointState()
                    msg_j.position = [0.0, 0.0, 0.0, 0.0, 0.0, -0.3] # Wrist3 관절을 -0.3(rad) 회전       
                    self.pub_joint_rel.publish(msg_j)

                elif name == "test2":
                    # ur5_rtde_bridge.py
                    # 미리 저장해둔 pose1 -> pose2 -> pose3 -> pose2
                    msg = String()
                    msg.data = "go pose1"
                    self.pub_cmd.publish(msg)

                    self._wait_status("MOVING", timeout_sec=1.0)

                    if not self._wait_status("IDLE", timeout_sec=20.0):
                        self.get_logger().warn("Timeout waiting IDLE (motion may be stuck or motion is too long.)")

                    msg = String()
                    msg.data = "go pose2"
                    self.pub_cmd.publish(msg)
                    
                    self._wait_status("MOVING", timeout_sec=1.0)

                    if not self._wait_status("IDLE", timeout_sec=20.0):
                        self.get_logger().warn("Timeout waiting IDLE (motion may be stuck or motion is too long.)")
                    
                    msg = String()
                    msg.data = "go pose3"
                    self.pub_cmd.publish(msg)
                    
                    self._wait_status("MOVING", timeout_sec=1.0)

                    if not self._wait_status("IDLE", timeout_sec=20.0):
                        self.get_logger().warn("Timeout waiting IDLE (motion may be stuck or motion is too long.)")

                    msg = String()
                    msg.data = "go pose2"
                    self.pub_cmd.publish(msg)
 
            except Exception as e:
                self.get_logger().error(str(e))
            finally:
                with self._lock:
                    self._busy = False
        
        threading.Thread(target=worker, daemon=True).start()

    def _load_pose_db(self):
        try:
            if os.path.exists(self.pose_db_path):
                with open(self.pose_db_path, "r") as f:
                    data = json.load(f)
                for k, v in data.items():
                    if isinstance(v, list) and len(v) == 6:
                        self.pose_db[str(k).lower()] = [float(x) for x in v]
            self.get_logger().info(f"Loaded pose DB: {self.pose_db_path} (keys={list(self.pose_db.keys())})")
        except Exception as e:
            self.get_logger().warn(f"Failed to load pose DB: {e}")


def main():
    rclpy.init()
    node = Dancer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
