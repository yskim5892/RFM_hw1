#!/usr/bin/env python3
"""
dance_master.py

- Loads motions from dance_routines.yaml (key: motions).
- If dance_routines.yaml contains routines (List[str]), publishes motions in that order.
  Otherwise, publishes random motions.
- After each publish, waits for a real "MOVING -> IDLE" cycle before sending the next.
"""

import os
import random
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import yaml


class DanceMaster(Node):
    def __init__(self):
        super().__init__("dance_master")

        self.motion_topic = self.declare_parameter("motion_topic", "/hw/motion").value
        self.status_topic = self.declare_parameter("status_topic", "/hw/status").value
        self.num_moves = int(self.declare_parameter("num_moves", 5).value)
        self.routines_file = self.declare_parameter("routines_file", "dance_routines.yaml").value

        # resolve routines_file (relative -> script dir)
        if not os.path.isabs(self.routines_file):
            self.routines_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.routines_file)

        self.motions, self.routines = self._load_yaml(self.routines_file)

        if not self.motions:
            raise RuntimeError(f"No motions loaded from {self.routines_file} (missing or empty key: motions)")

        # If routines exist, follow them in order and override num_moves
        if self.routines:
            missing = [m for m in self.routines if m not in self.motions]
            if missing:
                raise RuntimeError(f"routines contain unknown motion(s): {missing}. Available: {self.motions}")
            self.num_moves = len(self.routines)

        self.pub_motion = self.create_publisher(String, self.motion_topic, 10)
        self.sub_status = self.create_subscription(String, self.status_topic, self._on_status, 10)

        self._status: Optional[str] = None
        self._sent = 0
        self._need_idle = False
        self._seen_moving = False

        self.timer = self.create_timer(0.05, self._tick)

        self.get_logger().info(f"Loaded motions (N={len(self.motions)}): {self.motions}")
        if self.routines:
            self.get_logger().info(f"Loaded routines (N={len(self.routines)}): {self.routines}")
        else:
            self.get_logger().info("No routines found; will publish random motions.")
        self.get_logger().info(f"Publishing motion -> {self.motion_topic}")
        self.get_logger().info(f"Subscribing status -> {self.status_topic}")

    @staticmethod
    def _load_yaml(path: str) -> Tuple[List[str], List[str]]:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        motions = data.get("motions") or []
        routines = data.get("routines") or []
        # normalize
        motions = [str(x).strip() for x in motions if str(x).strip()]
        routines = [str(x).strip() for x in routines if str(x).strip()]
        return motions, routines

    def _on_status(self, msg: String):
        s = (msg.data or "").strip().upper()
        if not s:
            return
        self._status = s
        if s == "MOVING":
            self._seen_moving = True

    def _next_motion_name(self) -> str:
        if self.routines:
            return self.routines[self._sent]  # _sent is 0-based before increment
        return random.choice(self.motions)

    def _send_go(self):
        motion = self._next_motion_name()
        m = String()
        m.data = motion
        self.pub_motion.publish(m)

        self._sent += 1
        self._need_idle = True
        self._seen_moving = False
        self.get_logger().info(f"[{self._sent}/{self.num_moves}] -> {m.data}")

    def _tick(self):
        # send the first command immediately
        if self._sent == 0:
            self._send_go()
            return

        # finished
        if self._sent >= self.num_moves:
            if self._status == "IDLE":
                self.get_logger().info("DanceMaster done. Bye!")
                rclpy.shutdown()
            return

        if not self._need_idle:
            return

        # need a real "MOVING -> IDLE" cycle
        if self._status != "IDLE":
            return
        if not self._seen_moving:
            return

        self._need_idle = False
        self._send_go()


def main():
    rclpy.init()
    node = DanceMaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
