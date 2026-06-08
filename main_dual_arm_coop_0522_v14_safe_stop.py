#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_dual_arm_coop_0522_v14_safe_stop.py

A/B dual-arm cooperative inspection main controller.

Important change:
- Uses ONE ROS2 context/node in this main file.
- Does NOT call A_arm_fixed_v2.VisionROSBridge.start()
- Does NOT call B_arm.BInspectionROSBridge.start()
This avoids: RuntimeError: Context.init() must only be called once

Required files in the same folder:
- A_arm_fixed_v2.py
- B_arm.py
- high_level_ik.py
- robot_executor.py
- main_dual_arm_coop_0522_v14_safe_stop.py
"""

import importlib
import json
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from high_level_ik import HighLevelIK, HighLevelIKConfig
from robot_executor import RobotExecutor

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    ROS2_AVAILABLE = True
except Exception:
    rclpy = None
    Node = object
    String = None
    ROS2_AVAILABLE = False


A_MODULE_NAME = "A_arm_fixed_v2"
B_MODULE_NAME = "B_arm_v2_tolerance"

# Check with: ls /dev/ttyUSB*
A_PORT = "/dev/ttyUSB1"
B_PORT = "/dev/ttyUSB0"

BAUDRATE = 2000000

# True: A/B move at the same time. False: A moves first, then B moves.
PARALLEL_MOVE = True


A_POSES = {
    # A 초기/복귀 위치와 최종 이송 위치.
    # 실제값이 정해지면 여기만 수정.
    "A_HOME": {"q1": -66.0, "q2": 0.0, "q3": 70.0, "q4": 0.0, "q5": 0.0},
    "A_OK_MID": {"q1": 45.0, "q2": -13.0, "q3": 20.0, "q4": 0.0, "q5": 35.0},
    "A_NG_MID": {"q1": -60.0, "q2": -10.0, "q3": 20.0, "q4": 0.0, "q5": 35.0},
    "A_OK_PLACE": {"q1": 85.0, "q2": 43.0, "q3": 39.0, "q4": 0.0, "q5": 44.0},
    "A_NG_PLACE": {"q1": -85.0, "q2": 43.0, "q3": 39.0, "q4": 0.0, "q5": 44.0},

    "A_FRONT_VIEW": {"q1": 0.0, "q2": -30.0, "q3": 30.0, "q4": 0.0, "q5": 40.0},
    "A_LOWER_VIEW": {"q1": 0.0, "q2": 0.0, "q3": 8.0, "q4": 0.0, "q5": 5.0},
    "A_UPPER_VIEW": {"q1": 0.0, "q2": 10.0, "q3": 45.0, "q4": 0.0, "q5": 42.0},
    "A_LEFT_VIEW":  {"q1": -30.0, "q2": 19.0, "q3": 50.0, "q4": 0.0, "q5": 0.0},
    "A_RIGHT_VIEW": {"q1": 30.0, "q2": 19.0, "q3":50.0, "q4": 0.0, "q5": 0.0},
}


B_POSES = {
    # B 초기/복귀 위치.
    # 실제값이 정해지면 여기만 수정.
    "B_HOME": {"q1": 90.0, "q2": 0.0, "q3": 0.0, "q4": 0.0, "q5": 0.0, "q6": 0.0},

    "B_FRONT_VIEW": {"q1": 183.0, "q2": 0.0, "q3": 0.0,  "q4": 0.0, "q5": 98.0,  "q6": 0.0},
    "B_LOWER_VIEW": {"q1": 183.0,   "q2": 0.0, "q3": -100.0,  "q4": 0.0, "q5": -40.0,   "q6": 0.0},
    "B_UPPER_VIEW": {"q1": 183.0, "q2": 0.0, "q3": -5.0, "q4": 0.0, "q5": 117.0, "q6": 0.0},
    "B_LEFT_VIEW":  {"q1": 320.0,   "q2": 75.0, "q3": 85.0,  "q4": 210.0, "q5": 100.0,   "q6": -270.0},
    "B_RIGHT_VIEW": {"q1": 40.0,   "q2": -75.0, "q3": 75.0,  "q4": -24.0, "q5": -110.0,   "q6": 90.0},
}


COOP_SEQUENCE = [
    {"view": "front", "a_view": "A_FRONT_VIEW", "b_view": "B_FRONT_VIEW", "a_mid_before": None, "b_mid_before": None},
    {"view": "lower", "a_view": "A_LOWER_VIEW", "b_view": "B_LOWER_VIEW", "a_mid_before": None, "b_mid_before": None},
    {"view": "upper", "a_view": "A_UPPER_VIEW", "b_view": "B_UPPER_VIEW", "a_mid_before": None, "b_mid_before": None},
    {"view": "left",  "a_view": "A_LEFT_VIEW",  "b_view": "B_LEFT_VIEW",  "a_mid_before": None, "b_mid_before": None},
    {"view": "right", "a_view": "A_RIGHT_VIEW", "b_view": "B_RIGHT_VIEW", "a_mid_before": None, "b_mid_before": None},
]

A3_W_MM = 420.0
A3_H_MM = 297.0


@dataclass
class CameraRobotAffineCalibration:
    x_from_x: float = 0.015098
    x_from_y: float = -0.750252
    x_bias: float = 535.455012

    y_from_x: float = 1.017520
    y_from_y: float = 0.001678
    y_bias: float = -216.194994

    z_pick_mm: float = 20.0

    def _clamp01(self, v: float):
        return max(0.0, min(1.0, v))

    def camera_to_robot_xy(self, x_cam: float, y_cam: float):
        x_robot = self.x_from_x * x_cam + self.x_from_y * y_cam + self.x_bias
        y_robot = self.y_from_x * x_cam + self.y_from_y * y_cam + self.y_bias

        w_left_top = (
            self._clamp01((120.0 - x_cam) / 120.0)
            * self._clamp01((120.0 - y_cam) / 120.0)
        )

        w_left_bottom = (
            self._clamp01((120.0 - x_cam) / 120.0)
            * self._clamp01((y_cam - 180.0) / 90.0)
        )

        w_right_bottom = (
            self._clamp01((x_cam - 300.0) / 100.0)
            * self._clamp01((y_cam - 180.0) / 90.0)
        )

        w_right_top = (
            self._clamp01((x_cam - 300.0) / 100.0)
            * self._clamp01((120.0 - y_cam) / 120.0)
        )

        y_robot += w_left_top * (15.0)
        x_robot += w_left_top * (15.0)

        y_robot += w_left_bottom * (-20.0)
        x_robot += w_left_bottom * (10.0)

        y_robot += w_right_bottom * (13.0)
        x_robot += w_right_bottom * (-13.0)

        y_robot += w_right_top * (-19.0)
        x_robot += w_right_top * (-10.0)

        print(
            "[LOCAL CORR] "
            f"wLT={w_left_top:.2f}, "
            f"wLB={w_left_bottom:.2f}, "
            f"wRB={w_right_bottom:.2f}, "
            f"wRT={w_right_top:.2f} -> "
            f"robot=({x_robot:.1f}, {y_robot:.1f})"
        )

        return x_robot, y_robot

def is_inside_workspace(x_cam: float, y_cam: float) -> bool:
    return (0.0 <= x_cam <= A3_W_MM) and (0.0 <= y_cam <= A3_H_MM)


def build_a_planner(a_cfg) -> HighLevelIK:
    planner_cfg = HighLevelIKConfig(
        approach_offset_mm=100.0,
        min_approach_offset_mm=10.0,
        approach_offset_step_mm=5.0,
        base_x_mm=109.0,
        base_z_mm=171.0,
        link1_mm=175.0,
        link2_mm=233.23854913922,
        wrist_to_link5_mm=94.5,
        link5_to_tcp_mm=84.0,
        q2_zero_offset_deg=90.0,
        q3_zero_offset_deg=0.0,
        q2_sign=-1.0,
        q3_sign=-1.0,
        q4_fixed_deg=0.0,
        q5_a=0.41,
        q5_b=-0.20,
        q5_c=38.5,
        use_single_q5_for_approach_and_final=False,
        far_start_mm=430.0,
        far_full_mm=500.0,
        far_q5_offset_deg=8.0,
        far_tcp_z_offset_mm=0.0,
        far_tcp_x_offset_mm=0.0,
        far_tcp_y_offset_mm=0.0,
        debug=True,
        q_limits_deg=a_cfg.joint_limits_deg,
    )
    return HighLevelIK(cfg=planner_cfg)



@dataclass
class TopDetection:
    label: str
    x_mm: float
    y_mm: float
    judge: str
    stamp: float


@dataclass
class InspectionState:
    has_car_part: bool
    has_defect: bool
    defect_count: int
    judge: str
    stamp: float


class MainROSNode(Node):
    def __init__(self, state_holder, lock):
        super().__init__("dual_arm_main_listener")
        self.state_holder = state_holder
        self.lock = lock

        self.sub_detected = self.create_subscription(String, "/detected_object", self.detected_callback, 10)
        self.sub_top_inspection = self.create_subscription(String, "/inspection_state", self.top_inspection_callback, 10)
        self.sub_b_inspection = self.create_subscription(String, "/b_inspection_state", self.b_inspection_callback, 10)

        self.get_logger().info("Subscribed to /detected_object")
        self.get_logger().info("Subscribed to /inspection_state")
        self.get_logger().info("Subscribed to /b_inspection_state")

    def detected_callback(self, msg):
        try:
            raw = msg.data.strip()

            # detector_node variants may publish either JSON or comma-separated text.
            # JSON example: {"label":"Car_part","x_mm":123.4,"y_mm":56.7,"judge":"OK"}
            # CSV example : Car_part,123.4,56.7,0.0,12345,OK
            if raw.startswith("{"):
                data = json.loads(raw)
                judge = str(data.get("judge", data.get("result", "NO_OBJECT"))).upper()
                det = TopDetection(
                    label=str(data.get("label", data.get("class", ""))),
                    x_mm=float(data.get("x_mm", data.get("x", 0.0))),
                    y_mm=float(data.get("y_mm", data.get("y", 0.0))),
                    judge=judge,
                    stamp=time.time(),
                )
            else:
                parts = [p.strip() for p in raw.split(",")]
                if len(parts) < 3:
                    raise ValueError(f"not enough fields: {raw}")

                label = parts[0]
                x_mm = float(parts[1])
                y_mm = float(parts[2])
                judge = parts[5].upper() if len(parts) >= 6 else "NO_OBJECT"

                det = TopDetection(
                    label=label,
                    x_mm=x_mm,
                    y_mm=y_mm,
                    judge=judge,
                    stamp=time.time(),
                )

            with self.lock:
                self.state_holder["detected"] = det
        except Exception as e:
            self.get_logger().warning(f"/detected_object parse failed: {e} / raw={msg.data}")

    def top_inspection_callback(self, msg):
        try:
            data = json.loads(msg.data)
            state = InspectionState(
                has_car_part=bool(data.get("has_car_part", False)),
                has_defect=bool(data.get("has_defect", False)),
                defect_count=int(data.get("defect_count", 0)),
                judge=str(data.get("judge", "NO_OBJECT")).upper(),
                stamp=time.time(),
            )
            with self.lock:
                self.state_holder["top_inspection"] = state
        except Exception as e:
            self.get_logger().warning(f"/inspection_state parse failed: {e} / raw={msg.data}")

    def b_inspection_callback(self, msg):
        try:
            data = json.loads(msg.data)
            state = InspectionState(
                has_car_part=bool(data.get("has_car_part", False)),
                has_defect=bool(data.get("has_defect", False)),
                defect_count=int(data.get("defect_count", 0)),
                judge=str(data.get("judge", "NO_OBJECT")).upper(),
                stamp=time.time(),
            )
            with self.lock:
                self.state_holder["b_inspection"] = state
        except Exception as e:
            self.get_logger().warning(f"/b_inspection_state parse failed: {e} / raw={msg.data}")


class MainROSBridge:
    def __init__(self):
        self._thread = None
        self._node = None
        self._started = False
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._state_holder = {
            "detected": None,
            "top_inspection": None,
            "b_inspection": None,
        }

    def start(self):
        if not ROS2_AVAILABLE:
            print("[WARN] ROS2 Python 환경을 찾지 못했습니다.")
            return False

        if self._started:
            print("[OK] Main ROS bridge already running")
            return True

        self._thread = threading.Thread(target=self._spin_thread, daemon=True)
        self._thread.start()
        self._started = True
        time.sleep(0.5)
        return True

    def _spin_thread(self):
        if not rclpy.ok():
            rclpy.init(args=None)

        node = MainROSNode(self._state_holder, self._lock)
        self._node = node

        try:
            while rclpy.ok() and not self._stop_evt.is_set():
                rclpy.spin_once(node, timeout_sec=0.2)
        finally:
            node.destroy_node()
            self._node = None

    def stop(self):
        if not self._started:
            return
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._started = False
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

    def get_detected(self):
        with self._lock:
            return self._state_holder.get("detected")

    def get_top_inspection(self):
        with self._lock:
            return self._state_holder.get("top_inspection")

    def get_b_inspection(self):
        with self._lock:
            return self._state_holder.get("b_inspection")


class MainBInspectionBridgeAdapter:
    def __init__(self, main_ros_bridge):
        self.main_ros_bridge = main_ros_bridge
        self._started = True

    def start(self):
        return True

    def stop(self):
        return

    def get_inspection(self):
        return self.main_ros_bridge.get_b_inspection()


class DualArmMain:
    def __init__(self):
        self.a_mod = None
        self.b_mod = None

        self.a_arm = None
        self.a_ctrl = None

        self.b_arm = None
        self.b_ctrl = None

        self.a_planner = None
        self.a_executor = None
        self.cam_calib = CameraRobotAffineCalibration()

        self.ros = MainROSBridge()
        self.initialized = False

        print("[AUTO ROS] Starting /detected_object, /inspection_state, /b_inspection_state subscriber")
        self.ros.start()

        self.last_top_judge: Optional[str] = None
        self.last_coop_results: Optional[Dict[str, str]] = None
        self.last_final_judgement: Optional[str] = None

    def import_modules(self):
        print(f"[IMPORT] A module = {A_MODULE_NAME}")
        self.a_mod = importlib.import_module(A_MODULE_NAME)
        if not hasattr(self.a_mod, "DxlConfig"):
            raise AttributeError(f"{A_MODULE_NAME}.py 안에 DxlConfig가 없습니다.")

        print(f"[IMPORT] B module = {B_MODULE_NAME}")
        self.b_mod = importlib.import_module(B_MODULE_NAME)
        if not hasattr(self.b_mod, "DxlConfig"):
            raise AttributeError(f"{B_MODULE_NAME}.py 안에 DxlConfig가 없습니다.")

    def build_a(self):
        cfg = self.a_mod.DxlConfig(device_name=A_PORT, baudrate=BAUDRATE)
        if "q3" in cfg.joint_limits_deg:
            cfg.joint_limits_deg["q3"] = (-130.0, 220.0)
        if "q5" in cfg.joint_limits_deg:
            cfg.joint_limits_deg["q5"] = (-60.0, 60.0)

        calib = self.a_mod.RobotCalibration()
        poses = self.a_mod.PoseLibrary()
        vision_cfg = self.a_mod.VisionConfig()
        suction_cfg = self.a_mod.SuctionConfig()
        ik_cfg = self.a_mod.IKConfig()

        # Dummy ROS bridge for A controller. We do not start it.
        dummy_a_ros = self.a_mod.VisionROSBridge(vision_cfg.ros_topic, "/inspection_state")

        arm = self.a_mod.DynamixelArm(cfg, calib)
        ctrl = self.a_mod.VisionStandController(
            arm, poses, cfg, vision_cfg, suction_cfg, ik_cfg, dummy_a_ros
        )
        return arm, ctrl

    def build_b(self):
        cfg = self.b_mod.DxlConfig(device_name=B_PORT, baudrate=BAUDRATE)
        cfg.joint_limits_deg["q1"] = (-360.0, 360.0)
        cfg.joint_limits_deg["q5"] = (-130.0, 130.0)

        calib = self.b_mod.RobotCalibration()
        poses = self.b_mod.PoseLibrary()

        # Adapter reads /b_inspection_state from the single main ROS bridge.
        b_ros_adapter = MainBInspectionBridgeAdapter(self.ros)

        arm = self.b_mod.DynamixelArmB(cfg, calib)
        ctrl = self.b_mod.BArmController(arm, poses, cfg, b_ros_adapter)
        return arm, ctrl

    def init_robots(self):
        if self.initialized:
            print("[INIT] already initialized")
            return

        print("\n[INIT] Import modules")
        self.import_modules()

        print("\n[INIT] Build A/B controllers")
        self.a_arm, self.a_ctrl = self.build_a()
        self.b_arm, self.b_ctrl = self.build_b()

        print("\n[INIT] A arm open + Extended Position Mode")
        self.a_arm.open()
        self.a_arm.set_extended_position_mode_all()
        self.a_arm.set_current_as_zero()

        print("\n[INIT] B arm open + Extended Position Mode")
        self.b_arm.open()
        self.b_arm.set_extended_position_mode_all()
        self.b_arm.set_current_as_zero()

        print("\n[INIT] Build A IK planner and executor")
        self.a_planner = build_a_planner(self.a_arm.cfg)
        self.a_executor = RobotExecutor(self.a_arm, self.a_ctrl)

        print("\n[INIT] Single ROS subscriber node is already running")
        self.ros.start()

        self.initialized = True
        print("\n[OK] Dual-arm main initialized")
        self.show_status_once()

    def close(self):
        try:
            self.ros.stop()
        except Exception:
            pass

        for obj in [self.a_arm, self.b_arm]:
            try:
                if obj:
                    obj.close()
            except Exception:
                pass

    def require_init(self):
        if not self.initialized:
            raise RuntimeError("먼저 메뉴 1번으로 A/B 로봇을 초기화하세요.")

    def read_top_view_judge(self):
        self.require_init()

        det = self.ros.get_detected()
        insp = self.ros.get_top_inspection()

        judge = None
        source = None

        if det is not None:
            candidate = str(getattr(det, "judge", "")).upper()
            if candidate in {"OK", "NG"}:
                judge = candidate
                source = "/detected_object"

        if judge is None and insp is not None:
            candidate = str(getattr(insp, "judge", "")).upper()
            if candidate in {"OK", "NG"}:
                judge = candidate
                source = "/inspection_state"

        if judge is None:
            print("[TOP] 아직 유효한 top_view OK/NG 결과가 없습니다.")
            print("      Top-view detector topic /detected_object 또는 /inspection_state를 확인하세요.")
            return None

        self.last_top_judge = judge
        print(f"[TOP] top_view judge = {judge} from {source}")
        return judge

    def show_latest_ros_states(self):
        det = self.ros.get_detected()
        top = self.ros.get_top_inspection()
        b = self.ros.get_b_inspection()

        print("\n[LATEST ROS STATES]")
        print(f"  /detected_object: {det}")
        print(f"  /inspection_state: {top}")
        print(f"  /b_inspection_state: {b}")

    def move_a(self, pose_name):
        pose = A_POSES[pose_name]
        print(f"[A MOVE] {pose_name}: {pose}")
        self.a_arm.move_joint_deg(pose)

    def move_b(self, pose_name):
        pose = B_POSES[pose_name]
        print(f"[B MOVE] {pose_name}: {pose}")
        self.b_arm.move_joint_deg(pose)

    def move_a_b_parallel(self, a_pose_name=None, b_pose_name=None):
        errors = []

        def run_a():
            try:
                if a_pose_name is not None:
                    self.move_a(a_pose_name)
            except Exception as e:
                errors.append(("A", e))

        def run_b():
            try:
                if b_pose_name is not None:
                    self.move_b(b_pose_name)
            except Exception as e:
                errors.append(("B", e))

        threads = []

        if a_pose_name is not None:
            threads.append(threading.Thread(target=run_a, daemon=True))
        if b_pose_name is not None:
            threads.append(threading.Thread(target=run_b, daemon=True))

        for t in threads:
            t.start()

        try:
            for t in threads:
                while t.is_alive():
                    t.join(timeout=0.1)
        except KeyboardInterrupt:
            print("\n[INTERRUPT] Ctrl+C detected during A/B parallel move")
            raise

        if errors:
            msg = "; ".join([f"{label}: {err}" for label, err in errors])
            raise RuntimeError(f"parallel move failed: {msg}")

    def move_mid_if_needed(self, step):
        a_mid = step.get("a_mid_before")
        b_mid = step.get("b_mid_before")
        if a_mid is None and b_mid is None:
            return

        print("\n[MID MOVE] 중간 자세 경유")

        if PARALLEL_MOVE:
            self.move_a_b_parallel(a_mid, b_mid)
        else:
            if a_mid is not None:
                self.move_a(a_mid)
                time.sleep(0.2)
            if b_mid is not None:
                self.move_b(b_mid)

        time.sleep(0.8)

    def run_one_view(self, step, hold_sec=3.0, required_defect_sec=2.0):
        self.require_init()
        view = step["view"]

        print(f"\n========== COOP VIEW: {view.upper()} ==========")
        self.move_mid_if_needed(step)

        if PARALLEL_MOVE:
            print("[PARALLEL MOVE] A/B target view move")
            self.move_a_b_parallel(step["a_view"], step["b_view"])
        else:
            self.move_a(step["a_view"])
            time.sleep(0.2)
            self.move_b(step["b_view"])

        time.sleep(0.8)

        result = self.b_ctrl.inspect_current_view(
            view,
            hold_sec=hold_sec,
            required_defect_sec=required_defect_sec,
        )

        print(f"[VIEW RESULT] {view}: {result['judge']}")
        return result

    def get_latest_top_detection_for_pick(self):
        self.require_init()

        det = self.ros.get_detected()
        if det is None:
            print("[PICK] /detected_object 좌표가 아직 없습니다.")
            print("       Top-view detector가 좌표를 publish하는지 확인하세요.")
            return None

        age = time.time() - det.stamp
        if age > 3.0:
            print(f"[PICK] 경고: /detected_object가 오래됐습니다. age={age:.2f}s")

        if not is_inside_workspace(det.x_mm, det.y_mm):
            print(f"[PICK] 작업공간 밖 좌표: x={det.x_mm:.2f}, y={det.y_mm:.2f}")
            return None

        print("\n[PICK TARGET FROM TOP-VIEW]")
        print(f"  label = {det.label}")
        print(f"  camera xy = ({det.x_mm:.2f}, {det.y_mm:.2f})")
        print(f"  judge = {det.judge}")
        print(f"  age = {age:.2f}s")
        return det

    def execute_ik_pick_from_latest_top_view(self, z_mm=None):
        self.require_init()

        det = self.get_latest_top_detection_for_pick()
        if det is None:
            return False

        if z_mm is None:
            z_mm = self.cam_calib.z_pick_mm

        # Save top-view judgement before robot motion.
        top_judge = None
        top_state = self.ros.get_top_inspection()
        if top_state is not None and str(top_state.judge).upper() in {"OK", "NG"}:
            top_judge = str(top_state.judge).upper()
        elif str(det.judge).upper() in {"OK", "NG"}:
            top_judge = str(det.judge).upper()

        if top_judge is not None:
            self.last_top_judge = top_judge
            print(f"[TOP SAVE] top_view judge saved before pick = {self.last_top_judge}")
        else:
            print("[TOP SAVE] top_view OK/NG 판정은 아직 없고, 좌표만 사용합니다.")

        x_robot, y_robot = self.cam_calib.camera_to_robot_xy(det.x_mm, det.y_mm)

        print("\n[CAMERA -> ROBOT]")
        print(f"  camera xy = ({det.x_mm:.2f}, {det.y_mm:.2f})")
        print(f"  robot  xy = ({x_robot:.2f}, {y_robot:.2f})")
        print(f"  pick z    = {z_mm:.2f}")

        print("\n[IK PLAN START]")
        plan = self.a_planner.plan(x_robot, y_robot, z_mm)
        print(f"[PLAN RESULT] approach = {plan['approach']}")
        print(f"[PLAN RESULT] final    = {plan['final']}")
        print(f"[PLAN RESULT] used approach offset = {plan.get('used_approach_offset_mm', 0.0):.1f} mm")

        print("\n[A PICK] IK approach -> descend -> suction ON -> retreat")
        self.a_executor.execute_pick_soft_vertical(
            planner=self.a_planner,
            plan=plan,
            use_suction=True,
            retreat=True,
            q5_fn=self.a_planner.compute_q5,
            z_step_mm=5.0,
            step_dt=0.05,
            max_allow_err_mm=20.0,
        )

        print("[A PICK] complete. Object should be held by A suction.")
        return True

    def run_pick_then_front(self):
        self.require_init()
        ok = self.execute_ik_pick_from_latest_top_view()
        if not ok:
            print("[ABORT] pick failed or no valid top-view target")
            return None

        print("\n[NEXT] A/B parallel FRONT inspection")
        return self.run_front_only()

    def run_pick_then_coop_sequence(self):
        self.require_init()
        ok = self.execute_ik_pick_from_latest_top_view()
        if not ok:
            print("[ABORT] pick failed or no valid top-view target")
            return None

        print("\n[NEXT] A/B cooperative 5-view inspection")
        return self.run_coop_sequence()

    def run_front_only(self):
        self.require_init()
        result = self.run_one_view(COOP_SEQUENCE[0], hold_sec=3.0, required_defect_sec=2.0)
        self.last_coop_results = {"front": result["judge"]}
        print(f"[FRONT ONLY] front = {result['judge']}")
        return result

    def run_one_view_by_name(self, view_name):
        self.require_init()
        view_name = view_name.strip().lower()

        for step in COOP_SEQUENCE:
            if step["view"] == view_name:
                result = self.run_one_view(step, hold_sec=3.0, required_defect_sec=2.0)
                if self.last_coop_results is None:
                    self.last_coop_results = {}
                self.last_coop_results[view_name] = result["judge"]
                return result

        print(f"[ERROR] invalid view name: {view_name}")
        print("valid: front, lower, upper, left, right")
        return None

    def run_coop_sequence(self):
        self.require_init()
        print("\n================================================")
        print(" A/B COOPERATIVE 5-VIEW REAL SEQUENCE START")
        print(" order: front -> lower -> upper -> left -> right")
        print(" mid poses: none during inspection")
        print("================================================")

        results = []
        for i, step in enumerate(COOP_SEQUENCE, start=1):
            print(f"\n[COOP STEP {i}/{len(COOP_SEQUENCE)}] {step['view'].upper()}")
            result = self.run_one_view(step, hold_sec=3.0, required_defect_sec=2.0)
            results.append(result)
            print(f"[COOP STEP DONE] {step['view']} = {result['judge']}")

        result_map = {r["view"]: r["judge"] for r in results}
        coop_judge = "NG" if "NG" in result_map.values() else "OK"
        self.last_coop_results = result_map

        print("\n[COOP SUMMARY]")
        for k, v in result_map.items():
            print(f"  {k}: {v}")
        print(f"  coop_judgement: {coop_judge}")
        print("================================================")
        print(" A/B COOPERATIVE 5-VIEW REAL SEQUENCE END")
        print("================================================")
        return coop_judge, result_map

    def manual_move_a(self):
        self.require_init()
        print("\n[A MANUAL JOINT MOVE]")
        print("Enter A joint values in deg.")
        q = {
            "q1": float(input("A q1(deg): ")),
            "q2": float(input("A q2(deg): ")),
            "q3": float(input("A q3(deg): ")),
            "q4": float(input("A q4(deg): ")),
            "q5": float(input("A q5(deg): ")),
        }
        self.a_arm.move_joint_deg(q)

    def manual_move_b(self):
        self.require_init()
        print("\n[B MANUAL JOINT MOVE]")
        print("Enter B joint values in deg.")
        q = {
            "q1": float(input("B q1(deg): ")),
            "q2": float(input("B q2(deg): ")),
            "q3": float(input("B q3(deg): ")),
            "q4": float(input("B q4(deg): ")),
            "q5": float(input("B q5(deg): ")),
            "q6": float(input("B q6(deg): ")),
        }
        self.b_arm.move_joint_deg(q)

    def move_a_zero(self):
        self.require_init()
        q = {"q1": 0.0, "q2": 0.0, "q3": 0.0, "q4": 0.0, "q5": 0.0}
        print(f"[A ZERO MOVE] {q}")
        self.a_arm.move_joint_deg(q)

    def move_b_zero(self):
        self.require_init()
        q = {"q1": 0.0, "q2": 0.0, "q3": 0.0, "q4": 0.0, "q5": 0.0, "q6": 0.0}
        print(f"[B ZERO MOVE] {q}")
        self.b_arm.move_joint_deg(q)

    def move_a_named_pose(self):
        self.require_init()
        print("\n[A NAMED POSES]")
        for name in A_POSES:
            print(f"  {name}: {A_POSES[name]}")
        pose_name = input("A pose name: ").strip().upper()
        if pose_name not in A_POSES:
            print("[ERROR] invalid A pose name")
            return
        self.move_a(pose_name)

    def move_b_named_pose(self):
        self.require_init()
        print("\n[B NAMED POSES]")
        for name in B_POSES:
            print(f"  {name}: {B_POSES[name]}")
        pose_name = input("B pose name: ").strip().upper()
        if pose_name not in B_POSES:
            print("[ERROR] invalid B pose name")
            return
        self.move_b(pose_name)

    def move_ab_zero_parallel(self):
        self.require_init()
        errors = []

        def run_a():
            try:
                self.move_a_zero()
            except Exception as e:
                errors.append(("A", e))

        def run_b():
            try:
                self.move_b_zero()
            except Exception as e:
                errors.append(("B", e))

        t1 = threading.Thread(target=run_a, daemon=True)
        t2 = threading.Thread(target=run_b, daemon=True)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        if errors:
            msg = "; ".join([f"{label}: {err}" for label, err in errors])
            raise RuntimeError(f"AB zero move failed: {msg}")

    def move_ab_named_pose_pair(self):
        self.require_init()
        print("\n[AB NAMED POSE PAIR]")
        print("Valid view names: front, lower, upper, left, right")
        view = input("view name: ").strip().lower()
        step = None
        for s in COOP_SEQUENCE:
            if s["view"] == view:
                step = s
                break
        if step is None:
            print("[ERROR] invalid view name")
            return

        if PARALLEL_MOVE:
            print(f"[AB PAIR MOVE] {view} in parallel")
            self.move_a_b_parallel(step["a_view"], step["b_view"])
        else:
            print(f"[AB PAIR MOVE] {view} sequential")
            self.move_a(step["a_view"])
            self.move_b(step["b_view"])

    def final_judgement_preview(self):
        if self.last_top_judge is None:
            print("[FINAL] top_view 결과가 없습니다. 메뉴 2번을 먼저 실행하세요.")
            return None
        if self.last_coop_results is None:
            print("[FINAL] coop 결과가 없습니다. 메뉴 3, 4, 또는 5를 먼저 실행하세요.")
            return None

        all_results = {"top_view": self.last_top_judge, **self.last_coop_results}
        final = "NG" if "NG" in all_results.values() else "OK"
        self.last_final_judgement = final

        print("\n[FINAL JUDGEMENT PREVIEW]")
        for k, v in all_results.items():
            print(f"  {k}: {v}")
        print(f"  final_judgement: {final}")
        return final

    def compute_final_judgement_from_saved_results(self, allow_missing_top=False):
        if self.last_coop_results is None:
            print("[FINAL] coop 검사 결과가 없습니다. 먼저 4번/12번/27번 중 하나로 검사를 실행하세요.")
            return None

        all_results = dict(self.last_coop_results)

        if self.last_top_judge is None:
            if allow_missing_top:
                print("[FINAL] top_view 결과가 없어 coop 결과만으로 임시 판정합니다.")
            else:
                print("[FINAL] top_view 결과가 없습니다. 2번으로 top_view 결과를 먼저 저장하세요.")
                return None
        else:
            all_results = {"top_view": self.last_top_judge, **all_results}

        final = "NG" if "NG" in all_results.values() else "OK"
        self.last_final_judgement = final

        print("\n[FINAL JUDGEMENT]")
        for k, v in all_results.items():
            print(f"  {k}: {v}")
        print(f"  final_judgement: {final}")

        return final

    def safe_stop_return_home(self, reason="user stop"):
        """
        안전 정지/복귀:
        Ctrl+C 또는 메뉴 명령으로 현재 자동 공정을 중단한 뒤,
        흡착을 해제하고 A_HOME/B_HOME으로 복귀한다.
        """
        print("\n================================================")
        print(f" SAFE STOP REQUESTED: {reason}")
        print("================================================")

        if not self.initialized:
            print("[SAFE STOP] robots are not initialized. Nothing to move.")
            return False

        print("[SAFE STOP] suction OFF")
        try:
            self.a_ctrl.suction_off()
            time.sleep(0.3)
        except Exception as e:
            print(f"[SAFE STOP WARN] suction_off failed: {e}")

        print("[SAFE STOP] move A_HOME + B_HOME")
        try:
            if PARALLEL_MOVE:
                self.move_a_b_parallel("A_HOME", "B_HOME")
            else:
                self.move_a("A_HOME")
                self.move_b("B_HOME")
            print("[SAFE STOP] home return complete")
            return True

        except KeyboardInterrupt:
            print("[SAFE STOP] interrupted again during home return")
            return False

        except Exception as e:
            print(f"[SAFE STOP WARN] parallel home return failed: {e}")
            print("[SAFE STOP] retry sequential home return")

            ok = True
            try:
                self.move_a("A_HOME")
            except Exception as ea:
                ok = False
                print(f"[SAFE STOP ERROR] A_HOME failed: {ea}")

            try:
                self.move_b("B_HOME")
            except Exception as eb:
                ok = False
                print(f"[SAFE STOP ERROR] B_HOME failed: {eb}")

            if ok:
                print("[SAFE STOP] sequential home return complete")
            return ok

    def move_final_place_and_release_auto(self):
        self.require_init()

        if self.last_final_judgement is None:
            final = self.compute_final_judgement_from_saved_results(allow_missing_top=False)
            if final is None:
                return None
        else:
            final = self.last_final_judgement

        if final == "OK":
            a_mid = "A_OK_MID"
            a_target = "A_OK_PLACE"
        else:
            a_mid = "A_NG_MID"
            a_target = "A_NG_PLACE"

        print("\n[FINAL PLACE]")
        print(f"  final_judgement = {final}")
        print(f"  A mid    = {a_mid}")
        print(f"  A target = {a_target}")
        print("  B target = B_HOME")

        print(f"[FINAL TRANSFER MID] {a_mid}")
        self.move_a(a_mid)

        if PARALLEL_MOVE:
            self.move_a_b_parallel(a_target, "B_HOME")
        else:
            self.move_a(a_target)
            self.move_b("B_HOME")

        print("[SUCTION] OFF at final place")
        self.a_ctrl.suction_off()
        time.sleep(0.5)

        print("[RETURN HOME] A_HOME + B_HOME")
        if PARALLEL_MOVE:
            self.move_a_b_parallel("A_HOME", "B_HOME")
        else:
            self.move_a("A_HOME")
            self.move_b("B_HOME")

        return final

    def wait_for_new_top_target(self, min_stamp=0.0, timeout_sec=0.0, print_period_sec=1.0):
        """
        새 물체가 top-view에 들어올 때까지 대기한다.
        - /detected_object가 있어야 함
        - stamp가 min_stamp보다 커야 함
        - A3 작업공간 안 좌표여야 함
        - top_view judge가 OK 또는 NG로 확보되어야 함
        timeout_sec=0이면 무한 대기
        """
        self.require_init()

        start = time.time()
        last_print = 0.0

        while True:
            now = time.time()

            if timeout_sec > 0.0 and (now - start) >= timeout_sec:
                print("[WAIT TARGET] timeout")
                return None

            det = self.ros.get_detected()
            top = self.ros.get_top_inspection()

            judge = None
            if top is not None and str(top.judge).upper() in {"OK", "NG"}:
                judge = str(top.judge).upper()
            elif det is not None and str(det.judge).upper() in {"OK", "NG"}:
                judge = str(det.judge).upper()

            valid = False
            reason = "no /detected_object"

            if det is not None:
                if det.stamp <= min_stamp:
                    reason = f"old target stamp={det.stamp:.2f} <= min_stamp={min_stamp:.2f}"
                elif not is_inside_workspace(det.x_mm, det.y_mm):
                    reason = f"outside workspace x={det.x_mm:.1f}, y={det.y_mm:.1f}"
                elif judge not in {"OK", "NG"}:
                    reason = f"no valid top judge yet: {judge}"
                else:
                    valid = True

            if valid:
                self.last_top_judge = judge
                print("\n[WAIT TARGET] new valid target found")
                print(f"  label = {det.label}")
                print(f"  camera xy = ({det.x_mm:.2f}, {det.y_mm:.2f})")
                print(f"  top_view judge = {judge}")
                print(f"  target stamp = {det.stamp:.2f}")
                return det

            if now - last_print >= print_period_sec:
                print(f"[WAIT TARGET] waiting... {reason}")
                last_print = now

            time.sleep(0.1)

    def run_continuous_auto_loop(self):
        """
        연속 자동 공정:
        새 top-view target 감지
        -> 27번 1회 자동 공정 실행
        -> A_HOME/B_HOME 복귀 완료
        -> 다음 새 target 대기
        """
        self.require_init()

        max_raw = input("max cycles (blank = infinite): ").strip()
        max_cycles = None if max_raw == "" else int(max_raw)

        delay_raw = input("delay after each cycle sec (default 1.0): ").strip()
        delay_sec = 1.0 if delay_raw == "" else float(delay_raw)

        timeout_raw = input("target wait timeout sec (0 = no timeout, default 0): ").strip()
        timeout_sec = 0.0 if timeout_raw == "" else float(timeout_raw)

        print("\n================================================")
        print(" CONTINUOUS AUTO LOOP START")
        print(" Press Ctrl+C to stop safely.")
        print("================================================")

        cycle = 0
        min_target_stamp = 0.0

        try:
            while True:
                if max_cycles is not None and cycle >= max_cycles:
                    print(f"[LOOP] max_cycles reached: {max_cycles}")
                    break

                print(f"\n[LOOP] Waiting for next target... cycle={cycle + 1}")
                det = self.wait_for_new_top_target(
                    min_stamp=min_target_stamp,
                    timeout_sec=timeout_sec,
                    print_period_sec=1.0,
                )

                if det is None:
                    print("[LOOP] no target. continuous loop stopped.")
                    break

                final = self.run_full_one_cycle_auto()

                cycle += 1
                min_target_stamp = time.time()

                print(f"\n[LOOP] cycle {cycle} done. final={final}")
                print(f"[LOOP] wait {delay_sec:.1f}s before next target")
                time.sleep(delay_sec)

        except KeyboardInterrupt:
            print("\n[LOOP] stopped by user Ctrl+C")
            self.safe_stop_return_home("Ctrl+C during continuous auto loop")

        print("================================================")
        print(f" CONTINUOUS AUTO LOOP END. completed cycles={cycle}")
        print("================================================")

    def run_full_one_cycle_auto(self):
        """
        1회 자동 공정:
        A_HOME/B_HOME 대기
        -> top_view 결과 저장
        -> top_view 좌표 기반 A IK pick + suction ON
        -> A/B 5면 협동검사
        -> top_view + 5면 결과 최종판정
        -> A OK/NG 위치 이송 + suction OFF
        -> A_HOME/B_HOME 복귀
        """
        self.require_init()

        # Reset only per-cycle result buffers. Robot calibration/zero data is not changed.
        self.last_top_judge = None
        self.last_coop_results = None
        self.last_final_judgement = None

        print("\n================================================")
        print(" FULL ONE-CYCLE AUTO START")
        print("================================================")

        print("\n[STEP 0] Move to initial waiting poses: A_HOME + B_HOME")
        if PARALLEL_MOVE:
            self.move_a_b_parallel("A_HOME", "B_HOME")
        else:
            self.move_a("A_HOME")
            self.move_b("B_HOME")

        print("\n[STEP 1] Save current top_view judge")
        top = self.read_top_view_judge()
        if top is None:
            print("[ABORT] top_view OK/NG 결과가 없습니다.")
            return None

        print("\n[STEP 2] A IK pick from latest top-view target")
        ok = self.execute_ik_pick_from_latest_top_view()
        if not ok:
            print("[ABORT] IK pick 실패 또는 유효한 top-view 좌표 없음")
            return None

        print("\n[STEP 3] A/B cooperative 5-view inspection")
        self.run_coop_sequence()

        print("\n[STEP 4] Compute final judgement")
        final = self.compute_final_judgement_from_saved_results(allow_missing_top=False)
        if final is None:
            print("[ABORT] 최종판정 실패")
            return None

        print("\n[STEP 5] Move to OK/NG place, suction OFF, return home")
        moved_final = self.move_final_place_and_release_auto()

        print("\n================================================")
        print(f" FULL ONE-CYCLE AUTO END: final={final}, moved_final={moved_final}")
        print("================================================")
        return final

    def recover_compute_and_transfer_allow_missing_top(self):
        """
        복구용:
        right 검사까지 끝났거나 일부 coop 결과가 저장된 상태에서,
        top_view가 없어도 저장된 coop 결과만으로 판정하고 이송한다.
        """
        self.require_init()
        final = self.compute_final_judgement_from_saved_results(allow_missing_top=True)
        if final is None:
            return None
        return self.move_final_place_and_release_auto()

    def _read_one_motor_status(self, label, arm, mod, dxl_id):
        try:
            torque = arm.read1(dxl_id, mod.ADDR_TORQUE_ENABLE, f"{label}_torque_{dxl_id}")
            hwerr = arm.read1(dxl_id, mod.ADDR_HARDWARE_ERROR_STATUS, f"{label}_hwerr_{dxl_id}")
            volt_raw = arm.read2(dxl_id, mod.ADDR_PRESENT_INPUT_VOLTAGE, f"{label}_volt_{dxl_id}")
            temp = arm.read1(dxl_id, mod.ADDR_PRESENT_TEMPERATURE, f"{label}_temp_{dxl_id}")
            vel = arm.read4(dxl_id, mod.ADDR_PRESENT_VELOCITY, f"{label}_vel_{dxl_id}")

            current_text = "current=N/A"
            if hasattr(mod, "ADDR_PRESENT_CURRENT") and hasattr(arm, "read_present_current_raw"):
                current_raw = arm.read_present_current_raw(dxl_id)
                current_ma = current_raw * 2.69
                current_text = f"current={current_ma:.1f}mA(raw={current_raw})"

            print(
                f"  {label} ID {dxl_id}: torque={torque}, hwerr={hwerr}, "
                f"voltage={volt_raw/10.0:.1f}V, {current_text}, temp={temp}C, vel={vel}"
            )
        except Exception as e:
            print(f"  {label} ID {dxl_id}: status read failed: {e}")

    def show_status_once(self):
        self.require_init()
        print("\n[A ROBOT STATUS]")
        for dxl_id in self.a_arm.cfg.ids:
            self._read_one_motor_status("A", self.a_arm, self.a_mod, dxl_id)

        print("\n[B ROBOT STATUS]")
        for dxl_id in self.b_arm.cfg.ids:
            self._read_one_motor_status("B", self.b_arm, self.b_mod, dxl_id)

    def monitor_status(self, duration_sec=10.0, interval_sec=1.0):
        self.require_init()
        print(f"\n[DUAL STATUS MONITOR] duration={duration_sec:.1f}s, interval={interval_sec:.2f}s")
        start = time.time()
        while True:
            elapsed = time.time() - start
            if elapsed > duration_sec:
                break
            print(f"\n========== t = {elapsed:.2f}s ==========")
            self.show_status_once()
            time.sleep(interval_sec)


def print_menu():
    print("\n======================================")
    print(" DUAL ARM MAIN MENU")
    print("======================================")
    print(f"1. Initialize A/B robots + single ROS subscriber  | PARALLEL_MOVE={PARALLEL_MOVE}")
    print("2. Read and save current top_view judge")
    print("4. Run cooperative 5-view real sequence")
    print("5. Run one selected view real test")
    print("7. Read A/B status once")
    print("8. Monitor A/B status repeatedly")
    print("9. Show latest ROS topic states")
    print("10. A only: IK pick from latest top-view target")
    print("11. IK pick -> FRONT inspection")
    print("12. IK pick -> cooperative 5-view inspection")
    print("13. Manual move A joints")
    print("14. Manual move B joints")
    print("15. Move A to zero")
    print("16. Move B to zero")
    print("17. Move A named pose")
    print("18. Move B named pose")
    print("19. Move A/B to zero in parallel")
    print("20. Move A/B named view pose pair")
    print("27. FULL AUTO: top_view -> IK pick -> 5-view -> OK/NG place -> home")
    print("28. Move final OK/NG place + suction OFF + home")
    print("30. RECOVERY: compute judgement from saved coop results -> final transfer")
    print("31. CONTINUOUS AUTO LOOP: repeat full one-cycle process")
    print("32. SAFE STOP: suction OFF + A_HOME/B_HOME")
    print("0. Exit")
    print("======================================")


def main():
    system = DualArmMain()
    try:
        while True:
            print_menu()
            sel = input("select: ").strip()

            try:
                if sel == "1":
                    system.init_robots()
                elif sel == "2":
                    system.read_top_view_judge()
                elif sel == "4":
                    system.run_coop_sequence()
                elif sel == "5":
                    view = input("view name [front/lower/upper/left/right]: ").strip()
                    system.run_one_view_by_name(view)
                elif sel == "7":
                    system.show_status_once()
                elif sel == "8":
                    duration_raw = input("duration sec (default 10): ").strip()
                    interval_raw = input("interval sec (default 1): ").strip()
                    duration = 10.0 if duration_raw == "" else float(duration_raw)
                    interval = 1.0 if interval_raw == "" else float(interval_raw)
                    system.monitor_status(duration_sec=duration, interval_sec=interval)
                elif sel == "9":
                    system.show_latest_ros_states()
                elif sel == "10":
                    z_raw = input("pick z mm (default 50): ").strip()
                    z_mm = None if z_raw == "" else float(z_raw)
                    system.execute_ik_pick_from_latest_top_view(z_mm=z_mm)
                elif sel == "11":
                    system.run_pick_then_front()
                elif sel == "12":
                    system.run_pick_then_coop_sequence()
                elif sel == "13":
                    system.manual_move_a()
                elif sel == "14":
                    system.manual_move_b()
                elif sel == "15":
                    system.move_a_zero()
                elif sel == "16":
                    system.move_b_zero()
                elif sel == "17":
                    system.move_a_named_pose()
                elif sel == "18":
                    system.move_b_named_pose()
                elif sel == "19":
                    system.move_ab_zero_parallel()
                elif sel == "20":
                    system.move_ab_named_pose_pair()
                elif sel == "27":
                    system.run_full_one_cycle_auto()
                elif sel == "28":
                    system.move_final_place_and_release_auto()
                elif sel == "30":
                    system.recover_compute_and_transfer_allow_missing_top()
                elif sel == "31":
                    system.run_continuous_auto_loop()
                elif sel == "32":
                    system.safe_stop_return_home("menu 32")
                elif sel == "0":
                    break
                else:
                    print("[ERROR] invalid input")
            except KeyboardInterrupt:
                print("\n[MAIN] Ctrl+C detected")
                system.safe_stop_return_home("Ctrl+C during menu operation")
            except Exception as e:
                print(f"[ERROR] {e}")
    finally:
        system.close()


if __name__ == "__main__":
    main()
