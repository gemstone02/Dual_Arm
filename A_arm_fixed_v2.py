#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 0507 fixed-pose multi-view inspection robot controller

import math
import threading
import time
import urllib.request
import os
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncWrite, GroupSyncRead

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


ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_HARDWARE_ERROR_STATUS = 70
ADDR_POSITION_D_GAIN = 80
ADDR_POSITION_I_GAIN = 82
ADDR_POSITION_P_GAIN = 84
ADDR_FEEDFORWARD_2ND_GAIN = 88
ADDR_FEEDFORWARD_1ST_GAIN = 90
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_INPUT_VOLTAGE = 144
ADDR_PRESENT_TEMPERATURE = 146

LEN_GOAL_POSITION = 4
LEN_PRESENT_POSITION = 4

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
OPERATING_MODE_EXTENDED_POSITION = 4
PROTOCOL_VERSION = 2.0
DXL_RESOLUTION = 4096


@dataclass
class DxlConfig:
    device_name: str = "/dev/ttyUSB0"
    baudrate: int = 2000000
    ids: List[int] = field(default_factory=lambda: [3, 4, 5, 6, 7, 8])
    profile_velocity: int = 20
    profile_acceleration: int = 5
    position_tolerance_tick: int = 100
    move_timeout_sec: float = 25.0
    pose_dwell_sec: float = 0.2
    joint_limits_deg: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "q1": (-150.0, 150.0),
        "q2": (-85.0, 90.0),
        "q3": (-40.0, 200.0),
        "q4": (-260.0, 260.0),
        "q5": (-45.0, 45.0),
    })


@dataclass
class RobotCalibration:
    k_q1_from_m3: float = -0.22
    k_q2_from_m4: float = +0.065
    k_q3_from_diff56: float = +0.27
    k_q4_from_sum56: float = -0.263
    k_q5_from_diff78: float = +0.50
    motor_zero_tick: Dict[int, int] = field(default_factory=lambda: {3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0})


@dataclass
class PoseLibrary:
    HOME: Dict[str, float] = field(default_factory=lambda: {"q1": -70.0, "q2": 0.0, "q3": 70.0, "q4": 0.0, "q5": 0.0})

    # 고정 검사 공정용 자세
    PRE_GRASP: Dict[str, float] = field(default_factory=lambda: {"q1": 0.0, "q2": 20.0, "q3": 50.0, "q4": 0.0, "q5": 35.0})
    GRASP: Dict[str, float] = field(default_factory=lambda: {"q1": 0.0, "q2": 25.0, "q3": 105.0, "q4": 0.0, "q5": 25.0})

    FSD: Dict[str, float] = field(default_factory=lambda: {"q1": 0.0, "q2": 0.0, "q3": 65.0, "q4": 0.0, "q5": 0.0})
    SEC: Dict[str, float] = field(default_factory=lambda: {"q1": 0.0, "q2": 0.0, "q3": 65.0, "q4": 75.0, "q5": 0.0})
    TRD: Dict[str, float] = field(default_factory=lambda: {"q1": 0.0, "q2": 0.0, "q3": 65.0, "q4": 160.0, "q5": 0.0})
    FTH: Dict[str, float] = field(default_factory=lambda: {"q1": 0.0, "q2": 0.0, "q3": 65.0, "q4": 240.0, "q5": 0.0})
    STH: Dict[str, float] = field(default_factory=lambda: {"q1": 0.0, "q2": 0.0, "q3": 65.0, "q4": 160.0, "q5": 45.0})

    OK_PLACE: Dict[str, float] = field(default_factory=lambda: {"q1": 80.0, "q2": 10.0, "q3": 65.0, "q4": 160.0, "q5": -45.0})
    NG_PLACE: Dict[str, float] = field(default_factory=lambda: {"q1": -75.0, "q2": 10.0, "q3": 65.0, "q4": 160.0, "q5": -45.0})

    # 기존 메뉴 호환용 이름
    PRE_GRASP_1: Dict[str, float] = field(default_factory=lambda: {"q1": 0.0, "q2": 20.0, "q3": 50.0, "q4": 0.0, "q5": 20.0})
    PRE_GRASP_2: Dict[str, float] = field(default_factory=lambda: {"q1": 0.0, "q2": 20.0, "q3": 75.0, "q4": 0.0, "q5": 20.0})
    POST_GRASP_LIFT: Dict[str, float] = field(default_factory=lambda: {"q1": 0.0, "q2": 20.0, "q3": 50.0, "q4": 0.0, "q5": 20.0})
    HOLD_VERTICAL: Dict[str, float] = field(default_factory=lambda: {"q1": 0.0, "q2": 10.0, "q3": -12.0, "q4": 0.0, "q5": -20.0})


@dataclass
class VisionConfig:
    ref_x_mm: float = 225.8
    ref_y_mm: float = 155.8
    kx_q1: float = 0.03
    ky_q2: float = 0.03
    ky_q3: float = 0.015
    max_dx_mm: float = 40.0
    max_dy_mm: float = 40.0
    ros_topic: str = "/detected_object"
    default_pick_z_mm: float = 120.0
    allowed_labels_for_ik: tuple = ("Car_part", "car_part")

    ik_ref_robot_x_mm: float = 390.0
    ik_ref_robot_y_mm: float = 0.0
    ik_map_dy_to_robot_x_gain: float = -1.25
    ik_map_dx_to_robot_y_gain: float = 1.00
    ik_robot_x_min_mm: float = 360.0
    ik_robot_x_max_mm: float = 650.0
    ik_robot_y_min_mm: float = -200.0
    ik_robot_y_max_mm: float = 200.0


@dataclass
class SuctionConfig:
    enabled: bool = True
    base_url: str = "http://10.109.12.156"
    timeout_sec: float = 3.0


@dataclass
class IKConfig:
    base_height_mm: float = 91.0
    shoulder_to_elbow_mm: float = 179.0
    elbow_to_tool_mm: float = 288.0
    wrist_forward_offset_mm: float = 94.5
    wrist_lateral_offset_mm: float = 38.2
    wrist_vertical_offset_mm: float = 0.0
    q2_zero_offset_deg: float = 82.0
    q3_zero_offset_deg: float = 4.3
    fixed_q4_deg: float = 0.0
    fixed_q5_deg: float = 38.0
    q5_parallel_home_deg: float = 42.0
    q5_comp_from_q2_gain: float = 0.2
    q5_comp_from_q3_gain: float = 0.0
    z_input_ref_mm: float = 120.0
    z_input_gain: float = 1.00
    z_invert: bool = False


@dataclass
class DetectionData:
    label: str
    x_mm: float
    y_mm: float
    angle_deg: float
    area: int
    judge: str
    stamp: float


@dataclass
class InspectionState:
    has_car_part: bool
    has_defect: bool
    defect_count: int
    judge: str
    stamp: float


def deg_to_tick(deg: float) -> int:
    return int(round((deg / 360.0) * DXL_RESOLUTION))


def tick_to_deg(tick: int) -> float:
    return (tick / DXL_RESOLUTION) * 360.0


def uint32_from_int(value: int) -> int:
    return value & 0xFFFFFFFF


def dxl_signed32(value: int) -> int:
    if value & 0x80000000:
        value = -((~value + 1) & 0xFFFFFFFF)
    return value


def dxl_signed16(value: int) -> int:
    if value & 0x8000:
        value = -((~value + 1) & 0xFFFF)
    return value


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class JointMotorMapper:
    def __init__(self, calib: RobotCalibration):
        self.c = calib

    def motor_deg_to_joint_deg(self, m3, m4, m5, m6, m7, m8):
        q1 = self.c.k_q1_from_m3 * m3
        q2 = self.c.k_q2_from_m4 * m4
        q3 = self.c.k_q3_from_diff56 * ((m5 - m6) / 2.0)
        q4 = self.c.k_q4_from_sum56 * ((m5 + m6) / 2.0)
        q5 = self.c.k_q5_from_diff78 * ((m7 - m8) / 2.0)
        return {"q1": q1, "q2": q2, "q3": q3, "q4": q4, "q5": q5}

    def joint_deg_to_motor_deg(self, q1, q2, q3, q4, q5):
        m3 = q1 / self.c.k_q1_from_m3
        m4 = q2 / self.c.k_q2_from_m4
        a = q3 / self.c.k_q3_from_diff56
        b = q4 / self.c.k_q4_from_sum56
        m5 = a + b
        m6 = b - a
        c = q5 / self.c.k_q5_from_diff78
        m7 = c
        m8 = -c
        return {3: m3, 4: m4, 5: m5, 6: m6, 7: m7, 8: m8}

    def joint_deg_to_goal_tick(self, q):
        motor_deg = self.joint_deg_to_motor_deg(q["q1"], q["q2"], q["q3"], q["q4"], q["q5"])
        return {dxl_id: self.c.motor_zero_tick[dxl_id] + deg_to_tick(mdeg) for dxl_id, mdeg in motor_deg.items()}

    def present_tick_to_joint_deg(self, present_tick):
        m3 = tick_to_deg(present_tick[3] - self.c.motor_zero_tick[3])
        m4 = tick_to_deg(present_tick[4] - self.c.motor_zero_tick[4])
        m5 = tick_to_deg(present_tick[5] - self.c.motor_zero_tick[5])
        m6 = tick_to_deg(present_tick[6] - self.c.motor_zero_tick[6])
        m7 = tick_to_deg(present_tick[7] - self.c.motor_zero_tick[7])
        m8 = tick_to_deg(present_tick[8] - self.c.motor_zero_tick[8])
        return self.motor_deg_to_joint_deg(m3, m4, m5, m6, m7, m8)


class DynamixelArm:
    def __init__(self, cfg: DxlConfig, calib: RobotCalibration):
        self.cfg = cfg
        self.calib = calib
        self.mapper = JointMotorMapper(calib)
        self.port_handler = PortHandler(cfg.device_name)
        self.packet_handler = PacketHandler(PROTOCOL_VERSION)
        self.sync_write_goal = GroupSyncWrite(self.port_handler, self.packet_handler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
        self.sync_read_present = GroupSyncRead(self.port_handler, self.packet_handler, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
        self._opened = False

    def _check_comm(self, dxl_comm_result, dxl_error, context=""):
        if dxl_comm_result != 0:
            raise RuntimeError(f"[{context}] 통신 실패: {self.packet_handler.getTxRxResult(dxl_comm_result)}")
        if dxl_error != 0:
            raise RuntimeError(f"[{context}] DXL 에러: {self.packet_handler.getRxPacketError(dxl_error)}")

    def open(self):
        if not self.port_handler.openPort():
            raise RuntimeError(f"포트 열기 실패: {self.cfg.device_name}")
        if not self.port_handler.setBaudRate(self.cfg.baudrate):
            raise RuntimeError(f"Baudrate 설정 실패: {self.cfg.baudrate}")
        self._opened = True
        for dxl_id in self.cfg.ids:
            self.sync_read_present.addParam(dxl_id)

    def close(self):
        if self._opened:
            self.port_handler.closePort()
            self._opened = False

    def write1(self, dxl_id, addr, value, context="write1"):
        comm, err = self.packet_handler.write1ByteTxRx(self.port_handler, dxl_id, addr, value)
        self._check_comm(comm, err, context)

    def write2(self, dxl_id, addr, value, context="write2"):
        comm, err = self.packet_handler.write2ByteTxRx(self.port_handler, dxl_id, addr, value)
        self._check_comm(comm, err, context)

    def write4(self, dxl_id, addr, value, context="write4"):
        comm, err = self.packet_handler.write4ByteTxRx(self.port_handler, dxl_id, addr, value)
        self._check_comm(comm, err, context)

    def read1(self, dxl_id, addr, context="read1"):
        val, comm, err = self.packet_handler.read1ByteTxRx(self.port_handler, dxl_id, addr)
        self._check_comm(comm, err, context)
        return val

    def read2(self, dxl_id, addr, context="read2"):
        val, comm, err = self.packet_handler.read2ByteTxRx(self.port_handler, dxl_id, addr)
        self._check_comm(comm, err, context)
        return val

    def read_present_current_raw(self, dxl_id):
        val = self.read2(dxl_id, ADDR_PRESENT_CURRENT, f"read_current_id{dxl_id}")
        return dxl_signed16(val)

    def read_present_current_ma(self, dxl_id):
        return self.read_present_current_raw(dxl_id) * 2.69

    def read4(self, dxl_id, addr, context="read4"):
        val, comm, err = self.packet_handler.read4ByteTxRx(self.port_handler, dxl_id, addr)
        self._check_comm(comm, err, context)
        return dxl_signed32(val)

    def disable_torque_all(self):
        for dxl_id in self.cfg.ids:
            try:
                self.write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE, "disable_torque")
            except Exception as e:
                print(f"[WARN] disable torque 실패 ID {dxl_id}: {e}")

    def enable_torque_all(self):
        for dxl_id in self.cfg.ids:
            self.write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE, "enable_torque")

    def set_profile_velocity_individual(self):
        vel_map = {3: 60, 4: 120, 5: 45, 6: 45, 7: 12, 8: 12}
        acc_map = {3: 4, 4: 5, 5: 2, 6: 2, 7: 2, 8: 2}
        for dxl_id in self.cfg.ids:
            self.write4(dxl_id, ADDR_PROFILE_VELOCITY, vel_map[dxl_id], "vel")
            self.write4(dxl_id, ADDR_PROFILE_ACCELERATION, acc_map[dxl_id], "acc")
        print("[OK] 모터별 속도/가속도 설정 완료")

    def set_extended_position_mode_all(self):
        self.disable_torque_all()
        for dxl_id in self.cfg.ids:
            self.write1(dxl_id, ADDR_OPERATING_MODE, OPERATING_MODE_EXTENDED_POSITION, "set_mode")
        self.enable_torque_all()
        self.set_profile_velocity_individual()

    def apply_position_pid_gains_all(self, p_gain, i_gain=0, d_gain=0, ff1_gain=0, ff2_gain=0):
        for dxl_id in self.cfg.ids:
            self.write2(dxl_id, ADDR_POSITION_D_GAIN, d_gain, "set_pos_d")
            self.write2(dxl_id, ADDR_POSITION_I_GAIN, i_gain, "set_pos_i")
            self.write2(dxl_id, ADDR_POSITION_P_GAIN, p_gain, "set_pos_p")
            self.write2(dxl_id, ADDR_FEEDFORWARD_1ST_GAIN, ff1_gain, "set_ff1")
            self.write2(dxl_id, ADDR_FEEDFORWARD_2ND_GAIN, ff2_gain, "set_ff2")
        print(f"[OK] Position gains 적용: P={p_gain}, I={i_gain}, D={d_gain}, FF1={ff1_gain}, FF2={ff2_gain}")

    def read_present_ticks_individual(self, retry_per_id: int = 3, retry_dt: float = 0.04):
        """
        SyncRead가 계속 실패할 때 사용하는 fallback.
        각 모터의 Present Position을 개별 read4로 읽는다.
        느리지만 통신이 불안정할 때 프로그램이 바로 죽는 것을 줄인다.
        """
        out = {}
        failed = []

        for dxl_id in self.cfg.ids:
            ok = False
            last_error = None

            for attempt in range(1, retry_per_id + 1):
                try:
                    out[dxl_id] = self.read4(dxl_id, ADDR_PRESENT_POSITION, f"fallback_present_id{dxl_id}")
                    ok = True
                    break
                except Exception as e:
                    last_error = e
                    time.sleep(retry_dt)

            if not ok:
                failed.append((dxl_id, last_error))

        if failed:
            msg = ", ".join([f"ID {dxl_id}: {err}" for dxl_id, err in failed])
            raise RuntimeError(f"Fallback individual read 실패: {msg}")

        return out

    def read_present_ticks(self, retry: int = 8, retry_dt: float = 0.08, use_fallback: bool = True):
        """
        현재 모터 tick을 읽는다.

        1차: GroupSyncRead 재시도
        2차: SyncRead가 모두 실패하면 모터별 개별 read4 fallback

        이동 명령은 정상적으로 들어갔는데 이동 중 응답 패킷이 누락되는 경우가 있어,
        단순 SyncRead 실패만으로 프로그램이 바로 종료되지 않게 한다.
        """
        last_comm = None

        for attempt in range(1, retry + 1):
            comm = self.sync_read_present.txRxPacket()
            last_comm = comm

            if comm == 0:
                out = {}
                ok = True

                for dxl_id in self.cfg.ids:
                    if not self.sync_read_present.isAvailable(
                        dxl_id,
                        ADDR_PRESENT_POSITION,
                        LEN_PRESENT_POSITION
                    ):
                        ok = False
                        break

                    raw = self.sync_read_present.getData(
                        dxl_id,
                        ADDR_PRESENT_POSITION,
                        LEN_PRESENT_POSITION
                    )
                    out[dxl_id] = dxl_signed32(raw)

                if ok:
                    return out

            print(
                f"[WARN] SyncRead 재시도 {attempt}/{retry}: "
                f"{self.packet_handler.getTxRxResult(comm)}"
            )
            time.sleep(retry_dt)

        if use_fallback:
            print("[WARN] SyncRead 실패 -> 개별 모터 read fallback 시도")
            return self.read_present_ticks_individual(retry_per_id=3, retry_dt=0.05)

        raise RuntimeError(
            f"SyncRead 실패: {self.packet_handler.getTxRxResult(last_comm)}"
        )

    def write_goal_ticks(self, goal_tick):
        self.sync_write_goal.clearParam()
        for dxl_id in self.cfg.ids:
            raw = uint32_from_int(goal_tick[dxl_id])
            param = [raw & 0xFF, (raw >> 8) & 0xFF, (raw >> 16) & 0xFF, (raw >> 24) & 0xFF]
            self.sync_write_goal.addParam(dxl_id, bytes(param))
        comm = self.sync_write_goal.txPacket()
        if comm != 0:
            raise RuntimeError(f"SyncWrite 실패: {self.packet_handler.getTxRxResult(comm)}")

    def print_present_state(self):
        ticks = self.read_present_ticks()
        joints = self.mapper.present_tick_to_joint_deg(ticks)
        print("\n[현재 모터 tick]")
        for dxl_id in self.cfg.ids:
            print(f"  ID {dxl_id}: {ticks[dxl_id]}")
        print("[현재 가상관절 degree]")
        for k in ["q1", "q2", "q3", "q4", "q5"]:
            print(f"  {k}: {joints[k]: .3f}")

    def print_device_status(self):
        print("\n[장치 상태]")
        for dxl_id in self.cfg.ids:
            try:
                torque = self.read1(dxl_id, ADDR_TORQUE_ENABLE, "read_torque")
                hwerr = self.read1(dxl_id, ADDR_HARDWARE_ERROR_STATUS, "read_hwerr")
                volt_raw = self.read2(dxl_id, ADDR_PRESENT_INPUT_VOLTAGE, "read_voltage")
                current_raw = self.read_present_current_raw(dxl_id)
                current_ma = current_raw * 2.69
                temp = self.read1(dxl_id, ADDR_PRESENT_TEMPERATURE, "read_temp")
                vel = self.read4(dxl_id, ADDR_PRESENT_VELOCITY, "read_vel")
                print(
                    f"  ID {dxl_id}: torque={torque}, hwerr={hwerr}, "
                    f"voltage={volt_raw/10.0:.1f}V, current={current_ma:.1f}mA(raw={current_raw}), "
                    f"temp={temp}C, vel={vel}"
                )
            except Exception as e:
                print(f"  ID {dxl_id}: 상태 읽기 실패: {e}")

    def monitor_power(self, duration_sec=5.0, interval_sec=0.5):
        print(f"\n[A POWER MONITOR] duration={duration_sec:.1f}s, interval={interval_sec:.2f}s")
        start = time.time()
        while True:
            now = time.time()
            if now - start > duration_sec:
                break

            print(f"\n  t = {now - start:.2f}s")
            for dxl_id in self.cfg.ids:
                try:
                    volt_raw = self.read2(dxl_id, ADDR_PRESENT_INPUT_VOLTAGE, f"monitor_voltage_id{dxl_id}")
                    current_raw = self.read_present_current_raw(dxl_id)
                    current_ma = current_raw * 2.69
                    temp = self.read1(dxl_id, ADDR_PRESENT_TEMPERATURE, f"monitor_temp_id{dxl_id}")
                    print(f"    ID {dxl_id}: {volt_raw/10.0:.1f}V, {current_ma:.1f}mA(raw={current_raw}), temp={temp}C")
                except Exception as e:
                    print(f"    ID {dxl_id}: monitor failed: {e}")
            time.sleep(interval_sec)

    def set_current_as_zero(self):
        ticks = self.read_present_ticks()
        self.calib.motor_zero_tick = dict(ticks)
        print("[INFO] 현재 위치를 motor_zero_tick 으로 설정:")
        print(self.calib.motor_zero_tick)

    def _validate_joint_limits(self, q):
        for key in ["q1", "q2", "q3", "q4", "q5"]:
            lo, hi = self.cfg.joint_limits_deg[key]
            if not (lo <= q[key] <= hi):
                raise ValueError(f"{key}={q[key]:.2f} deg 가 제한 [{lo}, {hi}] 밖입니다.")

    def move_joint_deg(self, q, wait=True):
        self._validate_joint_limits(q)
        goal_tick = self.mapper.joint_deg_to_goal_tick(q)
        print("[MOVE] joint target(deg):", q)
        print("[MOVE] goal tick:", goal_tick)
        self.write_goal_ticks(goal_tick)
        if not wait:
            return
        start = time.time()
        last_present = None
        while True:
            try:
                present = self.read_present_ticks()
                last_present = present
            except Exception as e:
                print(f"[ERROR] present tick 읽기 실패: {e}")
                print("[WARN] 현재 위치 확인 실패. 모터 상태 출력 후, 이동 명령은 이미 전송된 상태입니다.")
                self.print_device_status()
                raise
            done = True
            for dxl_id in self.cfg.ids:
                if abs(present[dxl_id] - goal_tick[dxl_id]) > self.cfg.position_tolerance_tick:
                    done = False
                    break
            if done:
                print("[MOVE] 완료")
                return
            if time.time() - start > self.cfg.move_timeout_sec:
                print("[TIMEOUT] goal:", goal_tick)
                print("[TIMEOUT] present:", last_present)
                self.print_device_status()
                raise TimeoutError("move_joint_deg timeout")
            time.sleep(0.10)


class DetectionSubscriber(Node):
    def __init__(self, topic_name: str, inspection_topic: str, state_holder: Dict[str, object], lock: threading.Lock):
        super().__init__('robot_detected_object_listener')
        self._state_holder = state_holder
        self._lock = lock
        self.subscription = self.create_subscription(String, topic_name, self.callback, 10)
        self.inspection_subscription = self.create_subscription(String, inspection_topic, self.inspection_callback, 10)
        self.get_logger().info(f"Subscribed to {topic_name}")
        self.get_logger().info(f"Subscribed to {inspection_topic}")

    def callback(self, msg):
        try:
            parts = [p.strip() for p in msg.data.split(',')]
            if len(parts) < 6:
                self.get_logger().warning(f"Unexpected detected_object format: {msg.data}")
                return

            label = parts[0]
            x_mm = float(parts[1])
            y_mm = float(parts[2])
            angle_deg = float(parts[3])
            area = int(parts[4])
            judge = parts[5]

            det = DetectionData(
                label=label,
                x_mm=x_mm,
                y_mm=y_mm,
                angle_deg=angle_deg,
                area=area,
                judge=judge,
                stamp=time.time()
            )

            with self._lock:
                self._state_holder['latest'] = det

        except Exception as e:
            self.get_logger().warning(f"detected_object parse 실패: {e} / raw={msg.data}")

    def inspection_callback(self, msg):
        try:
            data = json.loads(msg.data)
            state = InspectionState(
                has_car_part=bool(data.get("has_car_part", False)),
                has_defect=bool(data.get("has_defect", False)),
                defect_count=int(data.get("defect_count", 0)),
                judge=str(data.get("judge", "NO_OBJECT")),
                stamp=time.time(),
            )
            with self._lock:
                self._state_holder["inspection"] = state
        except Exception as e:
            self.get_logger().warning(f"inspection_state parse 실패: {e} / raw={msg.data}")


class VisionROSBridge:
    def __init__(self, topic_name: str, inspection_topic: str = "/inspection_state"):
        self.topic_name = topic_name
        self.inspection_topic = inspection_topic
        self._thread = None
        self._lock = threading.Lock()
        self._state_holder: Dict[str, object] = {'latest': None, 'inspection': None}
        self._stop_evt = threading.Event()
        self._started = False

    def start(self):
        if not ROS2_AVAILABLE:
            print("[WARN] ROS2 Python 환경을 찾지 못했습니다. 이 터미널에서 source /opt/ros/humble/setup.bash 및 source ~/ros2_ws/install/setup.bash 후 다시 실행하세요.")
            return False
        if self._started:
            print("[OK] ROS2 detected_object subscriber 이미 실행 중")
            return True
        self._thread = threading.Thread(target=self._spin_thread, daemon=True)
        self._thread.start()
        self._started = True
        time.sleep(0.2)
        return True

    def _spin_thread(self):
        rclpy.init(args=None)
        node = DetectionSubscriber(self.topic_name, self.inspection_topic, self._state_holder, self._lock)
        try:
            while rclpy.ok() and not self._stop_evt.is_set():
                rclpy.spin_once(node, timeout_sec=0.2)
        finally:
            node.destroy_node()
            rclpy.shutdown()

    def stop(self):
        if not self._started:
            return
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._started = False

    def get_latest(self) -> Optional[DetectionData]:
        with self._lock:
            return self._state_holder.get('latest')

    def get_inspection(self) -> Optional[InspectionState]:
        with self._lock:
            return self._state_holder.get('inspection')


class VisionStandController:
    def __init__(self, arm: DynamixelArm, poses: PoseLibrary, cfg: DxlConfig,
                 vision_cfg: VisionConfig, suction_cfg: SuctionConfig,
                 ik_cfg: IKConfig, ros_bridge: VisionROSBridge):
        self.arm = arm
        self.poses = poses
        self.cfg = cfg
        self.vision_cfg = vision_cfg
        self.suction_cfg = suction_cfg
        self.ik_cfg = ik_cfg
        self.ros_bridge = ros_bridge

    def move_pose(self, pose_name, dwell=None):
        if dwell is None:
            dwell = self.cfg.pose_dwell_sec
        pose = getattr(self.poses, pose_name)
        print(f"\n[POSE] {pose_name}")
        self.arm.move_joint_deg(pose)
        time.sleep(dwell)

    def move_manual(self, q1, q2, q3, q4, q5):
        self.arm.move_joint_deg({"q1": q1, "q2": q2, "q3": q3, "q4": q4, "q5": q5})

    def _send_grip_state(self, state: int):
        if not self.suction_cfg.enabled:
            print(f"[SUCTION] disabled, state={state}")
            return
        url = f"{self.suction_cfg.base_url}/grip?state={state}"
        try:
            with urllib.request.urlopen(url, timeout=self.suction_cfg.timeout_sec) as resp:
                body = resp.read().decode("utf-8", errors="ignore").strip()
            print(f"[OK] grip state={state} 응답: {body}")
        except Exception as e:
            print(f"[WARN] grip state={state} 전송 실패: {e}")

    def suction_on(self):
        print("[SUCTION] ON")
        self._send_grip_state(1)

    def suction_off(self):
        print("[SUCTION] OFF")
        self._send_grip_state(0)

    def _apply_delta(self, base_pose, dq1, dq2, dq3):
        q = dict(base_pose)
        q["q1"] += dq1
        q["q2"] += dq2
        q["q3"] += dq3
        q["q1"] = clamp(q["q1"], *self.cfg.joint_limits_deg["q1"])
        q["q2"] = clamp(q["q2"], *self.cfg.joint_limits_deg["q2"])
        q["q3"] = clamp(q["q3"], *self.cfg.joint_limits_deg["q3"])
        return q

    def make_pick_poses_from_vision(self, x_mm, y_mm):
        dx = clamp(x_mm - self.vision_cfg.ref_x_mm, -self.vision_cfg.max_dx_mm, self.vision_cfg.max_dx_mm)
        dy = clamp(y_mm - self.vision_cfg.ref_y_mm, -self.vision_cfg.max_dy_mm, self.vision_cfg.max_dy_mm)
        dq1 = self.vision_cfg.kx_q1 * dx
        dq2 = self.vision_cfg.ky_q2 * dy
        dq3 = self.vision_cfg.ky_q3 * dy
        print("\n[VISION INPUT]")
        print(f"  x_mm={x_mm:.2f}, y_mm={y_mm:.2f}")
        print(f"  ref_x={self.vision_cfg.ref_x_mm:.2f}, ref_y={self.vision_cfg.ref_y_mm:.2f}")
        print(f"  dx={dx:.2f}, dy={dy:.2f}")
        print(f"  dq1={dq1:.3f}, dq2={dq2:.3f}, dq3={dq3:.3f}")
        pre1 = self._apply_delta(self.poses.PRE_GRASP_1, dq1, dq2, dq3)
        pre2 = self._apply_delta(self.poses.PRE_GRASP_2, dq1, dq2, dq3)
        grasp = self._apply_delta(self.poses.GRASP, dq1, dq2, dq3)
        post_lift = self._apply_delta(self.poses.POST_GRASP_LIFT, dq1, dq2, dq3)
        return pre1, pre2, grasp, post_lift

    def preview_pick_poses(self, x_mm, y_mm):
        pre1, pre2, grasp, post_lift = self.make_pick_poses_from_vision(x_mm, y_mm)
        print("\n[CALCULATED POSES]")
        print("PRE_GRASP_1:", pre1)
        print("PRE_GRASP_2:", pre2)
        print("GRASP:", grasp)
        print("POST_GRASP_LIFT:", post_lift)

    def move_joint_blend(self, q_target: Dict[str, float], steps: int = 30, step_dt: float = 0.02, final_dwell: float = 0.1):
        start_q = self.arm.mapper.present_tick_to_joint_deg(self.arm.read_present_ticks())
        for i in range(1, steps + 1):
            t = i / steps
            q = {
                k: start_q[k] + (q_target[k] - start_q[k]) * t
                for k in ["q1", "q2", "q3", "q4", "q5"]
            }
            self.arm.move_joint_deg(q, wait=False)
            time.sleep(step_dt)
        self.arm.move_joint_deg(q_target, wait=True)
        time.sleep(final_dwell)

    def _user_z_to_ik_z(self, z_user_mm: float) -> float:
        ref = self.ik_cfg.z_input_ref_mm
        gain = self.ik_cfg.z_input_gain
        dz = z_user_mm - ref
        if self.ik_cfg.z_invert:
            return ref - gain * dz
        return ref + gain * dz

    def solve_ik_3d(self, x_mm: float, y_mm: float, z_mm: float) -> Dict[str, float]:
        z_ik_mm = self._user_z_to_ik_z(z_mm)

        q1_deg = math.degrees(math.atan2(y_mm, x_mm))
        r_world = math.hypot(x_mm, y_mm)
        r_planar = r_world - self.ik_cfg.wrist_forward_offset_mm - self.ik_cfg.wrist_lateral_offset_mm
        z_planar = z_ik_mm - self.ik_cfg.base_height_mm - self.ik_cfg.wrist_vertical_offset_mm

        r_ref = 260.0
        z_ref = 30.0

        dr = r_planar - r_ref
        dz = z_planar - z_ref

        print("[IK INPUT]")
        print(f"  target xyz(mm) = ({x_mm:.2f}, {y_mm:.2f}, {z_mm:.2f})")
        print(f"  z_user->ik     = {z_mm:.2f} -> {z_ik_mm:.2f}")
        print(f"  r_world        = {r_world:.2f}")
        print(f"  r_planar       = {r_planar:.2f}")
        print(f"  z_planar       = {z_planar:.2f}")
        print(f"  dr,dz          = ({dr:.2f}, {dz:.2f})")

        base_q2 = 20.475
        base_q3 = -5.541

        dr_eff = 0.95 * dr
        dz_eff = 1.50 * dz

        dq2 = 0.188119 * dr_eff + (-0.148515) * dz_eff
        dq3 = 0.320000 * dr_eff + (-0.049505) * dz_eff

        q2_code = base_q2 + dq2
        q3_code = base_q3 + dq3

        q2_clamped = clamp(q2_code, *self.cfg.joint_limits_deg["q2"])
        q3_clamped = clamp(q3_code, *self.cfg.joint_limits_deg["q3"])
        q5_comp = (
            self.ik_cfg.q5_parallel_home_deg
            - self.ik_cfg.q5_comp_from_q2_gain * q2_clamped
            - self.ik_cfg.q5_comp_from_q3_gain * q3_clamped
        )

        q = {
            "q1": clamp(q1_deg, *self.cfg.joint_limits_deg["q1"]),
            "q2": q2_clamped,
            "q3": q3_clamped,
            "q4": self.ik_cfg.fixed_q4_deg,
            "q5": clamp(q5_comp, *self.cfg.joint_limits_deg["q5"]),
        }

        print(f"[IK SOLVED] {q}")
        return q

    def move_xyz_ik(self, x_mm: float, y_mm: float, z_mm: float):
        q = self.solve_ik_3d(x_mm, y_mm, z_mm)
        print("[IK MOVE] 수동 xyz 입력으로 이동")
        self.arm.move_joint_deg(q)

    def preview_latest_detection(self):
        det = self.ros_bridge.get_latest()
        if det is None:
            print("[VISION] 아직 /detected_object 메시지를 받지 못했습니다.")
            return
        age = time.time() - det.stamp
        print("\n[최신 비전 검출]")
        print(f"  label = {det.label}")
        print(f"  x_mm  = {det.x_mm:.1f}")
        print(f"  y_mm  = {det.y_mm:.1f}")
        print(f"  angle = {det.angle_deg:.1f}")
        print(f"  area  = {det.area}")
        print(f"  judge = {det.judge}")
        print(f"  age   = {age:.2f} sec")

    def get_inspection_state(self):
        state = self.ros_bridge.get_inspection()
        if state is None:
            print("[INSPECT] 아직 /inspection_state 메시지를 받지 못했습니다.")
            return None
        age = time.time() - state.stamp
        print("\n[최신 검사 상태]")
        print(f"  has_car_part = {state.has_car_part}")
        print(f"  has_defect   = {state.has_defect}")
        print(f"  defect_count = {state.defect_count}")
        print(f"  judge        = {state.judge}")
        print(f"  age          = {age:.2f} sec")
        return state

    def inspect_current_view(self, view_name: str, hold_sec: float = 3.0,
                             required_defect_sec: float = 2.0, sample_dt: float = 0.05,
                             max_state_age_sec: float = 0.5):
        print(f"\n[INSPECT] {view_name}: {hold_sec:.1f}초 관찰 시작")
        start = time.time()
        last_t = start
        defect_accum_sec = 0.0
        max_defect_count = 0
        sample_count = 0
        valid_sample_count = 0

        while True:
            now = time.time()
            if now - start >= hold_sec:
                break

            dt = now - last_t
            last_t = now
            sample_count += 1

            state = self.ros_bridge.get_inspection()
            if state is not None and (now - state.stamp) <= max_state_age_sec:
                valid_sample_count += 1
                max_defect_count = max(max_defect_count, state.defect_count)
                if state.has_defect:
                    defect_accum_sec += dt

            time.sleep(sample_dt)

        detected = defect_accum_sec >= required_defect_sec
        result = {
            "view": view_name,
            "detected": detected,
            "defect_accum_sec": round(defect_accum_sec, 3),
            "hold_sec": hold_sec,
            "required_defect_sec": required_defect_sec,
            "max_defect_count": max_defect_count,
            "sample_count": sample_count,
            "valid_sample_count": valid_sample_count,
        }

        print(
            f"[INSPECT RESULT] {view_name}: "
            f"defect_time={defect_accum_sec:.2f}/{hold_sec:.2f}s, "
            f"threshold={required_defect_sec:.2f}s -> {'NG' if detected else 'OK'}"
        )
        return result

    def _save_inspection_log(self, results, final_judgement: str):
        log_dir = os.path.join(os.getcwd(), "inspection_logs")
        os.makedirs(log_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(log_dir, f"inspection_{stamp}.json")
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "final_judgement": final_judgement,
            "results": results,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[LOG] 검사 결과 저장: {path}")
        return path

    def run_fixed_pose_multi_view_inspection(self):
        """
        고정 pose 기반 6면 검사 공정.
        - HOME에서 TOP 검사
        - PRE_GRASP -> GRASP -> suction ON -> PRE_GRASP
        - FSD/SEC/TRD/FTH/STH에서 각각 3초 관찰
        - 각 관찰 구간에서 defect 누적 2초 이상이면 해당 view NG
        - 하나라도 NG면 NG_PLACE, 모두 OK면 OK_PLACE로 이동 후 suction OFF
        """
        if not self.ros_bridge._started:
            print("[ROS] /inspection_state 구독 시작")
            self.ros_bridge.start()
            time.sleep(0.5)

        results = []

        print("\n========== FIXED POSE MULTI-VIEW INSPECTION START ==========")

        print("\n[STEP 0] HOME 자세에서 TOP 검사")
        self.move_pose("HOME", dwell=0.5)
        results.append(self.inspect_current_view("top", hold_sec=3.0, required_defect_sec=2.0))

        print("\n[STEP 1] PRE_GRASP -> GRASP -> suction ON -> PRE_GRASP")
        self.move_pose("PRE_GRASP", dwell=0.3)
        self.move_pose("GRASP", dwell=0.3)
        self.suction_on()
        time.sleep(0.7)
        self.move_pose("PRE_GRASP", dwell=0.5)

        for pose_name, view_name in [
            ("FSD", "fsd"),
            ("SEC", "sec"),
            ("TRD", "trd"),
            ("FTH", "fth"),
            ("STH", "sth"),
        ]:
            print(f"\n[STEP INSPECT] {pose_name} 자세 이동 후 {view_name} 검사")
            self.move_pose(pose_name, dwell=0.5)
            results.append(self.inspect_current_view(view_name, hold_sec=3.0, required_defect_sec=2.0))

        has_any_defect = any(r["detected"] for r in results)
        final_judgement = "NG" if has_any_defect else "OK"

        if final_judgement == "NG":
            print("\n[FINAL] 하나 이상의 view에서 defect 검출 -> 불량품 위치 이동")
            self.move_pose("NG_PLACE", dwell=0.5)
        else:
            print("\n[FINAL] 모든 view에서 defect 미검출 -> 정상품 위치 이동")
            self.move_pose("OK_PLACE", dwell=0.5)

        self.suction_off()
        log_path = self._save_inspection_log(results, final_judgement)

        print("\n[SUMMARY]")
        for r in results:
            print(f"  {r['view']}: {'NG' if r['detected'] else 'OK'} ({r['defect_accum_sec']:.2f}s)")
        print(f"  final_judgement: {final_judgement}")
        print("========== FIXED POSE MULTI-VIEW INSPECTION END ==========\n")

        return final_judgement, results, log_path

    def run_pick_and_sort_from_latest_detection(self, z_mm: Optional[float] = None):
        det = self.ros_bridge.get_latest()
        if det is None:
            raise RuntimeError("/detected_object 최신 메시지가 없습니다.")

        if det.label not in self.vision_cfg.allowed_labels_for_ik:
            raise RuntimeError(f"허용 라벨은 {self.vision_cfg.allowed_labels_for_ik} 인데 현재 검출은 {det.label} 입니다.")

        if z_mm is None:
            z_mm = self.vision_cfg.default_pick_z_mm

        vision_dx = det.x_mm - self.vision_cfg.ref_x_mm
        vision_dy = det.y_mm - self.vision_cfg.ref_y_mm

        x_robot_raw = self.vision_cfg.ik_ref_robot_x_mm + self.vision_cfg.ik_map_dy_to_robot_x_gain * vision_dy
        y_robot_raw = self.vision_cfg.ik_ref_robot_y_mm + self.vision_cfg.ik_map_dx_to_robot_y_gain * vision_dx

        x_robot = clamp(x_robot_raw, self.vision_cfg.ik_robot_x_min_mm, self.vision_cfg.ik_robot_x_max_mm)
        y_robot = clamp(y_robot_raw, self.vision_cfg.ik_robot_y_min_mm, self.vision_cfg.ik_robot_y_max_mm)

        print("\n[AUTO SORT]")
        print(f"  label = {det.label}")
        print(f"  judge = {det.judge}")
        print(f"  vision = ({det.x_mm:.1f}, {det.y_mm:.1f})")
        print(f"  robot target = ({x_robot:.1f}, {y_robot:.1f}, {z_mm:.1f})")
        

        self.move_xyz_ik(x_robot, y_robot, z_mm)
        time.sleep(0.5)

        self.suction_on()
        time.sleep(0.7)

        if det.judge == "OK":
            target_pose = {"q1": 0.0, "q2": -30.0, "q3": 130.0, "q4": 0.0, "q5": -35.0}
            print("[SORT] 정상 → 정상 위치로 이동")
        else:
            target_pose = {"q1": 66.0, "q2": 0.0, "q3": 0.0, "q4": 0.0, "q5": -35.0}
            print("[SORT] 불량 → 불량 위치로 이동")

        self.arm.move_joint_deg(target_pose)
        time.sleep(0.5)

        self.suction_off()
        print("[DONE] 분류 완료")


def build_robot():
    cfg = DxlConfig()
    calib = RobotCalibration()
    poses = PoseLibrary()
    vision_cfg = VisionConfig()
    suction_cfg = SuctionConfig()
    ik_cfg = IKConfig()
    ros_bridge = VisionROSBridge(vision_cfg.ros_topic)
    arm = DynamixelArm(cfg, calib)
    ctrl = VisionStandController(arm, poses, cfg, vision_cfg, suction_cfg, ik_cfg, ros_bridge)
    return arm, ctrl, poses, vision_cfg, ik_cfg, ros_bridge


def print_pose_library(poses):
    print("\n[현재 저장된 자세]")
    for name in ["HOME", "PRE_GRASP", "GRASP", "FSD", "SEC", "TRD", "FTH", "STH", "OK_PLACE", "NG_PLACE", "PRE_GRASP_1", "PRE_GRASP_2", "POST_GRASP_LIFT", "HOLD_VERTICAL"]:
        print(f"{name}: {getattr(poses, name)}")


def print_vision_config(vcfg):
    print("\n[현재 vision 보정 설정]")
    print(f"ref_x_mm = {vcfg.ref_x_mm}")
    print(f"ref_y_mm = {vcfg.ref_y_mm}")
    print(f"ros_topic= {vcfg.ros_topic}")
    print(f"default_pick_z_mm = {vcfg.default_pick_z_mm}")
    print(f"allowed_labels_for_ik = {vcfg.allowed_labels_for_ik}")


def print_ik_config(ikcfg):
    print("\n[IK 설정]")
    print(f"base_height_mm          = {ikcfg.base_height_mm}")
    print(f"shoulder_to_elbow_mm    = {ikcfg.shoulder_to_elbow_mm}")
    print(f"elbow_to_tool_mm        = {ikcfg.elbow_to_tool_mm}")
    print(f"wrist_forward_offset_mm = {ikcfg.wrist_forward_offset_mm}")
    print(f"wrist_lateral_offset_mm = {ikcfg.wrist_lateral_offset_mm}")
    print(f"q5_parallel_home_deg    = {ikcfg.q5_parallel_home_deg}")


def print_menu():
    print("\n==============================")
    print(" REAL ROBOT VISION + IK MENU")
    print("==============================")
    print("1. 포트 열기 + Extended Position Mode 설정")
    print("2. 현재 모터/관절 상태 읽기")
    print("3. 현재 자세를 zero로 저장")
    print("4. 장치 상태(전압/온도/hwerr/torque) 읽기")
    print("5. 저장된 자세 목록 보기")
    print("6. 저장된 자세로 이동")
    print("7. 수동 joint 입력 이동")
    print("8. 기본 dry run 시퀀스")
    print("10. Position PID/FF gain 설정")
    print("11. Vision 보정 설정 보기")
    print("18. IK 설정 보기")
    print("19. 수동 xyz 입력으로 IK 이동")
    print("20. ROS /detected_object subscriber 시작")
    print("21. 최신 비전 검출 보기")
    print("22. 최신 검사 상태 보기(/inspection_state)")
    print("24. 최신 비전 좌표로 자동 흡착 후 정상/불량 분류")
    print("25. 고정 자세 기반 다면 검사 공정 실행")
    print("23. 종료")
    print("==============================")


def main():
    arm, ctrl, poses, vision_cfg, ik_cfg, ros_bridge = build_robot()
    try:
        ctrl.suction_off()
        while True:
            print_menu()
            sel = input("선택: ").strip()
            if sel == "1":
                arm.open()
                arm.set_extended_position_mode_all()
                print("[OK] 포트 오픈 및 Extended Position Mode 설정 완료")
            elif sel == "2":
                arm.print_present_state()
            elif sel == "3":
                arm.set_current_as_zero()
            elif sel == "4":
                arm.print_device_status()
            elif sel == "5":
                print_pose_library(poses)
            elif sel == "6":
                pose_name = input("pose 이름 입력: ").strip().upper()
                valid = ["HOME", "PRE_GRASP", "GRASP", "FSD", "SEC", "TRD", "FTH", "STH", "OK_PLACE", "NG_PLACE", "PRE_GRASP_1", "PRE_GRASP_2", "POST_GRASP_LIFT", "HOLD_VERTICAL"]
                if pose_name not in valid:
                    print("잘못된 pose 이름")
                    continue
                ctrl.move_pose(pose_name)
            elif sel == "7":
                q1 = float(input("q1(deg): "))
                q2 = float(input("q2(deg): "))
                q3 = float(input("q3(deg): "))
                q4 = float(input("q4(deg): "))
                q5 = float(input("q5(deg): "))
                ctrl.move_manual(q1, q2, q3, q4, q5)
            elif sel == "8":
                ctrl.move_pose("HOME")
                ctrl.move_pose("GRASP")
                ctrl.move_pose("POST_GRASP_LIFT")
            elif sel == "10":
                p = int(input("Position P Gain (예: 300): "))
                i = int(input("Position I Gain (예: 0): "))
                d = int(input("Position D Gain (예: 0): "))
                ff1 = int(input("Feedforward 1st Gain (예: 0): "))
                ff2 = int(input("Feedforward 2nd Gain (예: 0): "))
                arm.apply_position_pid_gains_all(p_gain=p, i_gain=i, d_gain=d, ff1_gain=ff1, ff2_gain=ff2)
            elif sel == "11":
                print_vision_config(vision_cfg)
            elif sel == "18":
                print_ik_config(ik_cfg)
            elif sel == "19":
                x_mm = float(input("x_mm: "))
                y_mm = float(input("y_mm: "))
                z_mm = float(input("z_mm: "))
                ctrl.move_xyz_ik(x_mm, y_mm, z_mm)
            elif sel == "20":
                ros_bridge.start()
            elif sel == "21":
                ctrl.preview_latest_detection()
            elif sel == "22":
                ctrl.get_inspection_state()
            elif sel == "24":
                z_raw = input(f"z_mm (엔터시 기본 {vision_cfg.default_pick_z_mm}): ").strip()
                z_mm = None if z_raw == "" else float(z_raw)
                ctrl.run_pick_and_sort_from_latest_detection(z_mm=z_mm)
            elif sel == "25":
                ctrl.run_fixed_pose_multi_view_inspection()
            elif sel == "23":
                print("[종료]")
                break
            else:
                print("잘못된 입력")
    finally:
        ros_bridge.stop()
        arm.close()


if __name__ == '__main__':
    main()
