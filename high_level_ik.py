import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def norm3(x, y, z):
    return math.sqrt(x * x + y * y + z * z)


@dataclass
class HighLevelIKConfig:
    approach_offset_mm: float = 35.0
    min_approach_offset_mm: float = 10.0
    approach_offset_step_mm: float = 5.0

    # base / arm
    base_x_mm: float = 109.0
    base_z_mm: float = 171.0
    link1_mm: float = 175.0
    link2_mm: float = 233.23854913922

    # TCP model
    wrist_to_link5_mm: float = 94.5
    link5_to_tcp_mm: float = 84.0

    # joint mapping
    q2_zero_offset_deg: float = 90.0
    q3_zero_offset_deg: float = 0.0
    q2_sign: float = -1.0
    q3_sign: float = -1.0

    # wrist / tool
    q4_fixed_deg: float = 0.0

    # 기존 경험식 관련 값들
    # 남겨두지만 compute_q5_raw에서는 더 이상 사용하지 않음
    ref_flat_q2_deg: float = 21.589868769688138
    ref_flat_q3_deg: float = 82.10570346709484
    ref_flat_q5_deg: float = 28.33826075725441
    q5_gain_from_q2_delta: float = 5.0560
    q5_gain_from_q3_delta: float = 4.2035
    q5_delta_scale: float = 0.12
    q5_global_offset_deg: float = 0.0

    # 새 q5 선형 모델
    # q5 = a*q2 + b*q3 + c
    q5_a: float = 0.41
    q5_b: float = -0.2
    q5_c: float = 38.5

    # 접근/최종 모두 같은 평행 기준을 쓰고 싶으면 True
    use_single_q5_for_approach_and_final: bool = False

    # -------- far-region practical compensation --------
    far_start_mm: float = 430.0
    far_full_mm: float = 500.0
    far_q5_offset_deg: float = 8.0
    far_tcp_z_offset_mm: float = 0.0
    far_tcp_x_offset_mm: float = 0.0
    far_tcp_y_offset_mm: float = 0.0

    debug: bool = True
    q_limits_deg: Optional[Dict[str, Tuple[float, float]]] = None

    @property
    def tcp_offset_mm(self) -> float:
        return self.wrist_to_link5_mm + self.link5_to_tcp_mm


class HighLevelIK:
    def __init__(self, cfg: HighLevelIKConfig):
        self.cfg = cfg

        if self.cfg.q_limits_deg is None:
            self.cfg.q_limits_deg = {
                "q1": (-100, 100),
                "q2": (-85, 80),
                "q3": (-40, 200),
                "q4": (-90, 90),
                "q5": (-45, 45),
            }

    # ---------- q5 model ----------
    def compute_q5_raw(self, q2, q3):
        return (
            -0.39 * q2
            - 0.43 * q3
            + 73.42
            + 5.0
        )

    def compute_q5(self, q2, q3):
        q5 = self.compute_q5_raw(q2, q3)
        lo, hi = self.cfg.q_limits_deg["q5"]
        return clamp(q5, lo, hi)

    def _far_alpha(self, x, y):
        r = math.hypot(x, y)
        start = self.cfg.far_start_mm
        full = max(self.cfg.far_full_mm, start + 1e-6)

        if r <= start:
            return 0.0
        if r >= full:
            return 1.0
        return (r - start) / (full - start)

    def get_far_compensation(self, x, y):
        alpha = self._far_alpha(x, y)
        return {
            "alpha": alpha,
            "q5_offset_deg": alpha * self.cfg.far_q5_offset_deg,
            "tcp_z_offset_mm": alpha * self.cfg.far_tcp_z_offset_mm,
            "tcp_x_offset_mm": alpha * self.cfg.far_tcp_x_offset_mm,
            "tcp_y_offset_mm": alpha * self.cfg.far_tcp_y_offset_mm,
        }

    def choose_parallel_q5(self, final_q2, final_q3, approach_q2=None, approach_q3=None):
        q5_final = self.compute_q5(final_q2, final_q3)

        if self.cfg.use_single_q5_for_approach_and_final:
            return q5_final

        if approach_q2 is None or approach_q3 is None:
            return q5_final

        q5_app = self.compute_q5(approach_q2, approach_q3)
        return 0.5 * (q5_final + q5_app)

    # ---------- kinematics ----------
    def command_to_model_angles_deg(self, q2, q3):
        t2_deg = self.cfg.q2_sign * (q2 - self.cfg.q2_zero_offset_deg)
        t3_deg = self.cfg.q3_sign * (q3 - self.cfg.q3_zero_offset_deg)
        return t2_deg, t3_deg

    def model_to_command_angles_deg(self, t2_deg, t3_deg):
        q2 = self.cfg.q2_zero_offset_deg + (t2_deg / self.cfg.q2_sign)
        q3 = self.cfg.q3_zero_offset_deg + (t3_deg / self.cfg.q3_sign)
        return q2, q3

    def fk_wrist(self, q1, q2, q3):
        t1 = math.radians(q1)
        t2_deg, t3_deg = self.command_to_model_angles_deg(q2, q3)
        t2 = math.radians(t2_deg)
        t3 = math.radians(t3_deg)

        r_local = (
            self.cfg.link1_mm * math.cos(t2) +
            self.cfg.link2_mm * math.cos(t2 + t3)
        )
        z_local = (
            self.cfg.link1_mm * math.sin(t2) +
            self.cfg.link2_mm * math.sin(t2 + t3)
        )

        r_world = self.cfg.base_x_mm + r_local
        z_world = self.cfg.base_z_mm + z_local

        x = r_world * math.cos(t1)
        y = r_world * math.sin(t1)
        return x, y, z_world

    def fk_tcp(self, q1, q2, q3):
        wx, wy, wz = self.fk_wrist(q1, q2, q3)
        return wx, wy, wz - self.cfg.tcp_offset_mm

    def compute_wrist_target(self, x, y, z):
        return x, y, z + self.cfg.tcp_offset_mm

    # ---------- IK ----------
    def _candidate_penalty(self, q2, q3, ref_q2=None, ref_q3=None):
        lo2, hi2 = self.cfg.q_limits_deg["q2"]
        lo3, hi3 = self.cfg.q_limits_deg["q3"]

        p = 0.0

        if q2 < lo2:
            p += (lo2 - q2) * 100.0
        elif q2 > hi2:
            p += (q2 - hi2) * 100.0

        if q3 < lo3:
            p += (lo3 - q3) * 100.0
        elif q3 > hi3:
            p += (q3 - hi3) * 100.0

        if ref_q2 is not None:
            p += abs(q2 - ref_q2) * 0.5
        if ref_q3 is not None:
            p += abs(q3 - ref_q3) * 1.5

        if abs(q3) < 3.0:
            p += 20.0

        return p

    def solve_planar_2link(self, r_target_local, z_target_local, ref_q2=None, ref_q3=None):
        L1 = self.cfg.link1_mm
        L2 = self.cfg.link2_mm

        rr = r_target_local
        zz = z_target_local

        D_raw = (rr * rr + zz * zz - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
        D = clamp(D_raw, -1.0, 1.0)

        theta3_candidates = [math.acos(D), -math.acos(D)]
        candidates: List[Dict[str, float]] = []

        for theta3 in theta3_candidates:
            k1 = L1 + L2 * math.cos(theta3)
            k2 = L2 * math.sin(theta3)
            theta2 = math.atan2(zz, rr) - math.atan2(k2, k1)

            t2_deg = math.degrees(theta2)
            t3_deg = math.degrees(theta3)

            q2, q3 = self.model_to_command_angles_deg(t2_deg, t3_deg)
            penalty = self._candidate_penalty(q2, q3, ref_q2=ref_q2, ref_q3=ref_q3)

            candidates.append({
                "q2": q2,
                "q3": q3,
                "t2_deg": t2_deg,
                "t3_deg": t3_deg,
                "D_raw": D_raw,
                "penalty": penalty,
            })

        chosen = min(candidates, key=lambda c: c["penalty"])
        q2 = clamp(chosen["q2"], *self.cfg.q_limits_deg["q2"])
        q3 = clamp(chosen["q3"], *self.cfg.q_limits_deg["q3"])
        return q2, q3, chosen

    def solve(self, x, y, z, ref_q2=None, ref_q3=None):
        q1 = math.degrees(math.atan2(y, x))
        q1 = clamp(q1, *self.cfg.q_limits_deg["q1"])

        far = self.get_far_compensation(x, y)
        x_eff = x + far["tcp_x_offset_mm"]
        y_eff = y + far["tcp_y_offset_mm"]
        z_eff = z + far["tcp_z_offset_mm"]

        wx, wy, wz = self.compute_wrist_target(x_eff, y_eff, z_eff)

        r_world = math.hypot(wx, wy)
        r_target_local = r_world - self.cfg.base_x_mm
        z_target_local = wz - self.cfg.base_z_mm

        q2, q3, meta = self.solve_planar_2link(
            r_target_local,
            z_target_local,
            ref_q2=ref_q2,
            ref_q3=ref_q3,
        )

        q5_raw = self.compute_q5_raw(q2, q3) + far["q5_offset_deg"]
        q5 = clamp(q5_raw, *self.cfg.q_limits_deg["q5"])

        result = {
            "q1": q1,
            "q2": q2,
            "q3": q3,
            "q4": self.cfg.q4_fixed_deg,
            "q5": q5,
        }

        fx, fy, fz = self.fk_tcp(q1, q2, q3)
        ex, ey, ez = x_eff - fx, y_eff - fy, z_eff - fz
        err = norm3(ex, ey, ez)

        if self.cfg.debug:
            print("\n[TCP IK START]")
            print(f"  target tcp xyz(mm) = ({x:.2f}, {y:.2f}, {z:.2f})")
            print(f"  effective target   = ({x_eff:.2f}, {y_eff:.2f}, {z_eff:.2f})")
            print(f"  wrist target xyz   = ({wx:.2f}, {wy:.2f}, {wz:.2f})")
            print(f"  r_local, z_local   = ({r_target_local:.2f}, {z_target_local:.2f})")
            print(f"  tcp_offset_mm      = {self.cfg.tcp_offset_mm:.2f}")
            print(
                f"  chosen t2={meta['t2_deg']:.2f} deg, "
                f"t3={meta['t3_deg']:.2f} deg, D_raw={meta['D_raw']:.4f}, "
                f"penalty={meta['penalty']:.2f}"
            )
            print(f"  far_alpha={far['alpha']:.3f}")
            print(
                f"  far offsets        = "
                f"(x={far['tcp_x_offset_mm']:.2f}, y={far['tcp_y_offset_mm']:.2f}, "
                f"z={far['tcp_z_offset_mm']:.2f}, q5={far['q5_offset_deg']:.2f})"
            )
            print(f"  q5_raw={q5_raw:.2f}, q5_clamped={q5:.2f}")
            print(
                f"  q5 linear model    = "
                f"{self.cfg.q5_a:.4f}*q2 + "
                f"{self.cfg.q5_b:.4f}*q3 + "
                f"{self.cfg.q5_c:.4f}"
            )
            print(f"[TCP IK RESULT] q = {result}")
            print(f"  fk_tcp    = ({fx:.2f}, {fy:.2f}, {fz:.2f})")
            print(f"  final_err = ({ex:.2f}, {ey:.2f}, {ez:.2f}) |norm|={err:.2f}")

        return result, err, meta

    def plan(self, x, y, z):
        final_q, final_err, final_meta = self.solve(x, y, z)

        used_offset = self.cfg.approach_offset_mm
        approach_q = None
        approach_err = None
        approach_xyz = None
        approach_meta = None

        offset = self.cfg.approach_offset_mm
        while offset >= self.cfg.min_approach_offset_mm:
            ax, ay, az = x, y, z + offset
            q_app, err_app, meta_app = self.solve(
                ax, ay, az,
                ref_q2=final_q["q2"],
                ref_q3=final_q["q3"],
            )

            q3_alive = abs(q_app["q3"]) > 3.0
            acceptable = (err_app <= 35.0 and q3_alive) or (err_app <= 20.0)

            if acceptable:
                used_offset = offset
                approach_q = q_app
                approach_err = err_app
                approach_xyz = (ax, ay, az)
                approach_meta = meta_app
                break

            offset -= self.cfg.approach_offset_step_mm

        if approach_q is None:
            ax, ay, az = x, y, z + self.cfg.min_approach_offset_mm
            q_app, err_app, meta_app = self.solve(
                ax, ay, az,
                ref_q2=final_q["q2"],
                ref_q3=final_q["q3"],
            )
            used_offset = self.cfg.min_approach_offset_mm
            approach_q = q_app
            approach_err = err_app
            approach_xyz = (ax, ay, az)
            approach_meta = meta_app

        if self.cfg.use_single_q5_for_approach_and_final:
            q5_parallel = self.choose_parallel_q5(
                final_q2=final_q["q2"],
                final_q3=final_q["q3"],
                approach_q2=approach_q["q2"],
                approach_q3=approach_q["q3"],
            )

            final_far = self.get_far_compensation(x, y)
            q5_parallel = clamp(
                q5_parallel + final_far["q5_offset_deg"],
                *self.cfg.q_limits_deg["q5"],
            )

            approach_q["q5"] = q5_parallel
            final_q["q5"] = q5_parallel
        else:
            final_far = self.get_far_compensation(x, y)

        if self.cfg.debug:
            print("\n[PLAN SUMMARY]")
            print(f"  final q            = {final_q}")
            print(f"  final err          = {final_err:.2f} mm")
            print(f"  approach q         = {approach_q}")
            print(f"  approach err       = {approach_err:.2f} mm")
            print(f"  used approach z+   = {used_offset:.2f} mm")
            print(f"  final t3_deg       = {final_meta['t3_deg']:.2f}")
            print(f"  approach t3_deg    = {approach_meta['t3_deg']:.2f}")
            if self.cfg.use_single_q5_for_approach_and_final:
                print(f"  shared parallel q5 = {final_q['q5']:.2f}")
            else:
                print(f"  approach q5        = {approach_q['q5']:.2f}")
                print(f"  final q5           = {final_q['q5']:.2f}")
            print(
                f"  far-comp(final)    = "
                f"alpha={final_far['alpha']:.3f}, "
                f"x={final_far['tcp_x_offset_mm']:.2f}, "
                f"y={final_far['tcp_y_offset_mm']:.2f}, "
                f"z={final_far['tcp_z_offset_mm']:.2f}, "
                f"q5={final_far['q5_offset_deg']:.2f}"
            )

        return {
            "approach": approach_q,
            "final": final_q,
            "approach_err": approach_err,
            "final_err": final_err,
            "approach_xyz": approach_xyz,
            "final_xyz": (x, y, z),
            "used_approach_offset_mm": used_offset,
        }
