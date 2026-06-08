import time
import math


class RobotExecutor:
    def __init__(self, arm, ctrl, approach_warn_mm=25.0, final_warn_mm=15.0):
        self.arm = arm
        self.ctrl = ctrl
        self.approach_warn_mm = approach_warn_mm
        self.final_warn_mm = final_warn_mm

    def move_joint(self, q):
        print(f"[MOVE] direct -> {q}")
        self.arm.move_joint_deg(q)

    def suction_on(self):
        self.ctrl.suction_on()

    def suction_off(self):
        self.ctrl.suction_off()

    def go_home(self):
        print("[HOME] 이동")
        self.arm.move_joint_deg(self.ctrl.poses.HOME)

    def go_zero(self):
        """
        사용자가 시작 시점에 저장한 zero 자세로 복귀.
        motor_zero_tick 기준이므로 각 관절 0도로 보내면 된다.
        """
        print("[ZERO] 저장된 zero 자세로 이동")
        q_zero = {
            "q1": 0.0,
            "q2": 0.0,
            "q3": 0.0,
            "q4": 0.0,
            "q5": 0.0,
        }
        self.arm.move_joint_deg(q_zero)

    def get_current_joint_deg(self):
        ticks = self.arm.read_present_ticks()
        return self.arm.mapper.present_tick_to_joint_deg(ticks)

    def _interp_pose_with_orientation_control(self, start_q, end_q, t, q5_fn=None, locked_q5=None):
        pose = {}
        for k in ["q1", "q2", "q3", "q4"]:
            pose[k] = start_q[k] + (end_q[k] - start_q[k]) * t

        if q5_fn is not None:
            pose["q5"] = q5_fn(pose["q2"], pose["q3"])
        elif locked_q5 is not None:
            pose["q5"] = locked_q5
        else:
            pose["q5"] = start_q["q5"] + (end_q["q5"] - start_q["q5"]) * t

        return pose

    def move_joint_path_orientation_aware(
        self,
        start_q,
        end_q,
        steps=30,
        step_dt=0.02,
        final_dwell=0.1,
        q5_fn=None,
        locked_q5=None,
    ):
        for step in range(1, steps + 1):
            t = step / steps
            pose = self._interp_pose_with_orientation_control(
                start_q=start_q,
                end_q=end_q,
                t=t,
                q5_fn=q5_fn,
                locked_q5=locked_q5,
            )
            print(f"[MOVE] orient step {step:02d}/{steps:02d} -> {pose}")
            self.arm.move_joint_deg(pose)
            time.sleep(step_dt)

        final_pose = dict(end_q)
        if q5_fn is not None:
            final_pose["q5"] = q5_fn(final_pose["q2"], final_pose["q3"])
        elif locked_q5 is not None:
            final_pose["q5"] = locked_q5

        self.arm.move_joint_deg(final_pose)
        time.sleep(final_dwell)

    def execute_soft_vertical_descent(
        self,
        planner,
        start_xyz,
        final_xyz,
        q_start,
        use_suction=False,
        retreat=False,
        z_step_mm=5.0,
        step_dt=0.05,
        final_dwell=0.1,
        q5_fn=None,
        max_allow_err_mm=20.0,
        q2_limit_margin_deg=2.0,
        q3_limit_margin_deg=2.0,
    ):
        """
        작업공간 끝에서 final target이 바로 안 될 때,
        현재 가능한 자세를 유지하면서 z 방향으로 조금씩만 내린다.

        start_xyz: (x, y, z)
        final_xyz: (x, y, z)
        q_start  : 시작 관절값 (보통 approach q)
        """
        x_hold, y_hold, z_start = start_xyz
        _, _, z_final = final_xyz

        print("\n[SOFT DESCENT]")
        print(f"  start_xyz = {start_xyz}")
        print(f"  final_xyz = {final_xyz}")
        print(f"  z_step_mm = {z_step_mm}")

        current_q = dict(q_start)

        if q5_fn is not None:
            current_q["q5"] = q5_fn(current_q["q2"], current_q["q3"])

        if z_step_mm <= 0:
            raise ValueError("z_step_mm must be positive")

        lo_q2, hi_q2 = self.arm.cfg.joint_limits_deg["q2"]
        lo_q3, hi_q3 = self.arm.cfg.joint_limits_deg["q3"]

        z_values = []
        z_now = z_start
        while z_now > z_final:
            z_now = max(z_now - z_step_mm, z_final)
            z_values.append(z_now)

        last_good_q = dict(current_q)
        last_good_xyz = start_xyz

        for i, z_try in enumerate(z_values, start=1):
            print(f"\n[SOFT DESCENT] step {i}/{len(z_values)} -> target z={z_try:.2f}")

            q_try, err_try, meta_try = planner.solve(
                x_hold,
                y_hold,
                z_try,
                ref_q2=current_q["q2"],
                ref_q3=current_q["q3"],
            )

            if q5_fn is not None:
                q_try["q5"] = q5_fn(q_try["q2"], q_try["q3"])

            print(f"  q_try   = {q_try}")
            print(f"  err_try = {err_try:.2f} mm")

            near_q2_limit = (q_try["q2"] >= hi_q2 - q2_limit_margin_deg) or (q_try["q2"] <= lo_q2 + q2_limit_margin_deg)
            near_q3_limit = (q_try["q3"] >= hi_q3 - q3_limit_margin_deg) or (q_try["q3"] <= lo_q3 + q3_limit_margin_deg)

            if err_try > max_allow_err_mm:
                print(f"  [STOP] err {err_try:.2f} > {max_allow_err_mm:.2f}")
                break

            if near_q2_limit or near_q3_limit:
                print(f"  [STOP] joint limit near: q2={q_try['q2']:.2f}, q3={q_try['q3']:.2f}")
                break

            self.move_joint_path_orientation_aware(
                start_q=current_q,
                end_q=q_try,
                steps=4,
                step_dt=step_dt,
                final_dwell=final_dwell,
                q5_fn=q5_fn,
                locked_q5=None if q5_fn is not None else q_try["q5"],
            )

            current_q = dict(q_try)
            last_good_q = dict(q_try)
            last_good_xyz = (x_hold, y_hold, z_try)

        if use_suction:
            print("[SOFT DESCENT] suction ON")
            self.suction_on()
            time.sleep(0.4)

        if retreat:
            print("[SOFT DESCENT] retreat")
            self.move_joint_path_orientation_aware(
                start_q=last_good_q,
                end_q=q_start,
                steps=12,
                step_dt=step_dt,
                final_dwell=0.1,
                q5_fn=q5_fn,
                locked_q5=None if q5_fn is not None else q_start["q5"],
            )

        return {
            "last_good_q": last_good_q,
            "last_good_xyz": last_good_xyz,
        }

    def execute_pick_fixed_orientation(
        self,
        plan,
        use_suction=False,
        retreat=False,
        descent_steps=30,
        step_dt=0.02,
        q5_fn=None,
    ):
        print("\n[PLAN CHECK]")
        print(f"  approach xyz = {plan['approach_xyz']}")
        print(f"  final xyz    = {plan['final_xyz']}")
        print(f"  approach err = {plan['approach_err']:.2f} mm")
        print(f"  final err    = {plan['final_err']:.2f} mm")
        print(f"  used approach offset = {plan.get('used_approach_offset_mm', 0.0):.2f} mm")

        if plan["approach_err"] > self.approach_warn_mm:
            print("[WARN] 접근점 오차가 큼. 그래도 계속 진행.")

        if plan["final_err"] > self.final_warn_mm:
            print("[WARN] 최종점 오차가 큼. TCP 보정 재확인 권장.")

        approach_q = dict(plan["approach"])
        final_q = dict(plan["final"])

        if q5_fn is not None:
            approach_q["q5"] = q5_fn(approach_q["q2"], approach_q["q3"])
            final_q["q5"] = q5_fn(final_q["q2"], final_q["q3"])
            print(f"[ORIENT] dynamic q5 enabled")
            print(f"[ORIENT] approach q5 = {approach_q['q5']:.2f}")
            print(f"[ORIENT] final q5    = {final_q['q5']:.2f}")
        else:
            locked_q5 = final_q["q5"]
            approach_q["q5"] = locked_q5
            print(f"[ORIENT] locked final q5 = {locked_q5:.2f}")

        print("\n[STEP 1] 접근")
        self.move_joint(approach_q)
        time.sleep(0.2)

        print("[STEP 2] 자세유지 하강")
        self.move_joint_path_orientation_aware(
            start_q=approach_q,
            end_q=final_q,
            steps=descent_steps,
            step_dt=step_dt,
            final_dwell=0.1,
            q5_fn=q5_fn,
            locked_q5=None if q5_fn is not None else final_q["q5"],
        )

        if use_suction:
            print("[STEP 3] 흡착 ON")
            self.suction_on()
            time.sleep(0.4)

        if retreat:
            print("[STEP 4] 자세유지 상승")
            self.move_joint_path_orientation_aware(
                start_q=final_q,
                end_q=approach_q,
                steps=descent_steps,
                step_dt=step_dt,
                final_dwell=0.1,
                q5_fn=q5_fn,
                locked_q5=None if q5_fn is not None else final_q["q5"],
            )
        else:
            print("[STEP 4] 상승 생략 -> 현재 최종 높이에서 측정 가능")

    def execute_pick_soft_vertical(
        self,
        planner,
        plan,
        use_suction=False,
        retreat=False,
        q5_fn=None,
        z_step_mm=5.0,
        step_dt=0.05,
        max_allow_err_mm=20.0,
    ):
        print("\n[PLAN CHECK - SOFT VERTICAL]")
        print(f"  approach xyz = {plan['approach_xyz']}")
        print(f"  final xyz    = {plan['final_xyz']}")
        print(f"  approach err = {plan['approach_err']:.2f} mm")
        print(f"  final err    = {plan['final_err']:.2f} mm")

        approach_q = dict(plan["approach"])

        if q5_fn is not None:
            approach_q["q5"] = q5_fn(approach_q["q2"], approach_q["q3"])

        print("\n[STEP 1] 접근")
        self.move_joint(approach_q)
        time.sleep(0.2)

        print("[STEP 2] 소프트 수직 하강")
        return self.execute_soft_vertical_descent(
            planner=planner,
            start_xyz=plan["approach_xyz"],
            final_xyz=plan["final_xyz"],
            q_start=approach_q,
            use_suction=use_suction,
            retreat=retreat,
            z_step_mm=z_step_mm,
            step_dt=step_dt,
            final_dwell=0.1,
            q5_fn=q5_fn,
            max_allow_err_mm=max_allow_err_mm,
        )

    def execute_pick(self, plan, use_suction=False, retreat=False, q5_fn=None):
        self.execute_pick_fixed_orientation(
            plan,
            use_suction=use_suction,
            retreat=retreat,
            descent_steps=30,
            step_dt=0.02,
            q5_fn=q5_fn,
        )

    def execute_place(self, q_place, use_suction_off=True):
        print("\n[PLACE] 이동")
        self.move_joint(q_place)
        time.sleep(0.3)

        if use_suction_off:
            print("[PLACE] 흡착 OFF")
            self.suction_off()
            time.sleep(0.3)
