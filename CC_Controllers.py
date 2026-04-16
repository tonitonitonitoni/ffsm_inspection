from __future__ import annotations
import pinocchio as pin
import numpy as np
from types import SimpleNamespace
from tqdm import trange
from utils.params import make_params
from utils.pure import cross, saturate, sanitize_column, sanitize_matrix, as_flat3, as_col3, safe_normalize, save_npz, matrix_from_xz
from utils.quaternions import quat_eps_eta
from utils.data_classes import Desired, Gains, State
from utils.plotter import Plotter3D
from utils.logger import CCLogger
from com_base_guidance import COM_Trajectory, BaseGuidance
from utils.robot import GiordanoRobot

class CC_Controller:
    def __init__(self, traj=None, cfg=None, debug=False,
                 log_controls=True, log_actuators=True, log_metrics=False,
                 diagnostics=None, plot3=False, enable_base=False, enable_ee=False):
        self.enable_base = enable_base
        self.enable_ee = enable_ee
        self.cfg = cfg if cfg is not None else make_params(vps=True, vision=True)
        self.cfg.enable_base = self.enable_base
        self.cfg.enable_ee = self.enable_ee
        self.robot = GiordanoRobot()
        self.debug = debug
        self.diagnostics = diagnostics
        self.m = self.robot.m
        self.g = Gains(
            K_c=self.cfg.K_c,
            D_c=self.cfg.D_c,
            I_c=self.cfg.I_c)
        self.traj = traj if traj is not None else COM_Trajectory(self.cfg)
        self.log_controls = log_controls
        self.log_actuators = log_actuators
        self.log_metrics = log_metrics
        self.windows = self._assemble_windows()
        self.add_title = None
        self.logger = self.make_logger()
        self.motion = None
        self.dyn = None
        self.forces = None
        self.step = None
        self.plot3=plot3


    def _assemble_windows(self):
        step = self.cfg.sample_freq
        n = len(self.traj.orbit.path)
        windows = []
        for i0 in range(0, n - 1, step):
            i1 = min(i0 + step, n - 1)
            if i1 <= i0:
                continue
            windows.append((i0, i1))
        return windows

    def _resolve_window_bounds(self, idx: int | None = None):
        if len(self.windows) == 0:
            raise ValueError("COM controller assembled no orbit windows")
        if idx is None:
            idx = 0
        if idx < 0 or idx >= len(self.windows):
            raise IndexError(f"window index {idx} out of range for {len(self.windows)} windows")
        return self.windows[idx]

    def make_logger(self):
        logger = CCLogger(**self.log())
        if self.diagnostics is not None:
            for d in self.diagnostics:
                logger.add_key(d)
            if "sigma_min_J_plus" in logger.logs:
                logger.logs["sigma_min_J_plus"].metrics = ["rms", "min"]
        return logger

    def log(self):
        
        enable_base = self.enable_base
        enable_ee = self.enable_ee
        return {
            "enable_base": enable_base,
            "enable_ee": enable_ee,
            "control": self.log_controls,
            "actuator": self.log_actuators,
            "metrics": self.log_metrics,
            "add_title": self.add_title,
        }

    def clamp_com_integral(self, x_int: np.ndarray) -> np.ndarray:
        if not self.cfg.use_com_integral:
            return np.zeros((3, 1))
        return saturate(np.asarray(x_int, dtype=float), self.cfg.com_integral_limit)

    def com_reference_startup_scale(self, t: float | None) -> float:
        tau_s = self.cfg.com_reference_startup_tau
        if t is None:
            return 1.0
        return 1.0 - np.exp(-max(t, 0.0) / tau_s)

    # ── State sync ───────────────────────────────────────────────────────────

    def update_motion(self, st):
        self.motion = self.robot.all_motion_terms(st.q, st.v)
        return self.motion

    def update_dynamics(self, st):
        self.dyn = self.robot.all_dynamics_terms(st.q, st.v)
        self.motion = self.robot.motion
        return self.dyn

    def sync_robot_state(self, st, update_views=False):
        self.update_motion(st)
        if update_views:
            st.update_views(self.robot)
        return st

    # ── Error terms (world-frame; require update_motion first) ───────────────

    def x_c_tilde(self, des: Desired):
        return self.motion.p_tc - des.p_c

    def x_c_tilde_dot(self, des: Desired):
        return self.motion.v_tc - des.v_c

    # ── Force terms ──────────────────────────────────────────────────────────

    def f_c_fb(self, des: Desired):
        return as_col3((-self.g.K_c @ self.x_c_tilde(des))
                       - (self.g.D_c @ self.x_c_tilde_dot(des)))

    def compute_f_c_terms(self, des: Desired,
                          x_c_int: np.ndarray | None = None,
                          t: float | None = None) -> dict:
        m = self.robot.m
        ff_scale = 1.0
        if t is not None and self.cfg.ff_startup_tau > 1e-9:
            ff_scale -= np.exp(-max(t, 0.0) / self.cfg.ff_startup_tau)
        f_ff = as_col3(saturate(ff_scale * self.cfg.ff_com_alpha * m * as_col3(des.a_c),
                                self.cfg.ff_force_max))
        f_fb = self.f_c_fb(des)
        if self.cfg.use_com_integral and x_c_int is not None:
            f_i = -(self.g.I_c @ self.clamp_com_integral(x_c_int))
        else:
            f_i = np.zeros((3, 1))
        f_unsat = f_ff + f_fb + f_i
        f_c = as_col3(saturate(f_unsat.copy(), self.cfg.F_max))
        return {"f_ff": f_ff, "f_fb": f_fb, "f_i": f_i, "f_unsat": f_unsat, "f_c": f_c}

    def compute_f_c(self, des: Desired,
                    x_c_int: np.ndarray | None = None,
                    t: float | None = None) -> np.ndarray:
        return self.compute_f_c_terms(des, x_c_int=x_c_int, t=t)["f_c"]

    # ── Task-space force placeholders (overridden by subclasses) ─────────────

    def compute_tau_b_oplus(self, des: Desired):
        return np.zeros((3, 1))

    def compute_omega_e_oplus(self, des: Desired):
        return np.zeros((6, 1))

    # ── All control terms  ─────────────────────────────────

    def all_control_terms(self, st: State, des: Desired,
                          x_c_int: np.ndarray | None = None,
                          t: float | None = None):
        self.update_dynamics(st)

        f_c_terms = self.compute_f_c_terms(des, x_c_int, t)
        f_c       = f_c_terms["f_c"]
        tau_b     = self.compute_tau_b_oplus(des)
        omega_e   = self.compute_omega_e_oplus(des)

        G = np.vstack([f_c, tau_b, omega_e])           # (12, 1) task-space generalized force
        F = self.dyn.Gamma.T @ G                       # (12, 1) actuator-space

        self.forces = SimpleNamespace(
            **f_c_terms,
            tau_b_oplus=tau_b,
            omega_e_oplus=omega_e,
            G=G,
            f_b=F[:3],
            tau_b=F[3:6],
            tau=F[6:].reshape(-1, 1),
        )
        self.step = SimpleNamespace(
            v_c_dot=as_col3(f_c / self.m),
        )
        return self.forces

    # ── State reconstruction and integration ─────────────────────────────────

    def reconstruct_generalized_velocity(self, st: State) -> np.ndarray:
        """Map task-space state (v_c, omega_b, nu_e_oplus) → full generalized velocity."""
        y = np.concatenate([st.v_c, st.omega_b, st.nu_e_oplus])
        Gamma = self.robot.Gamma(st.q)
        return np.linalg.lstsq(Gamma, y.reshape(-1), rcond=None)[0]

    def project_reduced_state(self, st: State):
        st.v = self.reconstruct_generalized_velocity(st)
        self.sync_robot_state(st, update_views=True)
        return st

    def _apply_step_state_update(self, st: State) -> State:
        dt = self.cfg.dt_ctrl
        st.v_c = as_col3(saturate(st.v_c + self.step.v_c_dot * dt, self.cfg.vc_max))
        return st

    # ── Conditioning Helpers ───────────────────────────────────────────────────
    def _dyn_sigma_min_gamma(self, Gamma: np.ndarray | None = None) -> float:
        if self.dyn is not None:
            return self.dyn.s_min_G
        if Gamma is None:
            raise AttributeError("Gamma is unavailable and self.dyn has not been initialized")
        return np.linalg.svd(Gamma, compute_uv=False)[-1]

    def _dyn_sigma_min_jplus(self, J_plus: np.ndarray | None = None) -> float:
        if self.dyn is not None:
            return self.dyn.s_min_J
        if J_plus is None:
            raise AttributeError("J_plus is unavailable and self.dyn has not been initialized")
        return np.linalg.svd(J_plus, compute_uv=False)[-1]
    
    # ── Initialization ────────────────────────────────────────────────────────

    def initialize_window_state(self, st: State, des: Desired):
        st.v_c = as_col3(des.v_c)
        st.omega_b = np.zeros((3, 1))
        st.nu_e_oplus = np.zeros((6, 1))
        return self.project_reduced_state(st)

    def initial_state_for_desired(self, des0: Desired):
        q_init = self.robot.choose_q_init_by_sigma().copy()
        self.robot.update(q_init)
        p_tc_init = as_col3(self.robot.data.com[0])
        q_init[:3] += as_flat3(des0.p_c - p_tc_init)
        st = State(q=q_init, v=np.zeros((self.robot.model.nv,)))
        return self.initialize_window_state(st, des0)

    # ── Loop internals ────────────────────────────────────────────────────────

    def _loop_sync_uses_update_views(self) -> bool:
        return True

    def _update_loop_views(self, st: State) -> None:
        return None
    
    def _sync_loop_state(self, st: State, update_views: bool = False) -> State:
        self.sync_robot_state(st, update_views=update_views)
        self._update_loop_views(st)
        return st

    def build_desired_for_step(
        self, start_idx: int, end_idx: int,
        t: float, des_prev: Desired | None = None,
        st=None, idx=None, step_idx=None, smooth=True
    ) -> tuple[Desired, dict]:
        dt = self.cfg.dt_ctrl
        p_tc_now = self.motion.p_tc
        if des_prev is None:
            desired_speed = self.cfg.desired_com_speed
        else:
            desired_speed = self.cfg.desired_com_speed * self.com_reference_startup_scale(t)
        des_raw = self.traj.desired_at_window(
            p_tc=p_tc_now, start_idx=start_idx, end_idx=end_idx,
            desired_speed=desired_speed)
        if smooth: des = self.traj.smooth_desired_com_reference(des_raw, des_prev, dt)
        else: des = des_raw
        return des, {}

    def _update_com_integral_state(
        self, des: Desired, x_c_int: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        dt = self.cfg.dt_ctrl
        x_c_tilde_now = self.x_c_tilde(des)
        if self.cfg.use_com_integral and np.linalg.norm(x_c_tilde_now) <= self.cfg.com_integral_disable_err:
            x_c_int = self.clamp_com_integral(x_c_int + x_c_tilde_now * dt)
        return x_c_int, x_c_tilde_now

    def _terminal_progress_reached(self, end_idx: int) -> bool:
        progress = self.traj.closest_progress.progress
        if progress is None:
            return False
        return progress >= end_idx - 1e-6

    def _actual_path_finished(self, end_idx: int, des: Desired | None = None) -> bool:
        p_goal_end = as_col3(self.traj.orbit_p_cd[end_idx])
        pos_err = np.linalg.norm(as_col3(self.motion.p_tc) - p_goal_end)
        vel_mag = np.linalg.norm(as_col3(self.motion.v_tc))
        des_vel_mag = np.inf if des is None else np.linalg.norm(as_col3(des.v_c))

        pos_tol = max(self.cfg.com_integral_disable_err, 5e-2)
        vel_tol = max(0.1 * self.cfg.desired_com_speed, 2e-2)

        return (
            self._terminal_progress_reached(end_idx)
            and pos_err <= pos_tol
            and vel_mag <= vel_tol
            and des_vel_mag <= vel_tol
        )

    def log_control_snapshot(
        self, st: State, des_log: Desired, t_log: float,
    ) -> None:
        logger_step = {
            "t": t_log,
            "p_c": (self.motion.p_tc.reshape(-1), des_log.p_c.reshape(-1)),
            "v_c": (self.motion.v_tc.reshape(-1), des_log.v_c.reshape(-1)),
            "q": np.asarray(st.q, dtype=float).copy(),
            "v": np.asarray(st.v, dtype=float).copy(),
        }
        if self.log_actuators:
            logger_step["f_b"] = self.forces.f_b
        if self.log_controls:
            logger_step["f_c"] = self.forces.f_c
        return logger_step

    def _finalize_control_loop_result(self, st: State):
        if self.plot3:
            plot_logs = SimpleNamespace(
                t=np.asarray(self.logger.logs["t"].finalize(norm=False), dtype=float).reshape(-1),
                p_actual=np.asarray(self.logger.logs["p_c"].data),
                p_desired=np.asarray(self.logger.logs["p_c"].data2)
            )
            return st, self.logger.finalize(), plot_logs
        return st, self.logger.finalize()

    # ── Control loop ──────────────────────────────────────────────────────────

    def run_control_loop(
        self, st: State, idx: int | None = None,
        start_idx: int | None = None, end_idx: int | None = None, T=None,
    ):
        if idx is None:
            idx = 0
        if start_idx is None or end_idx is None:
            start_idx, end_idx = self._resolve_window_bounds(idx)

        st = st.copy()
        cfg = self.cfg
        dt = self.cfg.dt_ctrl
        self.logger = self.make_logger()
        if hasattr(self, "nu_e_des"):
            self.nu_e_des = np.zeros((6, 1))
        if hasattr(self, "nu_e_des_prev"):
            self.nu_e_des_prev = np.zeros((6, 1))

        if st.v_c is None or st.omega_b is None or st.nu_e_oplus is None:
            self._sync_loop_state(st, update_views=True)
        else:
            self._sync_loop_state(st, update_views=self._loop_sync_uses_update_views())

        if T is None:
            T = cfg.T
        N = int(T / dt)
        iterator = trange(N)
        x_c_int = np.zeros((3, 1))
        des_prev = None
        terminal_hold_steps = max(1, int(np.ceil(0.15 / max(dt, 1e-9))))
        terminal_hold_count = 0

        for k in iterator:
            t = k * dt
            st.v = self.reconstruct_generalized_velocity(st)
            self._sync_loop_state(st, update_views=self._loop_sync_uses_update_views())

            des, step_ctx = self.build_desired_for_step(
                start_idx=start_idx, end_idx=end_idx,
                t=t, des_prev=des_prev,  st=st, idx=idx, step_idx=k)
            des_prev = des.copy()

            x_c_int, x_c_tilde_now = self._update_com_integral_state(des, x_c_int)

            self.all_control_terms(st, des, x_c_int=x_c_int, t=t)

            should_log = (k == 0) or (((k + 1) % cfg.log_stride) == 0) or (k == N - 1)
            if should_log:
                logger_step = self.log_control_snapshot(st, des_log=des, t_log=t)
                self.logger.log_step(logger_step)

            self._apply_step_state_update(st)

            st.v = self.reconstruct_generalized_velocity(st)
            st.q = pin.integrate(self.robot.model, st.q, st.v * dt)
            self._sync_loop_state(st, update_views=self._loop_sync_uses_update_views())

            vc_correction = self.cfg.vc_correction_alpha
            if vc_correction > 0.0:
                st.v_c = st.v_c + vc_correction * (as_col3(self.motion.v_tc) - st.v_c)

            if self.cfg.use_com_integral and np.linalg.norm(self.forces.f_unsat) > self.cfg.F_max:
                x_c_int = self.clamp_com_integral(x_c_int - x_c_tilde_now * dt)

            if self._actual_path_finished(end_idx, des=des):
                terminal_hold_count += 1
            else:
                terminal_hold_count = 0

            if terminal_hold_count >= terminal_hold_steps:
                des_stop, step_ctx_stop = self.build_desired_for_step(
                    start_idx=start_idx, end_idx=end_idx,
                    step_idx=k, st=st, idx=idx,
                    t=t + dt, des_prev=des_prev)
                logger_step = self.log_control_snapshot(
                    st, des_log=des_stop, t_log=t + dt)
                self.logger.log_step(logger_step)
                break

        return self._finalize_control_loop_result(st)

    def run_all(
        self,
        start_idx=0,
        end_idx=None,
        plot=True,
        printout=True,
        add_title=None,
        debug_time_limit: float | None = None,
        save_figs = False,
        show_figs = True,
    ):
        self.add_title = add_title
        self.traj.reset_progress()
        if hasattr(self.traj, "reset_runtime_state"): 
            self.traj.reset_runtime_state()

        n_windows = len(self.windows)
        if start_idx < 0 or start_idx >= n_windows:
            raise IndexError(f"start_idx {start_idx} out of range for {n_windows} windows")
        if end_idx is None:
            end_idx = n_windows
        if end_idx <= start_idx or end_idx > n_windows:
            raise IndexError(
                f"end_idx {end_idx} must satisfy {start_idx} < end_idx <= {n_windows}")

        active_windows = self.windows[start_idx:end_idx]
        if len(active_windows) == 0:
            raise ValueError("run_all requires at least one window")

        orbit_start_idx = active_windows[0][0]
        orbit_end_idx = active_windows[-1][1]
        desired_speed0 = self.cfg.desired_com_speed
        des0 = self.traj.desired_at_window_start(
            start_idx=orbit_start_idx, end_idx=orbit_end_idx, desired_speed=desired_speed0)
        st0 = self.initial_state_for_desired(des0)
        T_nominal = self.traj.orbit_duration(
            start_idx=orbit_start_idx, end_idx=orbit_end_idx)
        speed_floor = 1.0
        if self.cfg.use_gamma_speed_derating:
            speed_floor = min(speed_floor, self.cfg.gamma_speed_floor)
        speed_floor = max(speed_floor, 1e-3)
        startup_pad = 4.0 * self.cfg.com_reference_startup_tau
        T_total = (T_nominal / speed_floor) + startup_pad
        if debug_time_limit is not None:
            T_total = min(T_total, max(debug_time_limit, self.cfg.dt_ctrl))

        results = self.run_control_loop(
            st0, idx=start_idx,
            start_idx=orbit_start_idx, end_idx=orbit_end_idx, T=T_total)

        n_active = end_idx - start_idx
        if start_idx == 0 and end_idx == n_windows:
            title = "All windows"
        elif n_active == 1:
            title = f"Window {start_idx}"
        else:
            title = f"Windows {start_idx}:{end_idx}"
        if add_title is not None:
            title += add_title

        if printout:
            self.logger.summarize_metrics(title=title)
        if plot:
            self.logger.plot_by_kind(save=save_figs, show=show_figs, debug_time_limit=debug_time_limit)

        return results

class BaseController(CC_Controller):
    def __init__(self, **kwargs):
        kwargs["enable_base"] = True
        super().__init__(**kwargs)
        self.enable_base = True
        self.cfg.enable_base = True
        self.traj = BaseGuidance(self.cfg)
        self.g = Gains(
            K_c=self.cfg.K_c,
            D_c=self.cfg.D_c,
            I_c=self.cfg.I_c,
            K_b=self.cfg.K_b,
            D_b=self.cfg.D_b)

    # ── Sync ─────────────────────────────────────────────────────────────────

    def _loop_sync_uses_update_views(self) -> bool:
        return False
    
    def _update_loop_views(self, st: State) -> None:
        self._update_breve_views(st)

    def _sync_loop_state(self, st: State, update_views: bool = False) -> State:
        # Full dynamics pass so self.dyn.Gamma is available for _update_breve_views.
        self.update_dynamics(st)
        self._update_loop_views(st)
        return st

    def _update_breve_views(self, st: State):
        st.omega_b = as_col3(self.motion.omega_b)
        y = self.dyn.Gamma @ st.v.reshape(-1, 1)
        st.nu_e_oplus = y[6:12]

    # ── Desired building ──────────────────────────────────────────────────────

    def build_desired_for_step(
        self, start_idx: int, end_idx: int,
        t: float, des_prev: Desired | None = None,
        st=None, idx=None, step_idx=None
    ) -> tuple[Desired, dict]:
        
        des_com, log = super().build_desired_for_step(start_idx=start_idx, end_idx=end_idx, t=t, st=st, idx=idx, step_idx=step_idx, smooth=False)
        dt = self.cfg.dt_ctrl
        #Derate by gamma
        sigma_now = self._dyn_sigma_min_gamma()
        des_com = self.derate_desired_by_gamma(des_com, sigma_now)

        des_com = self.traj.smooth_desired_com_reference(des_com, des_prev, dt)

        des = self.traj.sample_base_goal_on_interval(des_com, start_idx=start_idx, end_idx=end_idx)
        #Derate by J+
        sigma_jplus_now = self._dyn_sigma_min_jplus()
        des = self.blend_base_goal_by_jplus(des, sigma_jplus_now)
        log["sigma_now"] = sigma_now
        log["sigma_jplus_now"] = sigma_jplus_now
        return des, log

    # ── Error terms ───────────────────────────────────────────────────────────

    def x_b_tilde(self, des: Desired):
        R_bb_d = self.motion.R_tb.T @ des.R_b
        eps_bb_d, _ = quat_eps_eta(R_bb_d)
        return 2.0 * eps_bb_d

    def x_b_tilde_dot(self, des: Desired):
        return self.J_xb(des) @ (self.motion.omega_b - des.omega_b)

    def J_xb(self, des: Desired):
        R_bb_d = self.motion.R_tb.T @ des.R_b
        eps_bb_d, eta_bb_d = quat_eps_eta(R_bb_d)
        return -eta_bb_d * np.eye(3) + cross(as_flat3(eps_bb_d))

    def x_tilde(self, des: Desired):
        return np.vstack([self.x_b_tilde(des), self.x_e_tilde(des)])

    def J_x(self, des: Desired):
        return np.block([
            [self.J_xb(des),  np.zeros((3, 6))],
            [np.zeros((6, 3)),    self.J_xe(des)],
        ])

    def v_breve_damping_error(self, des: Desired) -> np.ndarray:
        omega_b_err = self.motion.omega_b - des.omega_b
        return np.vstack([omega_b_err, self.motion.nu_e])

    # Placeholders for EE terms
    def x_e_tilde(self, des: Desired):
        return np.zeros((6, 1))

    def J_xe(self, des: Desired):
        return np.zeros((6, 6))

    def v_breve_damping_error(self, des: Desired) -> np.ndarray:
        """
        Velocity error used in the reduced-dynamics damping term.

        Base/COM controllers override the relevant pieces. The default
        implementation is a placeholder.
        """
        return np.zeros((9, 1))

    # ── Task-space force computation ──────────────────────────────────────────

    def compute_tau_b_oplus(self, des: Desired):
        J_xb = self.J_xb(des)
        omega_b_err = self.motion.omega_b - des.omega_b
        tau_b = -(J_xb.T @ self.g.K_b @ self.x_b_tilde(des)) - (self.g.D_b @ omega_b_err)
        jplus_scale = self.jplus_condition_scale(self._dyn_sigma_min_jplus())
        tau_b *= jplus_scale
        return as_col3(saturate(tau_b, self.cfg.tau_b_max))

    # ── Dynamics RHS and v_breve_dot ─────────────────────────────────────────

    def RHS34b(self, st: State, des: Desired) -> np.ndarray:
        g = self.g
        C_breve = sanitize_matrix(self.dyn.C_breve)
        C_c     = sanitize_matrix(self.dyn.C_c)

        arm_scale = 1.0
        if self.cfg.gamma_derate_arm_motion:
            sigma = self._dyn_sigma_min_gamma()
            arm_scale = self.gamma_condition_scale(sigma)

        D_breve = np.block([
            [g.D_b,             np.zeros((3, 6))],
            [np.zeros((6, 3)),  arm_scale * g.D_e],
        ])
        K_breve = np.block([
            [g.K_b,             np.zeros((3, 6))],
            [np.zeros((6, 3)),  arm_scale * g.K_e],
        ])

        v_breve  = sanitize_column(st.v_breve)
        v_breve_err = sanitize_column(self.v_breve_damping_error(des))
        J_x      = sanitize_matrix(self.J_x(des))
        x_tilde  = sanitize_column(self.x_tilde(des))
        G_vc_breve = sanitize_matrix(self.motion.G_vc_breve)

        rhs = -(C_breve @ v_breve
                + D_breve @ v_breve_err
                + J_x.T @ K_breve @ x_tilde
                + (C_c + D_breve @ G_vc_breve) @ self.x_c_tilde_dot(des))
        
        return sanitize_column(rhs)

    def v_breve_dot(self, st: State, des: Desired) -> np.ndarray:
        M_breve = sanitize_matrix(self.dyn.M_breve)
        rhs = self.RHS34b(st, des)
        try:
            vdot = np.linalg.solve(M_breve, rhs)
        except np.linalg.LinAlgError:
            vdot = np.linalg.pinv(M_breve) @ rhs
        return sanitize_column(vdot, 1e6)

    # ── All control terms ─────────────────────────────────────────────────────

    def all_control_terms(self, st: State, des: Desired,
                          x_c_int: np.ndarray | None = None,
                          t: float | None = None):
        # dynamics already current from _sync_loop_state; super() refreshes
        # them once more (same q, v → same result, harmless)
        super().all_control_terms(st, des, x_c_int, t)

        v_breve_dot = self.v_breve_dot(st, des)
        self.step.omega_b_dot = as_col3(saturate(v_breve_dot[:3], self.cfg.omega_b_dot_max))
        self.step.nu_e_oplus_dot = saturate(v_breve_dot[3:], self.cfg.nu_e_oplus_dot_max)
        return self.forces

    def _apply_step_state_update(self, st: State) -> State:
        dt = self.cfg.dt_ctrl
        super()._apply_step_state_update(st)
        st.omega_b = as_col3(
            saturate(st.omega_b + self.step.omega_b_dot * dt, self.cfg.omega_b_max))
        st.nu_e_oplus = saturate(st.nu_e_oplus + self.step.nu_e_oplus_dot * dt, self.cfg.nu_e_oplus_max).reshape(6, 1)

        return st

    # ── State reconstruction ──────────────────────────────────────────────────

    def reconstruct_generalized_velocity(self, st: State) -> np.ndarray:
        y = np.concatenate([st.v_c, st.omega_b, st.nu_e_oplus])
        Gamma = self.dyn.Gamma if self.dyn is not None else self.robot.Gamma(st.q)

        if self.cfg.use_regularization:
            lam = self.cfg.gamma_reg
            if self.cfg.gamma_reg_floor > 0.0:
                sigma_min = self._dyn_sigma_min_gamma(Gamma)
                lam = max(lam, self.cfg.gamma_reg_floor ** 2 - sigma_min ** 2)
            GT = Gamma.T
            v = np.linalg.solve(GT @ Gamma + lam * np.eye(Gamma.shape[1]), GT @ y)
            if self.cfg.clip_velocity:
                v = np.clip(v, -50.0, 50.0)
            return v
        else:
            return np.linalg.lstsq(Gamma, y.reshape(-1), rcond=None)[0]

    # ── Conditioning ──────────────────────────────────────────────────────────

    def gamma_condition_scale(self, sigma: float) -> float:
        if not self.cfg.use_gamma_speed_derating:
            return 1.0
        s_lo = self.cfg.gamma_sigma_low
        s_hi = self.cfg.gamma_sigma_high
        floor = self.cfg.gamma_speed_floor
        if sigma <= s_lo:
            return floor
        if sigma >= s_hi:
            return 1.0
        return floor + (1.0 - floor) * (sigma - s_lo) / max(s_hi - s_lo, 1e-12)

    def derate_desired_by_gamma(self, des: Desired, sigma: float) -> Desired:
        scale = self.gamma_condition_scale(sigma)
        des_scaled = des.copy()
        des_scaled.v_c *= scale
        des_scaled.a_c *= scale ** 2
        des_scaled.omega_b *= scale
        return des_scaled

    def jplus_condition_scale(self, sigma: float) -> float:
        if not self.cfg.use_jplus_base_derating:
            return 1.0
        s_lo = self.cfg.jplus_sigma_low
        s_hi = self.cfg.jplus_sigma_high
        floor = self.cfg.jplus_base_floor
        if sigma <= s_lo:
            return floor
        if sigma >= s_hi:
            return 1.0
        return floor + (1.0 - floor) * (sigma - s_lo) / max(s_hi - s_lo, 1e-12)

    def blend_base_goal_by_jplus(self, des: Desired, sigma_jplus: float) -> Desired:
        scale = self.jplus_condition_scale(sigma_jplus)
        if scale >= 1.0:
            return des.copy()
        des_blend = des.copy()
        R_current = np.asarray(self.motion.R_tb, dtype=float).reshape(3, 3)
        R_desired = np.asarray(des.R_b, dtype=float).reshape(3, 3)
        z_inward = safe_normalize(-as_flat3(des_blend.p_c))
        if np.linalg.norm(z_inward) <= 1e-9:
            z_inward = safe_normalize(R_desired[:, 2])
        x_hint = (1.0 - scale) * R_current[:, 0] + scale * R_desired[:, 0]
        des_blend.R_b = matrix_from_xz(x_hint, z_inward)
        des_blend.z_b = as_col3(z_inward)
        des_blend.omega_b = scale * as_col3(des.omega_b)
        return des_blend

    # ── Logging ───────────────────────────────────────────────────────────────

    def log_control_snapshot(self, st: State, des_log: Desired, t_log: float) -> None:
        logger_step = super().log_control_snapshot(st, des_log, t_log)

        logger_step["z_b"] = (np.asarray(self.motion.R_tb)[:, 2], np.asarray(des_log.z_b))
        logger_step["omega_b"] = (self.motion.omega_b.reshape(-1), des_log.omega_b.reshape(-1))


        if self.log_metrics:
            logger_step["sigma_min_gamma"]  = self._dyn_sigma_min_gamma()
            logger_step["sigma_min_J_plus"] = self._dyn_sigma_min_jplus()
        

        if self.log_actuators:
            logger_step["tau_b"]  = self.forces.tau_b
            logger_step["tau"]    = self.forces.tau

        if self.log_controls:
            logger_step["tau_b_oplus"] = self.forces.tau_b_oplus
        
        return logger_step
    
    def _finalize_control_loop_result(self, st: State):
        if self.plot3:
            plot_logs = SimpleNamespace(
                t=np.asarray(self.logger.logs["t"].finalize(norm=False), dtype=float).reshape(-1),
                p_actual=np.asarray(self.logger.logs["p_c"].data),
                p_desired=np.asarray(self.logger.logs["p_c"].data2),
                z_actual=np.asarray(self.logger.logs["z_b"].data),
                z_desired=np.asarray(self.logger.logs["z_b"].data2),
            )
            return st, self.logger.finalize(), plot_logs
        return st, self.logger.finalize()
    
class EEController(BaseController):
    def __init__(self, **kwargs):
        kwargs["enable_ee"] = True
        super().__init__(**kwargs)
        self.enable_ee = True
        self.cfg.enable_ee = True

        from target_finder import EEGuidance
        self.traj = EEGuidance(self.cfg, robot=self.robot, debug=self.debug) 
        self.g = Gains(
            K_c=self.cfg.K_c,   D_c=self.cfg.D_c,
            I_c = self.cfg.I_c,
            K_b=self.cfg.K_b,   D_b=self.cfg.D_b, 
            K_ep=self.cfg.K_ep, D_ep=self.cfg.D_ep,
            K_eo=self.cfg.K_eo, D_eo=self.cfg.D_eo)
        self.nu_e_des = np.zeros((6, 1))
        self.nu_e_des_prev = np.zeros((6, 1))
        
    def R_ee_d(self, des: Desired):
        R_te = self.motion.R_te
        if des.R_e is not None:
            R_ted = np.asarray(des.R_e, dtype=float).reshape(3, 3)
        else:
            R_ted = matrix_from_xz(R_te[:, 0], des.z_e)
        return R_te.T @ R_ted
    
    def x_e_tilde(self, des: Desired):
        R_te = self.motion.R_te
        p_te = self.motion.p_te
        p_ted = des.p_e
        p_ee_d = R_te.T @ (p_te - p_ted)

        eps_ee_d, _ = quat_eps_eta(self.R_ee_d(des))
        e_point = 2.0 * eps_ee_d

        return np.vstack([p_ee_d, e_point])  # (6,1)
    
    def nu_e_tilde(self) -> np.ndarray:
        """
        EE twist error in the current actual EE local frame.

        This is the controller-side moving-reference quantity used in the
        damping term. It is not the same object as the paper's
        `dot(x_e_tilde) = J_xe * nu_e`.
        """

        return self.motion.nu_e - self.nu_e_des

    def v_breve_damping_error(self, des: Desired) -> np.ndarray:
        omega_b_err = self.motion.omega_b - des.omega_b
        return np.vstack([omega_b_err, self.nu_e_tilde()])

    def x_e_tilde_dot(self, des: Desired):
        """
        Paper-consistent kinematic map from actual EE twist to the chosen EE
        error coordinates.

        For a moving desired pose, this should be read as the instantaneous
        coordinate-rate map induced by the actual twist `nu_e`; the controller's
        moving-reference damping proxy is `x_e_tilde_dot_proxy()`.
        """
        return (self.J_xe(des) @ self.motion.nu_e).reshape(6, 1)

    def x_e_tilde_dot_proxy(self, des: Desired):
        """
        Controller-side moving-reference rate proxy formed from the twist error
        `nu_e_tilde = nu_e - nu_e_des`.
        """
        return (self.J_xe(des) @ self.nu_e_tilde()).reshape(6, 1)
    
    def x_tilde_dot(self, des: Desired):
        return np.vstack([self.x_b_tilde_dot(des), self.x_e_tilde_dot(des)])

    def x_tilde_dot_27(self, st: State, des: Desired):
        J_x = self.J_x(des)
        G_vc_breve = self.motion.G_vc_breve 
        v_breve = st.v_breve 
        v_c = self.motion.v_c
        return (J_x @ v_breve) + (J_x @ G_vc_breve @ v_c)
        
    # ── Desired building ──────────────────────────────────────────────────────

    def build_desired_for_step(
        self, start_idx: int, end_idx: int,
        t: float, des_prev: Desired | None,
        st=None, idx=None, step_idx=None
    ) -> tuple[Desired, dict]:
        
        des_base, log = super().build_desired_for_step(
            start_idx=start_idx,
            end_idx=end_idx,
            t=t,
            st=st,
            idx=idx,
            step_idx=step_idx,
            des_prev=des_prev,
        )

        ee_goal_result = self.traj.add_ee_goal(
            st=st,
            des_base=des_base,
            idx=idx,
            step_idx=step_idx,
            des_prev=des_prev,
        )
        if isinstance(ee_goal_result, tuple):
            des, ee_log = ee_goal_result
            log.update(ee_log)
        else:
            des = ee_goal_result

        if st is not None:
            self.nu_e_des = self.traj.desired_nu_e_oplus(
                st=st,
                des=des,
                des_prev=des_prev,
                dt=self.cfg.dt_ctrl,
                nu_e_des_prev=self.nu_e_des_prev,
            )
            self.nu_e_des_prev = self.nu_e_des.copy()
        else:
            self.nu_e_des = np.zeros((6, 1))
        
        return des, log
    
    # Jacobians
    def J_xe(self, des: Desired):
        J_11 = np.eye(3)
        J_22 = np.zeros((3, 3))

        R_ee_d = self.R_ee_d(des)
        eps_ee_d, eta_ee_d = quat_eps_eta(R_ee_d)
        J_22 = -eta_ee_d*np.eye(3) + cross(eps_ee_d)

        return np.block([
            [J_11, np.zeros((3, 3))],
            [np.zeros((3, 3)), J_22]
        ])

    def J_x(self, des: Desired):
        J_xb = self.J_xb(des)
        J_xe = self.J_xe(des)
        return np.block([
            [J_xb,             np.zeros((3, 6))],
            [np.zeros((6, 3)), J_xe]
        ])
    
    def compute_omega_e_oplus(self, des: Desired):
        x_e_tilde = self.x_e_tilde(des)
        g = self.g
        J_xe = self.J_xe(des)
        nu_e_tilde = self.nu_e_tilde()
        omega_e = -(J_xe.T @ g.K_e @ x_e_tilde) - (g.D_e @ nu_e_tilde)
        return omega_e

    def log_control_snapshot(self, st: State, des_log: Desired, t_log: float) -> None:
        logger_step = super().log_control_snapshot(st, des_log, t_log)
        logger_step.update({
            "p_e":      (self.motion.p_te.reshape(-1), des_log.p_e.reshape(-1)),
            "z_e":      (self.motion.z_te.reshape(-1), des_log.z_e.reshape(-1)),
        })

        if self.log_controls:
            logger_step["omega_e_oplus"] = self.forces.omega_e_oplus

        return logger_step
    
    def _finalize_control_loop_result(self, st: State):
        if self.plot3:
            plot_logs = SimpleNamespace(
                t=np.asarray(self.logger.logs["t"].finalize(norm=False), dtype=float).reshape(-1),
                p_actual=np.asarray(self.logger.logs["p_c"].data),
                p_desired=np.asarray(self.logger.logs["p_c"].data2),
                z_actual=np.asarray(self.logger.logs["z_b"].data),
                z_desired=np.asarray(self.logger.logs["z_b"].data2),
                p_e_actual=np.asarray(self.logger.logs["p_e"].data),
                p_e_desired=np.asarray(self.logger.logs["p_e"].data2),
                z_e_actual=np.asarray(self.logger.logs["z_e"].data),
                z_e_desired=np.asarray(self.logger.logs["z_e"].data2),
            )
            return st, self.logger.finalize(), plot_logs
        return st, self.logger.finalize()
   
if __name__ == "__main__":
    """ Use this file to run any of the controllers on any of the satellites.
    base is enabled by default when you run enable_ee.
    Turn on plot3 for 3D plots of the orbit (no central mesh)
    Turn on save_states to save states for later simulation."""
    enable_base = 0
    enable_ee = 1
    plot3 = 0
    plot_metrics = 1
    save_states = 0
    model = "GRO" # or "RCM", "acrim"

    if plot3: plotter = Plotter3D()
    
    diagnostics = ["sigma_min_J_plus"]
    cfg = make_params(model_key=model)
    ctrl_kwargs = dict(
        diagnostics=diagnostics,
        log_metrics=True,
        log_controls=True,
        log_actuators=True,
        plot3=plot3,
        cfg=cfg,
        )

    if enable_ee:
        ctrl = EEController(**ctrl_kwargs)
        if plot3: plot = plotter.plot_combined
    elif enable_base:
        ctrl = BaseController(**ctrl_kwargs)
        if plot3: plot = plotter.plot_base_com
    else:
        ctrl = CC_Controller(**ctrl_kwargs)
        if plot3: plot = plotter.plot_com

    results = ctrl.run_all(end_idx=None,
                           plot=plot_metrics,
                           debug_time_limit=100,
                           save_figs=True,
                           add_title=" RESTORED",
                           )
    #Return states and save for simulation and reconstruction
    if save_states:
        states = ctrl.logger.return_states()
        stem = f"states_{model}"
        save_npz(stem, add=True, states=states)

    if plot3:
        _, _, plot_logs = results
        plot(plot_logs)
