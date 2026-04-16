from __future__ import annotations
from typing import Any
from utils.data_classes import Desired, ProgressState, OrbitWindow
import numpy as np
from utils.pure import safe_normalize, saturate, as_flat3, as_col3, cross, vee, so3_log
from utils.curvature import *
from utils.orbit import OrbitGenerator

class COM_Trajectory:
    def __init__(self, cfg):
        self.cfg = cfg
        self.closest_progress = ProgressState(i0=0)
        self.orbit = OrbitGenerator(self.cfg)
        self.orbit_p_cd = np.asarray(self.orbit.path, dtype=float)
        self.orbit_v_cd = safe_normalize(np.asarray(self.orbit.tangents, dtype=float))
        self.orbit_kappa_cd = segment_curvature_vectors(
            np.asarray(self.orbit.curvature, dtype=float),
            len(self.orbit_p_cd),
        )
        self.orbit_s_cd = self._build_arclength_samples(self.orbit_p_cd)
        self.closest_idx = 0

    def _build_arclength_samples(self, p_cd: np.ndarray) -> np.ndarray:
        p_cd = np.asarray(p_cd, dtype=float)
        if len(p_cd) <= 1:
            return np.zeros(len(p_cd))
        ds = np.linalg.norm(np.diff(p_cd, axis=0), axis=1)
        return np.concatenate([[0.0], np.cumsum(ds)])

    def smooth_desired_com_reference(self, des_raw: Desired, prev_des: Desired | None, dt: float) -> Desired:
        """
        Low-pass the desired CoM velocity so the blended vector field does not
        inject sharp reference-velocity changes as the nearest-point index moves
        across polyline samples. Feedforward acceleration is rebuilt from the
        filtered velocity for dynamic consistency.
        """
        tau = self.cfg.desired_vc_smoothing_tau
        if tau <= 1e-9 or prev_des is None:
            return des_raw

        alpha = np.clip(dt / max(tau, dt), 0.0, 1.0)
        v_prev = as_col3(prev_des.v_c)
        v_raw = as_col3(des_raw.v_c)
        v_smooth = (1.0 - alpha) * v_prev + alpha * v_raw
        v_smooth = as_col3(saturate(v_smooth, self.cfg.vc_max))

        a_smooth = (v_smooth - v_prev) / max(dt, 1e-9)
        a_smooth = as_col3(saturate(a_smooth, self.cfg.vc_dot_max))

        return Desired(
            p_c=des_raw.p_c,
            v_c=v_smooth,
            a_c=a_smooth,
            R_b=des_raw.R_b,
            omega_b=des_raw.omega_b,
            p_e=des_raw.p_e,
            z_e=des_raw.z_e,
            R_e=des_raw.R_e,
        )
  
    def reset_progress(self, idx=0) -> None:
        if idx is None: idx=0
        self.closest_progress = ProgressState(i0=idx)
        self.closest_idx = idx

    def _clamp_window_bounds(
        self,
        start_idx: int = 0,
        end_idx: int | None = None,
    ) -> tuple[int, int]:
        n = len(self.orbit_p_cd)
        if n == 0:
            raise ValueError("orbit has no CoM samples")
        start_idx = np.clip(start_idx, 0, n - 1)
        if end_idx is None:
            end_idx = n - 1
        end_idx = np.clip(end_idx, start_idx, n - 1)
        return start_idx, end_idx

    def orbit_duration(
        self,
        start_idx: int = 0,
        end_idx: int | None = None,
    ) -> float:
        start_idx, end_idx = self._clamp_window_bounds(start_idx=start_idx, end_idx=end_idx)
        s_cd = self.orbit_s_cd[start_idx:end_idx + 1].reshape(-1)
        if s_cd.size <= 1:
            return 0.0

        kappa_cd = segment_curvature_samples(
            self.orbit_kappa_cd[start_idx:end_idx + 1],
            len(s_cd),
        )
        kappa_safe = np.maximum(kappa_cd, 0.0)
        v_mag_cd = np.full_like(s_cd, float(self.cfg.desired_com_speed))

        curved = kappa_safe > 1e-9
        v_mag_cd[curved] = np.minimum(
            v_mag_cd[curved],
            np.sqrt(self.cfg.ac_ff_max / kappa_safe[curved]),
        )
        v_mag_cd = np.maximum(v_mag_cd, 1e-6)

        ds = np.diff(s_cd)
        v_avg = 0.5 * (v_mag_cd[:-1] + v_mag_cd[1:])
        return np.sum(ds / np.maximum(v_avg, 1e-6))

    def desired_at_window_start(
        self,
        start_idx: int = 0,
        end_idx: int | None = None,
        desired_speed: float | None = None,
    ) -> Desired:
        start_idx, end_idx = self._clamp_window_bounds(start_idx=start_idx, end_idx=end_idx)
        p_start = as_flat3(self.orbit_p_cd[start_idx])
        return self.desired_at_window(
            p_tc=p_start,
            start_idx=start_idx,
            end_idx=end_idx,
            desired_speed=desired_speed,
        )

    def _closest_point_on_segment_polyline(self, p_cd: np.ndarray, p_tc: np.ndarray):
        """
        Find the closest point on the polyline defined by p_cd to the query point p_tc.

        Returns
        -------
        i0 : int
            Start index of the closest polyline segment.
        i1 : int
            End index of the closest polyline segment.
        alpha : float
            Interpolation parameter in [0, 1] along segment i0 -> i1.
        p_near : ndarray, shape (3,)
            Closest point on the polyline.
        """
        p_cd = np.asarray(p_cd, dtype=float)
        p_tc = as_flat3(p_tc)

        if len(p_cd) == 0:
            raise ValueError("polyline has no samples")
        if len(p_cd) == 1:
            return 0, 0, 0.0, p_cd[0].copy()

        best_dist2 = np.inf
        best_i0 = 0
        best_i1 = 1
        best_alpha = 0.0
        best_p = p_cd[0].copy()

        for i in range(len(p_cd) - 1):
            p0 = p_cd[i]
            p1 = p_cd[i + 1]
            d = p1 - p0
            denom = np.dot(d, d)

            if denom <= 1e-12:
                alpha = 0.0
                p_proj = p0
            else:
                alpha = np.dot(p_tc - p0, d) / denom
                alpha = np.clip(alpha, 0.0, 1.0)
                p_proj = p0 + alpha * d

            dist2 = np.dot(p_tc - p_proj, p_tc - p_proj)
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_i0 = i
                best_i1 = i + 1
                best_alpha = alpha
                best_p = p_proj.copy()

        return best_i0, best_i1, best_alpha, best_p
    
    def _sample_polyline_fields(
        self,
        p_cd: np.ndarray,
        v_cd: np.ndarray,
        kappa_vec_cd: np.ndarray,
        progress: float | None = None,
    ):
        if progress is None: progress = self.closest_progress.progress
        """
        Interpolate path position and associated fields at a continuous polyline
        progress value measured in sample indices. Tangent is derived from the
        polyline geometry so the position-based guidance law is not sensitive to
        inconsistent stored v_cd samples at segment boundaries.
        """
        n = len(p_cd)
        if n == 0:
            raise ValueError("polyline has no CoM samples")
        if n == 1:
            return (
                as_flat3(p_cd[0]),
                as_flat3(v_cd[0]),
                as_flat3(kappa_vec_cd[0]),
            )

        u = np.clip(progress, 0.0, n - 1)
        i0 = int(np.floor(u))
        i1 = min(i0 + 1, n - 1)
        alpha = u - i0

        p = (1.0 - alpha) * as_flat3(p_cd[i0]) + alpha * as_flat3(p_cd[i1])
        v = path_tangent_from_positions(p_cd, i0, i1, alpha)
        kappa = smooth_vector_field(kappa_vec_cd, i0, i1, alpha, radius=1)

        return p, v, kappa
    
    def desired_at_window(
        self,
        p_tc,
        start_idx: int = 0,
        end_idx: int | None = None,
        desired_speed: float | None = None,
        progress_state: ProgressState | None = None,
    ) -> Desired:
        if progress_state is None: progress_state = self.closest_progress
        """
        Progress-driven guidance on a bounded interval of the global orbit.
        Stored progress remains in global sample coordinates.
        """
        k_track = self.cfg.k_track
        k_progress = self.cfg.k_progress
        p_tc = as_flat3(p_tc)
        start_idx, end_idx = self._clamp_window_bounds(start_idx=start_idx, end_idx=end_idx)
        p_window = self.orbit_p_cd[start_idx:end_idx + 1]

        if desired_speed is None:
            desired_speed = float(self.cfg.desired_com_speed)

        if len(p_window) == 0:
            raise ValueError("orbit window has no CoM samples")

        ds_nom = np.linalg.norm(np.diff(p_window, axis=0), axis=1)
        ds_ref = np.median(ds_nom) if ds_nom.size > 0 else 0.0
        dt_ctrl = self.cfg.dt_ctrl
        progress_min = float(start_idx)
        progress_max = float(end_idx)

        if progress_state is not None and progress_state.progress is not None:
            progress_center = np.clip(progress_state.progress, progress_min, progress_max)
            local_radius = max(2, self.cfg.search_window)
            local_start = int(np.clip(np.floor(progress_center) - local_radius, start_idx, end_idx))
            local_end = int(np.clip(np.floor(progress_center) + local_radius + 1, local_start + 1, end_idx))
            p_local = self.orbit_p_cd[local_start:local_end + 1]
            i0_local, i1_local, alpha_local, p_near = self._closest_point_on_segment_polyline(
                p_local,
                p_tc,
            )
            progress_geom = local_start + i0_local + alpha_local
            progress_prev = np.clip(progress_state.progress, progress_min, progress_max)
        else:
            i0_init, i1_init, alpha_init, p_near = self._closest_point_on_segment_polyline(
                p_window,
                p_tc,
            )
            progress_geom = start_idx + i0_init + alpha_init
            progress_prev = progress_geom

        progress_geom = np.clip(progress_geom, progress_min, progress_max)
        progress_prev = np.clip(progress_prev, progress_min, progress_max)
        progress_phase = max(progress_prev, progress_geom)

        # `progress_geom` is the local geometric closest progress used for the
        # position error. `progress_phase` is the nondecreasing phase used for
        # lookahead/base guidance so the stored progress stays monotone.
        lookahead_time = self.cfg.lookahead_time
        if ds_ref > 1e-9 and lookahead_time > 0.0:
            lookahead_progress = desired_speed * lookahead_time / ds_ref
        else:
            lookahead_progress = 0.0
        ref_progress = np.clip(progress_phase + lookahead_progress, progress_min, progress_max)

        p_track, _t_track_raw, _ = self._sample_polyline_fields(
            p_cd=self.orbit_p_cd,
            v_cd=self.orbit_v_cd,
            kappa_vec_cd=self.orbit_kappa_cd,
            progress=progress_phase,
        )
        _p_ref, t_raw, kappa_vec = self._sample_polyline_fields(
            p_cd=self.orbit_p_cd,
            v_cd=self.orbit_v_cd,
            kappa_vec_cd=self.orbit_kappa_cd,
            progress=ref_progress,
        )

        t_hat = safe_normalize(t_raw)
        e_track = p_track - p_tc
        e_along = np.dot(e_track, t_hat)
        e_lateral = e_track - e_along * t_hat

        d = t_hat + k_track * e_lateral
        d_norm = np.linalg.norm(d)
        p_goal_end = as_flat3(self.orbit_p_cd[end_idx])
        dist_to_end = np.linalg.norm(p_goal_end - p_track)
        if d_norm < 1e-9:
            v_tcd = np.zeros(3)
        else:
            # If the vehicle lags behind the current progress point
            # (e_along > 0), slow the phase down so the reference does not run
            # away. If it gets ahead, speed the phase up slightly.
            v_cmd = desired_speed - k_progress * e_along
            v_cmd = np.clip(v_cmd, 0.0, float(self.cfg.vc_max))
            # Force the reference to come to rest at the terminal point instead
            # of carrying a finite tangent velocity into the end sample.
            v_cmd = min(v_cmd, dist_to_end / max(dt_ctrl, 1e-9))
            v_tcd = v_cmd * (d / d_norm)

        v_mag = np.linalg.norm(v_tcd)
        a_tcd = (v_mag ** 2) * as_flat3(kappa_vec)

        if progress_state is not None:
            i0_now = int(np.floor(progress_phase))
            i1_now = min(i0_now + 1, len(self.orbit_p_cd) - 1)
            alpha_now = progress_phase - i0_now

            progress_state.progress = progress_phase
            progress_state.raw_progress = progress_geom
            progress_state.i0 = i0_now
            progress_state.i1 = i1_now
            progress_state.alpha = alpha_now
            progress_state.p_near = p_near.copy()

        self.closest_idx = int(np.floor(progress_phase))

        return self._make_com_desired(
            p_c=p_track,
            v_c=v_tcd,
            a_c=a_tcd,
        )

    def _make_com_desired(
        self,
        p_c,
        v_c,
        a_c,
    ) -> Desired:
        return Desired(
            p_c=as_flat3(p_c),
            v_c=as_flat3(v_c),
            a_c=as_flat3(a_c),
            R_b=np.eye(3),
            omega_b=np.zeros(3),
            p_e=np.zeros(3),
            z_e=np.array([0.0, 0.0, 1.0]),
            R_e=None,
        )

class BaseGuidance(COM_Trajectory):
    """
    Combined COM + base-attitude guidance using the window formulation.

    Inherits the full COM trajectory (orbit, progress tracking, desired_at_window)
    and adds window management and interpolated base-attitude references from
    a precomputed inward-pointing frame sequence along the orbit.
    """
    def __init__(self, cfg, debug=False):
        super().__init__(cfg)
        self.debug = debug
        self.orbit_z_bd = self._assemble_z_axis_reference()
        self.windows = self._assemble_windows()
        self._window_context: list[OrbitWindow] | None = None

    # ── window management ──────────────────────────────────────────────

    def _assemble_windows(self) -> list[OrbitWindow]:
        step = self.cfg.sample_freq
        n = len(self.orbit_p_cd)
        windows: list[OrbitWindow] = []
        for i0 in range(0, n - 1, step):
            i1 = min(i0 + step, n - 1)
            if i1 <= i0:
                continue
            windows.append(OrbitWindow(i0=i0, i1=i1))
        return windows

    def set_window_context(self, windows: list[OrbitWindow] | None = None) -> None:
        self._window_context = None if windows is None else list(windows)

    def window_source(self) -> list[OrbitWindow]:
        return self.windows if self._window_context is None else self._window_context

    def _resolve_window(self, idx: int | None = None) -> OrbitWindow:
        windows = self.window_source()
        if len(windows) == 0:
            raise ValueError("BaseGuidance assembled no orbit windows")
        if idx is None:
            idx = 0
        if idx < 0 or idx >= len(windows):
            raise IndexError(f"window index {idx} out of range for {len(windows)} windows")
        return windows[idx]

    def window_bounds(
        self,
        idx: int | None = None,
        window: OrbitWindow | None = None,
    ) -> tuple[int, int]:
        if window is None:
            window = self._resolve_window(idx=idx)
        start_idx = window.i0
        if self._window_context is None:
            end_idx = window.i1
        else:
            end_idx = self.window_source()[-1].i1
        return start_idx, end_idx

    def window_duration(
        self,
        idx: int | None = None,
        window: OrbitWindow | None = None,
    ) -> float:
        start_idx, end_idx = self.window_bounds(idx=idx, window=window)
        return self.orbit_duration(start_idx=start_idx, end_idx=end_idx)

    # ── COM sampling through windows ───────────────────────────────────

    def desired_com_at_window_start(self, idx: int | None = None) -> Desired:
        start_idx, end_idx = self.window_bounds(idx=idx)
        return self.desired_at_window_start(
            start_idx=start_idx,
            end_idx=end_idx,
        )

    def sample_com_goal(
        self,
        p_tc,
        idx: int | None = None,
        desired_speed: float | None = None,
        progress_state=None,
    ) -> Desired:
        start_idx, end_idx = self.window_bounds(idx=idx)
        return self.desired_at_window(
            p_tc=p_tc,
            start_idx=start_idx,
            end_idx=end_idx,
            desired_speed=desired_speed,
            progress_state=progress_state,
        )

    def sample_progress_anchor(self, progress: float, idx: int | None = None) -> np.ndarray:
        fields = self.guidance_fields(idx=idx)
        progress = np.clip(progress, fields["start_idx"], fields["end_idx"])
        p_anchor, _, _ = self._sample_polyline_fields(
            p_cd=fields["p_orbit"],
            v_cd=fields["v_orbit"],
            kappa_vec_cd=fields["k_orbit"],
            progress=progress,
        )
        return as_flat3(p_anchor)

    def guidance_fields(self, idx: int | None = None) -> dict[str, Any]:
        start_idx, end_idx = self.window_bounds(idx=idx)
        return {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "p_orbit": self.orbit_p_cd,
            "v_orbit": self.orbit_v_cd,
            "k_orbit": self.orbit_kappa_cd,
            "z_bd_orbit": list(self.orbit_z_bd),
            "p_window": self.orbit_p_cd[start_idx:end_idx + 1],
        }

    def _current_progress(self) -> float:
        progress_state = self.closest_progress
        if progress_state.progress is not None:
            return progress_state.progress
        return progress_state.i0 + (0.0 if progress_state.alpha is None else progress_state.alpha)

    # ── base attitude helpers ──────────────────────────────────────────

    def angular_velocity_target(self, st, R: np.ndarray):
        if st.omega_b is None:
            return np.zeros((3, 1))
        omega_b = as_flat3(st.omega_b)
        R_dot = R @ cross(omega_b)
        Omega_bd = 0.5 * (R.T @ R_dot - R_dot.T @ R)
        return as_col3(vee(Omega_bd))

    def _frame_from_z_axis(self, z_axis, x_hint=None) -> np.ndarray:
        z_axis = safe_normalize(as_flat3(z_axis))
        if x_hint is None:
            x_hint = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(x_hint, z_axis)) > 0.8:
                x_hint = np.array([0.0, 1.0, 0.0])
        x_hint = as_flat3(x_hint)

        x_proj = x_hint - np.dot(x_hint, z_axis) * z_axis
        if np.linalg.norm(x_proj) < 1e-9:
            alt = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(alt, z_axis)) > 0.8:
                alt = np.array([0.0, 1.0, 0.0])
            x_proj = alt - np.dot(alt, z_axis) * z_axis

        x_axis = safe_normalize(x_proj)
        y_axis = np.cross(z_axis, x_axis)
        x_axis = np.cross(y_axis, z_axis)
        return np.column_stack((x_axis, y_axis, z_axis))

    def _rotation_aligning_vectors(self, a, b) -> np.ndarray:
        a = safe_normalize(as_flat3(a))
        b = safe_normalize(as_flat3(b))
        c = np.clip(np.dot(a, b), -1.0, 1.0)

        if c >= 1.0 - 1e-9:
            return np.eye(3)

        if c <= -1.0 + 1e-9:
            axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
            if np.linalg.norm(axis) < 1e-9:
                axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
            axis = safe_normalize(axis)
            K = cross(axis)
            return np.eye(3) + 2.0 * (K @ K)

        v = np.cross(a, b)
        s = np.linalg.norm(v)
        K = cross(v)
        return np.eye(3) + K + ((1.0 - c) / max(s * s, 1e-12)) * (K @ K)

    # ── z-axis reference assembly ──────────────────────────────────────

    def _assemble_z_axis_reference(self) -> list[np.ndarray]:
        z_seq = [safe_normalize(-p) for p in self.orbit_p_cd]
        if len(z_seq) == 0:
            return []

        frames = [self._frame_from_z_axis(z_seq[0])]
        for z_next in z_seq[1:]:
            R_prev = frames[-1]
            R_align = self._rotation_aligning_vectors(R_prev[:, 2], z_next)
            R_next = R_align @ R_prev
            frames.append(self._frame_from_z_axis(z_next, x_hint=R_next[:, 0]))

        return enforce_frame_sequence_continuity(frames)

    def _sample_z_bd_fields(
        self,
        z_bd,
        progress: float,
    ) -> tuple[np.ndarray, int, int, float]:
        n = len(z_bd)
        if n == 0:
            raise ValueError("z_bd field has no samples")
        if n == 1:
            return np.reshape(z_bd[0], (3, 3)), 0, 0, 0.0

        u = np.clip(progress, 0.0, n - 1)
        i0 = int(np.floor(u))
        i1 = min(i0 + 1, n - 1)
        alpha = u - i0

        if i0 == i1:
            R = np.reshape(z_bd[i0], (3, 3))
        else:
            R = interpolate_rotation(z_bd[i0], z_bd[i1], alpha)
        return R, i0, i1, alpha

    # ── base attitude sampling ─────────────────────────────────────────
    
    def sample_base_goal(self, des: Desired, idx: int | None = None) -> Desired:
        fields = self.guidance_fields(idx=idx)
        return self.sample_base_goal_on_interval(
            des,
            start_idx=fields["start_idx"],
            end_idx=fields["end_idx"],
        )

    def sample_base_goal_on_interval(
        self,
        des: Desired,
        start_idx: int,
        end_idx: int,
    ) -> Desired:
        start_idx, end_idx = self._clamp_window_bounds(start_idx=start_idx, end_idx=end_idx)
        progress = np.clip(self._current_progress(), start_idx, end_idx)
        R_hint, i0, i1, _alpha = self._sample_z_bd_fields(self.orbit_z_bd, progress)

        p_window = self.orbit_p_cd[start_idx:end_idx + 1]
        ds_nom = np.linalg.norm(np.diff(p_window, axis=0), axis=1)
        ds_ref = np.median(ds_nom) if ds_nom.size > 0 else 0.0
        speed = np.linalg.norm(as_flat3(des.v_c))
        progress_rate = 0.0 if ds_ref <= 1e-9 else speed / ds_ref

        des_new = des.copy()
        # Keep the desired base z-axis strictly inward and only borrow the
        # interpolated frame's x-axis as a roll-continuity hint.
        z_inward = safe_normalize(-as_flat3(des_new.p_c))
        des_new.R_b = self._frame_from_z_axis(z_inward, x_hint=R_hint[:, 0])
        des_new.z_b = as_col3(z_inward)
        des_new.omega_b = self._reference_angular_velocity(
            self.orbit_z_bd,
            progress=progress,
            progress_rate=progress_rate,
        )
        return des_new

    def _reference_angular_velocity(
        self,
        z_bd,
        progress: float,
        progress_rate: float,
    ) -> np.ndarray:
        if abs(progress_rate) <= 1e-9:
            return np.zeros((3, 1))

        n = len(z_bd)
        if n <= 1:
            return np.zeros((3, 1))

        h = min(1.0, 0.5 * max(n - 1, 1))
        u_minus = np.clip(progress - h, 0.0, n - 1)
        u_plus = np.clip(progress + h, 0.0, n - 1)
        if u_plus <= u_minus + 1e-12:
            return np.zeros((3, 1))

        R_minus, *_ = self._sample_z_bd_fields(z_bd, u_minus)
        R_plus, *_ = self._sample_z_bd_fields(z_bd, u_plus)
        phi = so3_log(R_minus.T @ R_plus)
        if not np.all(np.isfinite(phi)):
            return np.zeros((3, 1))
        return as_col3(progress_rate * phi / max(u_plus - u_minus, 1e-9))

    # ── public entry points ────────────────────────────────────────────

    def add_base_goal(self, st, des: Desired, idx: int | None = None):
        return self.sample_base_goal(des, idx=idx)
