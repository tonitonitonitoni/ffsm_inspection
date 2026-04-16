import pinocchio as pin, numpy as np
from utils.params import ROBOT_PARAMS, CORE_PARAMS
from types import SimpleNamespace
from utils.pure import as_col3, cross

class GiordanoRobot:
    def __init__(self, robot_name="UR3"):
        self.params = SimpleNamespace(pwd=ROBOT_PARAMS['pwd'], dt=CORE_PARAMS["dt_ctrl"], **ROBOT_PARAMS[robot_name])
        robot_path = self.params.pwd + self.params.robot_relative_path
        self.model, self.collision_model, _ = pin.buildModelsFromMJCF(robot_path)
        self.model.gravity = pin.Motion.Zero()
        self.data = self.model.createData()
        self.M0 = pin.crba(self.model, self.data, pin.neutral(self.model))
        M6 = self.M0[:6, :6].copy()
        self.m = float(np.trace(M6[:3, :3]) / 3.0)
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nx = self.nq + self.nv
        self.nu = self.model.nv
        self.n = self.model.nv - 6
        self.use_j_damping = False
        self.j_plus_damping = 1e-4
        self.j_plus_sigma_floor = 2e-2
        self.j_plus_sigma_hard = 5e-3
        self.j_plus_max_inv_norm = 1e3
        self.cam_frame = self.model.getFrameId(self.params.cam_frame)
        self.base_frame = self.model.getFrameId(self.params.base_frame)
        self.q0 = self.choose_q_init_by_gamma()
        
        self.joint_limits = [(self.model.lowerPositionLimit[i], self.model.upperPositionLimit[i])
                             for i in range(self.nq - self.params.n_DOF, self.nq)]
        joint_velocity_limits = self.params.joint_velocity_limits
        joint_torque_limits = self.params.joint_torque_limits

        self._last_J_plus_inv = np.zeros((self.n, 6))
        self.theta_dot_max = (joint_velocity_limits * np.ones(self.n)
                              if np.isscalar(joint_velocity_limits) else joint_velocity_limits)
        self.tau_max = (joint_torque_limits * np.ones(self.n)
                        if np.isscalar(joint_torque_limits) else joint_torque_limits)
        self.motion = None
        self.dyn = None
        self._motion_q = None
        self._motion_v = None
        self._dyn_q = None
        self._dyn_v = None

    # ── Pinocchio update ──────────────────────────────────────────────────────

    def update(self, q, v=None):
        if v is None:
            pin.forwardKinematics(self.model, self.data, q)
            pin.centerOfMass(self.model, self.data, q)
        else:
            pin.forwardKinematics(self.model, self.data, q, v)
        pin.updateFramePlacements(self.model, self.data)
        pin.centerOfMass(self.model, self.data, q, v)

    def twist(self, loc="base", ref="local"):
        if ref == "local":
            ref_frame = pin.ReferenceFrame.LOCAL
        elif ref == "lwa":
            ref_frame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        else:
            ref_frame = pin.ReferenceFrame.WORLD
        frame = self.base_frame if loc == "base" else self.cam_frame
        nu = pin.getFrameVelocity(self.model, self.data, frame, ref_frame)
        return np.vstack([as_col3(nu.linear), as_col3(nu.angular)])

    # ── All kinematic terms — one pass per step ───────────────────────────────

    def all_motion_terms(self, q, v):
        q = np.asarray(q, dtype=float).reshape(-1)
        v = np.asarray(v, dtype=float).reshape(-1)
        self.update(q, v)

        oMb = self.data.oMf[self.base_frame]
        p_tb = as_col3(oMb.translation)
        R_tb = oMb.rotation
        p_tc = as_col3(self.data.com[0])
        v_tc = as_col3(self.data.vcom[0])

        nu_b  = self.twist("base", "local")
        nu_tb = self.twist("base", "world")
        v_bb    = as_col3(nu_b[:3])
        omega_b = as_col3(nu_b[3:])
        omega_tb = as_col3(nu_tb[3:])

        R_cb = R_tb
        R_bc = R_tb.T
        p_bc = R_bc @ (p_tc - p_tb)
        v_cb = R_bc @ v_tc
        p_bc_dot = v_cb - v_bb - cross(omega_b) @ p_bc
        R_cb_dot = R_cb @ cross(omega_b)

        oMe = self.data.oMf[self.cam_frame]
        p_te = as_col3(oMe.translation)
        R_te = oMe.rotation
        z_te = as_col3(R_te[:, 2])

        nu_te    = self.twist("cam", "world")
        omega_te = nu_te[3:]
        v_te_lwa = self.twist("cam", "lwa")
        p_te_dot = v_te_lwa[:3]
        nu_e     = self.twist("cam", "local")
        omega_e  = nu_e[3:]

        R_ec = R_te.T
        R_eb = R_ec @ R_tb
        p_ec = R_ec @ (p_tc - p_te)
        p_ec_dot = R_te.T @ (v_tc - p_te_dot) - cross(as_col3(omega_e)) @ p_ec
        R_eb_dot = -cross(as_col3(omega_e)) @ R_eb + R_eb @ cross(omega_b)

        G_omega_b = np.vstack([cross(p_ec) @ R_eb, R_eb])
        G_omega_b_dot = np.vstack([
            cross(p_ec_dot) @ R_eb + cross(p_ec) @ R_eb_dot,
            R_eb_dot,
        ])
        G_vc          = np.vstack([R_ec, np.zeros((3, 3))])
        G_vc_breve    = np.block([[np.zeros((3, 3))], [G_vc]])
        G_omega_b_breve = np.hstack([-G_omega_b, np.eye(6)])

        self.motion = SimpleNamespace(
            p_tc=p_tc, v_tc=v_tc,
            p_tb=p_tb, R_tb=R_tb,
            omega_b=omega_b,
            p_bc=p_bc, p_bc_dot=p_bc_dot,
            R_cb=R_cb, R_bc=R_bc, R_cb_dot=R_cb_dot,
            nu_tb=nu_tb, omega_tb=omega_tb,
            p_te=p_te, p_te_dot=p_te_dot, R_te=R_te, z_te=z_te,
            R_eb=R_eb, R_eb_dot=R_eb_dot,
            nu_e=nu_e, omega_te=omega_te,
            p_ec=p_ec, p_ec_dot=p_ec_dot,
            G_omega_b=G_omega_b, G_omega_b_dot=G_omega_b_dot,
            G_vc=G_vc, G_vc_breve=G_vc_breve,
            G_omega_b_breve=G_omega_b_breve,
        )
        self._motion_q = q.copy()
        self._motion_v = v.copy()
        return self.motion
    
    def cam_terms(self, q, v):
        q = np.asarray(q, dtype=float).reshape(-1)
        v = np.asarray(v, dtype=float).reshape(-1)
        self.update(q, v)
        oMe = self.data.oMf[self.cam_frame]
        # Copy out of Pinocchio-backed storage so callers can safely cache poses.
        p_te = as_col3(np.array(oMe.translation, dtype=float, copy=True))
        R_te = oMe.rotation
        z_te = as_col3(np.array(R_te[:, 2], dtype=float, copy=True))
        return p_te, z_te


    # ── J_plus — standalone (used by conditioning / init) ────────────────────

    def _J_plus_inv(self, J):
        """
        Invert J_plus exactly when it is square and well-conditioned so the
        algebraic identities involving Gamma and Gamma_inv remain exact away
        from singularities. Near singularity, fall back to the damped SVD
        pseudo-inverse for robustness.
        """
        
        if not np.isfinite(J).all():
            return self._last_J_plus_inv

        try:
            U, s, Vt = np.linalg.svd(J, full_matrices=False)
        except np.linalg.LinAlgError:
            return self._last_J_plus_inv

        if s.size == 0:
            return self._last_J_plus_inv

        sigma_min = float(np.min(s))
        if not np.isfinite(sigma_min) or sigma_min <= self.j_plus_sigma_hard:
            return self._last_J_plus_inv

        # Exact inverse when safely away from singularity and J is square.
        if J.shape[0] == J.shape[1] and sigma_min >= self.j_plus_sigma_floor:
            try:
                J_inv = np.linalg.inv(J)
            except np.linalg.LinAlgError:
                return self._last_J_plus_inv
        else:
            lam = self.j_plus_damping
            if sigma_min < self.j_plus_sigma_floor:
                lam = max(lam, self.j_plus_sigma_floor - sigma_min)

            s_damped = s / (s * s + lam * lam)
            J_inv = Vt.T @ (s_damped[:, None] * U.T)

        if not np.isfinite(J_inv).all() or np.linalg.norm(J_inv, 2) > self.j_plus_max_inv_norm:
            return self._last_J_plus_inv

        self._last_J_plus_inv = J_inv
        return J_inv

    def J_plus(self, q):
        """Standalone J_plus — calls update + crba + frameJacobian internally."""
        self.update(q)
        M = pin.crba(self.model, self.data, q)
        Jv_bar = M[:3, 6:] / self.m
        oMb = self.data.oMf[self.base_frame]
        oMe = self.data.oMf[self.cam_frame]
        R_eb = oMe.rotation.T @ oMb.rotation
        R_eb0 = np.vstack([R_eb, np.zeros((3, 3))])
        J_nu_e = pin.computeFrameJacobian(
            self.model, self.data, q, self.cam_frame, pin.ReferenceFrame.LOCAL)[:, 6:]
        return J_nu_e - R_eb0 @ Jv_bar

    def J_plus_inv(self, q):
        return self._J_plus_inv(self.J_plus(q))

    # ── Gamma — standalone (used by gamma_sigma_min) ──────────────────────────

    def Gamma(self, q):
        """Standalone Gamma — one update + one crba + one frameJacobian call."""
        self.update(q)
        oMb = self.data.oMf[self.base_frame]
        R_tb = oMb.rotation
        p_tb = as_col3(oMb.translation)
        p_tc = as_col3(self.data.com[0])
        R_bc = R_tb.T
        p_bc = R_bc @ (p_tc - p_tb)
        
        oMe = self.data.oMf[self.cam_frame]
        R_te = oMe.rotation
        p_te = as_col3(oMe.translation)
        R_eb = R_te.T @ R_tb
        p_ec = R_te.T @ (p_tc - p_te)
        G_omega_b = np.vstack([cross(p_ec) @ R_eb, R_eb])

        M = pin.crba(self.model, self.data, q)
        Jv_bar = M[:3, 6:] / self.m
        R_eb0 = np.vstack([R_eb, np.zeros((3, 3))])
        J_nu_e = pin.computeFrameJacobian(
            self.model, self.data, q, self.cam_frame, pin.ReferenceFrame.LOCAL)[:, 6:]
        J_plus = J_nu_e - R_eb0 @ Jv_bar

        top = np.hstack([R_tb,              -R_tb @ cross(p_bc), R_tb @ Jv_bar])
        mid = np.hstack([np.zeros((3, 3)),   np.eye(3),           np.zeros((3, self.n))])
        bot = np.hstack([np.zeros((6, 3)),   G_omega_b,           J_plus])
        return np.vstack([top, mid, bot])

    # ── All dynamics terms — one pass per step ────────────────────────────────

    def all_dynamics_terms(self, q, v):
        """
        Calls all_motion_terms(q, v) then computes every dynamics quantity in a
        single forward pass — each pinocchio routine called exactly once.

        Returns a SimpleNamespace stored as self.dyn.
        """
        q = np.asarray(q, dtype=float).reshape(-1)
        v = np.asarray(v, dtype=float).reshape(-1)
        m = self.all_motion_terms(q, v)          # FK + frames + COM (sets self.motion)

        # ── Mass matrix + COM Jacobian in base frame ─────────────────────────
        M     = pin.crba(self.model, self.data, q)
        Jv_bar = M[:3, 6:] / self.m              # COM Jacobian, base frame, joint cols

        # ── EE frame Jacobian, joint columns ─────────────────────────────────
        J_nu_e = pin.computeFrameJacobian(
            self.model, self.data, q, self.cam_frame, pin.ReferenceFrame.LOCAL)[:, 6:]
        R_eb0  = np.vstack([m.R_eb, np.zeros((3, 3))])
        J_plus = J_nu_e - R_eb0 @ Jv_bar
        J_inv  = self._J_plus_inv(J_plus)        # n×6 damped pseudoinverse

        # ── Gamma ─────────────────────────────────────────────────────────────
        top = np.hstack([m.R_cb, -m.R_cb @ cross(m.p_bc), m.R_cb @ Jv_bar])
        mid = np.hstack([np.zeros((3, 3)),  np.eye(3),  np.zeros((3, self.n))])
        bot = np.hstack([np.zeros((6, 3)),  m.G_omega_b,  J_plus])
        Gamma = np.vstack([top, mid, bot])

        # ── Gamma_inv ─────────────────────────────────────────────────────────
        top_i = np.hstack([m.R_bc,
                            cross(m.p_bc) + Jv_bar @ J_inv @ m.G_omega_b,
                            -Jv_bar @ J_inv])
        mid_i = np.hstack([np.zeros((3, 3)), np.eye(3), np.zeros((3, 6))])
        bot_i = np.hstack([np.zeros((self.n, 3)), -J_inv @ m.G_omega_b, J_inv])
        Gam_inv = np.vstack([top_i, mid_i, bot_i])

        # ── Coriolis ──────────────────────────────────────────────────────────
        C = pin.computeCoriolisMatrix(self.model, self.data, q, v)

        # ── Jacobian time variation ───────────────────────────────────────────
        pin.computeJointJacobiansTimeVariation(self.model, self.data, q, v)
        J_nu_e_dot = pin.getFrameJacobianTimeVariation(
            self.model, self.data, self.cam_frame, pin.ReferenceFrame.LOCAL)[:, 6:]

        # ── COM Jacobian time variation ───────────────────────────────────────
        # computeCentroidalMapTimeVariation fills both data.Ag and data.dAg
        pin.computeCentroidalMapTimeVariation(self.model, self.data, q, v)
        Jq_w       = (self.data.Ag[:3,  :] / self.m)[:, 6:]   # world frame, joint cols
        Jcom_dot_w = (self.data.dAg[:3, :] / self.m)[:, 6:]
        R_bc_dot   = -m.R_bc @ cross(m.omega_tb)               # d/dt(R_tb^T)
        Jv_bar_dot = R_bc_dot @ Jq_w + m.R_bc @ Jcom_dot_w

        # ── J_plus_dot ───────────────────────────────────────────────────────
        R_eb0_dot = np.vstack([m.R_eb_dot, np.zeros((3, 3))])
        J_plus_dot = J_nu_e_dot - R_eb0_dot @ Jv_bar - R_eb0 @ Jv_bar_dot

        # ── Gamma_dot ────────────────────────────────────────────────────────
        top_d = np.hstack([m.R_cb_dot,
                            -m.R_cb_dot @ cross(m.p_bc) - m.R_cb @ cross(m.p_bc_dot),
                            m.R_cb_dot @ Jv_bar + m.R_cb @ Jv_bar_dot])
        mid_d = np.zeros((3, 3 + 3 + self.n))
        bot_d = np.hstack([np.zeros((6, 3)), m.G_omega_b_dot, J_plus_dot])
        Gam_dot = np.vstack([top_d, mid_d, bot_d])


        # ── Dynamics matrices ─────────────────────────────────────────────────
        P     = M @ Gam_inv
        M_hat = Gam_inv.T @ P
        Q     = C - P @ Gam_dot
        C_hat = Gam_inv.T @ Q @ Gam_inv

        # ── Conditioning ─────────────────────────────────────────────────
        s_min_G = np.linalg.svd(Gamma, compute_uv=False)[-1]
        s_min_J = np.linalg.svd(J_plus, compute_uv=False)[-1]

        self.dyn = SimpleNamespace(
            M=M,       Mtm=M[:3, 6:],
            M_hat=M_hat, M_breve=M_hat[3:, 3:],
            C=C,       C_hat=C_hat, C_c=C_hat[3:, :3], C_breve=C_hat[3:, 3:],
            Gamma=Gamma, Gam_inv=Gam_inv, Gam_dot=Gam_dot,
            J_plus=J_plus, J_inv=J_inv, Jv_bar=Jv_bar,
            s_min_G=s_min_G, s_min_J=s_min_J,
            sigma_min_G=s_min_G, sigma_min_J=s_min_J,
            s_min_Gamma=s_min_G, s_min_J_plus=s_min_J,
            sigma_min_Gamma=s_min_G, sigma_min_J_plus=s_min_J,
        )
        self._dyn_q = q.copy()
        self._dyn_v = v.copy()
        return self.dyn

    def motion_matches(self, q, v) -> bool:
        if self.motion is None or self._motion_q is None or self._motion_v is None:
            return False
        q = np.asarray(q, dtype=float).reshape(-1)
        v = np.asarray(v, dtype=float).reshape(-1)
        return np.array_equal(self._motion_q, q) and np.array_equal(self._motion_v, v)

    def dyn_matches(self, q, v) -> bool:
        if self.dyn is None or self._dyn_q is None or self._dyn_v is None:
            return False
        q = np.asarray(q, dtype=float).reshape(-1)
        v = np.asarray(v, dtype=float).reshape(-1)
        return np.array_equal(self._dyn_q, q) and np.array_equal(self._dyn_v, v)

    # ── Conditioning / diagnostics / initialization ───────────────────────────

    def J_plus_svals(self, q):
        return np.linalg.svd(self.J_plus(q), compute_uv=False)

    def sigma_min_J_plus(self, q):
        s = self.J_plus_svals(q)
        return float(np.min(s)) if s.size > 0 else np.nan

    def sample_sigma_min_J_plus(self, q_center, joint_step=0.05, samples=25, seed=0):
        rng = np.random.default_rng(seed)
        q_center = np.asarray(q_center, dtype=float).copy()
        out = []
        for idx in range(samples):
            q = q_center.copy()
            q[-self.n:] += rng.uniform(-joint_step, joint_step, size=self.n)
            out.append({"sample": idx,
                        "sigma_min": self.sigma_min_J_plus(q),
                        "q_joint": q[-self.n:].copy()})
        out.sort(key=lambda item: item["sigma_min"])
        return out

    def gamma_sigma_min(self, q):
        s = np.linalg.svd(self.Gamma(q), compute_uv=False)
        return s[-1]

    def choose_q_init_by_sigma(self, joint_step=0.35, samples=64, seed=0):
        q_neutral =  pin.neutral(self.model)#self.q0.copy()
        rows = self.sample_sigma_min_J_plus(q_neutral, joint_step=joint_step,
                                            samples=samples, seed=seed)
        q_best    = q_neutral.copy()
        sigma_best = self.sigma_min_J_plus(q_best)
        for row in rows:
            sigma = float(row["sigma_min"])
            if not np.isfinite(sigma) or sigma <= sigma_best:
                continue
            q_candidate = q_neutral.copy()
            q_candidate[-self.n:] = np.asarray(row["q_joint"], dtype=float)
            q_best, sigma_best = q_candidate, sigma
        return q_best

    def choose_q_init_by_gamma(self, joint_step=0.35, samples=96, seed=0):
        rng = np.random.default_rng(seed)
        q_neutral =  pin.neutral(self.model)
        q_best    = q_neutral.copy()
        sigma_best = self.gamma_sigma_min(q_best)

        q_candidates = [q_neutral.copy()]
        for _ in range(samples):
            q_candidate = q_neutral.copy()
            q_candidate[-self.n:] = (q_neutral[-self.n:]
                                     + rng.uniform(-joint_step, joint_step, size=self.n))
            q_candidates.append(q_candidate)

        for q_candidate in q_candidates:
            sigma = self.gamma_sigma_min(q_candidate)
            if not np.isfinite(sigma) or sigma <= sigma_best:
                continue
            q_best, sigma_best = q_candidate.copy(), sigma
        return q_best
