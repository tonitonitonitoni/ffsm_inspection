from __future__ import annotations
from utils.robot import GiordanoRobot
from typing import Any
import numpy as np
import open3d as o3d, open3d.core as o3c  # type:ignore
from utils.params import make_params
from utils.pure import safe_normalize, sample_hemisphere, as_flat3, as_col3, formatted_dict, saturate, so3_log, next_filename, last_npz
from utils.mesh_manipulation import sample_points_on_mesh, compute_triangle_areas, calc_up_vector
from utils.data_classes import CameraTarget, CameraPose, Desired, Target, State
from tqdm import tqdm
from com_base_guidance import BaseGuidance


class EETargetFinder:
    def __init__(
        self,
        cfg = None, 
        robot = None,
        debug=False,
        score_option="product",
        high_res=False
    ):
        self.cfg = cfg if cfg is not None else make_params(vps=1, vision=1)
        self.debug = debug
        self.score_option = score_option
        self.resolution = self.cfg.tsdf_resolution if high_res else self.cfg.resolution
        #Load the robot
        self.robot = robot if robot is not None else GiordanoRobot()

        #Triangle stats
        self.SEEN_TRIANGLES = set()
        self.TRIANGLE_STATS = {}
        #load the mesh and create the raycasting scene
        self.mesh = o3d.io.read_triangle_mesh(self.cfg.mesh_path)
        self.mesh.compute_vertex_normals()
        self.n_tris = len(self.mesh.triangles)
        self.max_triangle_area = np.max(compute_triangle_areas(self.mesh)) if self.n_tris > 0 else 0.0
        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)
        self.scene = o3d.t.geometry.RaycastingScene()
        _ = self.scene.add_triangles(tmesh)

        #Find surface targets and build a KD-tree for efficient searching
        self.targets = self.build_surface_targets()
        surf_pcd = o3d.geometry.PointCloud()
        surf_pcd.points = o3d.utility.Vector3dVector(np.array([t.x for t in self.targets]))
        self.surf_kdtree = o3d.geometry.KDTreeFlann(surf_pcd)
        self.r_reach = self.cfg.reachable_arm_radius * self.cfg.reach_ratio
        
        #Boundaries. As generic as possible at this stage 
        self.max_position_step = 0.5
        self.max_pointing_step = 0.5

        self.active = CameraTarget()

    def reset(self) -> None:
        self.SEEN_TRIANGLES.clear()
        self.TRIANGLE_STATS.clear()
        self.active = CameraTarget()

    def _motion_for_q(self, q: np.ndarray, v: np.ndarray | None = None):
        if v is None:
            v = np.zeros(self.robot.nv, dtype=float)
        q = np.asarray(q, dtype=float).reshape(-1)
        v = np.asarray(v, dtype=float).reshape(-1)
        if self.robot.motion_matches(q, v):
            return self.robot.motion
        return self.robot.all_motion_terms(q, v)

    # ===== Helpers
    def is_pose_reachable(self, p_c, p_e):
        """
        Simple geometric reachability filter for camera poses relative to the
        COM pose at the segment end.
        """
        com_pos = as_flat3(p_c)
        ee_pos = as_flat3(p_e)
        dist = np.linalg.norm(ee_pos - com_pos)
        return dist <= self.r_reach
    
    # ===== Step 0: Surface preprocessing
    def build_surface_targets(self, n_samples=None):
        
        if n_samples is None: n_samples = self.cfg.n_targets
        points, normals = sample_points_on_mesh(self.mesh, n_samples)
        vertices = np.asarray(self.mesh.vertices)
        triangles = np.asarray(self.mesh.triangles)
        tri_areas = compute_triangle_areas(self.mesh)

        tri_centroids = np.mean(vertices[triangles], axis=1)
        tri_centroid_pcd = o3d.geometry.PointCloud()
        tri_centroid_pcd.points = o3d.utility.Vector3dVector(tri_centroids)
        tri_kdtree = o3d.geometry.KDTreeFlann(tri_centroid_pcd)

        targets = []
        for i in tqdm(range(n_samples), total=n_samples, desc="Build surface targets", leave=False):
            target = Target()
            target.x = points[i]
            target.n = safe_normalize(normals[i])
            _, idx, _ = tri_kdtree.search_knn_vector_3d(target.x, 1)
            target.tri_idx = idx[0]
            target.tri_area = tri_areas[idx[0]]
            targets.append(target)

        self.targets = targets
        return targets
    
        #Step 4: Rank potential camera poses

    # ===== Step 1: Generate camera poses
    def candidate_camera_poses(self, p_c, n_top=3):
        """Given p_COM, generate potential camera poses."""
        p_anchor = as_flat3(p_c)
        e_in = safe_normalize(-p_anchor)

        default_return = CameraPose(p_e = p_anchor.copy(), z_e=e_in.copy())

        #sample hemisphere around e_in
        n_rays = self.cfg.n_rays
        dirs = sample_hemisphere(n_rays**2, e_in)
        if len(dirs) == 0:
            return [default_return]

        ranked: list[dict[str, Any]] = []
        radius_scale = self.cfg.camera_candidate_radius_scale
        sample_radius = np.clip(radius_scale, 1e-3, 1.0) * self.r_reach
        for direction in dirs:
            camera_point = p_anchor + sample_radius * direction
            camera_axis = safe_normalize(-camera_point)
            visible = self.query_candidate_targets_from_position(camera_point)
            ranked.append(
                {
                    "p_e": camera_point,
                    "z_e": camera_axis,
                    "visible_count": len(visible),
                }
            )

        ranked.sort(key=lambda item: item["visible_count"], reverse=True)

        candidates = [CameraPose(p_e=item["p_e"].copy(), z_e=item["z_e"].copy()) for item in ranked[:n_top]]
        if not candidates:
            candidates.append(default_return)
        return candidates

    def query_candidate_targets_from_position(self, p_e) -> list[int]:
        
        """
        Position-only candidate query.

        This ignores the camera optical axis and only checks whether a sampled
        surface target is in range, front-facing, and line-of-sight visible
        from the candidate camera position.
        """
        camera_point = as_flat3(p_e)
        cos_theta_max = np.cos(self.cfg.max_angle)

        _, idxs, _ = self.surf_kdtree.search_knn_vector_3d(camera_point, self.cfg.n_query)

        ray_rows = []
        ray_meta = []
        for idx in idxs:
            target = self.targets[idx]
            dir_vec = as_flat3(target.x) - camera_point
            dist = np.linalg.norm(dir_vec)
            if dist < self.cfg.d_min or dist > self.cfg.d_max:
                continue
            if dist <= 1e-12:
                continue

            dir_unit = dir_vec / dist

            # Target surface should face the camera.
            if np.dot(target.n, -dir_unit) < cos_theta_max:
                continue

            ray_rows.append(np.hstack((camera_point, dir_unit)))
            ray_meta.append((int(idx), dist))

        if not ray_rows:
            return []

        rays = np.asarray(ray_rows, dtype=np.float32)
        ray_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
        t_hits = self.scene.cast_rays(ray_tensor)["t_hit"].numpy()

        visible: list[int] = []
        tol = self.cfg.EPS
        for (idx, dist), t_hit in zip(ray_meta, t_hits):
            if np.isfinite(t_hit) and abs(t_hit - dist) < tol:
                visible.append(idx)

        return visible
    
    def query_visible_targets_from_pose(self, pose:CameraPose) -> list[int]:
        """
        Frustum-aware visibility query for candidate surface targets.

        Pipeline:
          - KD-tree nearest-neighbor query for candidate targets
          - range filtering
          - incidence-angle filtering against the target normal
          - camera-FOV filtering using the optical axis
          - one LOS ray per surviving target
          - exact visibility test with raycasting
        Returns list of visible target indices
        
        """
        p_e = pose.p_e
        z_e = pose.z_e

        camera_point = as_flat3(p_e)
        if z_e is None:
            camera_axis = safe_normalize(-camera_point)
        else:
            camera_axis = safe_normalize(z_e)

        half_fov_rad = 0.5 * np.deg2rad(self.cfg.camera_fov_deg)
        cos_half_fov = np.cos(half_fov_rad)
        visible = []
        for idx in self.query_candidate_targets_from_position(camera_point):
            target = self.targets[idx]
            dir_unit = safe_normalize(as_flat3(target.x) - camera_point)
            if np.dot(camera_axis, dir_unit) >= cos_half_fov:
                visible.append(idx)

        return visible
  
    # ===== Step 2: Scoring    
    def compute_triangle_camera_visibility(self, pose:CameraPose):
        # Uses pinhole ray generation from a camera pose and forward direction 
        # Records which mesh triangles are visible in the full rendered camera frustum
        cfg = self.cfg
        eye_pos = as_flat3(pose.p_e)
        forward = safe_normalize(pose.z_e)
        up = calc_up_vector(forward)

        width, height = self.resolution

        rays_tensor = self.scene.create_rays_pinhole(
            fov_deg=cfg.camera_fov_deg,
            center=(eye_pos + forward).tolist(),
            eye=eye_pos.tolist(),
            up=up.tolist(),
            width_px=width,
            height_px=height,
        ).to(o3c.Device("CPU:0"))

        result = self.scene.cast_rays(rays_tensor)
        t_hit = result["t_hit"].reshape((-1,))
        prim_ids = result["primitive_ids"].reshape((-1,))

        mask = t_hit.isfinite()
        if not bool(mask.any()):
            return np.array([], dtype=int)

        mask = mask & (t_hit <= cfg.d_far) & (t_hit >= cfg.d_near)
        mask_np = mask.numpy().astype(bool)
        if not mask_np.any():
            return np.array([], dtype=int)

        prim_np = prim_ids.numpy()[mask_np].astype(int)

        if prim_np.size > 0:
            unique_ids = np.unique(prim_np)
            self.SEEN_TRIANGLES.update(unique_ids.tolist())

            # Update angular diversity statistics
            for tri_id in unique_ids:
                if tri_id not in self.TRIANGLE_STATS:
                    self.TRIANGLE_STATS[tri_id] = {
                        "count": 0,
                        "dirs": []
                    }
                self.TRIANGLE_STATS[tri_id]["count"] += 1
                self.TRIANGLE_STATS[tri_id]["dirs"].append(forward.copy())

        return prim_np

    def compute_visibility_pointing_at_target(self, p_e, p_tgt):
        # Computes how many triangles are seen when the camera is pointing at the selected target.
        pose = CameraPose()
        pose.p_e = as_flat3(p_e)
        pose.z_e = safe_normalize(p_tgt - pose.p_e)
        return self.compute_triangle_camera_visibility(pose)

    def Kong_score(self, vp_dist, pixel_dist, theta):
        # -----------------------
        # Kong Score Metric
        # -----------------------
        params = self.cfg
        def q1_score(vp_dist):
            if vp_dist <= params.d_near or vp_dist >= params.d_far:
                return 0.
            elif vp_dist <= params.d_min:
                slope = 1./(params.d_min-params.d_near)
                return slope * (vp_dist - params.d_near)
            elif vp_dist <= params.d_max:
                return 1.
            else:
                slope = -1./(params.d_far-params.d_max)
                return 1 + slope*(vp_dist - params.d_max)
            
        def q2_score(theta):
            return np.cos(theta) if 0 < theta < params.max_angle else 0
        
        def q3_score(pixel_dist):
            max_pixel_distance = .5*np.linalg.norm(self.resolution)
            return 1 - pixel_dist/max_pixel_distance
        q1 = q1_score(vp_dist)
        q2 = q2_score(theta)
        q3 = q3_score(pixel_dist)
        return (params.w1*q1 + params.w2*q2 + params.w3*q3)/(params.w1 + params.w2 + params.w3)
    
    def score_target(self, target_idx: int, pose, p_c=None) -> float:
        p_e = pose.p_e
        z_e = pose.z_e
        target = self.targets[target_idx]

        camera_point = as_flat3(p_e)
        camera_axis = safe_normalize(z_e)

        dir_vec = as_flat3(target.x) - camera_point
        SCORE = {}
        # --- Kong score ----
        #Distance to target
        vp_dist = np.linalg.norm(dir_vec)
        if vp_dist <= 1e-9:
            return -np.inf

        #Incident angle from normal
        dir_unit = dir_vec / vp_dist
        cos_theta = np.clip(np.dot(target.n, -dir_unit), -1.0, 1.0)
        theta = np.arccos(cos_theta)

        #Off-axis distance
        cos_axis = np.clip(np.dot(camera_axis, dir_unit), -1.0, 1.0)
        axis_err = np.arccos(cos_axis)

        max_pixel_distance = 0.5 * np.linalg.norm(self.resolution)
        max_fov_rad = 0.5*np.deg2rad(self.cfg.camera_fov_deg)
        pixel_dist = max_pixel_distance * min(axis_err / max_fov_rad, 1.0)

        kong_score = self.Kong_score(vp_dist, pixel_dist, theta)
        SCORE["kong"] = kong_score

        # --- Novelty ----
        tri_id = target.tri_idx
        hit_count = 0
        if tri_id in self.TRIANGLE_STATS:
            hit_count = int(self.TRIANGLE_STATS[tri_id]["count"])
        novelty_score = 1.0 / (1.0 + hit_count)

        SCORE["novelty"] = novelty_score

        # --- Area ----
        area = target.tri_area
        area_score = 0.0 if self.max_triangle_area <= 0.0 else area / self.max_triangle_area
        SCORE["area"] = area_score

        # --- Stability ----
        # Smooth preference for staying near the previously selected target on the surface.
        stability_score = 0.0
        if self.active.p_tgt is not None:
            target_dist = np.linalg.norm(as_flat3(target.x) - as_flat3(self.active.p_tgt))
            stability_score = np.exp(-target_dist / self.cfg.stability_length)
        SCORE["stability"] = stability_score

        # --- Motion ----
        motion_score = 0.0
        if self.active.p_e is not None:
            motion_dist = np.linalg.norm(camera_point - as_flat3(self.active.p_e))
            motion_score = np.clip(1.0 - motion_dist / max(self.r_reach, 1e-9), 0.0, 1.0)
        SCORE["motion"] = motion_score

        # --- Anchor locality ---
        # Prefer surface points near the moving orbital anchor so the selected
        # target progresses over the body with the platform trajectory.
        anchor_score = 0.0
        if p_c is not None:
            anchor_dist = np.linalg.norm(as_flat3(target.x) - as_flat3(p_c))
            anchor_length = self.cfg.anchor_length
            anchor_score = np.exp(-anchor_dist / max(anchor_length, 1e-9))
        SCORE["anchor"] = anchor_score

        # --- Manipulability ---
        # Penalize arm configurations near full extension or full retraction,
        # where Gamma becomes singular.  Parabola peaks at mid-reach.
        manipulability_score = 1.0
        if p_c is not None:
            p_ce_dist = np.linalg.norm(camera_point - as_flat3(p_c))
            ratio = min(p_ce_dist / max(self.r_reach, 1e-9), 1.0)
            manipulability_score = 4.0 * ratio * (1.0 - ratio)
        SCORE["manipulability"] = manipulability_score

        # --- Final Tally ----
        if self.score_option=="mean":
            total_score = 0
            for v in SCORE.values():
                total_score +=v
            total_score /= len(SCORE)
        elif self.score_option=="product":
            total_score = 1.0
            for v in SCORE.values():
                total_score *=v
        elif self.score_option=="weighted":
            score_weights = {
                "kong": self.cfg.w_kong,
                "novelty": self.cfg.w_novelty,
                "area": self.cfg.w_area,
                "stability": self.cfg.w_stability,
                "motion": self.cfg.w_motion,
                "anchor": self.cfg.w_anchor,
                "manipulability": self.cfg.w_manipulability,
            }
            total_score = 0
            for k in SCORE.keys():
                total_score += score_weights[k] * SCORE[k]
        else: return RuntimeError("Pick one of 'mean', 'product', 'weighted'.")
        if self.debug: print(formatted_dict(SCORE))
        return total_score        

    # ===== Step 3: Final selection
    def choose_best_camera_target(self, candidates: list[CameraPose], p_c=None):

        """For each candidate camera pose:
        •	get visible targets from that position
        •	for each visible target, set the optical axis to point directly at that target
        •	score that target/pose pair
        •	keep the best one. 
        With hysteresis: if a new target is different from the currently active one, it must beat the current active score by at least camera_switch_hysteresis
        """
        best = CameraTarget()
        switch_margin = self.cfg.camera_switch_hysteresis
        active_score = self.active.score
        active_idx = self.active.idx

        for candidate in candidates:
            visible = self.query_candidate_targets_from_position(candidate.p_e)

            for target_idx in visible:
                target = self.targets[target_idx]
                z_to_target = safe_normalize(as_flat3(target.x) - as_flat3(candidate.p_e))
                pose = CameraPose(p_e=candidate.p_e.copy(), z_e=z_to_target)
                score = self.score_target(target_idx, pose, p_c=p_c)

                # end early if poor
                if score <= best.score:
                    continue

                # hysteresis
                if (
                    active_idx is not None
                    and target_idx != active_idx
                    and score <= (active_score + switch_margin)
                ):
                    continue

                best = CameraTarget(
                    idx=target_idx,
                    p_tgt=as_flat3(target.x).copy(),
                    n_tgt=as_flat3(target.n).copy(),
                    p_e=candidate.p_e.copy(),
                    z_e=z_to_target,
                    score=score,
                    switched=(active_idx != target_idx),
                    info={
                        "visible_count":    len(visible),
                        "triangle_area":    target.tri_area,
                        "tri_idx":          target.tri_idx,
                        "coverage_count":   target.coverage_count,
                        "fallback":         False,
                    }
                    
                )

        return best
    
    # ===== Step 4: Memory update
    def record_target_observation(self, selection: CameraTarget) -> None:
        if selection.idx is None or selection.p_e is None:
            return
        target = self.targets[selection.idx]
        if selection.p_tgt is None or not np.allclose(as_flat3(selection.p_tgt), as_flat3(target.x)):
            raise ValueError("selection target does not match stored target")
        self.compute_visibility_pointing_at_target(selection.p_e, selection.p_tgt)

        target.coverage_count +=1

    # ===== Step 5: Fallback in case no target is found.   
    def fallback(self, p_c): 
        """Returns the current end effector pose and axis.""" 
        q = self.robot.q0.copy()
        q[:3] = as_flat3(p_c) 
        motion = self._motion_for_q(q)

        tgt = self.active.copy()
        tgt.switched = False
        tgt.info = {"fallback": True}
        tgt.fallback=True
        tgt.p_e = motion.p_te
        tgt.z_e = motion.z_te

        return tgt

    def choose_goal(self, p_c):
        candidates = self.candidate_camera_poses(p_c)
        selection = self.choose_best_camera_target(candidates, p_c=p_c)
        if selection.idx is None:
            selection = self.fallback(p_c)
        return selection


class EEGuidance(BaseGuidance):
    def __init__(
        self,
        cfg,
        robot: GiordanoRobot | None = None,
        debug=False,
    ):
        super().__init__(cfg, debug)
        self.robot = robot if robot is not None else GiordanoRobot()
        
        self.target_finder = EETargetFinder(cfg=cfg, robot=self.robot, debug=debug)

        self.active_camera_target = CameraTarget()
        self._active_p_ce: np.ndarray | None = None
        self.last_target_update_step = -1
        self.last_target_switch_step = -1
        self._ee_guidance_step = -1
        self.filtered_camera_pose: CameraPose | None = None
        self._ee_kin_cache: dict[str, Any] | None = None
        self._ee_kin_cache_q: np.ndarray | None = None
        self._ee_kin_cache_v: np.ndarray | None = None
        self._ee_sigma_cache: float | None = None

    def reset_runtime_state(self) -> None:
        self.active_camera_target = CameraTarget()
        self._active_p_ce = None
        self.last_target_update_step = -1
        self.last_target_switch_step = -1
        self._ee_guidance_step = -1
        self.filtered_camera_pose = None
        self._ee_kin_cache = None
        self._ee_kin_cache_q = None
        self._ee_kin_cache_v = None
        self._ee_sigma_cache = None
        if self.target_finder is not None:
            self.target_finder.reset()

    def _frame_tracking_z_axis(
        self,
        z_axis,
        R_prev: np.ndarray | None = None,
        x_hint=None,
    ) -> np.ndarray:
        z_axis = safe_normalize(as_flat3(z_axis))
        if R_prev is not None:
            R_prev = np.asarray(R_prev, dtype=float).reshape(3, 3)
            R_align = self._rotation_aligning_vectors(R_prev[:, 2], z_axis)
            R_next = R_align @ R_prev
            return self._frame_from_z_axis(z_axis, x_hint=R_next[:, 0])
        return self._frame_from_z_axis(z_axis, x_hint=x_hint)

    def _state_velocity(self, st: State) -> np.ndarray:
        if st.v is None:
            return np.zeros(self.robot.nv, dtype=float)
        return np.asarray(st.v, dtype=float).reshape(-1)

    def _ee_cache_matches(self, q: np.ndarray, v: np.ndarray) -> bool:
        return (
            self._ee_kin_cache is not None
            and self._ee_kin_cache_q is not None
            and self._ee_kin_cache_v is not None
            and np.array_equal(self._ee_kin_cache_q, q)
            and np.array_equal(self._ee_kin_cache_v, v)
        )

    def _ee_kinematics(self, st: State) -> dict[str, Any]:
        q = np.asarray(st.q, dtype=float).reshape(-1)
        v = self._state_velocity(st)
        if not self._ee_cache_matches(q, v):
            if self.robot.motion_matches(q, v):
                motion = self.robot.motion
            elif self.robot.dyn_matches(q, v):
                motion = self.robot.motion
            else:
                motion = self.robot.all_motion_terms(q, v)
            self._ee_kin_cache = {
                "p_c": as_flat3(motion.p_tc),
                "p_e": as_flat3(motion.p_te),
                "z_e": safe_normalize(as_flat3(motion.z_te)),
                "R_e": np.asarray(motion.R_te, dtype=float).reshape(3, 3),
                "R_b": np.asarray(motion.R_tb, dtype=float).reshape(3, 3),
            }
            self._ee_kin_cache_q = q.copy()
            self._ee_kin_cache_v = v.copy()
            self._ee_sigma_cache = None
        return self._ee_kin_cache

    def _ee_sigma_min(self, st: State) -> float:
        q = np.asarray(st.q, dtype=float).reshape(-1)
        v = self._state_velocity(st)
        self._ee_kinematics(st)
        if self._ee_sigma_cache is None:
            if self.robot.dyn_matches(q, v):
                dyn = self.robot.dyn
            else:
                dyn = self.robot.all_dynamics_terms(q, v)
            motion = self.robot.motion
            self._ee_kin_cache = {
                "p_c": as_flat3(motion.p_tc),
                "p_e": as_flat3(motion.p_te),
                "z_e": safe_normalize(as_flat3(motion.z_te)),
                "R_e": np.asarray(motion.R_te, dtype=float).reshape(3, 3),
                "R_b": np.asarray(motion.R_tb, dtype=float).reshape(3, 3),
            }
            self._ee_kin_cache_q = q.copy()
            self._ee_kin_cache_v = v.copy()
            self._ee_sigma_cache = dyn.s_min_G
        return self._ee_sigma_cache

    def _default_ee_goal(self, st: State) -> CameraPose:
        kin = self._ee_kinematics(st)
        p_ted = as_col3(kin["p_e"])
        z_ted = as_col3(kin["z_e"])
        R_te = np.asarray(kin["R_e"], dtype=float).reshape(3, 3)
        R_b = np.asarray(kin["R_b"], dtype=float).reshape(3, 3)

        if self.cfg.enable_ee:
            z_ted = as_col3(R_b @ np.array([0.0, 0.0, 1.0], dtype=float))
            p_ce = 0.2 * R_b @ np.array([0.0, 0.0, 1.0], dtype=float)
            p_ted = as_col3(as_flat3(kin["p_c"]) + p_ce)

        R_ted = self._frame_tracking_z_axis(as_flat3(z_ted), R_prev=R_te)
        return CameraPose(p_e=p_ted, z_e=as_col3(R_ted[:, 2]), R_e=R_ted)

    def _sigma_blend_pose(self, st: State, raw_pose: CameraPose, fallback: CameraPose) -> CameraPose:
        """Conditioning-aware safety blend.
        If sigma_min(Gamma) is below the threshold, the raw target pose is blended toward the fallback pose. 
        If conditioning is good, the raw pose is used directly."""

        sigma_thresh = self.cfg.sigma_blend_threshold
        if sigma_thresh <= 0.0:
            return raw_pose
        sigma = self._ee_sigma_min(st)
        if sigma >= sigma_thresh:
            return raw_pose
        alpha = sigma / sigma_thresh
        p_blend = alpha * as_flat3(raw_pose.p_e) + (1.0 - alpha) * as_flat3(fallback.p_e)
        z_blend = safe_normalize(
            alpha * safe_normalize(as_flat3(raw_pose.z_e))
            + (1.0 - alpha) * safe_normalize(as_flat3(fallback.z_e))
        )
        R_prev = None if raw_pose.R_e is None else np.asarray(raw_pose.R_e, dtype=float).reshape(3, 3)
        R_blend = self._frame_tracking_z_axis(z_blend, R_prev=R_prev)
        return CameraPose(p_e=as_col3(p_blend), z_e=as_col3(z_blend), R_e=R_blend)

    def _should_use_target_finder(self) -> bool:
        if not self.cfg.enable_ee:
            return False
        return self.cfg.use_target_finder

    def _pointing_infeasible(self, st: State) -> bool:
        thresh = self.cfg.pointing_reselect_cos
        if thresh <= 0.0:
            return False
        tgt = self.active_camera_target
        if tgt.p_tgt is None or self._active_p_ce is None:
            return False
        p_e_now = self._ee_kinematics(st)["p_c"] + self._active_p_ce
        to_tgt = as_flat3(tgt.p_tgt) - p_e_now
        dist = np.linalg.norm(to_tgt)
        if dist < 1e-9:
            return False
        z_desired = to_tgt / dist
        if tgt.z_e is None:
            return False
        z_orig = safe_normalize(as_flat3(tgt.z_e))
        return np.dot(z_desired, z_orig) < thresh

    def _target_update_due(self, step_idx: int | None = None, st: State | None = None) -> bool:
        """Return True when the EE target should be reselected, False to hold."""
        # Bootstrap: no target yet, must pick one.
        if self.active_camera_target.p_e is None:
            return True
        # No timing info yet, force an update.
        if step_idx is None or self.last_target_update_step < 0:
            return True
        # Minimum hold window after a switch takes priority over every
        # other signal to suppress chattering.
        if (self.last_target_switch_step >= 0
                and (step_idx - self.last_target_switch_step) < self.cfg.camera_min_hold_steps):
            return False
        # Out of hold: force reselect if we can no longer point at the target.
        if st is not None and self._pointing_infeasible(st):
            return True
        # Otherwise update only on the normal period.
        return (step_idx - self.last_target_update_step) >= self.cfg.update_period_steps

    def _resolve_ee_guidance_step(self, step_idx: int | None = None) -> int:
        if step_idx is not None:
            return int(step_idx)
        self._ee_guidance_step += 1
        return self._ee_guidance_step

    def _ee_anchor_point(self, des: Desired, idx: int | None = None) -> np.ndarray:
        # Anchor point is selected forward along the trajectory from the current COM position.
        fields = self.guidance_fields(idx=idx)
        ds_nom = np.linalg.norm(np.diff(np.asarray(fields["p_window"], dtype=float), axis=0), axis=1)
        ds_ref = np.median(ds_nom) if ds_nom.size > 0 else 0.0
        ee_reach = self.cfg.ee_reach_horizon
        lookahead_time = ee_reach if ee_reach > 0.0 else self.cfg.lookahead_time
        speed_ref = np.linalg.norm(as_flat3(des.v_c))
        if ds_ref > 1e-9 and lookahead_time > 0.0:
            lookahead_progress = speed_ref * lookahead_time / ds_ref
        else:
            lookahead_progress = 0.0
        progress = self._current_progress() + lookahead_progress
        anchor = self.sample_progress_anchor(progress, idx=idx)
        if np.all(np.isfinite(anchor)):
            return anchor
        return as_flat3(des.p_c)

    def _clamp_camera_target_motion(self, st: State, selection: CameraTarget) -> CameraTarget:
        if selection.p_e is None:
            return selection

        kin = self._ee_kinematics(st)
        p_c_now = as_flat3(kin["p_c"])
        p_e_now = as_flat3(kin["p_e"])
        p_ce_now = p_e_now - p_c_now
        p_ce_raw = as_flat3(selection.p_e) - p_c_now
        delta = p_ce_raw - p_ce_now
        dist = np.linalg.norm(delta)
        if dist <= 1e-12:
            return selection

        max_step = self.cfg.max_delta_p_ce
        if max_step <= 0.0 or dist <= max_step:
            return selection

        p_ce_clamped = p_ce_now + (max_step / dist) * delta
        p_clamped = p_c_now + p_ce_clamped

        tgt = selection.copy()
        tgt.p_e = p_clamped

        return tgt

    def _held_target_at_com(self, st: State) -> CameraTarget:
        tgt = self.active_camera_target
        if self._active_p_ce is None or tgt.p_e is None:
            return tgt
        
        new_tgt = tgt.copy()

        p_com_now = as_flat3(self._ee_kinematics(st)["p_c"])
        new_tgt.p_e = p_com_now + self._active_p_ce
        new_tgt.switched = False

        return new_tgt

    def _clamp_z_drift(self, z_desired: np.ndarray, z_ref: np.ndarray) -> np.ndarray:
        max_deg = self.cfg.pointing_max_drift_deg
        if max_deg <= 0.0 or max_deg >= 180.0:
            return safe_normalize(z_desired)
        z_d = safe_normalize(z_desired)
        z_r = safe_normalize(z_ref)
        cos_angle = np.dot(z_d, z_r)
        cos_limit = np.cos(np.radians(max_deg))
        if cos_angle >= cos_limit:
            return z_d
        perp = z_d - cos_angle * z_r
        perp_norm = np.linalg.norm(perp)
        if perp_norm < 1e-12:
            return z_r
        perp = perp / perp_norm
        sin_limit = np.sin(np.radians(max_deg))
        return safe_normalize(cos_limit * z_r + sin_limit * perp)

    def _camera_goal_from_target(
        self,
        fallback_goal: CameraPose,
        selection: CameraTarget | None,
    ) -> CameraPose:
        if selection is None or selection.p_e is None:
            return fallback_goal

        p_ted = as_col3(selection.p_e)
        if selection.p_tgt is not None:
            z_axis = safe_normalize(as_flat3(selection.p_tgt) - as_flat3(selection.p_e))
        elif selection.z_e is not None:
            z_axis = safe_normalize(selection.z_e)
        else:
            z_axis = safe_normalize(fallback_goal.z_e)

        if self.filtered_camera_pose is not None and self.filtered_camera_pose.z_e is not None:
            z_axis = self._clamp_z_drift(z_axis, safe_normalize(as_flat3(self.filtered_camera_pose.z_e)))

        R_prev = None if fallback_goal.R_e is None else np.asarray(fallback_goal.R_e, dtype=float).reshape(3, 3)
        return CameraPose(
            p_e=p_ted,
            z_e=as_col3(z_axis),
            R_e=self._frame_tracking_z_axis(z_axis, R_prev=R_prev),
        )

    def _filter_camera_pose(self, raw_pose: CameraPose) -> CameraPose:
        """     
        Apply first-order smoothing and rate limiting to p_e, z_e
        Reconstruct a full rotation matrix consistent with the filtered axis."""
        p_raw = as_flat3(raw_pose.p_e)
        z_raw = safe_normalize(as_flat3(raw_pose.z_e))

        if self.filtered_camera_pose is None or self.filtered_camera_pose.p_e is None or self.filtered_camera_pose.z_e is None:
            R_init = (
                self._frame_tracking_z_axis(z_raw)
                if raw_pose.R_e is None
                else np.asarray(raw_pose.R_e, dtype=float).reshape(3, 3)
            )
            self.filtered_camera_pose = CameraPose(
                p_e=as_col3(p_raw),
                z_e=as_col3(R_init[:, 2]),
                R_e=R_init,
            )
            return self.filtered_camera_pose

        dt = self.cfg.dt_ctrl
        tau_p = self.cfg.ee_goal_pos_tau
        tau_z = self.cfg.ee_goal_axis_tau
        alpha_p = 1.0 if dt <= 0.0 or tau_p <= 1e-9 else 1.0 - np.exp(-dt / tau_p)
        alpha_z = 1.0 if dt <= 0.0 or tau_z <= 1e-9 else 1.0 - np.exp(-dt / tau_z)

        p_prev = as_flat3(self.filtered_camera_pose.p_e)
        z_prev = safe_normalize(as_flat3(self.filtered_camera_pose.z_e))
        R_prev = (
            self._frame_tracking_z_axis(z_prev)
            if self.filtered_camera_pose.R_e is None
            else np.asarray(self.filtered_camera_pose.R_e, dtype=float).reshape(3, 3)
        )

        p_cmd = p_prev + alpha_p * (p_raw - p_prev)
        pos_rate_max = self.cfg.ee_goal_pos_rate_max
        if dt > 0.0 and pos_rate_max > 0.0:
            max_step = pos_rate_max * dt
            delta_p = p_cmd - p_prev
            delta_norm = float(np.linalg.norm(delta_p))
            if delta_norm > max_step > 0.0:
                p_cmd = p_prev + (max_step / delta_norm) * delta_p

        z_blend = (1.0 - alpha_z) * z_prev + alpha_z * z_raw
        z_cmd = safe_normalize(z_blend)
        if np.linalg.norm(z_cmd) < 1e-9:
            z_cmd = z_prev

        axis_rate_max = self.cfg.ee_goal_axis_rate_max
        if dt > 0.0 and axis_rate_max > 0.0:
            dot_zz = float(np.clip(np.dot(z_prev, z_cmd), -1.0, 1.0))
            angle = float(np.arccos(dot_zz))
            max_angle = axis_rate_max * dt
            if angle > max_angle > 0.0:
                tangent = z_cmd - dot_zz * z_prev
                tan_norm = float(np.linalg.norm(tangent))
                if tan_norm > 1e-9:
                    tangent /= tan_norm
                    z_cmd = safe_normalize(np.cos(max_angle) * z_prev + np.sin(max_angle) * tangent)
                else:
                    z_cmd = z_prev

        R_cmd = self._frame_tracking_z_axis(z_cmd, R_prev=R_prev)
        self.filtered_camera_pose = CameraPose(
            p_e=as_col3(p_cmd),
            z_e=as_col3(R_cmd[:, 2]),
            R_e=R_cmd,
        )
        return self.filtered_camera_pose

    def _enforce_reach_limit(self, st: State, pose: CameraPose) -> CameraPose:
        """
        Final safety clamp on the commanded EE position.

        Target selection is clamped earlier, but subsequent sigma blending and
        low-pass filtering happen in Cartesian space and can drift a few
        millimeters outside the current reach ball as the CoM moves.
        """
        p_c_now = as_flat3(self._ee_kinematics(st)["p_c"])
        p_e = as_flat3(pose.p_e)
        p_ce = p_e - p_c_now
        reach = np.linalg.norm(p_ce)
        reach_max = self.cfg.reachable_arm_radius * self.cfg.reach_ratio
        if reach <= max(reach_max, 0.0) or reach < 1e-12:
            return pose

        p_clamped = p_c_now + (reach_max / reach) * p_ce
        clamped_pose = CameraPose(
            p_e=as_col3(p_clamped),
            z_e=None if pose.z_e is None else as_col3(as_flat3(pose.z_e)),
            R_e=None if pose.R_e is None else np.asarray(pose.R_e, dtype=float).reshape(3, 3),
        )
        self.filtered_camera_pose = clamped_pose
        return clamped_pose

    def _finalize_camera_pose(
        self,
        st: State,
        fallback_goal: CameraPose,
        selection: CameraTarget | None,
    ) -> CameraPose:
        raw_pose = self._camera_goal_from_target(fallback_goal, selection)
        blended_pose = self._sigma_blend_pose(st, raw_pose, fallback_goal)
        filtered_pose = self._filter_camera_pose(blended_pose)
        return self._enforce_reach_limit(st, filtered_pose)

    def _hold_current_target_pose(
        self,
        st: State,
        fallback_goal: CameraPose,
    ) -> CameraPose:
        held = self._held_target_at_com(st)
        return self._finalize_camera_pose(st, fallback_goal, held)
    
    def set_ee_target(
        self,
        st: State,
        des: Desired,
        idx: int | None = None,
        step_idx: int | None = None,
    ) -> CameraPose:

        # Default camera goal from current position and pointing axis.
        fallback_goal = self._default_ee_goal(st)
        if not self._should_use_target_finder():
            return fallback_goal

        # Tick internal step clock.
        step_idx = self._resolve_ee_guidance_step(step_idx)

        # Hold the current target (preserving its p_ce offset) unless an update is due.
        if not self._target_update_due(step_idx=step_idx, st=st):
            return self._hold_current_target_pose(st, fallback_goal)

        # Reselect: ask finder for a new goal along the anchor direction.
        finder = self.target_finder
        finder.active = self.active_camera_target
        p_anchor = self._ee_anchor_point(des, idx=idx)
        selection = finder.choose_goal(p_anchor)

        # Nothing viable — hold.
        if selection.idx is None:
            return self._hold_current_target_pose(st, fallback_goal)

        # Clamp motion to prevent jumps, then record the observation.
        selection = self._clamp_camera_target_motion(st, selection)
        finder.record_target_observation(selection)
        finder.active = selection

        # Track last switch step: bump it when the target changes, or on the first selection.
        prev_idx = self.active_camera_target.idx
        target_changed = (prev_idx is None) or (selection.idx != prev_idx)
        if target_changed or self.last_target_switch_step < 0:
            self.last_target_switch_step = step_idx

        self.active_camera_target = selection
        p_c_now = as_flat3(self._ee_kinematics(st)["p_c"])
        self._active_p_ce = (as_flat3(selection.p_e) - p_c_now).copy()
        self.last_target_update_step = step_idx

        return self._finalize_camera_pose(st, fallback_goal, selection)

    def sample_guidance_goal(self, st: State, des_base: Desired, idx: int | None = None, step_idx: int | None = None,
    ) -> Desired:
        des_new = self.sample_base_goal(des_base, idx=idx)
        sel = self.set_ee_target(st, des_new, idx=idx, step_idx=step_idx)
        des_new.p_e = as_col3(sel.p_e)
        if sel.R_e is None:
            des_new.R_e = self._frame_tracking_z_axis(as_flat3(sel.z_e))
        else:
            des_new.R_e = np.asarray(sel.R_e, dtype=float).reshape(3, 3)
        des_new.z_e = as_col3(des_new.R_e[:, 2])
        return des_new
    
    def add_ee_goal(self, st, des_base, idx, step_idx, des_prev):
        R_prev = None if des_prev is None else des_prev.R_e
        des_new = des_base.copy()
        sel = self.set_ee_target(st, des_base, idx=idx, step_idx=step_idx)
        des_new.p_e = as_col3(sel.p_e)
        if sel.R_e is None:
            des_new.R_e = self._frame_tracking_z_axis(as_flat3(sel.z_e), R_prev=R_prev)
        else:
            des_new.R_e = np.asarray(sel.R_e, dtype=float).reshape(3, 3)
        des_new.z_e = as_col3(des_new.R_e[:, 2])
        return des_new
    
    def _desired_omega_e_world(
        self,
        des: Desired,
        des_prev: Desired | None,
        dt: float | None,
    ) -> np.ndarray:
        """
        Finite-difference desired EE angular velocity in WORLD coordinates.

        The controller subtracts this from `self.motion.nu_e`, which is a LOCAL
        EE twist, so `desired_nu_e_oplus()` rotates the result into the current
        actual EE frame before returning it.
        """
        if des_prev is None or dt is None or dt <= 1e-9:
            return np.zeros((3, 1))
        if des.R_e is None or des_prev.R_e is None:
            return np.zeros((3, 1))

        R_curr = np.asarray(des.R_e, dtype=float).reshape(3, 3)
        R_prev = np.asarray(des_prev.R_e, dtype=float).reshape(3, 3)
        phi_world = so3_log(R_curr @ R_prev.T)
        if not np.all(np.isfinite(phi_world)):
            return np.zeros((3, 1))
        return as_col3(phi_world / float(dt))

    def desired_nu_e_oplus(
        self,
        st: State,
        des: Desired,
        des_prev: Desired | None = None,
        dt: float | None = None,
        nu_e_des_prev: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Desired EE twist expressed in the CURRENT ACTUAL EE local frame.

        This matches `self.motion.nu_e`, so the controller can form the damping
        term with `nu_e - nu_e_des` directly.
        """
        nu_e_des = np.zeros((6, 1))
        if not self.cfg.enable_ee: return nu_e_des

        R_te = np.asarray(self._ee_kinematics(st)["R_e"], dtype=float).reshape(3, 3)

        if self.cfg.ee_nu_des_match_com_velocity:
            # Desired camera point is carried with the platform, so the
            # translational feedforward follows the CoM velocity.
            v_des_world = as_col3(st.v_c)
        elif des_prev is not None and dt is not None and dt > 1e-9:
            v_des_world = (as_col3(des.p_e) - as_col3(des_prev.p_e)) / float(dt)
        else:
            v_des_world = np.zeros((3, 1))

        # Express the desired linear velocity in the current actual EE frame
        # so it can be subtracted from the LOCAL twist `nu_e`.
        nu_e_des[:3] = R_te.T @ v_des_world

        omega_des_world = self._desired_omega_e_world(des, des_prev, dt)
        nu_e_des[3:] = R_te.T @ omega_des_world

        if nu_e_des_prev is not None:
            tau = self.cfg.ee_nu_des_tau
            if tau > 1e-9 and dt is not None and dt > 0.0:
                alpha = float(1.0 - np.exp(-float(dt) / tau))
                nu_prev = np.asarray(nu_e_des_prev, dtype=float).reshape(6, 1)
                nu_e_des = nu_prev + alpha * (nu_e_des - nu_prev)

        nu_e_des_max = self.cfg.ee_nu_des_max
        if nu_e_des_max > 0.0:
            nu_e_des[:3] = as_col3(saturate(nu_e_des[:3], nu_e_des_max))
        return nu_e_des


# ===== End effector nominal guidance rollout
if __name__ == "__main__":
    from CC_Controllers import BaseController
    from utils.recon import RaycastMeshReconstructor
    from utils.pure import save_npz, load_npz

    def _require(condition, message):
        if not condition:
            raise AssertionError(message)

    def _resolve_guidance_window_indices(ctrl, window_indices):
        if window_indices is None:
            return list(range(len(ctrl.windows)))
        out = [int(idx) for idx in window_indices]
        _require(len(out) > 0, "no window indices provided")
        max_window_idx = len(ctrl.windows) - 1
        for idx in out:
            _require(0 <= idx <= max_window_idx, f"window index {idx} out of range for {len(ctrl.windows)} windows")
        return out

    def build_nominal_ee_guidance_rollout(
        traj, ctrl,
        window_indices=None,
        max_steps=None,
        rollout_step_stride=1,
    ):
        cfg = traj.cfg
        dt = cfg.dt_ctrl
        step_dt = dt * rollout_step_stride

        window_indices = _resolve_guidance_window_indices(ctrl, window_indices)
        first_idx = window_indices[0]
        des0 = traj.desired_com_at_window_start(idx=first_idx)
        st = ctrl.initial_state_for_desired(des0)
        ctrl.sync_robot_state(st, update_views=True)
        first_start_idx, _ = traj.window_bounds(idx=first_idx)
        traj.reset_progress(first_start_idx)

        frames = []
        prev_end_idx = None
        fine_step_idx = 0
        total_steps_est = 0
        for idx in window_indices:
            total_steps_est += np.ceil(max(traj.window_duration(idx=idx), step_dt) / step_dt)
        if max_steps is not None:
            total_steps_est = min(total_steps_est, int(max_steps))

        step_pbar = tqdm(total=total_steps_est, desc="Nominal EE rollout", dynamic_ncols=True)
        for idx in tqdm(window_indices, total=len(window_indices), desc="Guidance windows", leave=False, dynamic_ncols=True):
            start_idx, end_idx = traj.window_bounds(idx=idx)
            if prev_end_idx is not None and start_idx > (prev_end_idx + 1):
                traj.reset_progress(start_idx)
                p_restart = as_flat3(traj.orbit_p_cd[start_idx])
                p_now = as_flat3(ctrl.motion.p_tc)
                st.q[:3] += p_restart - p_now
                ctrl.sync_robot_state(st, update_views=True)

            T_window = max(traj.window_duration(idx=idx), step_dt)
            n_steps = int(np.ceil(T_window / step_dt))

            for _ in range(n_steps):
                if max_steps is not None and len(frames) >= int(max_steps):
                    break

                ctrl.sync_robot_state(st, update_views=True)
                p_tc_now = as_flat3(ctrl.motion.p_tc)
                des_com = traj.sample_com_goal(
                    p_tc_now,
                    idx=idx,
                    desired_speed=cfg.desired_com_speed,
                )
                des = traj.sample_guidance_goal(st, des_com, idx=idx, step_idx=fine_step_idx)

                active = traj.active_camera_target
                p_e = as_flat3(des.p_e)
                z_e = safe_normalize(as_flat3(des.z_e))
                if active.p_tgt is not None:
                    p_tgt = as_flat3(active.p_tgt)
                    ray_end = p_tgt.copy()
                else:
                    p_tgt = None
                    ray_end = p_e + cfg.d_far * z_e

                frames.append({
                    "step": len(frames),
                    "fine_step": fine_step_idx,
                    "window_idx": idx,
                    "p_c_now": p_tc_now.copy(),
                    "p_c_des": as_flat3(des.p_c),
                    "p_e": p_e.copy(),
                    "z_e": z_e.copy(),
                    "p_tgt": None if p_tgt is None else p_tgt.copy(),
                    "ray_end": ray_end.copy(),
                    "target_idx": active.idx,
                    "fallback": active.fallback,
                })

                st.q[:3] += step_dt * as_flat3(des.v_c)
                fine_step_idx += rollout_step_stride
                step_pbar.update(1)

            prev_end_idx = end_idx
            if max_steps is not None and len(frames) >= int(max_steps):
                break
        step_pbar.close()

        _require(len(frames) > 0, "nominal EE rollout assembled no frames")
        return frames

    sample_freq = 10
    do_rollout = 1
    models = [
        "GRO", 
        "RCM", 
        "acrim", 
        ]
    
    for model in models:
        cfg = make_params(model_key=model, vps=1, vision=1)
        stem = "rollout"
        if do_rollout:
            ctrl = BaseController(cfg=cfg)
            traj = EEGuidance(cfg, robot=ctrl.robot)
            frames = build_nominal_ee_guidance_rollout(traj=traj, ctrl=ctrl)
            save_npz(stem=stem, cfg=cfg, frames=frames)
        else: 
            try:
                frames = load_npz(fname_stem=stem, cfg=cfg, query="frames")
            except KeyError:
                frames = load_npz(fname_stem=stem, cfg=cfg, query="rollout")
            except FileNotFoundError:
                raise RuntimeError("Run with do_rollout=1/True before running this.")
    
        ee_path = np.asarray([frame["p_e"] for frame in frames], dtype=float)
        ee_pointing = np.asarray([frame["z_e"] for frame in frames], dtype=float)

        poses = np.stack((ee_path, ee_pointing), axis=1)

        recon = RaycastMeshReconstructor(model)
        mesh = recon.build_tsdf_from_poses(poses[::sample_freq])
        o3d.visualization.draw_geometries([mesh])
        mesh_stem = f"nominal_mesh_{model}"
        mesh_fname = next_filename(mesh_stem, "meshes", ".ply")
        o3d.io.write_triangle_mesh(mesh_fname, mesh)
