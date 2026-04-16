import time, mujoco, numpy as np, mediapy as media
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from utils.pure import next_filename
from utils.robot import GiordanoRobot
from utils.params import make_params

class FFSim:
    def __init__(self, cfg=None, robot_name="UR3", view_chaser=True, add_stars=False, live=False, q_pin=None):
        self.robot = GiordanoRobot(robot_name)
        self.cfg = cfg if cfg is not None else make_params()
        self.view_chaser = view_chaser
        self.add_stars = add_stars
        self.live = live
        self.spec = self.create_scene(view_chaser=view_chaser, add_stars=add_stars, live=live, q_pin=q_pin)
        self._bind_model()

    def _bind_model(self):
        self.mj_model = self.spec.compile()
        self.mj_data = mujoco.MjData(self.mj_model)
        self.nq = self.mj_model.nq
        self.nv = self.mj_model.nv
        self.nu = int(self.mj_model.nu)
        ctrlrange = np.asarray(self.mj_model.actuator_ctrlrange, dtype=float)
        if self.nu > 0 and ctrlrange.size > 0:
            self.ctrl_min = ctrlrange[:, 0].copy()
            self.ctrl_max = ctrlrange[:, 1].copy()
        else:
            self.ctrl_min = np.empty(0, dtype=float)
            self.ctrl_max = np.empty(0, dtype=float)
        return self

    def create_scene(self, view_chaser=True, add_stars=False, live=False, q_pin=None):
        spec = mujoco.MjSpec()
        spec.option.gravity = [0, 0, 0]
        spec.visual.global_.offwidth = 1200
        spec.visual.global_.offheight = 800
        add_directional_light(
            spec.worldbody,
            name="key_light",
            pos=[8, -6, 10],
            dir=[-0.5, 0.3, -1.0],
            diffuse=[0.9, 0.9, 0.9],
            specular=[0.25, 0.25, 0.25],
            ambient=[0.35, 0.35, 0.35],
        )
        add_directional_light(
            spec.worldbody,
            name="fill_light",
            pos=[-6, 4, 7],
            dir=[0.4, -0.2, -1.0],
            diffuse=[0.55, 0.55, 0.6],
            specular=[0.08, 0.08, 0.08],
            ambient=[0.18, 0.18, 0.2],
        )
        
        if add_stars:
            _ = spec.add_texture(
                name="stars",
                type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, 
                rgb1=[.4, .6, .8],
                rgb2 = [0, 0, 0],
                width=1200, height=800,
                mark = mujoco.mjtMark.mjMARK_RANDOM,
                markrgb = [1, 1, 1],
                )
        else:
            _ = spec.add_texture(
                name="sky",
                type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT, 
                rgb1=[1,1,1],
                width=1200, height=800,
                )

        q_mj = pin_to_mj_qpos(q_pin if q_pin is not None else self.robot.q0)
        chaser_spawn_pos = q_mj[:3]
        chaser_spawn_quat = q_mj[3:7]

        mesh_path = self.cfg.mesh_path
        if mesh_path is not None:
            target_mesh = spec.add_mesh(refpos=[0,0,0])
            target_mesh.name="target"
            target_mesh.file = mesh_path
            geom = spec.worldbody.add_geom()
            geom.type = mujoco.mjtGeom.mjGEOM_MESH
            geom.meshname="target"
            geom.name="target"
            geom.rgba = [0.72, 0.74, 0.78, 1.0]
        
        if view_chaser:
            robot_path = self.cfg.robot_path
            chaser = mujoco.MjSpec.from_file(robot_path)
            frame = spec.worldbody.add_frame(pos=chaser_spawn_pos, 
                                            quat=chaser_spawn_quat)
            frame.attach_body(chaser.body('box_base'))

        if live:
            viewpoint = spec.worldbody.add_body(name="vp_tgt", mocap=True)
            viewpoint.add_geom(name="red_box",
                            type=mujoco.mjtGeom.mjGEOM_BOX,
                            size=[.05, .05, .05], 
                            rgba=[1, 0, 0, .3])
        return spec

    def rebuild(self, q_pin=None, view_chaser=None, add_stars=None, live=None):
        if view_chaser is not None:
            self.view_chaser = view_chaser
        if add_stars is not None:
            self.add_stars = add_stars
        if live is not None:
            self.live = live
        self.spec = self.create_scene(
            view_chaser=self.view_chaser,
            add_stars=self.add_stars,
            live=self.live,
            q_pin=q_pin,
        )
        return self._bind_model()

    def _pin_state_to_mj(self, q_pin_or_x):
        x = np.asarray(q_pin_or_x, dtype=float).reshape(-1)
        q_pin = x[:self.nq]
        qpos = pin_to_mj_qpos(q_pin)
        qvel = np.zeros(self.nv, dtype=float)
        if x.size >= self.nq + self.nv:
            qvel[:] = x[self.nq:self.nq + self.nv]
        return qpos, qvel

    def display_config(self, q_pin):
        qpos, _ = self._pin_state_to_mj(q_pin)
        self.mj_data.qpos[:] = qpos[:self.nq]
        self.mj_data.qvel[:] = 0.0
        self.mj_data.ctrl[:] = 0.0
        self.mj_data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self.mj_model, self.mj_data)
        return self.mj_data

    def sim_config(self, q_pin, u):
        qpos, qvel = self._pin_state_to_mj(q_pin)
        self.mj_data.qpos[:] = qpos[:self.nq]
        self.mj_data.qvel[:] = qvel

        if self.nu == 0:
            self.mj_data.qfrc_applied[:] = 0.0
            mujoco.mj_forward(self.mj_model, self.mj_data)
            return self.mj_data

        u = np.asarray(u, dtype=float).reshape(-1)
        if u.size == 0:
            self.mj_data.ctrl[:] = 0.0
            self.mj_data.qfrc_applied[:] = 0.0
            mujoco.mj_forward(self.mj_model, self.mj_data)
            return self.mj_data
        if u.size < self.nu:
            u_full = np.zeros(self.nu, dtype=float)
            u_full[:u.size] = u
            u = u_full
        elif u.size > self.nu:
            u = u[:self.nu]

        np.clip(u, self.ctrl_min, self.ctrl_max, out=u)
        self.mj_data.ctrl[:] = 0.0
        self.mj_data.qfrc_applied[:] = 0.0
        self.mj_data.qfrc_applied[:self.nu] = u
        mujoco.mj_forward(self.mj_model, self.mj_data)
        return self.mj_data

    def collision_count(self, q_pin):
        self.display_config(q_pin)
        if self.mj_data.ncon == 0:
            return 0
        return count_contacts_no_capsules(self.mj_model, self.mj_data)

    def collisions(self, q_pin):
        return self.collision_count(q_pin)

    def collision_free(self, q_pin):
        return self.collision_count(q_pin) == 0

    def set_mocap_target(self, pos, quat=None):
        body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "vp_tgt")
        if body_id < 0:
            return False
        mocap_id = self.mj_model.body_mocapid[body_id]
        if mocap_id < 0:
            return False
        self.mj_data.mocap_pos[mocap_id] = np.asarray(pos, dtype=float)
        if quat is not None:
            self.mj_data.mocap_quat[mocap_id] = np.asarray(quat, dtype=float)
        return True

    def _named_body_position(self, names):
        for name in names:
            if not name:
                continue
            body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                return np.array(self.mj_data.xpos[body_id], dtype=float)
        return None

    def _named_geom_position(self, names):
        for name in names:
            if not name:
                continue
            geom_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom_id >= 0:
                return np.array(self.mj_data.geom_xpos[geom_id], dtype=float)
        return None

    def configure_camera(self, cam):
        base_names = [self.cfg.base_frame, "box_base", "base_link"]
        robot_pos = self._named_body_position(base_names)
        target_pos = self._named_geom_position(["target"])
        target_radius = float(self.cfg.radius)
        robot_radius = float(self.cfg.reachable_arm_radius) + 0.35

        if robot_pos is None and target_pos is None:
            lookat = np.zeros(3, dtype=float)
            distance = max(1.5, 0.3 * float(self.mj_model.stat.extent))
            azimuth = 135.0
            elevation = -18.0
        elif robot_pos is None:
            lookat = target_pos
            distance = max(2.0, 1.25 * max(target_radius, 0.4 * float(self.mj_model.stat.extent)))
            azimuth = 135.0
            elevation = -18.0
        elif target_pos is None:
            lookat = robot_pos.copy()
            lookat[2] += 0.2
            distance = 1.8
            azimuth = 135.0
            elevation = -18.0
        else:
            seg = robot_pos - target_pos
            separation = float(np.linalg.norm(seg))
            lookat = 0.65 * target_pos + 0.35 * robot_pos
            lookat[2] += 0.12 * target_radius

            seg_xy = seg.copy()
            seg_xy[2] = 0.0
            side = np.cross(np.array([0.0, 0.0, 1.0]), seg_xy)
            if np.linalg.norm(side) <= 1e-9:
                side = np.array([1.0, -1.0, 0.0])
            side /= np.linalg.norm(side)

            framing_radius = target_radius + 0.6 * separation + robot_radius
            distance = np.clip(1.8 * framing_radius, 6.0, 18.0)
            offset = distance * side + np.array([0.0, 0.0, 0.22 * distance])
            azimuth, elevation = azimuth_elevation_from_offset(offset)
            elevation = np.clip(elevation, -25.0, 20.0)

        cam.lookat = lookat
        cam.distance = distance
        cam.azimuth = azimuth
        cam.elevation = elevation
        return cam

    def display_traj(self, q_traj, u_traj=None, sim=True, live=None, fps=10, debug=None, body_cam=False):
        del debug
        if live is None:
            live = self.live
        if u_traj is not None:
            u_arr = np.asarray(u_traj, dtype=float)
            if u_arr.size == 0 or (u_arr.ndim >= 2 and u_arr.shape[-1] == 0):
                u_traj = None
                sim = False

        n_total = 0
        if live:
            from mujoco import viewer as mj_viewer
            with mj_viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
                camera_ready = False
                while viewer.is_running():
                    if sim and u_traj is not None:
                        for i, u in enumerate(u_traj):
                            self.sim_config(q_traj[i], u)
                            if not camera_ready:
                                self.configure_camera(viewer.cam)
                                camera_ready = True
                            viewer.user_scn.ngeom = 0
                            n_fr = count_contacts_no_capsules(self.mj_model, self.mj_data)
                            n_total += n_fr
                            time.sleep(1.0 / fps)
                            viewer.sync()
                    else:
                        for q in q_traj:
                            self.display_config(q)
                            if not camera_ready:
                                self.configure_camera(viewer.cam)
                                camera_ready = True
                            viewer.user_scn.ngeom = 0
                            n_fr = count_contacts_no_capsules(self.mj_model, self.mj_data)
                            n_total += n_fr
                            time.sleep(1.0 / fps)
                            viewer.sync()
        else:
            frames = []
            with mujoco.Renderer(self.mj_model, width=900, height=600) as renderer:
                cam = "body_camera" if body_cam else mujoco.MjvCamera()
                camera_ready = False
                if sim and u_traj is not None:
                    total = min(len(q_traj), len(u_traj))
                    for q, u in tqdm(zip(q_traj, u_traj), total=total, desc="Rendering"):
                        self.sim_config(q, u)
                        if not body_cam and not camera_ready:
                            self.configure_camera(cam)
                            camera_ready = True
                        renderer.update_scene(self.mj_data, camera=cam)
                        n_fr = count_contacts_no_capsules(self.mj_model, self.mj_data)
                        n_total += n_fr
                        pixels = renderer.render()
                        frames.append(add_text(pixels, f"collisions: {n_fr}"))
                else:
                    for q in tqdm(q_traj, total=len(q_traj), desc="Rendering"):
                        self.display_config(q)
                        if not body_cam and not camera_ready:
                            self.configure_camera(cam)
                            camera_ready = True
                        renderer.update_scene(self.mj_data, camera=cam)
                        n_fr = count_contacts_no_capsules(self.mj_model, self.mj_data)
                        n_total += n_fr
                        pixels = renderer.render()
                        frames.append(add_text(pixels, f"collisions: {n_fr}"))
            stem = "traj_sim"
            stem += f"_{self.cfg.model_key}"
            if body_cam: stem += "_ee_cam"
            fname = next_filename(stem, "videos", ".mp4")
            if n_total == 0:
                print("Great success!")
            media.write_video(fname, frames, fps=fps)
            print(f"Video written to {fname}")
        return n_total

    def run(self, states, live=False, body_cam=False, fps=30, N=None):
        x_traj = states.x
        u_traj = states.u
        if N is not None:
            while len(x_traj) > N:
                x_traj = x_traj[::2]
                u_traj = u_traj[::2]
        return self.display_traj(x_traj, u_traj=u_traj, fps=fps, live=live, body_cam=body_cam)
# -----------------------
# Mujoco <-> Pinocchio conversion
# -----------------------
def pin_to_mj_qpos(q_pin):
    pos = q_pin[0:3]
    x, y, z, w = q_pin[3:7]
    quat_wxyz = [w, x, y, z]
    joint_angles = q_pin[7:]
    return np.concatenate([pos, quat_wxyz, joint_angles])

def mj_to_pin_qpos(q_mj):
    pos = q_mj[0:3]
    w, x, y, z = q_mj[3:7]
    quat_xyzw = [x, y, z, w]
    joint_angles = q_mj[7:]
    return np.concatenate([pos, quat_xyzw, joint_angles])


def add_directional_light(worldbody, **kwargs):
    kwargs = dict(kwargs)
    kwargs.pop("directional", None)

    try:
        light_type = mujoco.mjtLight.mjLIGHT_DIRECTIONAL
    except AttributeError:
        light_type = 1

    kwargs["type"] = light_type
    return worldbody.add_light(**kwargs)


def azimuth_elevation_from_offset(offset):
    offset = np.asarray(offset, dtype=float)
    horiz = np.linalg.norm(offset[:2])
    if horiz <= 1e-9:
        azimuth = 135.0
    else:
        azimuth = np.degrees(np.arctan2(offset[1], offset[0]))
    elevation = np.degrees(np.arctan2(offset[2], max(horiz, 1e-9)))
    return azimuth, elevation


def count_contacts_no_capsules(mj_model, mj_data):
    mjOBJ_GEOM = mujoco.mjtObj.mjOBJ_GEOM.value
    seen = set()
    n_contacts = 0
    for ci in range(mj_data.ncon):
        contact = mj_data.contact[ci]
        g1, g2 = contact.geom[0], contact.geom[1]
        name1 = mujoco.mj_id2name(mj_model, mjOBJ_GEOM, g1)
        name2 = mujoco.mj_id2name(mj_model, mjOBJ_GEOM, g2)

        if name1 is None or name2 is None:
            continue
        if name1.endswith('_capsule') or name2.endswith('_capsule'):
            continue

        key = tuple(sorted((name1, name2)))
        if name1 == name2 or key in seen:
            continue
        seen.add(key)
        n_contacts += 1
    return n_contacts


def add_text(frame, text):
    image = Image.fromarray(np.array(frame, copy=True))
    draw = ImageDraw.Draw(image, "RGBA")
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
    except OSError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad = 10
    margin = 16
    x0 = margin
    y0 = margin
    x1 = x0 + text_w + 2 * pad
    y1 = y0 + text_h + 2 * pad

    draw.rounded_rectangle((x0, y0, x1, y1), radius=8, fill=(255, 255, 255, 170))
    draw.text((x0 + pad, y0 + pad), text, font=font, fill=(220, 30, 30, 255))
    return np.array(image, copy=False)
