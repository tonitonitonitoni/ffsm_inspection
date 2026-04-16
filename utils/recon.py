from pathlib import Path
import sys
import numpy as np
import open3d as o3d
import open3d.core as o3c # type: ignore
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.pure import safe_normalize, next_filename, last_npz, as_flat3
from utils.params import make_params
from utils.mesh_manipulation import calc_up_vector,prep_mesh

class RaycastMeshReconstructor:
    
    """
    Fuse Open3D tensor-raycasting hits into a TSDF volume and extract a mesh/point cloud.
    Works with rays created by o3d.t.geometry.RaycastingScene.create_rays_pinhole(...).
    """

    def __init__(self,
                 model_key: str = "GRO"):
        
        color_enum = o3d.pipelines.integration.TSDFVolumeColorType
        color_type = getattr(color_enum, 'NoColor')
        self.params = make_params(model_key, vps=True, vision=True)
        self._color_type = color_type
        self.volume = self._make_volume()
        self.width, self.height = self.params.tsdf_resolution
        self.fov_deg = self.params.camera_fov_deg
        
        self.gt_mesh = prep_mesh(self.params.mesh_path)
        self._fixed_gt_pts = None

        t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(self.gt_mesh)
        self.ray_scene = o3d.t.geometry.RaycastingScene()
        _ = self.ray_scene.add_triangles(t_mesh)

    def _make_volume(self):
        return o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.params.tsdf_voxel_length,
            sdf_trunc=self.params.tsdf_trunc,
            color_type=self._color_type
        )

    #@staticmethod
    def intrinsics_from_fov(self) -> o3d.camera.PinholeCameraIntrinsic:
        """
        Construct intrinsics assuming given horizontal FOV (deg) used in create_rays_pinhole.
        """
        width_px = self.width
        height_px = self.height
        fov = np.deg2rad(self.fov_deg)
        fx = width_px / (2.0 * np.tan(fov / 2.0))
        fy = fx 
        cx = width_px * 0.5
        cy = height_px * 0.5
        K = o3d.camera.PinholeCameraIntrinsic(width_px, height_px, fx, fy, cx, cy)
        return K

    @staticmethod
    def _world_to_camera_extrinsic(eye, forward, up_hint):
        eye = as_flat3(eye)
        f = safe_normalize(as_flat3(forward))
        up_hint = safe_normalize(as_flat3(up_hint))

        # real orthonormal up
        r = np.cross(up_hint, f)
        r = safe_normalize(r) 
        u = np.cross(f, r)
        u = safe_normalize(u)

        R_wc = np.column_stack((r, u, f))
        R_cw = R_wc.T
        t_c = -R_cw @ eye

        extrinsic = np.eye(4)
        extrinsic[:3,:3] = R_cw
        extrinsic[:3,3] = t_c
        return extrinsic

    def _depth_from_t_hit(self, result):
        width_px = self.width
        height_px = self.height

        t_hit = result["t_hit"].cpu().numpy()
        depth = np.asarray(t_hit.reshape(height_px, width_px), dtype=np.float32)
        invalid = ~np.isfinite(depth) | (depth <= 0.0)
        depth[invalid] = 0.0
        return o3d.geometry.Image(depth)
    
    def rgbd_from_depth(self, result, forward: np.ndarray, rays_tensor: o3c.Tensor | None = None):
        """Creates a dummy image and optionally filters for max incident angle"""
        depth_img = self._depth_from_t_hit(result)

        # Create a dummy grayscale image to satisfy RGBD interface if color_type is RGB        
        color_img = o3d.geometry.Image(np.full((self.height, self.width, 3), 128, dtype=np.uint8))

        # reject extremely near hits to avoid TSDF exploding surface noise
        depth_np = np.asarray(depth_img)
        near_thresh = self.params.tsdf_trunc #self.volume.voxel_length * self.params.near_thresh
        depth_np[depth_np < near_thresh] = 0.0
        depth_img = o3d.geometry.Image(depth_np)

        # mask based on angle threshold if rays_tensor is provided
        if rays_tensor is not None:
            rays_np = rays_tensor.cpu().numpy()
            ray_dirs = rays_np[...,3:6]
            ray_dirs = ray_dirs / (np.linalg.norm(ray_dirs,axis=-1,keepdims=True)+1e-9)
            fwd = safe_normalize(forward) #forward/ (np.linalg.norm(forward)+1e-9)
            dots = np.einsum('ijk,k->ij', ray_dirs, fwd)
            dot_mask = dots > self.params.dot_thresh_recon
            depth_np = np.asarray(depth_img)
            depth_np[~dot_mask] = 0.0
            depth_img = o3d.geometry.Image(depth_np)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img, depth_img,
            depth_scale=1.0,           # t_hit already in meters
            depth_trunc=1e6,           # large trunc so we don't clip legitimate hits
            convert_rgb_to_intensity=False
        )
        return rgbd
    
    def new_integrate_raycast(self, eye_pos, fwd):
        fov_deg = self.params.camera_fov_deg
        res_x, res_y = self.params.tsdf_resolution

        eye_pos = as_flat3(eye_pos)
        forward = safe_normalize(as_flat3(fwd))
        up = calc_up_vector(forward)
        
        rays_tensor = self.ray_scene.create_rays_pinhole(
            fov_deg=fov_deg,
            center=(eye_pos + forward).tolist(),
            eye=eye_pos.tolist(),
            up=up.tolist(),
            width_px=res_x,
            height_px=res_y,
        ).to(o3c.Device("CPU:0"))

        result = self.ray_scene.cast_rays(rays_tensor)
        intrinsic = self.intrinsics_from_fov()
        extrinsic = self._world_to_camera_extrinsic(eye_pos, forward, up)
        rgbd = self.rgbd_from_depth(result, forward, rays_tensor)
        self.volume.integrate(rgbd, intrinsic, extrinsic)

    def extract_mesh(self) -> o3d.geometry.TriangleMesh:
        mesh = self.volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        return mesh


    def build_tsdf_from_poses(self, poses=None):
        """Rebuild the TSDF volume from poses, then return the mesh.
        """
        for cam_pos, cam_dir in tqdm(poses):
            self.new_integrate_raycast(cam_pos, cam_dir)
        return self.extract_mesh()


def states_to_poses(states, robot=None, sample_freq=10):
    #states needs to be a namespace
    X = states.x
    if robot is None:

        from utils.robot import GiordanoRobot
        robot = GiordanoRobot()

    def split(x):
        arr = np.asarray(x).reshape(-1)
        q = arr[:robot.nq]
        v = arr[robot.nq:]
        return q, v

    poses = []
    for xk in X:
        p_te, z_te = robot.cam_terms(*split(xk))
        poses.append((
            np.asarray(p_te).copy(),
            np.asarray(z_te).copy(),
        ))

    return poses[::sample_freq]



if __name__ == "__main__":

    fname = last_npz("states", cfg=None, add=False)
    data = np.load(fname, allow_pickle=True)
    states = data["states"]
    if isinstance(states, np.ndarray) and states.dtype == object:
        states = states.item() if states.shape == () else list(states.flat)
    else:
        states = states

    poses = states_to_poses(states)

    recon = RaycastMeshReconstructor()
    mesh = recon.build_tsdf_from_poses(poses)
    o3d.visualization.draw_geometries([mesh])
    mesh_stem = "mesh_recon"
    mesh_fname = next_filename(mesh_stem, "meshes", ".ply")
    o3d.io.write_triangle_mesh(mesh_fname, mesh)
