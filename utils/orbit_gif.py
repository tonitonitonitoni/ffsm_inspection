from pathlib import Path
import sys
import numpy as np
import open3d as o3d
import pinocchio as pin
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if __package__ in {None, ""}:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.pure import next_filename, as_flat3, safe_normalize, last_npz


def load_states(state_path: Path):
    return np.load(state_path, allow_pickle=True)["states"].item()

def sample_indices(n_samples: int, max_frames: int) -> np.ndarray:
    n_frames = min(n_samples, max_frames)
    return np.unique(np.linspace(0, n_samples - 1, n_frames, dtype=int))

def load_com_positions(state_path: Path, robot_path: Path) -> np.ndarray:
    states = load_states(state_path)
    model, _, _ = pin.buildModelsFromMJCF(str(robot_path))
    data = model.createData()

    q_all = np.asarray(states.x[:, :model.nq], dtype=float)
    orbit = np.empty((len(q_all), 3), dtype=float)

    for out_i, q in enumerate(q_all):
        pin.centerOfMass(model, data, q)
        orbit[out_i] = np.asarray(data.com[0], dtype=float)

    return orbit

def make_orbit_lines(points: np.ndarray, color: tuple[float, float, float]) -> o3d.geometry.LineSet:
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    if len(points) >= 2:
        lines = np.column_stack([np.arange(len(points) - 1), np.arange(1, len(points))])
        line_set.lines = o3d.utility.Vector2iVector(lines)
        colors = np.repeat(np.asarray(color, dtype=float)[None, :], len(lines), axis=0)
        line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def load_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(str(path))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.72, 0.74, 0.78])
    return mesh

def make_marker(position: np.ndarray) -> o3d.geometry.TriangleMesh:
    marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.06)
    marker.compute_vertex_normals()
    marker.paint_uniform_color([0.86, 0.19, 0.17])
    marker.translate(position)
    return marker

def configure_camera(vis: o3d.visualization.Visualizer, orbit: np.ndarray) -> None:
    ctr = vis.get_view_control()
    lookat = orbit.mean(axis=0)
    radius = 100*max(np.linalg.norm(orbit - lookat, axis=1).max(), 1e-6)

    eye = lookat + radius * np.array([2.4, -1.8, 1.0], dtype=float)
    front = safe_normalize(lookat - eye)

    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    right = np.cross(front, world_up)
    if np.linalg.norm(right) < 1e-9:
        world_up = np.array([0.0, 1.0, 0.0], dtype=float)
        right = np.cross(front, world_up)
    right = safe_normalize(right)
    up = safe_normalize(np.cross(right, front))

    ctr.set_lookat(as_flat3(lookat).tolist())
    ctr.set_front(as_flat3(front).tolist())
    ctr.set_up(as_flat3(up).tolist())
    ctr.set_zoom(0.72)
    ctr.set_constant_z_far(1000.0)

def capture_frame(vis: o3d.visualization.Visualizer) -> Image.Image:
    vis.poll_events()
    vis.update_renderer()
    frame = np.asarray(vis.capture_screen_float_buffer(do_render=True), dtype=float)
    frame = np.clip(255.0 * frame, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(frame)

def create_window(vis: o3d.visualization.Visualizer) -> None:
    width, height = IMAGE_SIZE
    try:
        vis.create_window(width=width, height=height, visible=False)
    except TypeError:
        vis.create_window(width=width, height=height)

def render_gif(
    mesh: o3d.geometry.TriangleMesh,
    display_orbit: np.ndarray,
    marker_orbit: np.ndarray,
    output_path: Path,
) -> None:
    vis = o3d.visualization.Visualizer()
    create_window(vis)

    full_orbit = make_orbit_lines(display_orbit, (0.16, 0.39, 0.73))

    vis.add_geometry(mesh)
    if len(display_orbit) >= 2:
        vis.add_geometry(full_orbit)
    vis.add_geometry(marker := make_marker(marker_orbit[0]))

    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([0.96, 0.97, 0.98], dtype=float)
    render_opt.mesh_show_back_face = True
    render_opt.light_on = True

    configure_camera(vis, display_orbit)
    vis.poll_events()
    vis.update_renderer()

    frames = []
    prev = marker_orbit[0].copy()
    for point in tqdm(marker_orbit, total=len(marker_orbit)):
        marker.translate(point - prev)
        prev = point.copy()

        vis.update_geometry(marker)
        frames.append(capture_frame(vis))

    vis.destroy_window()

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=35,
        loop=0,
        optimize=False,
    )

if __name__ == "__main__":
    from utils.params import MODEL_LIBRARY
    ROOT = PROJECT_ROOT
    MODEL_PATH = ROOT / "models" 
    FIGURE_DIR = ROOT / "sample_figures"
    STATE_DIR = ROOT / "npz"
    ROBOT_PATH = MODEL_PATH / "ur3" / "ur3_box_limited_with_capsules.xml"
    MAX_FRAMES = 360
    DISPLAY_ORBIT_SAMPLES = 12000
    IMAGE_SIZE = (960, 720)
    
    models = [
        "GRO", 
        "RCM", 
        "acrim"
        ]
    for model in models:
        state_path = last_npz("states", model_key=model)
        
        mesh_path = MODEL_PATH / MODEL_LIBRARY[model]["path"]

        output_path = FIGURE_DIR / next_filename(f"com_orbit_{model}", folder=FIGURE_DIR, suffix=".gif")
        all_com = load_com_positions(state_path, ROBOT_PATH)
        display_idx = sample_indices(len(all_com), DISPLAY_ORBIT_SAMPLES)
        marker_idx = sample_indices(len(all_com), MAX_FRAMES)

        render_gif(
            load_mesh(mesh_path),
            all_com[display_idx],
            all_com[marker_idx],
            output_path,
        )
        print(f"Rendered GIF to {output_path}")
