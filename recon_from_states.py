from utils.recon import RaycastMeshReconstructor, states_to_poses
from utils.pure import next_filename, load_npz
import open3d as o3d


if __name__ == "__main__":
    sample_freq = 25 #decrease to include all the samples. Careful, it increases processing time
    models = [
            "RCM",
            "GRO",
            "acrim",
            ]
    for model in models:
        try:
            states = load_npz(f"states", model_key=model)
        except FileNotFoundError:
            raise RuntimeError("Run all_models.py or CC_Controllers.py with save_states = 1 ")
        poses = states_to_poses(states, sample_freq=sample_freq)
        recon = RaycastMeshReconstructor(model_key=model)
        mesh = recon.build_tsdf_from_poses(poses)
        o3d.visualization.draw_geometries([mesh])
        mesh_stem = f"mesh_recon_{model}"
        mesh_fname = next_filename(mesh_stem, "meshes", ".ply")
        o3d.io.write_triangle_mesh(mesh_fname, mesh)
