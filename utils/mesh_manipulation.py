import open3d as o3d, numpy as np
from typing import Tuple
from utils.pure import safe_normalize

# STEP 0 - Preprocess Satellite
def sample_points_on_mesh(mesh: o3d.geometry.TriangleMesh, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample exactly n_samples points on a triangle mesh surface and return points and normals.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")
    if len(mesh.triangles) == 0:
        raise ValueError("mesh must contain triangles")

    mesh.compute_triangle_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=n_samples, use_triangle_normal=True)
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    if points.shape[0] != n_samples:
        raise RuntimeError(f"Expected {n_samples} samples, got {points.shape[0]}")
    print(f"Sampled {points.shape[0]} points uniformly.")
    return points, normals


# -------------------------------------------------
# Triangle area weights
# -------------------------------------------------

def compute_triangle_areas(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """
    Compute area of each triangle in the mesh.
    Returns:
        areas: (n_tri,) array of triangle areas
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]

    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)

    return areas


def compute_triangle_area_weights(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """
    Normalized triangle area weights (sum to 1).
    """
    areas = compute_triangle_areas(mesh)
    total_area = np.sum(areas)
    if total_area <= 0:
        raise ValueError("Mesh has zero total area.")
    return areas / total_area

# -----------------------
# Mesh manipulation
# -----------------------
def prep_mesh(mesh_path):
    o3d_mesh = o3d.io.read_triangle_mesh(mesh_path)
    o3d_mesh = make_watertight(o3d_mesh)
    o3d_mesh.orient_triangles()
    o3d_mesh.compute_triangle_normals()
    return o3d_mesh

def make_watertight(mesh):
    """ in: o3d mesh. 
        out: o3d mesh."""
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.compute_triangle_normals()
    return mesh.compute_vertex_normals()

# -----------------------
# Open3D helper
# -----------------------
def calc_up_vector(dirn):
    forward = np.asarray(dirn, dtype=float)
    norm = np.linalg.norm(forward)
    if norm < 1e-9:
        return None
    forward /= norm

    up_world = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(np.dot(forward, up_world)) > 0.99:
        up_world = np.array([0.0, 1.0, 0.0], dtype=float)
    right = np.cross(forward, up_world)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-9:
        return None
    right /= right_norm

    return safe_normalize(np.cross(right, forward))



def smooth_radial_profile(r: np.ndarray, window: int = 9) -> np.ndarray:
    """Simple moving-average smoothing for r(s)."""
    r = _interp_nan_1d(r)
    w = int(window)
    if w <= 1:
        return r
    if w % 2 == 0:
        w += 1
    pad = w // 2
    rp = np.pad(r, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(rp, kernel, mode="valid")

def _interp_nan_1d(y: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs in a 1D array (endpoints extended)."""
    y = np.asarray(y, dtype=float)
    x = np.arange(y.size)
    good = np.isfinite(y)
    if np.all(good):
        return y
    if np.sum(good) < 2:
        return np.nan_to_num(y, nan=0.0)
    y2 = y.copy()
    y2[~good] = np.interp(x[~good], x[good], y[good])
    return y2

def pca_frame_from_points(points: np.ndarray, eps: float = 1e-12):
    """Compute a right-handed PCA frame from 3D points.

    Args:
        points: (N,3) array of 3D points.
        eps: small value to guard degeneracy.

    Returns:
        centroid: (3,) mean of points
        axes: (3,3) columns are principal directions [a0, a1, a2] (unit),
              ordered by descending variance.
        evals: (3,) eigenvalues (variances along axes), descending.
    """
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3 or P.shape[0] < 3:
        raise ValueError(f"points must be (N,3) with N>=3, got {P.shape}")

    centroid = P.mean(axis=0)
    X = P - centroid

    # Covariance (normalized by N)
    C = (X.T @ X) / max(X.shape[0], 1)

    # Symmetric eigendecomposition
    evals, evecs = np.linalg.eigh(C)

    # Sort descending
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    # Normalize eigenvectors (should already be unit, but be safe)
    for k in range(3):
        n = np.linalg.norm(evecs[:, k])
        if n > eps:
            evecs[:, k] /= n

    # Enforce right-handed frame: a2 = a0 x a1, and re-orthonormalize a1
    a0 = evecs[:, 0]
    a1 = evecs[:, 1]
    a2 = np.cross(a0, a1)
    n2 = np.linalg.norm(a2)
    if n2 > eps:
        a2 /= n2
        # Recompute a1 to be exactly orthogonal
        a1 = np.cross(a2, a0)
        n1 = np.linalg.norm(a1)
        if n1 > eps:
            a1 /= n1
    else:
        # Degenerate case: points nearly collinear; pick an arbitrary orthogonal completion
        # Choose a vector not parallel to a0
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, a0)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        a2 = np.cross(a0, tmp)
        a2 /= max(np.linalg.norm(a2), eps)
        a1 = np.cross(a2, a0)
        a1 /= max(np.linalg.norm(a1), eps)

    axes = np.column_stack([a0, a1, a2])
    return centroid, axes, evals


# --- Radial slicing and profile plotting ---
def slice_mesh_radial_profile(
    mesh,
    axis: np.ndarray,
    centroid: np.ndarray,
    n_slices: int = 80,
    r_percentile: float = 95.0,
):
    """Slice mesh along a principal axis and compute a radial profile r(s).

    Args:
        mesh: open3d TriangleMesh (already loaded)
        axis: (3,) unit vector (principal axis, e.g. PC1)
        centroid: (3,) mesh centroid
        n_slices: number of axial slices
        r_percentile: percentile of radius per slice (robust to outliers)

    Returns:
        s_centers: (n_slices,) slice center coordinates along axis
        r_vals: (n_slices,) radial profile
    """
    if not mesh.has_vertices():
        raise RuntimeError("Mesh has no vertices")

    axis = axis / np.linalg.norm(axis)
    verts = np.asarray(mesh.vertices, dtype=float)

    # Project vertices onto axis
    rel = verts - centroid[None, :]
    s = rel @ axis  # (N,)

    s_min, s_max = np.min(s), np.max(s)
    bins = np.linspace(s_min, s_max, n_slices + 1)

    s_centers = 0.5 * (bins[:-1] + bins[1:])
    r_vals = np.zeros(n_slices)

    for i in range(n_slices):
        mask = (s >= bins[i]) & (s < bins[i + 1])
        if not np.any(mask):
            r_vals[i] = np.nan
            continue

        # Radial distance to axis: ||x - (x·a)a||
        rel_i = rel[mask]
        proj = np.outer(rel_i @ axis, axis)
        radial = np.linalg.norm(rel_i - proj, axis=1)

        r_vals[i] = np.percentile(radial, r_percentile)

    return s_centers, r_vals

def generate_envelope_profile(cfg):
          
    mesh = o3d.io.read_triangle_mesh(cfg.mesh_path)

    verts = np.asarray(mesh.vertices, dtype=float)

    # --- PCA on mesh ---
    centroid, axes, evals = pca_frame_from_points(verts)
    axis = axes[:, 0]
    # print(np.round(evals, 2))
    # --- Slice mesh to get envelope radius profile ---
    
    s_centers, r_vals = slice_mesh_radial_profile(
        mesh,
        axis=axis,
        centroid=centroid,
        n_slices=80,
        r_percentile=cfg.r_percentile,
    )

    # --- Smooth envelope ---
    r_smooth = smooth_radial_profile(r_vals, window=9)

    # --- Attach envelope to OrbitGenerator ---
    envelope_profile = dict(
        s_centers=np.asarray(s_centers, float),
        r_profile=np.asarray(r_smooth, float),
        axis=safe_normalize(np.asarray(axis, float)),
        centroid=np.asarray(centroid, float),
    )

    return envelope_profile