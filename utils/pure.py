# =======================
# All the pure utility functions and imports
# =======================

import inspect, os, re, numpy as np
from types import SimpleNamespace
EPS = 1e-6

# =======================
# Geometry transformations
# =======================
def as_flat3(x):
    return np.asarray(x, dtype=float).reshape(3,)

def as_col3(x):
    return as_flat3(x).reshape(3, 1)

def safe_normalize(v, eps=1e-9):
    """Vectorized, automatic-debugging.   Slop or Gold? """
    v = np.asarray(v, dtype=float)

    caller = inspect.currentframe().f_back
    caller_info = f"{os.path.basename(caller.f_code.co_filename)}:{caller.f_lineno}"

    if v.ndim == 1:
        n = np.linalg.norm(v)
        if n < eps:
            print(f"safe_normalize: returned zeros for 1 vector; shape={v.shape}; caller={caller_info}")
            return np.zeros_like(v)
        return v / n

    if v.ndim == 2 and 1 in v.shape:
        v_flat = v.reshape(-1)
        n = np.linalg.norm(v_flat)
        if n < eps:
            print(f"safe_normalize: returned zeros for 1 vector; shape={v.shape}; caller={caller_info}")
            return np.zeros_like(v)
        return (v_flat / n).reshape(v.shape)

    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    mask = norms < eps
    if np.any(mask):
        zero_count = int(np.count_nonzero(np.squeeze(mask, axis=-1)))
        print(f"safe_normalize: returned zeros for {zero_count} vector(s); shape={v.shape}; caller={caller_info}")
    safe_norms = np.where(mask, 1.0, norms)
    out = v / safe_norms
    return np.where(mask, 0.0, out)

def matrix_from_xz(x, z):
    x = as_flat3(x)
    z = as_flat3(z)
    z_axis = safe_normalize(z)
    x_proj = x - np.dot(x, z_axis) * z_axis
    if np.linalg.norm(x_proj) < EPS:
        alt = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(alt, z_axis)) > 0.8:
            alt = np.array([0.0, 1.0, 0.0])
        x_proj = alt - np.dot(alt, z_axis) * z_axis

    x_axis = safe_normalize(x_proj)
    y_axis = np.cross(z_axis, x_axis)
    x_axis = np.cross(y_axis, z_axis)  # re-orthogonalize

    return np.column_stack((x_axis, y_axis, z_axis))


def matrix_from_vector(direction, up_ref=np.array([0,0,1])):
    """
    Construct a rotation matrix with 'direction' as forward (z-axis),
    using 'up_ref' to resolve roll.
    """
    d = as_flat3(direction)
    up_ref = as_flat3(up_ref)
    if np.linalg.norm(d) < 1e-12:
        return np.eye(3)
    fwd = d / np.linalg.norm(d)

    # pick a reference up (avoid collinearity)
    up = up_ref
    if abs(np.dot(up, fwd)) > 0.99:
        up = np.array([0,1,0])  # fallback if nearly collinear

    right = np.cross(up, fwd)
    right /= np.linalg.norm(right)
    up = np.cross(fwd, right)

    return np.column_stack((right, up, fwd))

def cross(w):
    """w in R3 -> w^x in R3x3"""
    w = as_flat3(w)
    wx, wy, wz = w
    return np.array([[0, -wz, wy],
                     [wz, 0, -wx],
                     [-wy, wx, 0]], dtype=float)

def vee(S):
    """vee operator for a skew-symmetric matrix"""
    return np.array([S[2, 1], S[0, 2], S[1, 0]])

def so3_log(R):
    """
    Log map from SO(3) to so(3), returned as a 3-vector phi (axis*angle).
    Robust for small angles and near pi.
    """
    # Clamp trace for numerical safety
    tr = np.trace(R)
    c = (tr - 1.0) * 0.5
    c = np.clip(c, -1.0, 1.0)
    theta = np.arccos(c)

    if theta < 1e-7:
        # Small-angle: log(R) ≈ 0.5*(R - R^T)
        return 0.5 * vee(R - R.T)

    if np.pi - theta < 1e-6:
        # Near pi: use diagonal-based extraction (more stable)
        # Find axis from (R + I)
        Rp = (R + np.eye(3))
        col_norms = np.sum(Rp**2, axis=0)
        i = int(np.argmax(col_norms))
        axis = safe_normalize(Rp[:, i])
        return theta * axis

    # Generic case
    W = (R - R.T) * (0.5 / np.sin(theta))
    return theta * vee(W)

def sanitize_column(x, max_norm=None):
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if max_norm is None:
        return x
    nrm = np.linalg.norm(x)
    if nrm > max_norm and nrm > 0.0:
        x = x * (max_norm / nrm)
    return x

def sanitize_matrix(A):
    return np.nan_to_num(np.asarray(A, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

def saturate(v, v_max, flag=False):
    sat = False
    v_norm = np.linalg.norm(v)
    if v_norm > v_max:
        sat = True
        v *= v_max/v_norm
    if flag:
        return v, sat
    else:
        return v

def sample_hemisphere(n_dirs, e_in):
    dirs = []
    phi = (1 + 5**0.5) / 2

    for i in range(n_dirs):
        z = 1 - (i + 0.5) / n_dirs * 2
        theta = 2 * np.pi * i / phi

        r = np.sqrt(max(0, 1 - z * z))
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        v = safe_normalize(np.array([x, y, z]))

        if np.dot(v, e_in) >= 0:
            dirs.append(v)

    return np.array(dirs)

# =======================
# File saving and loading utilities
# =======================

def last_npz(stem, cfg=None, add=True, model_key="GRO"):
    """finds the last NPZ for a given filename stem"""
    if cfg is None: cfg = SimpleNamespace(model_key = model_key if model_key is not None else "GRO")
    if add: stem +=f"_{cfg.model_key}"
    idx = find_last_idx(stem, "npz", ".npz")
    fname = f"npz/{stem}_{idx}.npz"
    return fname

def save_npz(stem, cfg=None, add=True, model_key="GRO", **kwargs):
    # Save keywords to an NPZ
    if cfg is None: cfg = SimpleNamespace(model_key = model_key if model_key is not None else "GRO")
    if add: stem +=f"_{cfg.model_key}"
    fname = next_filename(stem)
    np.savez(fname, **kwargs)
    print(f"Saved results to {fname}")

def load_npz(fname_stem, cfg=None, model_key="GRO", add=True, query=None, debug=False):

    # "fname_stem" is the filename, query is a list of string names of variables to load
    if cfg is None: cfg = SimpleNamespace(model_key = model_key if model_key is not None else "GRO")
    fname = last_npz(fname_stem, cfg=cfg, add=add)
    data = np.load(fname, allow_pickle=True)
    if debug: print(f"Loaded {fname}")

    if query is None: query = [fname_stem]
    if not isinstance(query, list): query = [query]
   
    results = []
    for q in query:
        states = data[q]
        if isinstance(states, np.ndarray) and states.dtype == object:
            states = states.item() if states.shape == () else list(states.flat)
        else:
            states = states
        results.append(states)
    return results[0] if len(results) == 1 else results

def next_filename(stem, folder="npz", suffix=".npz"):
    """
    Return the next available filename with numeric suffix.
    Example: prefix='good_poses' -> 'good_poses_0.pkl' -> 'good_poses_1.pkl', etc.
    """

    if folder is not None:
        os.makedirs(folder, exist_ok=True)

    max_num = find_last_idx(stem, folder, suffix)
    next_num = max_num + 1
    if folder is not None:
        return f"{folder}/{stem}_{next_num}{suffix}"
    else:
        return f"{stem}_{next_num}{suffix}"

def find_last_idx(stem, folder="npz", suffix=".npz"):
    """
    Return the last numeric suffix for the given filename
    Example: prefix='good_poses' -> 'good_poses_3.npz' returns 3
    """

    pattern = re.compile(rf"^{re.escape(stem)}_(\d+){re.escape(suffix)}$")
    max_num = -1

    for fname in os.listdir(folder):
        match = pattern.match(fname)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
    return max_num

def formatted_dict(d, title='', precision=2):
    """prints an aligned dict with floats rounded to 2 decimal places"""
    maxlen = max((len(str(k)) for k in d.keys()), default=0) +1
    return_string = title
    for k,v in d.items():
        if isinstance(v, float):
            return_string += f"\n\t{str(k):<{maxlen}}: {v:.{precision}g}"
        else:
            return_string += f"\n\t{str(k):<{maxlen}}: {v}"
    return return_string
