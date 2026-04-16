import numpy as np, pinocchio as pin

#SEGMENT HELPER FUNCTIONS
def enforce_frame_sequence_continuity(frames):
    out = []
    prev = None
    for frame in frames:
        R = np.asarray(frame, dtype=float).reshape(3, 3)
        if prev is None:
            out.append(R)
            prev = R
            continue

        alt = R.copy()
        alt[:, :2] *= -1.0

        if np.linalg.norm(prev - alt) < np.linalg.norm(prev - R):
            R = alt

        out.append(R)
        prev = R
    return out

def segment_curvature_samples(arr, n_expected: int):
    kappa = np.asarray(arr, dtype=float)
    if kappa.ndim == 0:
        return np.full(n_expected, float(kappa), dtype=float)
    if kappa.ndim == 1:
        if kappa.size == n_expected:
            return kappa.astype(float, copy=False)
        if kappa.size == 3 * n_expected:
            return np.linalg.norm(kappa.reshape(n_expected, 3), axis=1)
        raise ValueError("segment curvature must have N or 3N entries")
    if kappa.ndim == 2:
        if kappa.shape == (n_expected, 3):
            return np.linalg.norm(kappa, axis=1)
        if kappa.shape == (3, n_expected):
            return np.linalg.norm(kappa.T, axis=1)
        if kappa.shape == (n_expected, 1):
            return kappa[:, 0]
        if kappa.shape == (1, n_expected):
            return kappa[0]
    raise ValueError("segment curvature must be scalar, (N,), (N,3), or (3,N)")

def path_tangent_from_positions(p_cd: np.ndarray, i0: int, i1: int, alpha: float) -> np.ndarray:
    n = len(p_cd)
    if n <= 1:
        return np.zeros(3, dtype=float)

    def local_tangent(i: int) -> np.ndarray:
        i_prev = max(i - 1, 0)
        i_next = min(i + 1, n - 1)
        if i_next == i_prev:
            return np.zeros(3, dtype=float)
        return np.asarray(p_cd[i_next], dtype=float) - np.asarray(p_cd[i_prev], dtype=float)

    t0 = local_tangent(i0)
    t1 = local_tangent(i1)
    return (1.0 - alpha) * t0 + alpha * t1

def smooth_vector_field(samples: np.ndarray, i0: int, i1: int, alpha: float, radius: int = 1) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    n = len(samples)
    if n == 0:
        return np.zeros(3, dtype=float)

    def local_average(i: int) -> np.ndarray:
        lo = max(int(i) - int(radius), 0)
        hi = min(int(i) + int(radius) + 1, n)
        return np.mean(samples[lo:hi], axis=0)

    s0 = local_average(i0)
    s1 = local_average(i1)
    return (1.0 - alpha) * s0 + alpha * s1


def segment_curvature_vectors(arr, n_expected: int):
    kappa = np.asarray(arr, dtype=float)
    if kappa.ndim == 0:
        return np.zeros((n_expected, 3), dtype=float)
    if kappa.ndim == 1:
        if kappa.size == 3:
            return np.tile(kappa.reshape(1, 3), (n_expected, 1))
        if kappa.size == 3 * n_expected:
            return kappa.reshape(n_expected, 3)
        if kappa.size == n_expected:
            return np.zeros((n_expected, 3), dtype=float)
        raise ValueError("segment curvature vectors must have 3 or 3N entries")
    if kappa.ndim == 2:
        if kappa.shape == (n_expected, 3):
            return kappa.astype(float, copy=False)
        if kappa.shape == (3, n_expected):
            return kappa.T.astype(float, copy=False)
        if kappa.shape == (1, 3):
            return np.tile(kappa, (n_expected, 1))
        if kappa.shape == (3, 1):
            return np.tile(kappa.reshape(1, 3), (n_expected, 1))
    raise ValueError("segment curvature vectors must be scalar, (N,3), (3,N), or broadcastable 3-vectors")
    
def segment_samples(arr):
    samples = np.asarray(arr, dtype=float)
    if samples.ndim != 2:
        raise ValueError("segment samples must be 2D")
    if samples.shape[-1] == 3:
        return samples
    if samples.shape[0] == 3:
        return samples.T
    raise ValueError("segment samples must have shape (N, 3) or (3, N)")


def interpolate_rotation(R0, R1, alpha: float):
    R0 = np.asarray(R0, dtype=float).reshape(3, 3)
    R1 = np.asarray(R1, dtype=float).reshape(3, 3)
    rel = R0.T @ R1
    return R0 @ pin.exp3(alpha * pin.log3(rel))
