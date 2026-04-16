from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.pure import safe_normalize, matrix_from_vector
EPS = 1e-6

# Quaternions are all Scalar-LAST
# Quaternion evolution
def quat_mul(q, r):
    x0, y0, z0, w0 = q
    x1, y1, z1, w1 = r
    return np.array([
        w0*x1 + w1*x0 + y0*z1 - z0*y1,
        w0*y1 + w1*y0 + z0*x1 - x0*z1,
        w0*z1 + w1*z0 + x0*y1 - y0*x1,
        w0*w1 - x0*x1 - y0*y1 - z0*z1
    ])

def quat_from_vector(direction, wxyz=False):
    return R_to_quat((matrix_from_vector(direction)).as_quat(scalar_first=wxyz))

def quat_normalize(q):
    return safe_normalize(q)

def quat_from_axes(x, y, z):
    R = np.column_stack((x,y,z))
    # convert rotation matrix to quaternion
    qw = np.sqrt(1 + np.trace(R))/2
    qx = (R[2,1] - R[1,2])/(4*qw)
    qy = (R[0,2] - R[2,0])/(4*qw)
    qz = (R[1,0] - R[0,1])/(4*qw)
    return quat_normalize(np.array([qx,qy,qz,qw]))

def quat_error(q, q_des):
    # quaternion difference q_err = q_des * inv(q)
    q_inv = np.array([-q[0], -q[1], -q[2], q[3]])
    return quat_mul(q_des, q_inv)

def quat_to_rotvec(q):
    # small-angle conversion
    q = quat_normalize(q)
    angle = 2*np.arccos(np.clip(q[3], -1,1))
    if angle < 1e-6:
        return np.zeros(3)
    axis = q[:3] / np.sin(angle/2)
    return axis * angle

def quat_to_R(q):
    """
    Convert a unit quaternion [x, y, z, w] to a 3x3 rotation matrix.
    """
    x, y, z, w = q
    # Standard quaternion to rotation matrix formula
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),         2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),         1 - 2*(x*x + y*y)]
    ])
    return R

def R_to_quat(R):
    """
    Convert a 3x3 rotation matrix to a unit quaternion [x, y, z, w].
    """
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]

    trace = m00 + m11 + m22
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    return quat_normalize(np.array([x, y, z, w]))

def old_R_to_quat(R):
    """
    Convert a 3x3 rotation matrix to a unit quaternion [x, y, z, w].
    """
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]

    trace = m00 + m11 + m22
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    return quat_normalize(np.array([x, y, z, w]))
def quat_integrate(q, omega, dt, eps=1e-9):
    """
    Integrate quaternion forward given body-frame angular velocity.

    Quaternion convention: q = [x, y, z, w] (scalar last)
    omega: body angular velocity [rad/s], shape (3,)
    “Quaternion integration is done via the exponential map 
    using body-frame angular velocity, which guarantees unit norm and avoids numerical drift.”
    """
    

    wx, wy, wz = omega
    w_norm = np.linalg.norm(omega)

    if w_norm < eps:
        # Small-angle approximation
        dq = np.array([
            0.5 * wx * dt,
            0.5 * wy * dt,
            0.5 * wz * dt,
            1.0
        ])
    else:
        theta = w_norm * dt
        axis = omega / w_norm
        half = 0.5 * theta
        s = np.sin(half)

        dq = np.array([
            axis[0] * s,
            axis[1] * s,
            axis[2] * s,
            np.cos(half)
        ])

    # Quaternion multiplication: q_next = q ⊗ dq
    q_next = quat_mul(q, dq)

    return quat_normalize(q_next)

def rotate_vec_by_quat_conj(v, q):
    import casadi
    """
    Rotate vector v from world frame into body frame using quaternion q (xyzw).
    CasADi-compatible.
    """
    x, y, z, w = q[0], q[1], q[2], q[3]

    # quaternion conjugate
    qc = casadi.vertcat(-x, -y, -z, w)

    # embed vector as quaternion
    vq = casadi.vertcat(v[0], v[1], v[2], 0)

    def quat_mul(a, b):
        ax, ay, az, aw = a[0], a[1], a[2], a[3]
        bx, by, bz, bw = b[0], b[1], b[2], b[3]
        return casadi.vertcat(
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
            aw*bw - ax*bx - ay*by - az*bz
        )

    v_body_q = quat_mul(qc, quat_mul(vq, q))
    return v_body_q[:3]

def enforce_quat_continuity_xyzw(qs: np.ndarray) -> np.ndarray:
    """
    qs: (M,4) xyzw scalar-last quaternions (numpy)
    Returns qs with sign flips removed so consecutive dots are >= 0.
    """
    out = qs.copy()
    for i in range(1, out.shape[0]):
        if np.dot(out[i-1], out[i]) < 0.0:
            out[i] *= -1.0
    return out

def quat_eps_eta(R):
    q = np.asarray(R_to_quat(R), dtype=float).reshape(4,)
    if q[3] < 0.0:
        q = -q
    eps = q[:3].reshape(3, 1)
    eta = float(q[3])
    return eps, eta

if __name__ == '__main__':
    q_test = safe_normalize([1, 2, 3, 4])
    rng = np.random.default_rng(0)

    def _assert_close(a, b, atol=1e-6):
        if not np.allclose(a, b, atol=atol, rtol=0.0):
            raise AssertionError(f"mismatch\n{a}\n!=\n{b}")

    def _random_quats(n):
        q = rng.normal(size=(n, 4))
        return safe_normalize(q)

    def _quat_to_rotmat(q):
        return Rotation.from_quat(q).as_matrix()

    qs = _random_quats(10)
    rs = _random_quats(10)

    for q, r in zip(qs, rs):
        ours = quat_mul(q, r)
        scipy_R = _quat_to_rotmat(q) @ _quat_to_rotmat(r)
        _assert_close(quat_to_R(ours), scipy_R)

    for _ in range(10):
        R = Rotation.random(random_state=rng).as_matrix()
        ours = quat_from_axes(R[:, 0], R[:, 1], R[:, 2])
        _assert_close(quat_to_R(ours), R)

    for q, q_des in zip(qs, rs):
        ours = quat_error(q, q_des)
        scipy_R = _quat_to_rotmat(q_des) @ _quat_to_rotmat(q).T
        _assert_close(quat_to_R(ours), scipy_R)

    for q in qs:
        q_c = q if q[3] >= 0.0 else -q
        ours = quat_to_rotvec(q_c)
        scipy = Rotation.from_quat(q_c).as_rotvec()
        _assert_close(ours, scipy)

    for q in qs:
        _assert_close(quat_to_R(q), _quat_to_rotmat(q))

    for _ in range(10):
        R = Rotation.random(random_state=rng).as_matrix()
        ours = R_to_quat(R)
        scipy = Rotation.from_matrix(R).as_quat()
        if np.dot(ours, scipy) < 0.0:
            scipy = -scipy
        _assert_close(ours, scipy)

    for q in qs:
        for _ in range(3):
            omega = rng.normal(size=3)
            dt = rng.uniform(0.0, 0.1)
            ours = quat_integrate(q, omega, dt)
            scipy = (Rotation.from_quat(q) * Rotation.from_rotvec(omega * dt)).as_quat()
            if np.dot(ours, scipy) < 0.0:
                scipy = -scipy
            _assert_close(ours, scipy)

    print("utils_new quaternion tests passed")
