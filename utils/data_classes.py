import numpy as np
from utils.robot import GiordanoRobot
from dataclasses import dataclass, field
from utils.pure import as_col3, safe_normalize

#DATACLASSES
def _clone(obj):
    new = object.__new__(type(obj))
    for name in obj.__dataclass_fields__:
        v = getattr(obj, name)
        if isinstance(v, np.ndarray):
            v = v.copy()
        elif isinstance(v, dict):
            v = dict(v)
        setattr(new, name, v)
    return new

@dataclass
class State:
    q: np.ndarray               # (nq,) Pinocchio config
    v: np.ndarray               # (nv,) Pinocchio velocity

    # cached circumcentroidal components (update each tick)
    v_c: np.ndarray = None          # (3,1)
    omega_b: np.ndarray = None      # (3,1)
    nu_e_oplus: np.ndarray = None   # (6,1)

    def update_views(self, model: GiordanoRobot):
        y = model.Gamma(self.q) @ self.v.reshape(-1, 1)
        self.v_c = y[0:3]
        self.omega_b = y[3:6]
        self.nu_e_oplus = y[6:12]
        return self.copy()

    @property
    def v_breve(self):
        return np.vstack([self.omega_b, self.nu_e_oplus])
    
    def copy(self):
        return _clone(self)

@dataclass
class Desired:
    p_c: np.ndarray     = field(default_factory=lambda: np.zeros(3, dtype=float))               # (3,1) desired CoM position
    v_c: np.ndarray     = field(default_factory=lambda: np.zeros(3, dtype=float))               # (3,1) desired CoM velocity
    a_c: np.ndarray     = field(default_factory=lambda: np.zeros(3, dtype=float))               # (3,1) desired CoM acceleration
    R_b: np.ndarray     = field(default_factory=lambda: np.eye(3, dtype=float))                 # (3,3) desired base attitude
    z_b: np.ndarray     = field(default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=float)) # (3,1) desired base pointing direction
    omega_b: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))               # (3,1) desired base angular velocity in B_d
    p_e: np.ndarray     = field(default_factory=lambda: np.zeros(3, dtype=float))               # (3,1) desired EE position
    z_e: np.ndarray     = field(default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=float)) # (3,1) desired EE pointing direction
    R_e: np.ndarray | None = None                                                               # (3,3) desired EE attitude, optional

    def copy(self):
        return _clone(self)

    def __post_init__(self):
        self.p_c = as_col3(self.p_c)
        self.v_c = as_col3(self.v_c)
        self.a_c = as_col3(self.a_c)
        self.R_b = np.asarray(self.R_b, dtype=float).reshape(3, 3)
        self.z_b = as_col3(safe_normalize(self.R_b[:, 2]))
        self.omega_b = as_col3(self.omega_b)
        self.p_e = as_col3(self.p_e)
        self.R_e = None if self.R_e is None else np.asarray(self.R_e, dtype=float).reshape(3, 3)
        self.z_e = as_col3(safe_normalize(self.z_e if self.R_e is None else self.R_e[:, 2]))

@dataclass
class CameraPose:
    p_e: np.ndarray | None = None
    z_e: np.ndarray | None = None
    R_e: np.ndarray | None = None

@dataclass 
class Target:
    x: np.ndarray | None = None
    n: np.ndarray | None = None
    score: float = -np.inf
    switched: bool = False
    info: dict | None = None
    tri_idx: int = -1
    tri_area: float | None = None
    coverage_count: int = 0

@dataclass
class CameraTarget:
    idx: int | None = None #target index
    p_tgt: np.ndarray | None = None
    n_tgt: np.ndarray | None = None
    p_e: np.ndarray | None = None
    z_e: np.ndarray | None = None
    score: float = -np.inf
    switched: bool = False
    info: dict | None = None
    fallback: bool = False
    
    def copy(self):
        return _clone(self)

@dataclass
class Gains:
    K_c:  np.ndarray | float = 0
    D_c:  np.ndarray | float = 0
    I_c:  np.ndarray | float = 0

    K_b:  np.ndarray | float = 0
    D_b:  np.ndarray | float = 0

    K_ep: np.ndarray | float = 0
    D_ep: np.ndarray | float = 0
    K_eo: np.ndarray | float = 0
    D_eo: np.ndarray | float = 0
            
    @staticmethod
    def _as_gain_matrix(value, size):
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return float(arr) * np.eye(size, dtype=float)
        return arr

    def __post_init__(self):
        for name in type(self).__dataclass_fields__:
            setattr(self, name, self._as_gain_matrix(getattr(self, name), 3))

    @property
    def K_e(self):
        return np.block([
            [self.K_ep, np.zeros((3, 3))],
            [np.zeros((3, 3)), self.K_eo],
        ])

    @property
    def D_e(self):
        return np.block([
            [self.D_ep, np.zeros((3, 3))],
            [np.zeros((3, 3)), self.D_eo],
        ])

@dataclass
class ProgressState:
    """
    Mutable progress bookkeeping for CoM tracking along the global orbit.
    """
    i0: int = 0
    i1: int | None = None
    alpha: float | None = None
    progress: float | None = None
    raw_progress: float | None = None
    p_near: np.ndarray | None = None

    def copy(self):
        return _clone(self)

@dataclass
class OrbitWindow:
    i0: int
    i1: int
    p_e: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    z_e: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=float))
