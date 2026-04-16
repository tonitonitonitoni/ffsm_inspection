from pathlib import Path
import sys
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.pure import safe_normalize
from utils.params import make_params

class OrbitGenerator:
    def __init__(self, config):
        self.cfg = config
        self.clearance = self.cfg.clearance 
        self.n_turns, self.omega = self.calc_n_turns()
        self.N = int(self.cfg.N_pts)
        self.arclength_samples = np.zeros((0,), dtype=float)
        self.compute_curvature()
        
    def calc_n_turns(self):
        cfg = self.cfg
        n_turns = cfg.n_rev
        omega = 2*np.pi*n_turns
        return n_turns, omega
    
    def generate_coverage_orbit(self, mesh=None):
        if hasattr(self, 'path'):
            return self.path
        
        self.path = self.generate_spherical_helix()
        self.path = self.resample_path_uniform_arclength()
        self.arclength_samples = self._build_open_arclength_samples(self.path)
        return self.path
    
    def generate_spherical_helix(self):
        cfg = self.cfg
        R = cfg.orbital_radius

        psi = np.radians(cfg.psi_deg)
        n_rev = self.n_turns
        
        t = np.linspace(0, cfg.T, self.N+1)
        theta = 2*np.pi * n_rev * (t / cfg.T)  # n_rev wraps
        phi0 = -(np.pi/2 - psi)
        phi1 = +(np.pi/2 - psi)
        u = t/cfg.T
        phi = phi0 + (phi1 - phi0) * u

        x = R * np.cos(phi) * np.cos(theta)
        y = R * np.cos(phi) * np.sin(theta)
        z = R * np.sin(phi)
        return np.vstack([x, y, z]).T
    
    def resample_path_uniform_arclength(self):
        """
        Resample a discrete 3D path to N points uniformly spaced in arc length.
        """
        N = self.N
        path = np.asarray(self.path, float)

        diffs = np.diff(path, axis=0)
        seg_len = np.linalg.norm(diffs, axis=1)
        cum_s = np.concatenate(([0.0], np.cumsum(seg_len)))
        total_len = cum_s[-1]

        s_uniform = np.linspace(0.0, total_len, N)

        path_u = np.zeros((N, 3))
        for k in range(3):
            path_u[:, k] = np.interp(s_uniform, cum_s, path[:, k])
        return path_u

    def _build_arclength_table(self, path: np.ndarray):
        """
        Build cumulative arc-length table for a closed discrete path.
        Returns:
          - pathN: (N,3) path points (last duplicate removed if present)
          - cum_s: (N+1,) cumulative arc-length, cum_s[0]=0, cum_s[-1]=total length
          - seg_len: (N,) segment lengths between i and i+1 (with wrap)
        """
        pathN = np.asarray(path, dtype=float)

        # If path includes a duplicated final point (common when using linspace with N+1),
        # drop it so that indexing wraps cleanly.
        if len(pathN) >= 2 and np.linalg.norm(pathN[-1] - pathN[0]) < 1e-9:
            pathN = pathN[:-1]

        diffs = np.roll(pathN, -1, axis=0) - pathN
        seg_len = np.linalg.norm(diffs, axis=1)
        cum_s = np.concatenate(([0.0], np.cumsum(seg_len)))

        return cum_s

    def _build_open_arclength_samples(self, path: np.ndarray):
        path = np.asarray(path, dtype=float)
        if len(path) <= 1:
            return np.zeros((len(path),), dtype=float)
        ds = np.linalg.norm(np.diff(path, axis=0), axis=1)
        return np.concatenate(([0.0], np.cumsum(ds)))

    def spherical_helix_curvature(self, s):
        """
        Analytical curvature of a spherical helix.

        Parameters
        ----------
        s : float or ndarray
            Axial coordinate (must satisfy |s| < R).
        R : float
            Sphere radius.
        omega : float
            Helix winding rate dtheta/ds.
        eps : float
            Small safety margin to avoid singularities near the poles.

        Returns
        -------
        kappa : float or ndarray
            Curvature κ(s).
        """
        cfg = self.cfg
        eps = cfg.EPS
        R = cfg.orbital_radius
        omega = self.omega
        psi = np.radians(cfg.psi_deg)
        # Path construction avoids poles by psi, so use that bound for stable curvature evaluation
        s_max = R * np.cos(psi)  # since z = R sin(phi), |phi| <= pi/2 - psi => |z| <= R cos(psi)
        s = np.clip(s, -s_max + eps, s_max - eps)

        # Radius profile and powers
        r  = np.sqrt(R*R - s*s)
        r2 = r*r
        r3 = r2*r

        # First and second derivatives of r(s)
        rs  = -s / r
        rss = -R*R / r3

        # --- Curvature numerator terms ---
        term1 = (rss - omega*omega * r)**2
        term2 = 4.0 * omega*omega * rs*rs
        term3 = omega*omega * (omega*omega * r2 + 2.0*rs*rs - r*rss)**2

        num = term1 + term2 + term3

        # --- Denominator ---
        den = (1.0 + rs*rs + omega*omega * r2)**1.5

        return np.sqrt(num) / den
    
    def compute_curvature(self):
        """
        Analytical curvature drop-in replacement for compute_FD_curvature().

        Returns:
            path_raw      : (N,3) reference path (before arclength resampling)
            tangents      : (N,3) unit tangents along the path
            curvature_vecs: (N,3) curvature vectors (normal direction)

            'Analytical expressions are used to compute curvature magnitudes, while tangent and normal directions are computed from the discretized, 
            arc-length-parameterized path to ensure geometric consistency with the executed trajectory.' 
        """
        cfg = self.cfg

        # --- Build reference path exactly as FD version ---
        self.generate_coverage_orbit()
        path = self.path
        N = len(path)
        if N == 0:
            self.tangents = np.zeros((0, 3), dtype=float)
            self.curvature = np.zeros((0, 3), dtype=float)
            return

        if N == 1:
            self.tangents = np.zeros((1, 3), dtype=float)
            self.curvature = np.zeros((1, 3), dtype=float)
            return

        ds_seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        cum_s = np.concatenate(([0.0], np.cumsum(ds_seg)))

        # Open-path tangents: use one-sided differences at the ends and centered
        # differences in the interior to avoid wrapping the terminal sample back
        # onto the initial sample.
        tangent_dirs = np.zeros_like(path)
        tangent_dirs[0] = path[1] - path[0]
        tangent_dirs[-1] = path[-1] - path[-2]
        if N > 2:
            tangent_dirs[1:-1] = path[2:] - path[:-2]
        tangents = safe_normalize(tangent_dirs)
        self.tangents = tangents
    
        # s corresponds to z = R sin(phi)
        s_param = path[:, 2]
        kappa_mag = self.spherical_helix_curvature(s_param)

        # --- Build curvature vectors ---
        # Frenet formula: kappa * normal, with open-path tangent derivatives.
        dt_ds = np.zeros_like(tangents)
        dt_ds[0] = (tangents[1] - tangents[0]) / max(ds_seg[0], cfg.EPS)
        dt_ds[-1] = (tangents[-1] - tangents[-2]) / max(ds_seg[-1], cfg.EPS)
        if N > 2:
            span = np.maximum(cum_s[2:] - cum_s[:-2], cfg.EPS)
            dt_ds[1:-1] = (tangents[2:] - tangents[:-2]) / span[:, None]

        curvature_vecs = np.zeros_like(tangents)
        for i in range(N):
            n_dir = dt_ds[i] - np.dot(dt_ds[i], tangents[i]) * tangents[i]
            n_norm = np.linalg.norm(n_dir)
            if n_norm > cfg.EPS:
                curvature_vecs[i] = kappa_mag[i] * (n_dir / n_norm)

        self.tangents = tangents
        self.curvature = curvature_vecs

if __name__ == "__main__":
    cfg = make_params()
    orbit = OrbitGenerator(cfg)
    print(orbit.tangents[0])
