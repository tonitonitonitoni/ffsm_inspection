import os
from pathlib import Path
import sys
import numpy as np
from types import SimpleNamespace

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")

CORE_PARAMS = {
    'T':                    500,     #duration
    "dt":                   0.01,    #timestep
    "dt_ctrl":              0.03,
}

CONSTRAINTS = {
    # Actuator constraints
    "F_max":                 20.0,
    "tau_b_max":             5.0,

    #Velocity constraints
    "vc_max":               0.75,   # base linear velocity
    "omega_b_max":          0.5,    # base angular velocity
    "theta_dot_max":        2.0,    # joint velocity

    "k_ff":                 0.5,
    "ac_ff_max":            0.65,
    "theta_ddot_max":       2.0,
}

CONTROLLER_PARAMS = {
    "desired_com_speed":    0.7,
    
    "enable_base":          True,
    "enable_ee":            True,

    'sample_freq':          10,

    "K_c":                  10.0,
    "D_c":                  28.0,
    "K_b":                  1.0,
    "D_b":                  0.5,
    "I_c":                  1.0,
    "K_ep":                 2,
    "D_ep":                 2e-1,
    "K_eo":                 1.0,
    "D_eo":                 1.0,

    "log_stride":                   1,

    #COM tracking
    "use_com_integral":             True,
    "clip_velocity":                True,
    "com_integral_limit":   0.12,
    "com_integral_disable_err": 0.08,

    "k_track":                      0.7,
    "k_progress":                   2.0,
    "desired_vc_smoothing_tau":     0.30,
    "com_reference_startup_tau":    1.0,
    "vc_correction_alpha":          0.3,

    "ff_com_alpha":         1.0,
    "ff_force_max":         8.0,
    "ff_startup_tau":       0.5,


    # Camera tracking
    "use_target_finder":            True,
    "ee_nu_des_match_com_velocity": True,
    "ee_nu_des_tau":        0.0,
    "ee_nu_des_max":        0.0,
    "ee_reach_horizon":     1.0,

    "update_period_steps":  100,
    "pointing_reselect_cos": 0.7,
    "pointing_max_drift_deg": 60.0,

    "lookahead_time":       0.25,
    "search_window":        8,
    "camera_candidate_radius_scale": 0.8,
    
    "max_delta_p_ce":       0.15,
    "ee_goal_pos_tau":      0.8,
    "ee_goal_axis_tau":     0.2,
    "ee_goal_pos_rate_max": 0.0,
    "ee_goal_axis_rate_max": 0.0,
    "stability_length":     0.3,
    "camera_switch_hysteresis": 4e-1,
    "camera_min_hold_steps": 500,
    "sigma_blend_threshold": 0.0,

    #Gamma and J+ Regularization

    "use_regularization":               True,
    "use_gamma_speed_derating":         True,
    "gamma_derate_arm_motion":          True,
    "gamma_reg":            1e-4,
    "gamma_sigma_low":      1e-2,
    "gamma_sigma_high":     5e-2,
    "gamma_speed_floor":    0.25,
    "gamma_reg_floor":       5e-2,

    "use_jplus_base_derating":          False,
    "jplus_sigma_low":                  2e-2,
    "jplus_sigma_high":                 5e-2,
    "jplus_base_floor":                 0.0,

}

VP_PARAMS = {
    'n_rays':               6,
    'n_query':              50,
    'n_targets':            5000,
    'max_angle':            np.deg2rad(45),

    # Kong metric parameters - based on working depth ranges for a generic depth camera
    'd_far':                5.0,
    'd_max':                4.0,
    'd_min':                0.5,
    'd_near':               0.1,

    #Weights for q-scores (set equal for now)
    'w1':                   1,
    'w2':                   1,
    'w3':                   1,

    #More weights (Gotta love them)
    "w_kong":               1,
    "w_novelty":            0.2,
    "w_motion":             3.0,
    "w_stability":          3.0,
    "w_area":               0.0,
    "w_anchor":             0.0,
    "w_manipulability":     10.0,  
}

ORBIT_PARAMS = {
    "clearance":            1.0,

    # Path shaping
    "psi_deg":              5, #pole avoidance margin
    "n_rev":                30,
    "r_percentile":         99.0,

    # Numerical tolerances / noise
    "EPS":                  1e-7,
}

VISION_PARAMS = {
    #Camera
    'resolution':           (80, 60),
    'camera_fov_deg':       45,

    #Reconstruction
    'tsdf_resolution':      (800, 600), # larger is WAY better
    'rough_k':              12,
    'dot_thresh_recon':     0.55,       #not sure
}

ROBOT_PARAMS = {

    "pwd":                          os.path.join(MODELS_DIR, ""),

    'UR3': {
        "cam_frame":                "camera_link",
        "base_frame":               "box_base",
        "reachable_arm_radius":     0.5562,
        "mass":                     15.68,
        "I_body":                   [[ 0.49976701, -0.13571864, -0.36427287], [-0.13571864,  1.07045287, -0.1255856 ], [-0.36427287, -0.1255856, 0.74516702]],
        "I_inv":                    [[3.51289457, 0.65990448, 1.82848496], [0.65990448, 1.07699207, 0.50410175], [1.82848496, 0.50410175, 2.32078895]],
        "robot_relative_path":      "ur3/ur3_box_limited_with_capsules.xml",
        "n_DOF":                    6,
        "joint_velocity_limits":    np.pi*np.array([1, 1, 1, 2, 2, 2], dtype=float),
        "joint_torque_limits":      np.array([56]*2 +[28] +[12]*3, dtype=float),
        "reach_ratio":              0.9,
    },
}

MODEL_LIBRARY = {

    'GRO': {
        'name':             'Gamma Ray',
        'path':             'GRO.stl', 
        'radius':           3.87
    },

    'acrim': {
        'name':             'AcrimSat',
        'path':             'AcrimSAT.stl',
        'radius':           3.75,
    },

    'RCM': {
        'name':             'RadarSat',
        'path':             'RCM.stl',
        'radius':           3.92,
    },

    }

def make_params(model_key='GRO', 
                robot="UR3",
                vps=True, vision=True,
                ):
    BASE_PARAMS = dict(ROBOT_PARAMS[robot])
    BASE_PARAMS['pwd'] = ROBOT_PARAMS['pwd']
    BASE_PARAMS.update(CORE_PARAMS)
    BASE_PARAMS.update(MODEL_LIBRARY[model_key])
    BASE_PARAMS.update(ORBIT_PARAMS)
    BASE_PARAMS.update(CONTROLLER_PARAMS)
    BASE_PARAMS.update(CONSTRAINTS)

    if vps:
        BASE_PARAMS.update(VP_PARAMS)
    if vision:
        BASE_PARAMS.update(VISION_PARAMS)

    cfg = SimpleNamespace(**BASE_PARAMS)

    # TIME_PARAMS
    cfg.N_pts = cfg.T / cfg.dt

    # ROBOT_PARAMS
    cfg.model_key = model_key
    cfg.mesh_path = cfg.pwd + cfg.path     
    cfg.robot_path = cfg.pwd + cfg.robot_relative_path

    cfg.anchor_length = cfg.reachable_arm_radius * cfg.reach_ratio

    #ORBIT_PARAMS
    cfg.orbital_radius = cfg.radius + cfg.clearance

    #CONSTRAINTS
    I_body = np.array(cfg.I_body)
    I_diag = np.diag(I_body)
    cfg.alpha_b_max = cfg.tau_b_max / np.min(I_diag)
    cfg.omega_b_dot_max = cfg.alpha_b_max
    cfg.vc_dot_max = cfg.F_max / cfg.mass
    cfg.ac_ff_max = cfg.k_ff * cfg.vc_dot_max

    cfg.nu_e_oplus_max = cfg.theta_dot_max
    cfg.nu_e_oplus_dot_max = cfg.theta_ddot_max

    if vision:
        cfg.tsdf_voxel_length = 0.01
        cfg.epsilon = 3*cfg.tsdf_voxel_length
        cfg.tsdf_trunc = 4*cfg.tsdf_voxel_length

    return cfg
