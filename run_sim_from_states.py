from utils.mj import FFSim
from utils.pure import load_npz
from utils.params import make_params

# Render a Mujoco simulation from saved states.
models = [
        "RCM",
        "GRO",
        "acrim",
        ]
for model in models:
    cfg = make_params(model_key=model)
    try:
        states = load_npz(f"states", cfg=cfg)
    except FileNotFoundError:
        raise RuntimeError("Run all_models.py or CC_Controllers.py with save_states = 1/True")

    sim = FFSim(cfg=cfg, add_stars=True)
    for bc in [0,1]:
        sim.run(states, fps=50, N=5000, body_cam=bc)

    # Renders two videos:
    #   bird's eye view when bc=0
    #   inspection cam when bc=1
    # Reduce fps and/or increase N(number of rendered frames) to make motion less choppy
    # Note this will increase rendering time and video duration
    #  It is possible to use "live" here to open the Mujoco passive viewer, but it has not been fully tested
