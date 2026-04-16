from CC_Controllers import EEController, CC_Controller
from utils.params import make_params

COM_only = 0
save_states = 1

if __name__ == "__main__":
    models = [
        "RCM",
        "GRO",
        "acrim",
        ]
    for model in models:
        cfg = make_params(model_key=model, vps=True, vision=True)
        ctrl = CC_Controller(cfg=cfg) if COM_only else EEController(cfg=cfg)
        add_title = f" — {model}"
        if COM_only: add_title+= " (COM)"

        results = ctrl.run_all(end_idx=None,
                               add_title=add_title,
                               save_figs=1,
                               show_figs=False
                               )

        # # Return states and save for simulation and reconstruction
        if save_states:
            from utils.pure import save_npz
            states = ctrl.logger.return_states()
            stem = "states"
            if COM_only: stem +="_COM"
            save_npz(stem, cfg=cfg, states=states)
