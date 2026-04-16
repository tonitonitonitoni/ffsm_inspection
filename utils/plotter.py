import numpy as np, matplotlib.pyplot as plt

class Plotter3D:
    def __init__(self, figsize=(8, 6)):
        self.figsize = figsize

    def _new_axes(self):
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection="3d")
        return fig, ax

    def _path_quiver_data(self, p, z=None, quiver_count=64):
        p = np.asarray(p, dtype=float)
        path_ok = p.ndim == 2 and p.shape[1] == 3
        if not path_ok:
            return None
        if len(p) <= 0:
            return None

        out = {"p": p, "path_ok": True, "quiver_ok": False}
        if z is None: return out

        z = np.asarray(z, dtype=float)
        quiver_ok = z.ndim == 2 and z.shape[1] == 3
        if not quiver_ok: return out

        n = min(len(p), len(z))
        if n <= 0: return None

        p = p[:n]
        z = z[:n]
        step = max(1, n // max(1, int(quiver_count)))
        out.update({
            "p": p,
            "z": z,
            "quiver_ok": True,
            "p_q": p[::step],
            "z_q": z[::step],
        })
        return out

    def _quiver_length(self, pos_batches):
        span = np.ptp(np.vstack(pos_batches), axis=0)
        return span, max(0.08 * float(np.max(span)), 1e-3)

    def _finalize(self, ax, span, title):
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(title)
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect(tuple(np.maximum(span, 1e-6)))
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_quiver(self, ax, series, colour, quiver_len=1.0, quiver_width=1.0):
        actual = series
        ax.quiver(
                actual["p_q"][:, 0], actual["p_q"][:, 1], actual["p_q"][:, 2],
                actual["z_q"][:, 0], actual["z_q"][:, 1], actual["z_q"][:, 2],
                length=quiver_len, normalize=True, color=colour, 
                linewidth=quiver_width, alpha=0.45,
            )
        return ax

    def plot_base_com(self, log, title=None, quiver_count=64):
        actual = self._path_quiver_data(log.p_actual, log.z_actual, quiver_count)
        desired = self._path_quiver_data(log.p_desired, log.z_desired, quiver_count)
        if actual is None and desired is None:
            return

        fig, ax = self._new_axes()
        pos_batches = []

        if actual is not None:
            pos_batches.append(actual["p"])
            ax.plot(actual["p"][:, 0], actual["p"][:, 1], actual["p"][:, 2], color="red", label="st.p_c")
        if desired is not None:
            pos_batches.append(desired["p"])
            ax.plot(desired["p"][:, 0], desired["p"][:, 1], desired["p"][:, 2], color="blue", alpha=0.85, label="des.p_tcd")

        if not pos_batches:
            plt.close(fig)
            return

        span, quiver_len = self._quiver_length(pos_batches)

        if actual is not None and actual["quiver_ok"]:
            ax = self.plot_quiver(ax, actual, "red", quiver_len)
        if desired is not None and desired["quiver_ok"]:
            ax = self.plot_quiver(ax, desired, "blue", quiver_len)

        self._finalize(ax, span, "Base-CoM Trajectory" if title is None else title)

    def plot_com(self, log, title=None):
        actual = self._path_quiver_data(log.p_actual)
        desired = self._path_quiver_data(log.p_desired)
        if actual is None and desired is None: return
        fig, ax = self._new_axes()
        pos_batches = []

        if actual is not None:
            pos_batches.append(actual["p"])
            ax.plot(
                actual["p"][:, 0],
                actual["p"][:, 1],
                actual["p"][:, 2],
                color="red",
                label="st.p_c",
            )

        if desired is not None:
            pos_batches.append(desired["p"])
            ax.plot(
                desired["p"][:, 0],
                desired["p"][:, 1],
                desired["p"][:, 2],
                color="blue",
                alpha=0.85,
                label="des.p_c",
            )

        if not pos_batches:
            plt.close(fig)
            return

        span = np.ptp(np.vstack(pos_batches), axis=0)

        self._finalize(ax, span, "CoM Trajectory" if title is None else title)

    def plot_combined(self, log, title=None, quiver_count=64):
        actual_color = "black"
        desired_color = "purple"
        actual_alpha = 0.8
        desired_alpha = 0.75
        quiver_width = 2.6

        actual = self._path_quiver_data(log.p_actual, log.z_actual, quiver_count)
        desired = self._path_quiver_data(log.p_desired, log.z_desired, quiver_count)

        ee_series = []
        for p_key, z_key, color, label, alpha in (
            ("p_e", "z_e", desired_color, "p_e", desired_alpha),
            ("p_e_actual", "z_e_actual", actual_color, "st.p_e", actual_alpha),
            ("p_e_desired", "z_e_desired", desired_color, "des.p_e", desired_alpha),
        ):
            p_e = getattr(log, p_key, None)
            z_e = getattr(log, z_key, None)
            ee = None if p_e is None or z_e is None else self._path_quiver_data(p_e, z_e, quiver_count)
            if ee is not None:
                ee_series.append((ee, color, label, alpha))

        if actual is None and desired is None and not ee_series:
            return

        fig, ax = self._new_axes()
        pos_batches = []

        if actual is not None:
            pos_batches.append(actual["p"])
            ax.plot(
                actual["p"][:, 0], actual["p"][:, 1], actual["p"][:, 2],
                color=actual_color, linewidth=1.8, alpha=0.95, label="st.p_c",
            )
        if desired is not None:
            pos_batches.append(desired["p"])
            ax.plot(
                desired["p"][:, 0], desired["p"][:, 1], desired["p"][:, 2],
                color=desired_color, linewidth=1.8, alpha=0.9, label="des.p_tcd",
            )
        for ee, _, _, _ in ee_series:
            pos_batches.append(ee["p"])

        if not pos_batches:
            plt.close(fig)
            return

        span, quiver_len = self._quiver_length(pos_batches)

        if actual is not None and actual["quiver_ok"]:

            ax = self.plot_quiver(ax, actual, actual_color, quiver_len, quiver_width)
            
        if desired is not None and desired["quiver_ok"]:
            ax = self.plot_quiver(ax, desired, desired_color, quiver_len, quiver_width)


        for ee, color, label, alpha in ee_series:
            ax.plot(
                ee["p"][:, 0], ee["p"][:, 1], ee["p"][:, 2],
                color=color, alpha=min(alpha + 0.1, 0.95), linewidth=1.8, label=label,
            )
            if ee["quiver_ok"]:

                ax = self.plot_quiver(ax, ee, color, quiver_len, quiver_width)

        self._finalize(ax, span, "Combined Trajectory" if title is None else title)

    def plot_desired_base_com(self, log, title=None, quiver_count=64):
        desired = self._path_quiver_data(log.p_desired, log.z_desired, quiver_count)
        if desired is None:
            return

        fig, ax = self._new_axes()
        ax.plot(
            desired["p"][:, 0], desired["p"][:, 1], desired["p"][:, 2],
            color="blue", alpha=0.9, label="des.p_c",
        )

        span = np.ptp(desired["p"], axis=0)
        quiver_len = max(0.08 * float(np.max(span)), 1e-3)
        if desired["quiver_ok"]:
            ax.quiver(
                desired["p_q"][:, 0], desired["p_q"][:, 1], desired["p_q"][:, 2],
                desired["z_q"][:, 0], desired["z_q"][:, 1], desired["z_q"][:, 2],
                length=quiver_len, normalize=True, color="blue", linewidth=1.1, alpha=0.55,
            )

        self._finalize(ax, span, "Desired Base-CoM Trajectory" if title is None else title)

