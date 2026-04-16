from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
from types import SimpleNamespace
from utils.pure import next_filename

PROJECT_ROOT = Path(__file__).resolve().parent.parent

@dataclass(slots=True)
class LogEntry:
    variable_name: str
    kind: str | None = None #state, variable, control, actuator
    subsystem: str | None = None # COM, base, EE, None
    data: list = field(default_factory=list)
    double: bool = False
    data2: list | None = None
    diff: str | None = None
    formatted_title: str | None = None
    metrics: list[str] = field(default_factory=lambda: ["rms", "max", "final"])
    axis_idx: int | tuple[int, ...] | None = None
    ylabel: str | None = None
    xlabel: str = "Time (s)"
    override_norm: bool = False

    def append(self, value):
        if value is None:
            self.data.append(None)
            return
        self.data.append(np.asarray(value, dtype=float).copy())
    
    def append2(self, value1, value2):
        if self.data2 is None:
            self.data2 = []
        self.data.append(np.asarray(value1, dtype=float).copy())
        self.data2.append(np.asarray(value2, dtype=float).copy())

    def has_secondary(self):
        return self.data2 is not None and len(self.data2) > 0

    def finalize(self, series=None, norm=True):
        if series is None: series = self.data
        if series is None or len(series) == 0:
            return np.array([])
        if series[0] is None:
            return list(series)
        if norm: series = self.calc_norm(series)
        return np.asarray(series)

    def finalize2(self, norm=True):
        series1 = self.finalize(self.data, norm=False)
        series2 = self.finalize(self.data2, norm=False)

        if self.diff == "diff":
            diff_series = series1 - series2
            if norm: diff_series = self.calc_norm(diff_series)
            return np.asarray(diff_series)
        
        elif self.diff == "pointing":
            flat1 = np.asarray(series1, dtype=float).reshape(series1.shape[0], -1)
            flat2 = np.asarray(series2, dtype=float).reshape(series2.shape[0], -1)
            e_point = np.ones(series1.shape[0]) - (np.clip(np.einsum("ij,ij->i", flat1, flat2), -1.0, 1.0))
            return e_point
        
        elif self.diff == None:
            if norm and not self.override_norm:
                series1 = self.calc_norm(series1)
                series2 = self.calc_norm(series2)
            return [series1, series2]
        
    def calc_norm(self, values=None):
        if values is None:
            values = self.finalize()
        values = np.asarray(values, dtype=float)
        if values.size == 0 or values.ndim <= 1:
            return values
        flat = values.reshape(values.shape[0], -1)
        if flat.shape[1] == 1:
            return flat[:, 0]
        norms = np.empty(flat.shape[0], dtype=flat.dtype)
        np.sqrt(np.einsum("ij,ij->i", flat, flat), out=norms)
        return norms
    
    def plot(self, norm=True, ax=None, x=None):
        has_secondary = self.has_secondary()
        if not has_secondary:
            values = self.finalize(norm=norm)
        else:
            values = self.finalize2(norm=norm)
        
        created_ax = ax is None
        if created_ax:
            _, ax = plt.subplots(figsize=(6, 3))
        x_vals = None if x is None else np.asarray(x, dtype=float).reshape(-1)
        if not has_secondary:
            values = np.asarray(values, dtype=float)
            if values.ndim == 0:
                values = values.reshape(1)
            x_plot = np.arange(values.shape[0], dtype=float) if x_vals is None else x_vals[:values.shape[0]]
            if values.ndim == 1:
                ax.plot(x_plot, values)
            else:
                coord_labels = ["x", "y", "z"]
                coord_colors = ["tab:blue", "tab:orange", "tab:green"]
                flat = values.reshape(values.shape[0], -1)
                if flat.shape[1] == 3:
                    for idx in range(flat.shape[1]):
                        ax.plot(x_plot, flat[:, idx], label=coord_labels[idx], color=coord_colors[idx])
                    ax.legend(
                        loc="center right",
                        bbox_to_anchor=(-0.15, 0.5),
                        borderaxespad=0.0,
                    )
                else:
                    ax.plot(x_plot, flat)
        elif self.diff is not None:
            x_plot = np.arange(values.shape[0], dtype=float) if x_vals is None else x_vals[:values.shape[0]]
            if not norm: 
                coord_labels = ["x", "y", "z"]
                coord_colors = ["tab:blue", "tab:orange", "tab:green"]
                flat = values.reshape(values.shape[0], -1)
                if flat.shape[1] == 3:
                    for idx in range(flat.shape[1]):
                        ax.plot(x_plot, flat[:, idx], label=coord_labels[idx], color=coord_colors[idx])
                    ax.legend(
                        loc="center right",
                        bbox_to_anchor=(-0.15, 0.5),
                        borderaxespad=0.0,
                    )
                else:
                    ax.plot(x_plot, flat)
            else:
                ax.plot(x_plot, values)
        else:
            v_actual, v_desired = values
            v_actual = np.asarray(v_actual, dtype=float)
            v_desired = np.asarray(v_desired, dtype=float)
            if v_actual.ndim == 0:
                v_actual = v_actual.reshape(1)
            if v_desired.ndim == 0:
                v_desired = v_desired.reshape(1)
            n = min(v_actual.shape[0], v_desired.shape[0])
            if x_vals is None:
                x_plot = np.arange(n, dtype=float)
            else:
                n = min(n, x_vals.shape[0])
                x_plot = x_vals[:n]
            v_actual = v_actual[:n]
            v_desired = v_desired[:n]

            coord_labels = ["x", "y", "z"]
            coord_colors = ["tab:blue", "tab:orange", "tab:green"]
            if v_actual.ndim == 1:
                ax.plot(x_plot, v_actual, color="tab:blue", label="actual")
                ax.plot(x_plot, v_desired, color="tab:orange", linestyle="--", label="desired")
            else:
                flat_actual = v_actual.reshape(n, -1)
                flat_desired = v_desired.reshape(n, -1)
                n_coord = min(flat_actual.shape[1], flat_desired.shape[1], len(coord_labels))
                for j in range(n_coord):
                    color = coord_colors[j]
                    label = coord_labels[j]
                    ax.plot(x_plot, flat_actual[:, j], color=color, label=f"actual {label}")
                    ax.plot(x_plot, flat_desired[:, j], color=color, linestyle="--", label=f"desired {label}")
            ax.legend()

        title = self.formatted_title
        if self.has_secondary():
            add_title = " Error" if self.diff is not None else " Actual vs. Desired"
            title += add_title

        ax.set_title(title)
        if self.xlabel is not None: ax.set_xlabel(self.xlabel)
        if self.ylabel is not None: ax.set_ylabel(self.ylabel)

        if created_ax:
            plt.tight_layout()
            plt.show()
        return ax
    
    def choose_ylabel(self):
        if self.variable_name.startswith("p_"):
            self.ylabel = "m"
        elif self.variable_name.startswith("f_"):
            self.ylabel = "N"
        elif self.variable_name.startswith("tau"):
            self.ylabel = "Nm"
        elif self.variable_name.startswith("v_"):
            self.ylabel = "m/s"
        elif self.variable_name.startswith("omega_"):
            self.ylabel = "rad/s"
        else: self.ylabel = "-"
    
    def choose_metrics(self):
        if self.double:
            if self.variable_name.startswith("z_"):
                self.diff = "pointing"
            else: self.diff = "diff"
        if self.variable_name == "sigma_min_gamma": self.metrics = ["rms", "min"]
        else:
            self.metrics = ["rms", "max", "final"]
            if self.kind =="control" or self.kind=="actuator" : self.metrics.extend(['cumulative'])

    def __post_init__(self):
        if self.double and self.data2 is None:
            self.data2 = []
        if self.formatted_title is None:
            self.formatted_title = formatted_title(self.variable_name)
        self.choose_ylabel()
        self.choose_metrics()
    
LOG_ORG = {
    "state": ["t", "q", "v"],
    "COM": {
        "control": ["f_c"],
        "vars": ["p_c", "v_c"],
        "actuator": ["f_b"],
    },
    "base": {
        "control": ["tau_b_oplus"],
        "vars": ["z_b", "omega_b"],
        "actuator": ["tau_b"],
    },
    "EE": {
        "control": ["omega_e_oplus"],
        "vars": ["p_e", "z_e", "nu_e_oplus"],
        "actuator": ["tau"],
    },
    "metrics": {
        "other": ["sigma_min_gamma"],
    },
}
DEFAULT_DIR = PROJECT_ROOT / "figures"
class CCLogger:
    def __init__(self, log_config=LOG_ORG, enable_base=0, enable_ee=0, metrics=0, actuator=0, control=0, add_title=None):
        self.add_title=add_title
        self.state_keys = log_config["state"]
        self.logs = {}

        for k in self.state_keys:
            self.logs[k]=LogEntry(variable_name=k, 
                                  subsystem=None,
                                  kind="state", 
                                  double=False)
        subsystems = ["COM"]
        if enable_base: subsystems += ["base"]
        if enable_ee: subsystems += ["EE"]
        
        for ss in subsystems:                
            for var in log_config[ss]["vars"]:
                self.logs[var]= LogEntry(variable_name=var, 
                                          subsystem=ss,
                                          kind="variable", 
                                          double=True)
            if actuator:
                for var in log_config[ss]["actuator"]:
                    self.logs[var]= LogEntry(variable_name=var, 
                                          subsystem=ss,
                                          kind="actuator", 
                                          double=False)
            if control:
                for var in log_config[ss]["control"]:
                    self.logs[var]=LogEntry(variable_name=var, 
                                          subsystem=ss,
                                          kind="control", 
                                          double=False)
        if metrics:
            for var in log_config.get("metrics", {}).get("other", []):
                self.logs[var] = LogEntry(
                    variable_name=var,
                    subsystem=None,
                    kind="metrics",
                    double=False,
                )

    def add_key(
        self,
        key: str,
        *,
        kind: str = "metrics",
        subsystem: str | None = None,
        double: bool = False,
        diff: str | None = None,
        formatted_title: str | None = None,
        metrics: list[str] | tuple[str, ...] | None = None,
        ylabel: str | None = None,
        xlabel: str = "Time (s)",
        overwrite: bool = False,
    ) -> LogEntry:
        if key in self.logs and not overwrite:
            return self.logs[key]

        entry = LogEntry(
            variable_name=key,
            kind=kind,
            subsystem=subsystem,
            double=double,
            formatted_title=formatted_title,
            ylabel=ylabel,
            xlabel=xlabel,
        )
        if diff is not None:
            entry.diff = diff
        if metrics is not None:
            entry.metrics = [self._normalize_metric_name(metric) for metric in metrics]

        self.logs[key] = entry
        if kind == "state" and key not in self.state_keys:
            self.state_keys.append(key)
        return entry
        
    def log_step(self, step_results: dict):
        for k, v in step_results.items():
            if k in self.logs.keys():
                entry = self.logs[k]
                if entry.double and isinstance(v, (tuple, list)) and len(v) == 2:
                    entry.append2(v[0], v[1])
                else:
                    entry.append(np.asarray(v, dtype=float).copy())
            else:
                self.add_key(k)
                print(f"WARNING: {k} was not in log keys. Default attributes assigned") 
                continue

    def finalize(self):
        out = {}
        for k, entry in self.logs.items():
            if entry.has_secondary():
                out[k] = entry.finalize2()
            else:
                out[k] = entry.finalize()
        return out
    

    def extend(self, log: dict, t_offset: float = 0.0):
        """
        Append a finalized log dictionary (arrays/lists) into this logger.
        Useful for stitching whole-segment logs together.

        If `t_offset` is provided, it is added to the incoming time vector before
        appending so multiple segment logs can be placed on a single global time axis.
        """
        t_offset = float(t_offset)
        for k, entry in self.logs.items():
            if k not in log:
                continue
            source = log[k]

            if isinstance(source, LogEntry):
                if source.has_secondary():
                    series1 = source.data
                    series2 = source.data2
                    if series1 is None or series2 is None or len(series1) == 0:
                        continue
                    for value1, value2 in zip(series1, series2):
                        if value1 is None or value2 is None:
                            continue
                        sample1 = np.asarray(value1, dtype=float).copy()
                        sample2 = np.asarray(value2, dtype=float).copy()
                        if k == "t":
                            sample1 = sample1 + t_offset
                            sample2 = sample2 + t_offset
                        entry.append2(sample1, sample2)
                    continue

                series = source.data
                if series is None or len(series) == 0:
                    continue
                for value in series:
                    if value is None:
                        entry.append(None)
                        continue
                    sample = np.asarray(value, dtype=float).copy()
                    if k == "t":
                        sample = sample + t_offset
                    entry.append(sample)
                continue

            if entry.double and isinstance(source, (tuple, list)) and len(source) == 2:
                arr1 = np.asarray(source[0], dtype=float)
                arr2 = np.asarray(source[1], dtype=float)
                if arr1.size == 0 or arr2.size == 0:
                    continue
                n = min(arr1.shape[0], arr2.shape[0]) if arr1.ndim > 0 and arr2.ndim > 0 else 1
                if arr1.ndim == 0:
                    arr1 = arr1.reshape(1)
                if arr2.ndim == 0:
                    arr2 = arr2.reshape(1)
                if k == "t":
                    arr1 = arr1 + t_offset
                    arr2 = arr2 + t_offset
                for row1, row2 in zip(arr1[:n], arr2[:n]):
                    entry.append2(np.asarray(row1, dtype=float).copy(), np.asarray(row2, dtype=float).copy())
                continue

            arr = np.asarray(source, dtype=float)
            if arr.size == 0:
                continue

            if k == "t":
                arr = arr + t_offset

            if arr.ndim == 0:
                entry.append(arr.copy())
            else:
                for row in arr:
                    entry.append(np.asarray(row, dtype=float).copy())

    @staticmethod
    def _normalize_metric_name(metric: str):
        aliases = {
            "peak": "max",
            "maximum": "max",
            "cum": "cumulative",
        }
        return aliases.get(metric.lower(), metric.lower())

    def _time_vector(self):
        if "t" not in self.logs:
            raise RuntimeError("No time variable in log.")
        return np.asarray(self.logs["t"].finalize(norm=False), dtype=float).reshape(-1)

    def _entry_values(self, key: str, norm=False):
        entry = self.logs[key]
        if entry.has_secondary():
            return entry.finalize2(norm=norm)
        return entry.finalize(norm=norm)

    def _metric_signal(self, key: str, magnitude: bool = True):
        values = self._entry_values(key, norm=False)
        if isinstance(values, list):
            values = np.asarray(values[0], dtype=float)
        else:
            values = np.asarray(values, dtype=float)

        if values.size == 0:
            return np.array([])
        if values.ndim <= 1:
            signal = values.reshape(-1)
            return np.abs(signal) if magnitude else signal
        flat = values.reshape(values.shape[0], -1)
        return np.linalg.norm(flat, axis=1)

    def metric_series(self, key: str, metric: str):
        metric_name = self._normalize_metric_name(metric)
        signal = np.asarray(
            self._metric_signal(key, magnitude=(metric_name != "min")),
            dtype=float,
        ).reshape(-1)
        if signal.size == 0:
            return signal
        if metric_name == "norm":
            return signal
        if metric_name == "min":
            return np.minimum.accumulate(signal)
        if metric_name == "max":
            return np.maximum.accumulate(signal)
        if metric_name == "cumulative":
            t = self._time_vector()
            if len(t) != len(signal):
                return np.cumsum(signal)
            out = np.zeros_like(signal, dtype=float)
            if len(signal) > 1:
                dt = np.diff(t)
                out[1:] = np.cumsum(0.5 * (signal[:-1] + signal[1:]) * dt)
            return out
        return signal

    def _compute_metric(self, key: str, metric: str):
        metric_name = self._normalize_metric_name(metric)
        signal = np.asarray(
            self._metric_signal(key, magnitude=(metric_name != "min")),
            dtype=float,
        ).reshape(-1)
        if signal.size == 0:
            return float("nan")
        if metric_name == "rms":
            return float(np.sqrt(np.mean(signal ** 2)))
        if metric_name == "final":
            return float(signal[-1])
        if metric_name == "max":
            return float(np.max(signal))
        if metric_name == "min":
            return float(np.min(signal))
        if metric_name == "norm":
            return float(signal[-1])
        if metric_name == "cumulative":
            return float(self.metric_series(key, "cumulative")[-1])
        raise ValueError(f"Unsupported metric {metric!r}")

    @staticmethod
    def _sanitize_plot_name(name: str):
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
        return sanitized or "plot"

    def _plot_keys(self, plot_keys, title=None, save_dir=None, file_stem="plot", show=True):
        plot_items = [key for key in plot_keys if len(self.logs[key].data) > 0]
        if not plot_items:
            return

        t_vals = self._time_vector() if len(self.logs["t"].data) > 0 else None
        output_dir = None
        if save_dir is not None:
            output_dir = Path(save_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        max_plots_per_figure = 16
        if len(plot_items) > max_plots_per_figure:
            figure_groups = [
                plot_items[idx:idx + max_plots_per_figure]
                for idx in range(0, len(plot_items), max_plots_per_figure)
            ]
        else:
            figure_groups = [plot_items]

        def plot_group(items, figure_title=None, figure_index=None):
            nk = len(items)
            if nk < 3: n_cols = nk
            if nk < 12: n_cols = 3 
            else: n_cols = 4

            n_rows = nk // n_cols if nk % n_cols == 0 else nk // n_cols + 1
            fig, axs = plt.subplots(
                n_rows,
                n_cols,
                figsize=(6 * n_cols, 3 * n_rows),
                constrained_layout=True,
            )
            axes = np.asarray(axs).reshape(-1)
            for idx, key in enumerate(items):
                self.logs[key].plot(ax=axes[idx], x=t_vals)
            for ax in axes[nk:]:
                ax.set_visible(False)

            if figure_title is not None:
                fig.suptitle(figure_title)
            if output_dir is not None:
                suffix = f"_{figure_index}" if len(figure_groups) > 1 else ""

                filename = next_filename(f"{self._sanitize_plot_name(file_stem)}{suffix}", folder=str(output_dir), suffix=".png")
                fig.savefig(filename, dpi=150, bbox_inches="tight")
                print(f"Saved figure to {filename}")
            if show:
                plt.show()
            else:
                plt.close(fig)

        if len(figure_groups) == 1:
            plot_group(figure_groups[0], figure_title=title, figure_index=1)
            return

        for idx, group in enumerate(figure_groups, start=1):
            group_title = title if title is None else f"{title} ({idx}/{len(figure_groups)})"
            plot_group(group, figure_title=group_title, figure_index=idx)

    def plot_all(self, save=False, directory=DEFAULT_DIR):
        plot_keys = [key for key, entry in self.logs.items() if entry.kind != "state"]
        self._plot_keys(
            plot_keys,
            save_dir=directory if save else None,
            file_stem="all_plots",
        )


    def plot_by_subsystem(self, save=False, directory=DEFAULT_DIR):
        add_title = self.add_title
        subsystems = []
        for entry in self.logs.values():
            if entry.kind == "state" or entry.subsystem is None or entry.subsystem in subsystems:
                continue
            subsystems.append(entry.subsystem)

        for subsystem in subsystems:
            plot_keys = [
                key for key, entry in self.logs.items()
                if entry.kind != "state" and entry.subsystem == subsystem
            ]
            title=f"{subsystem} Plots"
            if add_title is not None: title += f" - {add_title}"
            self._plot_keys(
                plot_keys,
                title=title,
                save_dir=directory if save else None,
                file_stem=f"{subsystem}_plots",
            )

    def plot_by_kind(self, save=False, directory=DEFAULT_DIR, show=True, debug_time_limit=None):
        add_title = self.add_title
        kind_titles = {
            "metrics": "Metrics",
            "variable": "Variables",
            "control": "Controls",
            "actuator": "Actuators",
        }
        kind_stems = {
            "metrics": "metrics",
            "variable": "error",
            "control": "controls",
            "actuator": "actuators",
        }
        stem_prefix = f"{self._sanitize_plot_name(add_title)}_" if add_title is not None else ""
        stem_suffix = f"_{debug_time_limit}s" if debug_time_limit is not None else ""
        for kind, title in kind_titles.items():
            plot_keys = [key for key, entry in self.logs.items() if entry.kind == kind]
            if add_title is not None: title += f"{add_title}"
            self._plot_keys(
                plot_keys,
                title=title,
                save_dir=directory if save else None,
                file_stem=f"{stem_prefix}{kind_stems[kind]}{stem_suffix}",
                show=show,
            )
    

    def return_states(self):
        def matrix_for(key):
            if key not in self.logs:
                return None
            values = np.asarray(self.logs[key].finalize(norm=False), dtype=float)
            if values.size == 0:
                return None
            if values.ndim == 1:
                return values.reshape(-1, 1)
            return values.reshape(values.shape[0], -1)

        t_vals = self._time_vector()
        x_parts = [matrix_for("q"), matrix_for("v")]
        x_parts = [part for part in x_parts if part is not None]
        u_parts = [matrix_for(key) for key in ("f_b", "tau_b", "tau")]
        u_parts = [part for part in u_parts if part is not None]

        lengths = [len(t_vals)]
        lengths.extend(part.shape[0] for part in x_parts)
        lengths.extend(part.shape[0] for part in u_parts)
        n = min(lengths) if lengths else 0

        x_vals = np.concatenate([part[:n] for part in x_parts], axis=1) if x_parts else np.zeros((n, 0))
        u_vals = np.concatenate([part[:n] for part in u_parts], axis=1) if u_parts else np.zeros((n, 0))
        return SimpleNamespace(t=t_vals[:n], x=x_vals, u=u_vals)

    def aligned_text(
        self,
        metrics: dict,
        title: str | None = None,
        head_text="quantity",
        keys: list[str] | None = None,
        metric_columns: list[str] | tuple[str, ...] | None = None,
    ) -> str:
        if keys is None:
            keys = [k for k, entry in self.logs.items() if entry.kind != "state" and len(entry.data) > 0]
        if metric_columns is None:
            metric_columns = []
            for key in keys:
                for metric in self.logs[key].metrics:
                    metric_name = self._normalize_metric_name(metric)
                    if metric_name not in metric_columns:
                        metric_columns.append(metric_name)
        if len(keys) == 0:
            return "" if title is None else title

        def fmt(value):
            return f"{value:>12.3g}" if np.isfinite(value) else f"{'nan':>12}"

        quantity_width = max([len(head_text), *[len(k) for k in keys]])
        display_names = {"max": "peak"}
        header = f"{head_text:<{quantity_width}} | " + " | ".join(
            f"{display_names.get(metric, metric):>12}" for metric in metric_columns
        )
        divider = "-" * len(header)
        rows = [header, divider] if title is None else [title, header, divider]
        for k in keys:
            entry_metrics = metrics.get(k, {})
            row = [f"{k:<{quantity_width}}"]
            for metric in metric_columns:
                value = entry_metrics.get(metric)
                row.append(fmt(value) if value is not None else f"{'':>12}")
            rows.append(" | ".join(row))
        return "\n".join(rows)
    
    def summarize_metrics(self, printout=True, title=None):
        metrics_by_key = {}
        metrics = {}
        variable_keys = []
        metrics_keys = []
        control_keys = []
        actuator_keys = []

        for key, entry in self.logs.items():
            if entry.kind == "state" or len(entry.data) == 0:
                continue
            entry_metrics = {}
            for metric in entry.metrics:
                metric_name = self._normalize_metric_name(metric)
                value = self._compute_metric(key, metric_name)
                entry_metrics[metric_name] = value
                metrics[f"{key}_{metric_name}"] = value
            metrics_by_key[key] = entry_metrics

            if entry.kind == "control":
                control_keys.append(key)
            elif entry.kind == "actuator":
                actuator_keys.append(key)
            elif entry.kind == "metrics":
                metrics_keys.append(key)
            else:
                variable_keys.append(key)

        if printout:
            sections = []
            if title is not None:
                sections.append(title)

            grouped_sections = [
                ("Metrics", metrics_keys),
                ("Variables", variable_keys),
                ("Controls", control_keys),
                ("Actuators", actuator_keys),
            ]
            for section_title, section_keys in grouped_sections:
                if not section_keys:
                    continue
                if self.add_title is not None:
                    section_title = f"{section_title}{self.add_title}"
                sections.append(self.aligned_text(metrics_by_key, title=section_title, keys=section_keys))
            print("\n\n".join(section for section in sections if section))
        return metrics

# ==== LaTeX formatting
def latex_token(token: str):
    token_map = {
        "sigma": r"\sigma",
        "gamma": r"\Gamma",
        "tau": r"\tau",
        "omega": r"\omega",
        "nu": r"\nu",
        "min": r"\min",
    }
    if token in token_map:
        return token_map[token]
    if len(token) == 1 and token.isalpha():
        return token
    return rf"\mathrm{{{token}}}"

def formatted_title(key: str):
    if key is None:
        return None

    body = key.split(".", 1)[-1]
    special_titles = {
        "xb_tilde": r"$\tilde{x}_b$",
        "xb_tilde_dot": r"$\dot{\tilde{x}}_b$",
        "com_err": r"$\tilde{x}_c$",
        "vc_err": r"$\dot{\tilde{x}}_c$",
    }
    if body in special_titles:
        return special_titles[body]
    if body == "sigma_min_gamma":
        return r"$\sigma_{\min}(\Gamma)$"

    if body.endswith("_oplus"):
        base = formatted_title(body[:-6])
        if base is None:
            return None
        base_expr = base[1:-1] if base.startswith("$") and base.endswith("$") else base
        return rf"${base_expr}^{{\oplus}}$"

    tokens = body.split("_")
    if len(tokens) == 1:
        return rf"${latex_token(tokens[0])}$"

    head = latex_token(tokens[0])
    subscripts = ", ".join(latex_token(token) for token in tokens[1:])
    return rf"${head}_{{{subscripts}}}$"


# """
# import logging
# from datetime import datetime
# def today():
#     date = datetime.now().strftime("%m%d")
#     if date[0] == '0': date = date[1:]
#     return date
# def start_logging(fname):

#     "Start a log with the name you give it in logs directory. Suffix is .log "
#     logging.basicConfig(
#         filename=f'logs/{fname}.log',  # Specify the log file name
#         filemode='a',        # 'a' for append mode, 'w' for write (overwrite) mode
#         format='%(asctime)s - %(levelname)s - %(message)s', # Define log message format
#         level=logging.INFO,   # Set the minimum logging level to capture
#         datefmt='%m/%d/%Y %I:%M:%S %p'
#         )
#     # Get a logger instance
#     logger = logging.getLogger(__name__)
#     return logger
# """
