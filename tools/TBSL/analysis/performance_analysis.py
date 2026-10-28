import os
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


from ._config import DATA_ROOT, OUTPUT_ROOT, REPORTS_DIR, PERFORMANCE_DIR, data_path

FIG_DIR = PERFORMANCE_DIR
TODO_DIR = REPORTS_DIR

plt.rcParams.update(
    {
        "font.family": "serif",
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


@dataclass(frozen=True)
class CpuRun:
    label: str
    path: Path
    cores: int


@dataclass(frozen=True)
class TimingRun:
    label: str
    path: Path


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TODO_DIR.mkdir(parents=True, exist_ok=True)


def format_hours(hours: float) -> str:
    return f"{hours:.3f}"


def format_cpu_hours(cpu_hours: float) -> str:
    return f"{cpu_hours:,.1f}"


def parse_last_cpu_seconds(path: Path) -> tuple[int, float]:
    last_step = None
    last_cpu = None
    with path.open() as handle:
        for raw_line in handle:
            parts = raw_line.split()
            if len(parts) < 4:
                continue
            if not parts[0].isdigit():
                continue
            try:
                last_step = int(parts[0])
                last_cpu = float(parts[3])
            except ValueError:
                continue
    if last_step is None or last_cpu is None:
        raise ValueError(f"Could not find trailing Step/CPU row in {path}")
    return last_step, last_cpu


def parse_thermo_history(path: Path) -> dict[str, np.ndarray]:
    dt_fs: list[float] = []
    time_fs: list[float] = []
    cpu_s: list[float] = []
    vcpu: list[float] = []
    tpcpu: list[float] = []
    spcpu: list[float] = []
    with path.open() as handle:
        for raw_line in handle:
            parts = raw_line.split()
            if len(parts) < 7:
                continue
            if not parts[0].isdigit():
                continue
            try:
                dt_fs.append(float(parts[1]))
                time_fs.append(float(parts[2]))
                cpu_s.append(float(parts[3]))
                vcpu.append(float(parts[4]))
                tpcpu.append(float(parts[5]))
                spcpu.append(float(parts[6]))
            except ValueError:
                continue
    if not time_fs or not cpu_s:
        raise ValueError(f"Could not parse thermo history from {path}")
    return {
        "dt_fs": np.asarray(dt_fs, dtype=float),
        "time_fs": np.asarray(time_fs, dtype=float),
        "cpu_s": np.asarray(cpu_s, dtype=float),
        "vcpu": np.asarray(vcpu, dtype=float),
        "tpcpu": np.asarray(tpcpu, dtype=float),
        "spcpu": np.asarray(spcpu, dtype=float),
    }


def parse_total_wall_time_seconds(path: Path) -> float | None:
    match = re.search(r"Total wall time:\s*(\d+):(\d+):(\d+)", path.read_text())
    if not match:
        return None
    hours, minutes, seconds = map(int, match.groups())
    return float(hours * 3600 + minutes * 60 + seconds)


def parse_timing_table(path: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.rstrip()
            if "|" not in line or line.startswith("-") or line.startswith("Section"):
                continue
            parts = [part.strip() for part in line.split("|")]
            if len(parts) < 6:
                continue
            section = parts[0]
            if not section:
                continue
            avg_time = float(parts[2]) if parts[2] else 0.0
            pct_total = float(parts[5]) if parts[5] else 0.0
            rows[section] = {"avg_time": avg_time, "pct_total": pct_total}
    if not rows:
        raise ValueError(f"Could not parse timing table from {path}")
    return rows


def disjoint_wall_clock_seconds(rows: dict[str, dict[str, float]]) -> float:
    disjoint_sections = ["Pair", "Bond", "Neigh", "Comm", "Output", "Modify", "Other"]
    return sum(rows.get(section, {}).get("avg_time", 0.0) for section in disjoint_sections)


def detailed_breakdown(rows: dict[str, dict[str, float]]) -> dict[str, float]:
    pair = rows["Pair"]["pct_total"]
    ioniz = rows.get("Ioniz", {}).get("pct_total", 0.0)
    neigh = rows["Neigh"]["pct_total"]
    comm = rows["Comm"]["pct_total"]
    modify_total = rows["Modify"]["pct_total"]
    output = rows.get("Output", {}).get("pct_total", 0.0)
    bond = rows.get("Bond", {}).get("pct_total", 0.0)
    other_native = rows["Other"]["pct_total"]
    modify_residual = modify_total - ioniz
    other_collapsed = output + bond + other_native
    return {
        "Pair": pair,
        "Ionization": ioniz,
        "Neigh": neigh,
        "Communication": comm,
        "Modify": modify_residual,
        "Other": other_collapsed,
    }


def plot_timing_breakdown(timing_data: dict[str, dict[str, float]]) -> Path:
    sections = ["Other", "Neigh", "Communication", "Pair", "Ionization", "Modify"]
    colors = {
        "Other": "#EEE7D8",
        "Neigh": "#C7D8C6",
        "Communication": "#95B8B0",
        "Pair": "#D6B98A",
        "Ionization": "#C98C5B",
        "Modify": "#8E6C88",
    }
    labels = list(timing_data.keys())
    display_labels = {
        "1e6": r"$N=10^6$",
        "1e7": r"$N=10^7$",
        "1e8": r"$N=10^8$",
    }
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8.0, 3.5))
    left = np.zeros(len(labels), dtype=float)
    for section in sections:
        values = np.array([timing_data[label][section] for label in labels], dtype=float)
        bars = ax.barh(
            y,
            values,
            left=left,
            height=0.76,
            color=colors[section],
            edgecolor="white",
            linewidth=1.2,
            label=section,
        )
        for bar, value, x0 in zip(bars, values, left):
            if value < 3.0:
                continue
            x_center = x0 + value / 2.0
            text_color = "black" if section in {"Other", "Neigh", "Communication", "Pair"} else "white"
            ax.text(
                x_center,
                bar.get_y() + bar.get_height() / 2.0,
                f"{value:.0f}%",
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
            )
        left += values

    ax.set_xlim(0.0, 100.0)
    ax.set_yticks(y)
    ax.set_yticklabels([display_labels[label] for label in labels])
    ax.invert_yaxis()
    ax.set_xlabel("Runtime share (%)")
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, ncols=6, loc="upper center", bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout()

    out_path = FIG_DIR / "P1_timing_breakdown.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_ionization_overhead(overhead_hours: dict[str, float]) -> Path:
    labels = [
        "No ionization",
        "Short-range ionization",
        "Full ionization\n(LJ + DSF Coulomb)",
    ]
    keys = ["no ionization", "short ionization", "full ionization"]
    values = [overhead_hours[key] for key in keys]
    ratios = [value / values[0] for value in values]
    colors = ["#C7D8C6", "#C98C5B", "#8E6C88"]

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors, height=0.7, edgecolor="white", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Wall-clock time (h)")

    x_pad = max(values) * 0.03
    for bar, value, ratio in zip(bars, values, ratios):
        ax.text(
            bar.get_width() + x_pad,
            bar.get_y() + bar.get_height() / 2.0,
            f"{value:.2f} h ({ratio:.2f}x)",
            va="center",
            ha="left",
            fontsize=10,
        )

    fig.tight_layout()
    out_path = FIG_DIR / "P2_ionization_overhead.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def smooth_series(values: np.ndarray, window: int = 21) -> np.ndarray:
    if window <= 1 or values.size < window:
        return values.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def throughput_minimum_stats(cpu_runs: list[CpuRun], window: tuple[float, float] | None = None) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for run in cpu_runs:
        history = parse_thermo_history(run.path)
        time_ns = history["time_fs"] * 1.0e-6
        spcpu = history["spcpu"]
        valid = (time_ns > 0.0) & (spcpu > 0.0)
        time_ns = time_ns[valid]
        spcpu = spcpu[valid]
        if window is not None:
            wmask = (time_ns >= window[0]) & (time_ns <= window[1])
            time_ns = time_ns[wmask]
            spcpu = spcpu[wmask]
        min_idx = int(np.argmin(spcpu))
        stats[run.label] = {
            "time_ns": float(time_ns[min_idx]),
            "spcpu": float(spcpu[min_idx]),
        }
    return stats


def plot_step_throughput(cpu_runs: list[CpuRun], normalization: str, filename: str) -> Path:
    colors = {"1e6": "#4C78A8", "1e7": "#F58518", "1e8": "#54A24B"}
    legend_labels = {"1e6": r"$N = 10^6$", "1e7": r"$N = 10^7$", "1e8": r"$N = 10^8$"}

    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    for run in cpu_runs:
        history = parse_thermo_history(run.path)
        time_ns = history["time_fs"] * 1.0e-6
        spcpu = history["spcpu"]
        valid = (time_ns > 0.0) & (spcpu > 0.0)
        time_ns = time_ns[valid]
        spcpu = spcpu[valid]
        smoothed_spcpu = smooth_series(spcpu, window=21)

        if normalization == "median":
            baseline_mask = time_ns <= 1.0
            if np.any(baseline_mask):
                scale = float(np.median(smoothed_spcpu[baseline_mask]))
            else:
                positive = smoothed_spcpu[smoothed_spcpu > 0.0]
                scale = float(np.median(positive[: min(len(positive), 50)]))
        elif normalization == "min":
            collapse_mask = (time_ns >= 4.0) & (time_ns <= 5.0)
            if np.any(collapse_mask):
                scale = float(np.min(smoothed_spcpu[collapse_mask]))
            else:
                scale = float(np.min(smoothed_spcpu))
        else:
            raise ValueError(f"Unknown normalization mode: {normalization}")

        normalized_spcpu = smoothed_spcpu / scale
        ax.plot(time_ns, normalized_spcpu, color=colors[run.label], linewidth=2.0, label=legend_labels.get(run.label, run.label))

    ax.set_xlim(0.0, 10.1)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel(r"Normalized $S/CPU$")
    ax.grid(True, alpha=0.25)
    if filename == "P3_time_vs_throughput_min_norm.pdf":
        ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()

    out_path = FIG_DIR / filename
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_timestep_size(cpu_runs: list[CpuRun]) -> Path:
    colors = {"1e6": "#4C78A8", "1e7": "#F58518", "1e8": "#54A24B"}

    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    for run in cpu_runs:
        history = parse_thermo_history(run.path)
        time_ns = history["time_fs"] * 1.0e-6
        dt_fs = history["dt_fs"]
        valid = time_ns > 0.0
        ax.plot(time_ns[valid], dt_fs[valid], color=colors[run.label], linewidth=2.0)

    ax.set_xlim(0.0, 10.1)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Dt (fs)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    out_path = FIG_DIR / "P4_time_vs_dt.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_report(
    cpu_summary: dict[str, dict[str, float | int | str]],
    timing_rows: dict[str, dict[str, dict[str, float]]],
    collapsed: dict[str, dict[str, float]],
    overhead_hours: dict[str, float],
    p1_path: Path,
    p2_path: Path,
    p3_path: Path,
    p3_min_path: Path,
    p4_path: Path,
    throughput_minima: dict[str, dict[str, float]],
    segment_paths: list[Path],
) -> Path:
    report_path = TODO_DIR / "performance_results_summary.md"

    nseg = len(segment_paths)
    nrestart = max(nseg - 1, 0)

    lines = [
        "# Performance Results Summary",
        "",
        "## Task 1 — CPU-hours per run",
        "",
        f"N=1e6: wall-clock = {format_hours(float(cpu_summary['1e6']['hours']))} h, cores = 128, CPU-hours = {format_cpu_hours(float(cpu_summary['1e6']['cpu_hours']))}",
        f"N=1e7: wall-clock = {format_hours(float(cpu_summary['1e7']['hours']))} h, cores = 2048, CPU-hours = {format_cpu_hours(float(cpu_summary['1e7']['cpu_hours']))}",
        f"N=1e8: wall-clock = {format_hours(float(cpu_summary['1e8']['hours']))} h, cores = 2048, CPU-hours = {format_cpu_hours(float(cpu_summary['1e8']['cpu_hours']))} (N_seg = {nseg} job segments = {nrestart} restart files after the initial run)",
        "",
        "Log files used:",
        f"- `N=1e6`: `{cpu_summary['1e6']['path']}`",
        f"- `N=1e7`: `{cpu_summary['1e7']['path']}`",
        f"- `N=1e8`: `{cpu_summary['1e8']['path']}`",
        "- `N=1e8` segment count:",
    ]
    for path in segment_paths:
        lines.append(f"  - `{os.path.relpath(path, DATA_ROOT)}`")

    lines.extend(
        [
            "",
            "## Task 2 — LAMMPS timing breakdown (1e6, 1e7, and 1e8)",
            "",
            "Important parsing note:",
            "- `Ioniz` is reported as a nested sub-entry inside `Modify`, not as an extra disjoint section. For plotting and reporting it separately, `Modify` is therefore redefined as the non-ionization remainder: `Modify_residual = Modify - Ioniz`.",
            "- `Bond` is dropped from the six-way plot because it is negligible here, and `Output` is folded into `Other` as requested.",
            "",
            "N=1e6 timing breakdown (%total, from available performance snippet):",
            f"- Pair (LJ + DSF): {timing_rows['1e6']['Pair']['pct_total']:.2f} %",
            f"- Ionization: {timing_rows['1e6']['Ioniz']['pct_total']:.2f} %",
            f"- Neigh (list): {timing_rows['1e6']['Neigh']['pct_total']:.2f} %",
            f"- Communication: {timing_rows['1e6']['Comm']['pct_total']:.2f} %",
            f"- Modify (non-ionization residual): {collapsed['1e6']['Modify']:.2f} %",
            f"- Other (native Other + Output, Bond dropped): {collapsed['1e6']['Other']:.2f} %",
            "",
            "N=1e7 timing breakdown (%total):",
            f"- Pair (LJ + DSF): {timing_rows['1e7']['Pair']['pct_total']:.2f} %",
            f"- Ionization: {timing_rows['1e7']['Ioniz']['pct_total']:.2f} %",
            f"- Neigh (list): {timing_rows['1e7']['Neigh']['pct_total']:.2f} %",
            f"- Communication: {timing_rows['1e7']['Comm']['pct_total']:.2f} %",
            f"- Modify (non-ionization residual): {collapsed['1e7']['Modify']:.2f} %",
            f"- Other (native Other + Output, Bond dropped): {collapsed['1e7']['Other']:.2f} %",
            "",
            "N=1e8 timing breakdown (%total):",
            f"- Pair (LJ + DSF): {timing_rows['1e8']['Pair']['pct_total']:.2f} %",
            f"- Ionization: {timing_rows['1e8']['Ioniz']['pct_total']:.2f} %",
            f"- Neigh (list): {timing_rows['1e8']['Neigh']['pct_total']:.2f} %",
            f"- Communication: {timing_rows['1e8']['Comm']['pct_total']:.2f} %",
            f"- Modify (non-ionization residual): {collapsed['1e8']['Modify']:.2f} %",
            f"- Other (native Other + Output, Bond dropped): {collapsed['1e8']['Other']:.2f} %",
            "",
            "Timing breakdown files used:",
            "- `N=1e6`: `performance/1e6_full_0.5.lammps`",
            "- `N=1e7`: `performance/1e7_full.lammps`",
            "- `N=1e8`: `performance/1e8_full.lammps`",
            "",
            "## Task 3 — Ionization model overhead at N=1e6",
            "",
            f"- N=1e6, no ionization: wall-clock = {overhead_hours['no ionization']:.3f} h",
            f"- N=1e6, short ionization: wall-clock = {overhead_hours['short ionization']:.3f} h ({overhead_hours['short ionization'] / overhead_hours['no ionization']:.3f}x no-ionization)",
            f"- N=1e6, full ionization: wall-clock = {overhead_hours['full ionization']:.3f} h ({overhead_hours['full ionization'] / overhead_hours['no ionization']:.3f}x no-ionization)",
            "",
            "Overhead files used:",
            "- `performance/1e6_no_0.5.lammps`",
            "- `performance/1e6_short_0.5.lammps`",
            "- `performance/1e6_full_0.5.lammps`",
            "- These wall-clock values are reconstructed from the disjoint timing rows (`Pair + Bond + Neigh + Comm + Output + Modify + Other`), because the provided comparison snippets do not include a final glued `CPU` column or `Total wall time` footer.",
            "",
            "## Figure Outputs",
            "",
            f"- Fig. P1: `{os.path.relpath(p1_path, OUTPUT_ROOT)}`",
            f"- Fig. P2: `{os.path.relpath(p2_path, OUTPUT_ROOT)}`",
            f"- Fig. P3: `{os.path.relpath(p3_path, OUTPUT_ROOT)}`",
            f"- Alternate min-normalized P3: `{os.path.relpath(p3_min_path, OUTPUT_ROOT)}`",
            f"- Fig. P4: `{os.path.relpath(p4_path, OUTPUT_ROOT)}`",
            "- Fig. P3 uses physical time on the x-axis and normalized `S/CPU` on the y-axis, where `S/CPU` is the LAMMPS step-rate metric (`timesteps per CPU second`). The startup artifact at `t = 0` is dropped because LAMMPS reports `S/CPU = 0` before the first elapsed interval exists.",
            "- The canonical P3 version is lightly smoothed with a 21-point moving average, then normalized by each run's own median smoothed `S/CPU` over `0 < t <= 1 ns`.",
            "- The alternate P3 version is lightly smoothed with a 21-point moving average, then normalized by each run's own minimum smoothed `S/CPU` inside the collapse window `4 <= t <= 5 ns`, so each curve's plotted collapse bottleneck sits at 1.",
            "- Fig. P4 uses the same Time axis and line styling as P3, but plots the raw adaptive timestep size `Dt` without normalization.",
            "",
            "Collapse-window throughput minima (`S/CPU`, excluding the initialization point at `t=0`):",
            f"- N=1e6: min at `t = {throughput_minima['1e6']['time_ns']:.3f} ns`, `S/CPU = {throughput_minima['1e6']['spcpu']:.3f}`",
            f"- N=1e7: min at `t = {throughput_minima['1e7']['time_ns']:.3f} ns`, `S/CPU = {throughput_minima['1e7']['spcpu']:.3f}`",
            f"- N=1e8: min at `t = {throughput_minima['1e8']['time_ns']:.3f} ns`, `S/CPU = {throughput_minima['1e8']['spcpu']:.3f}`",
            "",
            "## Anomalies / Notes",
            "",
            "- The available 1e6 timing-breakdown file is `performance/1e6_full_0.5.lammps`, while the Task 1 glued wall-clock run is `1e6/boundary/run_1e6_01_08_alpha_1.lammps`; the timing-share and wall-clock numbers therefore come from different but related 1e6 runs.",
            "- The performance directory does not contain a file literally named `performance/1e6_full.lammps`; the available full-ionization snippet on disk is `performance/1e6_full_0.5.lammps`.",
            "- `performance/1e6_full_0.5.lammps` omits the `MPI task timing breakdown:` header line, but the timing table itself is intact.",
            "- `performance/1e7_full.lammps` confirms the interpretation of the `CPU` column: its footer wall-clock (`27:10:49`) matches the disjoint timing-table sum and agrees with the final `CPU` value in `1e7/run_1e7_01_07.lammps` to within a fraction of a second.",
            "- The `1e8` production run spans 7 job segments total (1 initial job + 6 restart files).",
            "- In the available 1e6 overhead snippets, the short-range ionization run is slower than the full-range ionization run. The timing table shows this is driven mainly by a larger `Pair` share in the short-range case, not by a larger `Ioniz` share in the full-range case.",
            "- For adaptive-timestep runs, LAMMPS `tpcpu` is not the same as step throughput. The source documentation defines `tpcpu` as simulation time per CPU second and notes that it does not track timestep-size changes under `fix dt/reset`; `spcpu` is the correct raw step-rate metric.",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def main() -> None:
    ensure_dirs()

    cpu_runs = [
        CpuRun("1e6", Path(data_path('1e6', 'boundary', 'run_1e6_01_08_alpha_1.lammps')), 128),
        CpuRun("1e7", Path(data_path('1e7', 'run_1e7_01_07.lammps')), 2048),
        CpuRun("1e8", Path(data_path('1e8', 'run_1e8_full.lammps')), 2048),
    ]

    cpu_summary: dict[str, dict[str, float | int | str]] = {}
    for run in cpu_runs:
        _, cpu_seconds = parse_last_cpu_seconds(run.path)
        hours = cpu_seconds / 3600.0
        cpu_summary[run.label] = {
            "path": os.path.relpath(run.path, DATA_ROOT),
            "seconds": cpu_seconds,
            "hours": hours,
            "cores": run.cores,
            "cpu_hours": hours * run.cores,
        }

    timing_runs = {
        "1e6": TimingRun("1e6", Path(data_path('performance', '1e6_full_0.5.lammps'))),
        "1e7": TimingRun("1e7", Path(data_path('performance', '1e7_full.lammps'))),
        "1e8": TimingRun("1e8", Path(data_path('performance', '1e8_full.lammps'))),
    }
    timing_rows = {label: parse_timing_table(run.path) for label, run in timing_runs.items()}
    collapsed = {label: detailed_breakdown(rows) for label, rows in timing_rows.items()}

    p1_path = plot_timing_breakdown(collapsed)

    overhead_files = {
        "no ionization": Path(data_path('performance', '1e6_no_0.5.lammps')),
        "short ionization": Path(data_path('performance', '1e6_short_0.5.lammps')),
        "full ionization": Path(data_path('performance', '1e6_full_0.5.lammps')),
    }
    overhead_hours = {
        label: disjoint_wall_clock_seconds(parse_timing_table(path)) / 3600.0
        for label, path in overhead_files.items()
    }
    p2_path = plot_ionization_overhead(overhead_hours)
    p3_path = plot_step_throughput(cpu_runs, normalization="median", filename="P3_time_vs_throughput.pdf")
    p3_min_path = plot_step_throughput(cpu_runs, normalization="min", filename="P3_time_vs_throughput_min_norm.pdf")
    p4_path = plot_timestep_size(cpu_runs)
    throughput_minima = throughput_minimum_stats(cpu_runs, window=(4.0, 5.0))

    segment_paths = sorted((Path(data_path('anvil_1e8'))).glob("run_1e8_*restart.lammps"))
    segment_paths = [Path(data_path('anvil_1e8', 'run_1e8_12_03.lammps'))] + segment_paths

    report_path = write_report(
        cpu_summary,
        timing_rows,
        collapsed,
        overhead_hours,
        p1_path,
        p2_path,
        p3_path,
        p3_min_path,
        p4_path,
        throughput_minima,
        segment_paths,
    )

    # Sanity printout for terminal use.
    print(f"Wrote {p1_path}")
    print(f"Wrote {p2_path}")
    print(f"Wrote {p3_path}")
    print(f"Wrote {p3_min_path}")
    print(f"Wrote {p4_path}")
    print(f"Wrote {report_path}")
    for label in ["1e6", "1e7", "1e8"]:
        print(
            f"{label}: hours={cpu_summary[label]['hours']:.6f}, "
            f"cpu_hours={cpu_summary[label]['cpu_hours']:.3f}"
        )


if __name__ == "__main__":
    main()
