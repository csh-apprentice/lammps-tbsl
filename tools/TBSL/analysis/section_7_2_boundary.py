import math
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


from ._config import RESULTS_DIR, data_path, align_steps

FIG_DIR = RESULTS_DIR

from .utils.parse_data import parse_ave_chunk_output, parse_global_output
from .utils.plot_data import _auto_vmax_row_weighted_percentile, _auto_vmin_smallest_positive


PINFTY = 101325.0
INITIAL_RADIUS_M = 4.5e-6
KB = 1.380649e-23
TINF = 300.0
TOTAL_PHYSICAL_PARTICLES = PINFTY * (4.0 / 3.0) * math.pi * INITIAL_RADIUS_M**3 / (KB * TINF)
SCALE_FACTOR_1E6 = TOTAL_PHYSICAL_PARTICLES / 1.0e6
FS_TO_NS = 1.0e-6
ANGSTROM_TO_UM = 1.0e-4
K_TO_KILO_K = 1.0e-3

GLOBAL_X_MIN_NS = 4.0
GLOBAL_X_MAX_NS = 5.0
HEATMAP_REL_X_MIN_NS = -0.8
HEATMAP_REL_X_MAX_NS = 0.5
HEATMAP_Y_MAX_UM = 1.2
HEATMAP_VMAX_Q = 0.999
ALPHA_SWEEP_COLORS = {
    "0.0": "#85C1E9",
    "0.5": "#2980B9",
    "1.0": "#1A5276",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


@dataclass
class BoundaryCase:
    name: str
    color: str
    global_file: str
    distribution_file: str
    heatmap_filename: str


def compute_shell_edges(radii: np.ndarray) -> np.ndarray:
    edges = np.empty(len(radii) + 1, dtype=float)
    edges[0] = 0.0
    edges[1:-1] = 0.5 * (radii[:-1] + radii[1:])
    edges[-1] = radii[-1] + 0.5 * (radii[-1] - radii[-2])
    return edges


def compute_time_edges(times: np.ndarray) -> np.ndarray:
    if times.size == 1:
        dt = 1.0
        return np.array([times[0] - 0.5 * dt, times[0] + 0.5 * dt], dtype=float)

    mids = 0.5 * (times[:-1] + times[1:])
    left = times[0] - (mids[0] - times[0])
    right = times[-1] + (times[-1] - mids[-1])
    return np.concatenate(([left], mids, [right]))


def save_current_figure(filename: str) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), bbox_inches="tight")
    plt.close()


def load_case(case: BoundaryCase) -> dict[str, np.ndarray]:
    global_data, _ = parse_global_output(case.global_file)
    dist_steps, radii, temp_values, _ = parse_ave_chunk_output(case.distribution_file, "temp")
    _, _, ncount_values, _ = parse_ave_chunk_output(case.distribution_file, "Ncount")

    global_steps = np.asarray(global_data["Step"])
    global_time = np.asarray(global_data["Time"])
    step_indices = align_steps(global_steps, dist_steps)

    dist_time = global_time[step_indices]
    radius = np.asarray(global_data["v_SLradius"])
    tmin_index = int(np.argmin(radius))
    tmin_fs = global_time[tmin_index]

    return {
        "global_time_ns": global_time * FS_TO_NS,
        "dist_time_ns": dist_time * FS_TO_NS,
        "relative_time_ns": (dist_time - tmin_fs) * FS_TO_NS,
        "avg_temperature_k": np.asarray(global_data["Temp"]) / SCALE_FACTOR_1E6,
        "center_temperature_k": np.asarray(global_data["c_coretemp"]) / SCALE_FACTOR_1E6,
        "wall_temperature_k": np.asarray(global_data["v_SLTwall"]),
        "radius_um": radius * ANGSTROM_TO_UM,
        "mapped_radius_um": radius[step_indices] * ANGSTROM_TO_UM,
        "radii_um": radii * ANGSTROM_TO_UM,
        "temp_k": temp_values / SCALE_FACTOR_1E6,
        "ncount": ncount_values,
        "tmin_fs": tmin_fs,
        "tmin_ns": tmin_fs * FS_TO_NS,
    }


def plot_global_timeseries(cases: list[BoundaryCase], loaded: dict[str, dict[str, np.ndarray]], key: str, ylabel: str, filename: str) -> None:
    plt.figure(figsize=(4.4, 3.4))
    for case in cases:
        data = loaded[case.name]
        time = data["global_time_ns"]
        mask = (time >= GLOBAL_X_MIN_NS) & (time <= GLOBAL_X_MAX_NS)
        plt.plot(time[mask], data[key][mask] * K_TO_KILO_K, color=case.color, linewidth=2.0, label=case.name)
    plt.xlim(GLOBAL_X_MIN_NS, GLOBAL_X_MAX_NS)
    plt.xlabel("Time (ns)")
    plt.ylabel(ylabel)
    if filename == "boundary_avgT.pdf":
        plt.legend(frameon=False, loc="upper right")
    save_current_figure(filename)


def plot_boundary_heatmap(case: BoundaryCase, data: dict[str, np.ndarray]) -> None:
    time_mask = (data["relative_time_ns"] >= HEATMAP_REL_X_MIN_NS) & (data["relative_time_ns"] <= HEATMAP_REL_X_MAX_NS)
    times = data["relative_time_ns"][time_mask]
    mapped_radius = data["mapped_radius_um"][time_mask]
    temp = data["temp_k"][time_mask] * K_TO_KILO_K
    ncount = data["ncount"][time_mask]

    shell_edges = compute_shell_edges(data["radii_um"])
    time_edges = compute_time_edges(times)
    x_edges = np.repeat(time_edges[:, None], shell_edges.size, axis=1)
    y_edges = np.repeat(shell_edges[None, :], time_edges.size, axis=0)

    vmin = _auto_vmin_smallest_positive(temp)
    vmax = _auto_vmax_row_weighted_percentile(temp, ncount, q=HEATMAP_VMAX_Q, ignore_zeros=True)
    if vmin is None or not np.isfinite(vmin):
        raise ValueError(f"No positive heatmap temperatures available for {case.name}")
    if vmax is None or not np.isfinite(vmax):
        raise ValueError(f"No finite heatmap vmax could be computed for {case.name}")

    plt.figure(figsize=(4.6, 3.6))
    mesh = plt.pcolormesh(
        x_edges,
        y_edges,
        temp,
        cmap="hot",
        shading="auto",
        vmin=float(vmin),
        vmax=float(vmax),
        rasterized=True,
    )

    y_centers = 0.5 * (shell_edges[:-1] + shell_edges[1:])
    void_mask = y_centers[None, :] > mapped_radius[:, None]
    void_overlay = np.ma.masked_where(~void_mask, np.ones_like(void_mask, dtype=float))
    void_cmap = mcolors.ListedColormap(["lightgray"])
    plt.pcolormesh(
        x_edges,
        y_edges,
        void_overlay,
        cmap=void_cmap,
        shading="auto",
        vmin=0,
        vmax=1,
        rasterized=True,
    )
    plt.plot(times, mapped_radius, color="black", linewidth=2.0, linestyle="-")
    plt.colorbar(mesh, label=r"Temperature ($10^3$ K)")
    plt.xlim(HEATMAP_REL_X_MIN_NS, HEATMAP_REL_X_MAX_NS)
    plt.ylim(0.0, HEATMAP_Y_MAX_UM)
    plt.xlabel(r"Time relative to $t_{\min}$ (ns)")
    plt.ylabel(r"$R_b$ ($\mu$m)")
    plt.grid(False)
    save_current_figure(case.heatmap_filename)


def print_summary(cases: list[BoundaryCase], loaded: dict[str, dict[str, np.ndarray]]) -> None:
    print("Section 7.2 boundary-condition summary")
    for case in cases:
        data = loaded[case.name]
        print(
            f"{case.name}: "
            f"t_min={data['tmin_ns']:.4f} ns, "
            f"T_avg,max={np.max(data['avg_temperature_k']):.1f} K, "
            f"T_center,max={np.max(data['center_temperature_k']):.1f} K, "
            f"T_wall,max={np.max(data['wall_temperature_k']):.1f} K"
        )


def main() -> None:
    cases = [
        BoundaryCase(
            name=r"$\alpha_t = 0.0$",
            color=ALPHA_SWEEP_COLORS["0.0"],
            global_file=data_path("1e6", "boundary", "run_1e6_01_08_alpha_0.lammps"),
            distribution_file=data_path("1e6", "boundary", "1e6_comdis_01_08_alpha_0_minN100_smooth.out"),
            heatmap_filename="boundary_heatmap_alpha0.pdf",
        ),
        BoundaryCase(
            name=r"$\alpha_t = 0.5$",
            color=ALPHA_SWEEP_COLORS["0.5"],
            global_file=data_path("1e6", "boundary", "run_1e6_01_08_alpha_0_5.lammps"),
            distribution_file=data_path("1e6", "boundary", "1e6_comdis_01_08_alpha_0_5_minN100_smooth.out"),
            heatmap_filename="boundary_heatmap_alpha05.pdf",
        ),
        BoundaryCase(
            name=r"$\alpha_t = 1.0$",
            color=ALPHA_SWEEP_COLORS["1.0"],
            global_file=data_path("1e6", "boundary", "run_1e6_01_08_alpha_1.lammps"),
            distribution_file=data_path("1e6", "boundary", "1e6_comdis_01_08_alpha_1_minN100_smooth.out"),
            heatmap_filename="boundary_heatmap_alpha1.pdf",
        ),
    ]

    loaded = {case.name: load_case(case) for case in cases}

    plot_global_timeseries(cases, loaded, "avg_temperature_k", r"$\langle T \rangle$ ($10^3$ K)", "boundary_avgT.pdf")
    plot_global_timeseries(cases, loaded, "center_temperature_k", r"$T_{\mathrm{center}}$ ($10^3$ K)", "boundary_Tcenter.pdf")
    plot_global_timeseries(cases, loaded, "wall_temperature_k", r"$T_{\mathrm{wall}}$ ($10^3$ K)", "boundary_Twall.pdf")

    for case in cases:
        plot_boundary_heatmap(case, loaded[case.name])

    print_summary(cases, loaded)


if __name__ == "__main__":
    main()
