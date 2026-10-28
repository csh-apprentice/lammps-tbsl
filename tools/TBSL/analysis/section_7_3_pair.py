import math
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


from ._config import RESULTS_DIR, data_path, align_steps

FIG_DIR = RESULTS_DIR

from .utils.parse_data import parse_ave_chunk_output, parse_global_output
from .utils.plot_data import _auto_vmax_row_weighted_percentile, _auto_vmin_smallest_positive
from .utils.sl_saha_emission import band_limited_emission_from_P_closed_form, compute_emission_from_lammps_number


PINFTY = 101325.0
INITIAL_RADIUS_M = 4.5e-6
KB = 1.380649e-23
TINF = 300.0
TOTAL_PHYSICAL_PARTICLES = PINFTY * (4.0 / 3.0) * math.pi * INITIAL_RADIUS_M**3 / (KB * TINF)
ENSEMBLE_PARTICLES = 1.0e6
SCALE_FACTOR_1E6 = TOTAL_PHYSICAL_PARTICLES / ENSEMBLE_PARTICLES

FS_TO_NS = 1.0e-6
ANGSTROM_TO_UM = 1.0e-4
K_TO_KILO_K = 1.0e-3
EMISSION_TO_MW_PER_UM3 = 1.0e-15

CHARGE_X_MIN_NS = 4.20
CHARGE_X_MAX_NS = 4.65
HEATMAP_X_MIN_NS = 4.0
HEATMAP_X_MAX_NS = 5.0
HEATMAP_Y_MAX_UM = 1.2
HEATMAP_VMAX_Q = 0.999
IONIZATION_ENERGY_EV = 15.76
SAHA_G_RATIO = 0.5
EMISSION_PREFACTOR = 1.57e-40

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
class PairCase:
    label: str
    legend_label: str
    color: str
    linestyle: str
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


def compute_shell_volumes(radii: np.ndarray) -> np.ndarray:
    edges = compute_shell_edges(radii)
    return (4.0 / 3.0) * math.pi * (edges[1:] ** 3 - edges[:-1] ** 3)


def save_current_figure(filename: str) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), bbox_inches="tight")
    plt.close()


def load_case(case: PairCase) -> dict[str, np.ndarray]:
    global_data, _ = parse_global_output(case.global_file)
    dist_steps, radii, temp_values, _ = parse_ave_chunk_output(case.distribution_file, "temp")
    _, _, ncount_values, _ = parse_ave_chunk_output(case.distribution_file, "Ncount")
    _, _, charge_values, _ = parse_ave_chunk_output(case.distribution_file, "v_slcharge")
    _, _, density_number_values, _ = parse_ave_chunk_output(case.distribution_file, "density/number")

    global_steps = np.asarray(global_data["Step"])
    global_time_fs = np.asarray(global_data["Time"])
    step_indices = align_steps(global_steps, dist_steps)

    dist_time_fs = global_time_fs[step_indices]
    charge_sum = np.sum(charge_values * ncount_values, axis=1)

    p_vals, _, _, _ = compute_emission_from_lammps_number(
        temp_vals=temp_values / SCALE_FACTOR_1E6,
        number_density_lammps=density_number_values * SCALE_FACTOR_1E6,
        radii=radii,
        ionization_energy_eV=IONIZATION_ENERGY_EV,
        prefactor=EMISSION_PREFACTOR,
        g_ratio=SAHA_G_RATIO,
    )
    p_band, _ = band_limited_emission_from_P_closed_form(
        p_vals,
        temp_values / SCALE_FACTOR_1E6,
        lambda_c_nm=200.0,
    )
    total_emission_mw = np.sum(p_band * compute_shell_volumes(radii) * 1.0e-30, axis=1) * 1.0e3

    return {
        "global_time_ns": global_time_fs * FS_TO_NS,
        "dist_time_ns": dist_time_fs * FS_TO_NS,
        "avg_temperature_k": np.asarray(global_data["Temp"]) / SCALE_FACTOR_1E6,
        "center_temperature_k": np.asarray(global_data["c_coretemp"]) / SCALE_FACTOR_1E6,
        "radius_um": np.asarray(global_data["v_SLradius"]) * ANGSTROM_TO_UM,
        "mapped_radius_um": np.asarray(global_data["v_SLradius"])[step_indices] * ANGSTROM_TO_UM,
        "radii_um": radii * ANGSTROM_TO_UM,
        "ncount": ncount_values,
        "charge_sum": charge_sum,
        "charge_percent": charge_sum * 100.0 / ENSEMBLE_PARTICLES,
        "emission_mw_um3": p_band * EMISSION_TO_MW_PER_UM3,
        "total_emission_mw": total_emission_mw,
    }


def plot_temperature_series(
    cases: list[PairCase],
    loaded: dict[str, dict[str, np.ndarray]],
    key: str,
    ylabel: str,
    filename: str,
) -> None:
    plt.figure(figsize=(4.4, 3.4))
    for case in cases:
        data = loaded[case.label]
        plt.plot(
            data["global_time_ns"],
            data[key] * K_TO_KILO_K,
            color=case.color,
            linestyle=case.linestyle,
            linewidth=2.0,
            label=case.legend_label,
        )
    plt.xlabel("Time (ns)")
    plt.ylabel(ylabel)
    if filename == "pair_avgT.pdf":
        plt.legend(frameon=False, loc="upper right")
    save_current_figure(filename)


def plot_charge_series(cases: list[PairCase], loaded: dict[str, dict[str, np.ndarray]]) -> None:
    plt.figure(figsize=(4.4, 3.4))
    for case in cases:
        data = loaded[case.label]
        mask = (data["dist_time_ns"] >= CHARGE_X_MIN_NS) & (data["dist_time_ns"] <= CHARGE_X_MAX_NS)
        plt.plot(
            data["dist_time_ns"][mask],
            data["charge_percent"][mask],
            color=case.color,
            linestyle=case.linestyle,
            linewidth=2.0,
            label=case.legend_label,
        )
    plt.xlim(CHARGE_X_MIN_NS, CHARGE_X_MAX_NS)
    plt.xlabel("Time (ns)")
    plt.ylabel("Total ionization (%)")
    save_current_figure("pair_chargesum.pdf")


def compute_emission_limits(case: PairCase, data: dict[str, np.ndarray]) -> tuple[float, float]:
    mask = (data["dist_time_ns"] >= HEATMAP_X_MIN_NS) & (data["dist_time_ns"] <= HEATMAP_X_MAX_NS)
    emission = data["emission_mw_um3"][mask]
    ncount = data["ncount"][mask]
    vmax = _auto_vmax_row_weighted_percentile(emission, ncount, q=HEATMAP_VMAX_Q, ignore_zeros=True)
    if vmax is None or not np.isfinite(vmax):
        raise ValueError(f"No finite robust emission vmax available for {case.label}")
    return 0.0, float(vmax)


def plot_emission_heatmap(case: PairCase, data: dict[str, np.ndarray], vmin: float, vmax: float) -> None:
    mask = (data["dist_time_ns"] >= HEATMAP_X_MIN_NS) & (data["dist_time_ns"] <= HEATMAP_X_MAX_NS)
    times = data["dist_time_ns"][mask]
    shell_edges = compute_shell_edges(data["radii_um"])
    mapped_radius = data["mapped_radius_um"][mask]
    emission = data["emission_mw_um3"][mask]

    time_edges = compute_time_edges(times)
    x_edges = np.repeat(time_edges[:, None], shell_edges.size, axis=1)
    y_edges = np.repeat(shell_edges[None, :], time_edges.size, axis=0)

    plt.figure(figsize=(4.6, 3.6))
    mesh = plt.pcolormesh(
        x_edges,
        y_edges,
        emission,
        cmap="hot",
        shading="auto",
        vmin=vmin,
        vmax=vmax,
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
    plt.plot(times, mapped_radius, color="black", linewidth=2.0)
    plt.colorbar(mesh, label=r"Emission (mW/$\mu$m$^3$)")
    plt.xlim(HEATMAP_X_MIN_NS, HEATMAP_X_MAX_NS)
    plt.ylim(0.0, HEATMAP_Y_MAX_UM)
    plt.xlabel("Time (ns)")
    plt.ylabel(r"Radius ($\mu$m)")
    plt.grid(False)
    save_current_figure(case.heatmap_filename)


def print_summary(cases: list[PairCase], loaded: dict[str, dict[str, np.ndarray]]) -> None:
    print("Section 7.3 pair-interaction summary")
    for case in cases:
        data = loaded[case.label]
        avg_idx = int(np.argmax(data["avg_temperature_k"]))
        center_idx = int(np.argmax(data["center_temperature_k"]))
        charge_idx = int(np.argmax(data["charge_percent"]))
        emission_idx = int(np.argmax(data["total_emission_mw"]))
        vmin, vmax = compute_emission_limits(case, data)
        print(
            f"{case.label}: "
            f"T_avg,max={data['avg_temperature_k'][avg_idx]:.1f} K at {data['global_time_ns'][avg_idx]:.4f} ns, "
            f"T_center,max={data['center_temperature_k'][center_idx]:.1f} K at {data['global_time_ns'][center_idx]:.4f} ns, "
            f"charge,max={data['charge_percent'][charge_idx]:.4f} % at {data['dist_time_ns'][charge_idx]:.4f} ns, "
            f"emission,max={data['total_emission_mw'][emission_idx]:.6e} mW at {data['dist_time_ns'][emission_idx]:.4f} ns, "
            f"heatmap_display_range={vmin:.6e} to {vmax:.6e} mW/um^3"
        )


def main() -> None:
    cases = [
        PairCase(
            label="No ionization",
            legend_label="None",
            color="#1f77b4",
            linestyle=":",
            global_file=data_path("1e6", "pair", "run_1e6_01_10_alpha_1_noio.lammps"),
            distribution_file=data_path("1e6", "pair", "1e6_comdis_01_08_alpha_1_noio_minN100_smooth.out"),
            heatmap_filename="pair_emission_noio.pdf",
        ),
        PairCase(
            label="Short-range ionization",
            legend_label="Short-range",
            color="#1f77b4",
            linestyle="--",
            global_file=data_path("1e6", "pair", "run_1e6_01_13_alpha_1_shortio.lammps"),
            distribution_file=data_path("1e6", "pair", "1e6_comdis_01_13_alpha_1_shortio_minN100_smooth.out"),
            heatmap_filename="pair_emission_shortio.pdf",
        ),
        PairCase(
            label="Full-range ionization",
            legend_label="Full-range",
            color="#1f77b4",
            linestyle="-",
            global_file=data_path("1e6", "boundary", "run_1e6_01_08_alpha_1.lammps"),
            distribution_file=data_path("1e6", "boundary", "1e6_comdis_01_08_alpha_1_minN100_smooth.out"),
            heatmap_filename="pair_emission_fullio.pdf",
        ),
    ]

    loaded = {case.label: load_case(case) for case in cases}

    plot_temperature_series(cases, loaded, "avg_temperature_k", r"$\langle T \rangle$ ($10^3$ K)", "pair_avgT.pdf")
    plot_temperature_series(cases, loaded, "center_temperature_k", r"$T_{\mathrm{center}}$ ($10^3$ K)", "pair_Tcenter.pdf")
    plot_charge_series(cases, loaded)

    for case in cases:
        vmin, vmax = compute_emission_limits(case, loaded[case.label])
        plot_emission_heatmap(case, loaded[case.label], vmin=vmin, vmax=vmax)

    print_summary(cases, loaded)


if __name__ == "__main__":
    main()
