import math
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np


from ._config import APPENDIX_DIR as CONFIG_APPENDIX_DIR, data_path, align_steps

APPENDIX_DIR = CONFIG_APPENDIX_DIR

from .utils.parse_data import compute_pressure_from_virial_density, parse_ave_chunk_output, parse_global_output
from .utils.plot_data import _auto_vmax_row_weighted_percentile, _auto_vmin_smallest_positive
from .section_7_4_table import TableCase, compute_case_metrics
from .utils.sl_saha_emission import band_limited_emission_from_P_closed_form, compute_emission_from_lammps_number


PINFTY = 101325.0
INITIAL_RADIUS_M = 4.5e-6
KB = 1.380649e-23
TINF = 300.0
TOTAL_PHYSICAL_PARTICLES = PINFTY * (4.0 / 3.0) * math.pi * INITIAL_RADIUS_M**3 / (KB * TINF)

FS_TO_NS = 1.0e-6
ANGSTROM_TO_UM = 1.0e-4
K_TO_KILO_K = 1.0e-3
ATM_TO_GPA = 101325.0 / 1.0e9
EMISSION_TO_MW_PER_UM3 = 1.0e-15
RADIAL_TRIM_BINS = 2

TEMP_HEATMAP_REL_X_MIN_NS = -0.8
TEMP_HEATMAP_REL_X_MAX_NS = 0.5
EMISSION_HEATMAP_X_MIN_NS = 4.0
EMISSION_HEATMAP_X_MAX_NS = 5.0
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


@dataclass(frozen=True)
class AppendixCase:
    label: str
    panel_label: str
    color: str
    ensemble_particles: float
    global_file: str
    distribution_file: str
    linestyle: str = "-"

    @property
    def scale_factor(self) -> float:
        return TOTAL_PHYSICAL_PARTICLES / self.ensemble_particles


def nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


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


def trim_stop_index(radii: np.ndarray, bubble_radius: float) -> int:
    stop = np.searchsorted(radii, bubble_radius, side="right")
    return max(stop - RADIAL_TRIM_BINS, 1)


def save_current_figure(filename: str) -> None:
    os.makedirs(APPENDIX_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(APPENDIX_DIR, filename), bbox_inches="tight")
    plt.close()


def annotate_panel_label(text: str) -> None:
    plt.text(
        0.03,
        0.97,
        text,
        transform=plt.gca().transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 2.0},
    )


def load_case(case: AppendixCase) -> dict[str, np.ndarray]:
    global_data, _ = parse_global_output(case.global_file)
    dist_steps, radii, temp_values, _ = parse_ave_chunk_output(case.distribution_file, "temp")
    _, _, ncount_values, _ = parse_ave_chunk_output(case.distribution_file, "Ncount")
    _, _, density_mass_values, _ = parse_ave_chunk_output(case.distribution_file, "density/mass")
    _, _, density_number_values, _ = parse_ave_chunk_output(case.distribution_file, "density/number")
    _, _, virial_values, _ = parse_ave_chunk_output(case.distribution_file, "v_slvirial")

    global_steps = np.asarray(global_data["Step"], dtype=int)
    global_time_fs = np.asarray(global_data["Time"], dtype=float)
    step_indices = align_steps(global_steps, dist_steps)

    dist_time_fs = global_time_fs[step_indices]
    radius_a = np.asarray(global_data["v_SLradius"], dtype=float)
    tmin_global_idx = int(np.argmin(radius_a))
    tmin_dist_idx = nearest_index(dist_time_fs, global_time_fs[tmin_global_idx])

    center_temperature_k = np.asarray(global_data["c_coretemp"], dtype=float) / case.scale_factor
    tcenter_max_global_idx = int(np.argmax(center_temperature_k))
    tcenter_max_dist_idx = nearest_index(dist_time_fs, global_time_fs[tcenter_max_global_idx])

    pressure_gpa = compute_pressure_from_virial_density(virial_values, density_number_values, 1.0) * ATM_TO_GPA

    rb_tcenter = radius_a[step_indices][tcenter_max_dist_idx]
    rb_tmin = radius_a[step_indices][tmin_dist_idx]
    temp_stop = trim_stop_index(radii, rb_tcenter)
    rmin_stop = trim_stop_index(radii, rb_tmin)

    return {
        "dist_time_ns": dist_time_fs * FS_TO_NS,
        "relative_time_ns": (dist_time_fs - global_time_fs[tmin_global_idx]) * FS_TO_NS,
        "mapped_radius_um": radius_a[step_indices] * ANGSTROM_TO_UM,
        "radii_um": radii * ANGSTROM_TO_UM,
        "ncount": ncount_values,
        "temp_k": temp_values / case.scale_factor,
        "density_1e3kgm3": density_mass_values,
        "pressure_gpa": pressure_gpa,
        "density_number_scaled": density_number_values * case.scale_factor,
        "tcenter_temp_frac_r": radii[:temp_stop] / rb_tcenter,
        "tcenter_temp_kilo_k": temp_values[tcenter_max_dist_idx, :temp_stop] / case.scale_factor * K_TO_KILO_K,
        "rmin_frac_r": radii[:rmin_stop] / rb_tmin,
        "rmin_pressure_gpa": pressure_gpa[tmin_dist_idx, :rmin_stop],
        "rmin_density_1e3kgm3": density_mass_values[tmin_dist_idx, :rmin_stop],
    }


def compute_emission_field(case: AppendixCase, data: dict[str, np.ndarray]) -> np.ndarray:
    p_vals, _, _, _ = compute_emission_from_lammps_number(
        temp_vals=data["temp_k"],
        number_density_lammps=data["density_number_scaled"],
        radii=data["radii_um"] / ANGSTROM_TO_UM,
        ionization_energy_eV=IONIZATION_ENERGY_EV,
        prefactor=EMISSION_PREFACTOR,
        g_ratio=SAHA_G_RATIO,
    )
    p_band, _ = band_limited_emission_from_P_closed_form(
        p_vals,
        data["temp_k"],
        lambda_c_nm=200.0,
    )
    return p_band * EMISSION_TO_MW_PER_UM3


def plot_heatmap(
    quantity: np.ndarray,
    data: dict[str, np.ndarray],
    filename: str,
    colorbar_label: str,
    panel_label: str,
    time_key: str,
    x_min: float,
    x_max: float,
    xlabel: str,
    use_zero_vmin: bool = False,
) -> None:
    time_vals = data[time_key]
    time_mask = (time_vals >= x_min) & (time_vals <= x_max)
    times = time_vals[time_mask]
    mapped_radius = data["mapped_radius_um"][time_mask]
    field = quantity[time_mask]
    ncount = data["ncount"][time_mask]

    shell_edges = compute_shell_edges(data["radii_um"])
    time_edges = compute_time_edges(times)
    x_edges = np.repeat(time_edges[:, None], shell_edges.size, axis=1)
    y_edges = np.repeat(shell_edges[None, :], time_edges.size, axis=0)

    if use_zero_vmin:
        vmin = 0.0
    else:
        vmin = _auto_vmin_smallest_positive(field)
        if vmin is None or not np.isfinite(vmin):
            raise ValueError(f"No positive values available for heatmap {filename}")

    vmax = _auto_vmax_row_weighted_percentile(field, ncount, q=HEATMAP_VMAX_Q, ignore_zeros=True)
    if vmax is None or not np.isfinite(vmax):
        raise ValueError(f"No finite robust vmax available for heatmap {filename}")

    plt.figure(figsize=(4.6, 3.6))
    mesh = plt.pcolormesh(
        x_edges,
        y_edges,
        field,
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
    plt.plot(times, mapped_radius, color="black", linewidth=2.0)
    plt.colorbar(mesh, label=colorbar_label)
    plt.xlim(x_min, x_max)
    plt.ylim(0.0, HEATMAP_Y_MAX_UM)
    plt.xlabel(xlabel)
    plt.ylabel(r"$R_b$ ($\mu$m)")
    annotate_panel_label(panel_label)
    plt.grid(False)
    save_current_figure(filename)


def plot_radial_sweep(
    cases: list[AppendixCase],
    loaded: dict[str, dict[str, np.ndarray]],
    x_key: str,
    y_key: str,
    ylabel: str,
    filename: str,
    show_legend: bool = False,
) -> None:
    plt.figure(figsize=(4.4, 3.4))
    for case in cases:
        data = loaded[case.label]
        plt.plot(data[x_key], data[y_key], color=case.color, linestyle=case.linestyle, linewidth=2.0, label=case.label)
    plt.xlim(0.0, 1.02)
    plt.xlabel(r"$r / R_b$")
    plt.ylabel(ylabel)
    if show_legend:
        plt.legend(frameon=False)
    save_current_figure(filename)


def build_appendix_a() -> None:
    appendix_a1_cases = [
        AppendixCase(
            label=r"$N_{\mathrm{ensem}} = 10^6$",
            panel_label=r"$N_{\mathrm{ensem}} = 10^6$",
            color="#1f77b4",
            ensemble_particles=1.0e6,
            global_file=data_path("1e6", "boundary", "run_1e6_01_08_alpha_1.lammps"),
            distribution_file=data_path("1e6", "boundary", "1e6_comdis_01_08_alpha_1_minN100_smooth.out"),
        ),
        AppendixCase(
            label=r"$N_{\mathrm{ensem}} = 10^7$",
            panel_label=r"$N_{\mathrm{ensem}} = 10^7$",
            color="#ff7f0e",
            ensemble_particles=1.0e7,
            global_file=data_path("1e7", "run_1e7_01_07.lammps"),
            distribution_file=data_path("1e7", "1e7_comdis_01_07_minN100_smooth.out"),
        ),
        AppendixCase(
            label=r"$N_{\mathrm{ensem}} = 10^8$",
            panel_label=r"$N_{\mathrm{ensem}} = 10^8$",
            color="#2ca02c",
            ensemble_particles=1.0e8,
            global_file=data_path("1e8", "run_1e8_full.lammps"),
            distribution_file=data_path("1e8", "1e8_comdis_full_minN100_smooth.out"),
        ),
    ]

    a1_filenames = [
        "A1_heatmap_T_N1e6.pdf",
        "A1_heatmap_T_N1e7.pdf",
        "A1_heatmap_T_N1e8.pdf",
    ]

    loaded_a1 = {case.label: load_case(case) for case in appendix_a1_cases}
    for case, filename in zip(appendix_a1_cases, a1_filenames):
        plot_heatmap(
            quantity=loaded_a1[case.label]["temp_k"] * K_TO_KILO_K,
            data=loaded_a1[case.label],
            filename=filename,
            colorbar_label=r"Temperature ($10^3$ K)",
            panel_label=case.panel_label,
            time_key="relative_time_ns",
            x_min=TEMP_HEATMAP_REL_X_MIN_NS,
            x_max=TEMP_HEATMAP_REL_X_MAX_NS,
            xlabel=r"Time relative to $t_{\min}$ (ns)",
        )

    appendix_a2_cases = [
        AppendixCase(
            label="No ionization",
            panel_label="No ionization",
            color="#d62728",
            ensemble_particles=1.0e6,
            global_file=data_path("1e6", "pair", "run_1e6_01_10_alpha_1_noio.lammps"),
            distribution_file=data_path("1e6", "pair", "1e6_comdis_01_08_alpha_1_noio_minN100_smooth.out"),
        ),
        AppendixCase(
            label="Short-range ionization",
            panel_label="Short-range ionization",
            color="#ff7f0e",
            ensemble_particles=1.0e6,
            global_file=data_path("1e6", "pair", "run_1e6_01_13_alpha_1_shortio.lammps"),
            distribution_file=data_path("1e6", "pair", "1e6_comdis_01_13_alpha_1_shortio_minN100_smooth.out"),
        ),
        AppendixCase(
            label="Full-range ionization",
            panel_label="Full-range ionization + DSF",
            color="#1f77b4",
            ensemble_particles=1.0e6,
            global_file=data_path("1e6", "boundary", "run_1e6_01_08_alpha_1.lammps"),
            distribution_file=data_path("1e6", "boundary", "1e6_comdis_01_08_alpha_1_minN100_smooth.out"),
        ),
    ]

    a2_filenames = [
        "A2_heatmap_T_noio.pdf",
        "A2_heatmap_T_shortio.pdf",
        "A2_heatmap_T_fullio.pdf",
    ]
    loaded_a2 = {case.label: load_case(case) for case in appendix_a2_cases}
    for case, filename in zip(appendix_a2_cases, a2_filenames):
        plot_heatmap(
            quantity=loaded_a2[case.label]["temp_k"] * K_TO_KILO_K,
            data=loaded_a2[case.label],
            filename=filename,
            colorbar_label=r"Temperature ($10^3$ K)",
            panel_label=case.panel_label,
            time_key="relative_time_ns",
            x_min=TEMP_HEATMAP_REL_X_MIN_NS,
            x_max=TEMP_HEATMAP_REL_X_MAX_NS,
            xlabel=r"Time relative to $t_{\min}$ (ns)",
        )

    appendix_a3_cases = [
        AppendixCase(
            label=r"$\alpha_t = 0.0$",
            panel_label=r"$\alpha_t = 0.0$",
            color="#85C1E9",
            ensemble_particles=1.0e6,
            global_file=data_path("1e6", "boundary", "run_1e6_01_08_alpha_0.lammps"),
            distribution_file=data_path("1e6", "boundary", "1e6_comdis_01_08_alpha_0_minN100_smooth.out"),
        ),
        AppendixCase(
            label=r"$\alpha_t = 0.5$",
            panel_label=r"$\alpha_t = 0.5$",
            color="#2980B9",
            ensemble_particles=1.0e6,
            global_file=data_path("1e6", "boundary", "run_1e6_01_08_alpha_0_5.lammps"),
            distribution_file=data_path("1e6", "boundary", "1e6_comdis_01_08_alpha_0_5_minN100_smooth.out"),
        ),
        AppendixCase(
            label=r"$\alpha_t = 1.0$",
            panel_label=r"$\alpha_t = 1.0$",
            color="#1A5276",
            ensemble_particles=1.0e6,
            global_file=data_path("1e6", "boundary", "run_1e6_01_08_alpha_1.lammps"),
            distribution_file=data_path("1e6", "boundary", "1e6_comdis_01_08_alpha_1_minN100_smooth.out"),
        ),
    ]
    a3_filenames = [
        "A3_heatmap_emission_alpha0.pdf",
        "A3_heatmap_emission_alpha05.pdf",
        "A3_heatmap_emission_alpha1.pdf",
    ]
    loaded_a3 = {case.label: load_case(case) for case in appendix_a3_cases}
    for case, filename in zip(appendix_a3_cases, a3_filenames):
        emission_field = compute_emission_field(case, loaded_a3[case.label])
        plot_heatmap(
            quantity=emission_field,
            data=loaded_a3[case.label],
            filename=filename,
            colorbar_label=r"Emission (mW/$\mu$m$^3$)",
            panel_label=case.panel_label,
            time_key="dist_time_ns",
            x_min=EMISSION_HEATMAP_X_MIN_NS,
            x_max=EMISSION_HEATMAP_X_MAX_NS,
            xlabel="Time (ns)",
            use_zero_vmin=True,
        )


def build_appendix_b() -> None:
    appendix_b1_cases = [
        AppendixCase(
            label=r"$\alpha_t = 0.0$",
            panel_label=r"$\alpha_t = 0.0$",
            color="#85C1E9",
            ensemble_particles=1.0e6,
            global_file=data_path("1e6", "boundary", "run_1e6_01_08_alpha_0.lammps"),
            distribution_file=data_path("1e6", "boundary", "1e6_comdis_01_08_alpha_0_minN100_smooth.out"),
        ),
        AppendixCase(
            label=r"$\alpha_t = 0.5$",
            panel_label=r"$\alpha_t = 0.5$",
            color="#2980B9",
            ensemble_particles=1.0e6,
            global_file=data_path("1e6", "boundary", "run_1e6_01_08_alpha_0_5.lammps"),
            distribution_file=data_path("1e6", "boundary", "1e6_comdis_01_08_alpha_0_5_minN100_smooth.out"),
        ),
        AppendixCase(
            label=r"$\alpha_t = 1.0$",
            panel_label=r"$\alpha_t = 1.0$",
            color="#1A5276",
            ensemble_particles=1.0e6,
            global_file=data_path("1e6", "boundary", "run_1e6_01_08_alpha_1.lammps"),
            distribution_file=data_path("1e6", "boundary", "1e6_comdis_01_08_alpha_1_minN100_smooth.out"),
        ),
    ]
    loaded_b1 = {case.label: load_case(case) for case in appendix_b1_cases}
    plot_radial_sweep(appendix_b1_cases, loaded_b1, "tcenter_temp_frac_r", "tcenter_temp_kilo_k", r"$T$ ($10^3$ K)", "B1_radial_T_alpha_sweep.pdf", show_legend=True)
    plot_radial_sweep(appendix_b1_cases, loaded_b1, "rmin_frac_r", "rmin_pressure_gpa", r"$P$ (GPa)", "B1_radial_P_alpha_sweep.pdf")
    plot_radial_sweep(appendix_b1_cases, loaded_b1, "rmin_frac_r", "rmin_density_1e3kgm3", r"$\rho$ ($10^3$ kg m$^{-3}$)", "B1_radial_rho_alpha_sweep.pdf")

    appendix_b2_cases = [
        AppendixCase(
            label="None",
            panel_label="None",
            color="#1f77b4",
            ensemble_particles=1.0e6,
            global_file=data_path("1e6", "pair", "run_1e6_01_10_alpha_1_noio.lammps"),
            distribution_file=data_path("1e6", "pair", "1e6_comdis_01_08_alpha_1_noio_minN100_smooth.out"),
            linestyle=":",
        ),
        AppendixCase(
            label="Short-range",
            panel_label="Short-range",
            color="#1f77b4",
            ensemble_particles=1.0e6,
            global_file=data_path("1e6", "pair", "run_1e6_01_13_alpha_1_shortio.lammps"),
            distribution_file=data_path("1e6", "pair", "1e6_comdis_01_13_alpha_1_shortio_minN100_smooth.out"),
            linestyle="--",
        ),
        AppendixCase(
            label="Full-range",
            panel_label="Full-range",
            color="#1f77b4",
            ensemble_particles=1.0e6,
            global_file=data_path("1e6", "boundary", "run_1e6_01_08_alpha_1.lammps"),
            distribution_file=data_path("1e6", "boundary", "1e6_comdis_01_08_alpha_1_minN100_smooth.out"),
            linestyle="-",
        ),
    ]
    loaded_b2 = {case.label: load_case(case) for case in appendix_b2_cases}
    plot_radial_sweep(appendix_b2_cases, loaded_b2, "tcenter_temp_frac_r", "tcenter_temp_kilo_k", r"$T$ ($10^3$ K)", "B2_radial_T_pair_sweep.pdf", show_legend=True)
    plot_radial_sweep(appendix_b2_cases, loaded_b2, "rmin_frac_r", "rmin_pressure_gpa", r"$P$ (GPa)", "B2_radial_P_pair_sweep.pdf")
    plot_radial_sweep(appendix_b2_cases, loaded_b2, "rmin_frac_r", "rmin_density_1e3kgm3", r"$\rho$ ($10^3$ kg m$^{-3}$)", "B2_radial_rho_pair_sweep.pdf")

    appendix_b3_cases = [
        AppendixCase(
            label=r"$N_{\mathrm{ensem}} = 10^6$",
            panel_label=r"$N_{\mathrm{ensem}} = 10^6$",
            color="#1f77b4",
            ensemble_particles=1.0e6,
            global_file=data_path("1e6", "boundary", "run_1e6_01_08_alpha_1.lammps"),
            distribution_file=data_path("1e6", "boundary", "1e6_comdis_01_08_alpha_1_minN100_smooth.out"),
        ),
        AppendixCase(
            label=r"$N_{\mathrm{ensem}} = 10^7$",
            panel_label=r"$N_{\mathrm{ensem}} = 10^7$",
            color="#ff7f0e",
            ensemble_particles=1.0e7,
            global_file=data_path("1e7", "run_1e7_01_07.lammps"),
            distribution_file=data_path("1e7", "1e7_comdis_01_07_minN100_smooth.out"),
        ),
        AppendixCase(
            label=r"$N_{\mathrm{ensem}} = 10^8$",
            panel_label=r"$N_{\mathrm{ensem}} = 10^8$",
            color="#2ca02c",
            ensemble_particles=1.0e8,
            global_file=data_path("1e8", "run_1e8_full.lammps"),
            distribution_file=data_path("1e8", "1e8_comdis_full_minN100_smooth.out"),
        ),
    ]
    loaded_b3 = {case.label: load_case(case) for case in appendix_b3_cases}
    plot_radial_sweep(appendix_b3_cases, loaded_b3, "tcenter_temp_frac_r", "tcenter_temp_kilo_k", r"$T$ ($10^3$ K)", "B3_radial_T_n_sweep.pdf", show_legend=True)
    plot_radial_sweep(appendix_b3_cases, loaded_b3, "rmin_frac_r", "rmin_pressure_gpa", r"$P$ (GPa)", "B3_radial_P_n_sweep.pdf")
    plot_radial_sweep(appendix_b3_cases, loaded_b3, "rmin_frac_r", "rmin_density_1e3kgm3", r"$\rho$ ($10^3$ kg m$^{-3}$)", "B3_radial_rho_n_sweep.pdf")


def build_appendix_c() -> None:
    cases = [
        TableCase("1e6", 1.0e6, "0.0", "none", data_path("1e6", "pair", "run_1e6_01_10_alpha_0_noio.lammps"), data_path("1e6", "pair", "1e6_comdis_01_08_alpha_0_noio_minN100_smooth.out")),
        TableCase("1e6", 1.0e6, "0.0", "short", data_path("1e6", "pair", "run_1e6_01_13_alpha_0_shortio.lammps"), data_path("1e6", "pair", "1e6_comdis_01_13_alpha_0_shortio_minN100_smooth.out")),
        TableCase("1e6", 1.0e6, "0.0", "full", data_path("1e6", "boundary", "run_1e6_01_08_alpha_0.lammps"), data_path("1e6", "boundary", "1e6_comdis_01_08_alpha_0_minN100_smooth.out")),
        TableCase("1e6", 1.0e6, "0.5", "none", data_path("1e6", "pair", "run_1e6_01_10_alpha_0_5_noio.lammps"), data_path("1e6", "pair", "1e6_comdis_01_08_alpha_0_5_noio_minN100_smooth.out")),
        TableCase("1e6", 1.0e6, "0.5", "short", data_path("1e6", "pair", "run_1e6_01_13_alpha_0_5_shortio.lammps"), data_path("1e6", "pair", "1e6_comdis_01_13_alpha_0_5_shortio_minN100_smooth.out")),
        TableCase("1e6", 1.0e6, "0.5", "full", data_path("1e6", "boundary", "run_1e6_01_08_alpha_0_5.lammps"), data_path("1e6", "boundary", "1e6_comdis_01_08_alpha_0_5_minN100_smooth.out")),
        TableCase("1e6", 1.0e6, "1.0", "none", data_path("1e6", "pair", "run_1e6_01_10_alpha_1_noio.lammps"), data_path("1e6", "pair", "1e6_comdis_01_08_alpha_1_noio_minN100_smooth.out")),
        TableCase("1e6", 1.0e6, "1.0", "short", data_path("1e6", "pair", "run_1e6_01_13_alpha_1_shortio.lammps"), data_path("1e6", "pair", "1e6_comdis_01_13_alpha_1_shortio_minN100_smooth.out")),
        TableCase("1e6", 1.0e6, "1.0", "full", data_path("1e6", "boundary", "run_1e6_01_08_alpha_1.lammps"), data_path("1e6", "boundary", "1e6_comdis_01_08_alpha_1_minN100_smooth.out")),
        TableCase("1e7", 1.0e7, "1.0", "full", data_path("1e7", "run_1e7_01_07.lammps"), data_path("1e7", "1e7_comdis_01_07_minN100_smooth.out")),
        TableCase("1e8", 1.0e8, "1.0", "full", data_path("1e8", "run_1e8_full.lammps"), data_path("1e8", "1e8_comdis_full_minN100_smooth.out")),
    ]
    rows = [compute_case_metrics(case) for case in cases]
    row_lookup = {(row["nensem"], row["alpha_t"], row["pair"]): row for row in rows}
    ordered_rows = [
        row_lookup[("1e6", "0.0", "none")],
        row_lookup[("1e6", "0.0", "short")],
        row_lookup[("1e6", "0.0", "full")],
        row_lookup[("1e6", "0.5", "none")],
        row_lookup[("1e6", "0.5", "short")],
        row_lookup[("1e6", "0.5", "full")],
        row_lookup[("1e6", "1.0", "none")],
        row_lookup[("1e6", "1.0", "short")],
        row_lookup[("1e6", "1.0", "full")],
        row_lookup[("1e7", "1.0", "full")],
        row_lookup[("1e8", "1.0", "full")],
    ]

    x_positions = np.array([0.0, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 9.0, 10.0, 12.0, 13.0], dtype=float)
    x_labels = ["none", "short", "full", "none", "short", "full", "none", "short", "full", r"$10^7$", r"$10^8$"]
    alpha_colors = {"0.0": "#c7dff2", "0.5": "#6aaed6", "1.0": "#1d4f91"}
    nconv_colors = {"1e7": "#ff7f0e", "1e8": "#2ca02c"}
    hatch_map = {"none": "", "short": "////", "full": "xxxx"}
    top_label_positions = [1.0, 5.0, 9.0, 12.5]
    top_label_text = ["alpha=0.0", "alpha=0.5", r"alpha=1.0 ($10^6$)", "N conv"]
    panels = [
        ("C1a_Tmax_bar.pdf", "T_max_K", r"$T_{\max}$ ($10^3$ K)", 18.870, K_TO_KILO_K),
        ("C1b_Tavmax_bar.pdf", "T_av_max_K", r"$\langle T \rangle_{\max}$ ($10^3$ K)", 12.269, K_TO_KILO_K),
        ("C1c_fion_bar.pdf", "f_ion_percent", r"$f_{\mathrm{ion}}$ (%)", None, 1.0),
        ("C1d_Rmin_bar.pdf", "R_min_um", r"$R_{\min}$ ($\mu$m)", 0.750, 1.0),
        ("C1e_vWmax_bar.pdf", "v_W_max_m_per_s", r"$v_{W,\max}$ (m s$^{-1}$)", 1510.0, 1.0),
        ("C1f_pWmax_bar.pdf", "p_W_max_GPa", r"$p_{W,\max}$ (GPa)", 3.30, 1.0),
    ]

    for filename, key, ylabel, ref_val, y_scale in panels:
        plt.figure(figsize=(6.8, 3.8))
        ax = plt.gca()
        values = np.array([row[key] for row in ordered_rows], dtype=float) * y_scale

        for x, row, y in zip(x_positions, ordered_rows, values):
            facecolor = nconv_colors[row["nensem"]] if row["nensem"] != "1e6" else alpha_colors[row["alpha_t"]]
            ax.bar(
                x,
                y,
                width=0.82,
                color=facecolor,
                edgecolor="black",
                linewidth=0.9,
                hatch=hatch_map[row["pair"]],
                zorder=3,
            )

        if ref_val is not None:
            ax.axhline(ref_val, color="black", linestyle="--", linewidth=1.5, zorder=5)

        ax.set_xlim(-0.8, 13.8)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
        ax.grid(False, axis="x")
        text_transform = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        for x_text, label_text in zip(top_label_positions, top_label_text):
            ax.text(
                x_text,
                1.04,
                label_text,
                transform=text_transform,
                ha="center",
                va="bottom",
                fontsize=10,
                clip_on=False,
            )

        ymax = max(np.max(values), ref_val if ref_val is not None else 0.0)
        ax.set_ylim(0.0, ymax * 1.16 if ymax > 0 else 1.0)
        save_current_figure(filename)


def main() -> None:
    build_appendix_a()
    build_appendix_b()
    build_appendix_c()
    print(f"Wrote appendix figures to {APPENDIX_DIR}")


if __name__ == "__main__":
    main()
