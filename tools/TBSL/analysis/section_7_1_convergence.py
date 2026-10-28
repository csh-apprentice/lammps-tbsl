import math
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


from ._config import RESULTS_DIR, data_path, align_steps

FIG_DIR = RESULTS_DIR

from .utils.parse_data import compute_pressure_from_virial_density, parse_ave_chunk_output, parse_global_output

PINFTY = 101325.0
INITIAL_RADIUS_M = 4.5e-6
KB = 1.380649e-23
TINF = 300.0
TOTAL_PHYSICAL_PARTICLES = PINFTY * (4.0 / 3.0) * math.pi * INITIAL_RADIUS_M**3 / (KB * TINF)
RADIAL_TRIM_BINS = 2
FS_TO_NS = 1.0e-6
K_TO_KILO_K = 1.0e-3
ATM_TO_GPA = 101325.0 / 1.0e9
DENSITY_TO_1E3_KG_PER_M3 = 1.0e-3

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
class SweepCase:
    label: str
    ensemble_particles: float
    color: str
    global_file: str
    distribution_file: str

    @property
    def scale_factor(self) -> float:
        return TOTAL_PHYSICAL_PARTICLES / self.ensemble_particles


def compute_shell_edges(radii: np.ndarray) -> np.ndarray:
    edges = np.empty(len(radii) + 1, dtype=float)
    edges[0] = 0.0
    edges[1:-1] = 0.5 * (radii[:-1] + radii[1:])
    edges[-1] = radii[-1] + 0.5 * (radii[-1] - radii[-2])
    return edges


def compute_shell_volumes(radii: np.ndarray) -> np.ndarray:
    edges = compute_shell_edges(radii)
    return (4.0 / 3.0) * math.pi * (edges[1:] ** 3 - edges[:-1] ** 3)


def nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def save_current_figure(filename: str) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), bbox_inches="tight")
    plt.close()


def load_case(case: SweepCase) -> dict[str, np.ndarray]:
    global_data, _ = parse_global_output(case.global_file)
    dist_steps, radii, temp_values, _ = parse_ave_chunk_output(case.distribution_file, "temp")
    _, _, ncount_values, _ = parse_ave_chunk_output(case.distribution_file, "Ncount")
    _, _, density_mass_values, _ = parse_ave_chunk_output(case.distribution_file, "density/mass")
    _, _, density_number_values, _ = parse_ave_chunk_output(case.distribution_file, "density/number")
    _, _, virial_values, _ = parse_ave_chunk_output(case.distribution_file, "v_slvirial")

    global_steps = np.asarray(global_data["Step"])
    global_time = np.asarray(global_data["Time"])
    step_indices = align_steps(global_steps, dist_steps)

    dist_time = global_time[step_indices]
    pressure_values = compute_pressure_from_virial_density(virial_values, density_number_values, 1.0) * ATM_TO_GPA

    radius_values = np.asarray(global_data["v_SLradius"]) * 1.0e-4
    avg_temperature = np.asarray(global_data["Temp"]) / case.scale_factor
    avg_pressure = np.asarray(global_data["v_pressbubble"]) * ATM_TO_GPA
    center_temperature = np.asarray(global_data["c_coretemp"]) / case.scale_factor

    radius_min_index = int(np.argmin(np.asarray(global_data["v_SLradius"])))
    radius_min_dist_index = nearest_index(dist_time, global_time[radius_min_index])
    r_min = np.asarray(global_data["v_SLradius"])[step_indices[radius_min_dist_index]]
    radius_min_ridx = np.searchsorted(radii, r_min, side="right")

    temp_max_index = int(np.argmax(temp_values[:, 0]))
    r_temp = np.asarray(global_data["v_SLradius"])[step_indices[temp_max_index]]
    temp_max_ridx = np.searchsorted(radii, r_temp, side="right")

    tavg_max_index = int(np.argmax(avg_temperature))
    tavg_max_dist_index = nearest_index(dist_time, global_time[tavg_max_index])
    r_tavg = np.asarray(global_data["v_SLradius"])[step_indices[tavg_max_dist_index]]
    tavg_max_ridx = np.searchsorted(radii, r_tavg, side="right")

    tcenter_max_global_index = int(np.argmax(center_temperature))
    tcenter_target_time = global_time[tcenter_max_global_index]
    tcenter_max_index = nearest_index(dist_time, tcenter_target_time)
    r_tcenter = np.asarray(global_data["v_SLradius"])[step_indices[tcenter_max_index]]
    tcenter_max_ridx = np.searchsorted(radii, r_tcenter, side="right")

    temp_stop = max(temp_max_ridx - RADIAL_TRIM_BINS, 1)
    tavg_stop = max(tavg_max_ridx - RADIAL_TRIM_BINS, 1)
    tcenter_stop = max(tcenter_max_ridx - RADIAL_TRIM_BINS, 1)
    radius_stop = max(radius_min_ridx - RADIAL_TRIM_BINS, 1)

    return {
        "radius_time_ns": global_time * FS_TO_NS,
        "radius_um": radius_values,
        "dist_time_ns": dist_time * FS_TO_NS,
        "avg_temperature": avg_temperature,
        "center_temperature": center_temperature,
        "avg_pressure": avg_pressure,
        "temperature_frac_r": radii[:temp_stop] / r_temp,
        "temperature_frac_r_tavgmax": radii[:tavg_stop] / r_tavg,
        "temperature_frac_r_tcentermax": radii[:tcenter_stop] / r_tcenter,
        "temperature_frac_r_rmin": radii[:radius_stop] / r_min,
        "pressure_frac_r": radii[:radius_stop] / r_min,
        "density_frac_r": radii[:radius_stop] / r_min,
        "radial_temperature": temp_values[temp_max_index, :temp_stop] / case.scale_factor * K_TO_KILO_K,
        "radial_temperature_tavgmax": temp_values[tavg_max_dist_index, :tavg_stop] / case.scale_factor * K_TO_KILO_K,
        "radial_temperature_tcentermax": temp_values[tcenter_max_index, :tcenter_stop] / case.scale_factor * K_TO_KILO_K,
        "radial_temperature_rmin": temp_values[radius_min_dist_index, :radius_stop] / case.scale_factor * K_TO_KILO_K,
        "radial_pressure": pressure_values[radius_min_dist_index, :radius_stop],
        "radial_density": density_mass_values[radius_min_dist_index, :radius_stop],
        "tmin_fs": global_time[radius_min_index],
        "rmin_um": np.asarray(global_data["v_SLradius"])[radius_min_index] * 1.0e-4,
        "tmax_k": float(np.nanmax(temp_values / case.scale_factor)),
        "tavg_max_k": float(np.nanmax(avg_temperature)),
        "pavg_max_gpa": float(np.nanmax(avg_pressure)),
    }


def load_theory() -> dict[str, np.ndarray]:
    theory_data, _ = parse_global_output(data_path("theory", "theory.txt"))
    return {
        "time_ns": np.asarray(theory_data["Time"]) * FS_TO_NS,
        "radius_um": np.asarray(theory_data["Radius"]) * 1.0e6,
        "avg_temperature": np.asarray(theory_data["Temp_avg_uniform"]),
        "avg_pressure": np.asarray(theory_data["Press_avg"]) * ATM_TO_GPA,
    }


def plot_global_radius(cases: list[SweepCase], loaded: dict[str, dict[str, np.ndarray]], theory: dict[str, np.ndarray]) -> None:
    x_min = 3.75
    x_max = 5.25

    plt.figure(figsize=(4.4, 3.4))
    for case in cases:
        case_data = loaded[case.label]
        mask = (case_data["radius_time_ns"] >= x_min) & (case_data["radius_time_ns"] <= x_max)
        plt.plot(case_data["radius_time_ns"][mask], case_data["radius_um"][mask], color=case.color, linewidth=2.0, label=case.label)

    theory_mask = (theory["time_ns"] >= x_min) & (theory["time_ns"] <= x_max)
    plt.plot(theory["time_ns"][theory_mask], theory["radius_um"][theory_mask], color="black", linestyle="--", linewidth=2.0, label="KM theory")
    plt.xlim(x_min, x_max)
    plt.xlabel("Time (ns)")
    plt.ylabel(r"$R_b$ ($\mu$m)")
    plt.legend(frameon=False, loc="upper right")
    save_current_figure("scaling_global_radius.pdf")


def plot_global_scalar(
    cases: list[SweepCase],
    loaded: dict[str, dict[str, np.ndarray]],
    theory: dict[str, np.ndarray] | None,
    key: str,
    ylabel: str,
    filename: str,
    y_scale: float = 1.0,
    x_min: float | None = None,
    x_max: float | None = None,
) -> None:
    plt.figure(figsize=(4.4, 3.4))
    for case in cases:
        case_data = loaded[case.label]
        x_vals = case_data["dist_time_ns"]
        y_vals = case_data[key] * y_scale
        if x_min is not None or x_max is not None:
            mask = np.ones_like(x_vals, dtype=bool)
            if x_min is not None:
                mask &= x_vals >= x_min
            if x_max is not None:
                mask &= x_vals <= x_max
            x_vals = x_vals[mask]
            y_vals = y_vals[mask]
        plt.plot(x_vals, y_vals, color=case.color, linewidth=2.0, label=case.label)
    if theory is not None and key in theory:
        x_vals = theory["time_ns"]
        y_vals = theory[key] * y_scale
        if x_min is not None or x_max is not None:
            mask = np.ones_like(x_vals, dtype=bool)
            if x_min is not None:
                mask &= x_vals >= x_min
            if x_max is not None:
                mask &= x_vals <= x_max
            x_vals = x_vals[mask]
            y_vals = y_vals[mask]
        plt.plot(x_vals, y_vals, color="black", linestyle="--", linewidth=2.0)
    if x_min is not None or x_max is not None:
        plt.xlim(x_min, x_max)
    plt.xlabel("Time (ns)")
    plt.ylabel(ylabel)
    save_current_figure(filename)


def plot_radial_profile(
    cases: list[SweepCase],
    loaded: dict[str, dict[str, np.ndarray]],
    x_key: str,
    y_key: str,
    ylabel: str,
    filename: str,
) -> None:
    plt.figure(figsize=(4.4, 3.4))
    for case in cases:
        case_data = loaded[case.label]
        plt.plot(case_data[x_key], case_data[y_key], color=case.color, linewidth=2.0, label=case.label)
    plt.xlabel(r"$r / R_b$")
    plt.ylabel(ylabel)
    plt.xlim(0.0, 1.02)
    if filename == "scaling_radial_T.pdf":
        plt.legend(frameon=False, loc="upper right")
    save_current_figure(filename)


def print_summary(cases: list[SweepCase], loaded: dict[str, dict[str, np.ndarray]]) -> None:
    print("Section 7.1 convergence summary")
    for case in cases:
        case_data = loaded[case.label]
        print(
            f"{case.label}: "
            f"t_min={case_data['tmin_fs'] * FS_TO_NS:.4f} ns, "
            f"R_min={case_data['rmin_um']:.4f} um, "
            f"T_max={case_data['tmax_k']:.1f} K, "
            f"T_avg,max={case_data['tavg_max_k']:.1f} K, "
            f"P_avg,max={case_data['pavg_max_gpa']:.4f} GPa"
        )


def main() -> None:
    cases = [
        SweepCase(
            label=r"$N = 10^6$",
            ensemble_particles=1.0e6,
            color="#1f77b4",
            global_file=data_path("1e6", "boundary", "run_1e6_01_08_alpha_1.lammps"),
            distribution_file=data_path("1e6", "boundary", "1e6_comdis_01_08_alpha_1_minN100_smooth.out"),
        ),
        SweepCase(
            label=r"$N = 10^7$",
            ensemble_particles=1.0e7,
            color="#ff7f0e",
            global_file=data_path("1e7", "run_1e7_01_07.lammps"),
            distribution_file=data_path("1e7", "1e7_comdis_01_07_minN100_smooth.out"),
        ),
        SweepCase(
            label=r"$N = 10^8$",
            ensemble_particles=1.0e8,
            color="#2ca02c",
            global_file=data_path("1e8", "run_1e8_full.lammps"),
            distribution_file=data_path("1e8", "1e8_comdis_full_minN100_smooth.out"),
        ),
    ]

    loaded = {case.label: load_case(case) for case in cases}
    theory = load_theory()

    plot_global_radius(cases, loaded, theory)
    plot_global_scalar(cases, loaded, theory, "avg_temperature", r"$\langle T \rangle$ ($10^3$ K)", "scaling_global_avgT.pdf", y_scale=K_TO_KILO_K)
    plot_global_scalar(cases, loaded, theory, "avg_pressure", r"$\langle P \rangle$ (GPa)", "scaling_global_avgP.pdf")
    plot_radial_profile(cases, loaded, "temperature_frac_r_tcentermax", "radial_temperature_tcentermax", r"$T$ ($10^3$ K)", "scaling_radial_T.pdf")
    plot_radial_profile(cases, loaded, "temperature_frac_r_tcentermax", "radial_temperature_tcentermax", r"$T$ ($10^3$ K)", "scaling_radial_T_tcentermax.pdf")
    plot_radial_profile(cases, loaded, "temperature_frac_r_tavgmax", "radial_temperature_tavgmax", r"$T$ ($10^3$ K)", "scaling_radial_T_tavgmax.pdf")
    plot_radial_profile(cases, loaded, "temperature_frac_r_rmin", "radial_temperature_rmin", r"$T$ ($10^3$ K)", "scaling_radial_T_rmin.pdf")
    plot_radial_profile(cases, loaded, "pressure_frac_r", "radial_pressure", r"$P$ (GPa)", "scaling_radial_P.pdf")
    plot_radial_profile(cases, loaded, "density_frac_r", "radial_density", r"$\rho$ ($10^3$ kg m$^{-3}$)", "scaling_radial_rho.pdf")
    print_summary(cases, loaded)


if __name__ == "__main__":
    main()
