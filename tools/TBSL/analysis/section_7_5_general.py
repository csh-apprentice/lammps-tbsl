import math
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


from ._config import RESULTS_DIR, REPORTS_DIR, data_path, align_steps

FIG_DIR = RESULTS_DIR
TODO_DIR = REPORTS_DIR

from .utils.parse_data import compute_pressure_from_virial_density, parse_ave_chunk_output, parse_global_output
from .utils.sl_saha_emission import band_limited_emission_from_P_closed_form, compute_emission_from_lammps_number


PINFTY = 101325.0
INITIAL_RADIUS_M = 4.5e-6
KB = 1.380649e-23
TINF = 300.0
TOTAL_PHYSICAL_PARTICLES = PINFTY * (4.0 / 3.0) * math.pi * INITIAL_RADIUS_M**3 / (KB * TINF)

FS_TO_NS = 1.0e-6
ANGSTROM_TO_UM = 1.0e-4
ANGSTROM_PER_FS_TO_M_PER_S = 1.0e5
K_TO_KILO_K = 1.0e-3
ATM_TO_GPA = 101325.0 / 1.0e9
RADIAL_TRIM_BINS = 2

GENERAL_DYNAMICS_X_HALFSPAN_NS = 1.5
EMISSION_X_MIN_NS = 4.10
EMISSION_X_MAX_NS = 4.70

IONIZATION_ENERGY_EV = 15.76
SAHA_G_RATIO = 0.5
EMISSION_PREFACTOR = 1.57e-40
EMISSION_TO_MW_PER_UM3 = 1.0e-15

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
class EmissionCase:
    label: str
    ensemble_particles: float
    color: str
    global_file: str
    distribution_file: str

    @property
    def scale_factor(self) -> float:
        return TOTAL_PHYSICAL_PARTICLES / self.ensemble_particles


def save_current_figure(filename: str) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), bbox_inches="tight")
    plt.close()


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


def trim_stop_index(radii: np.ndarray, bubble_radius: float) -> int:
    stop = np.searchsorted(radii, bubble_radius, side="right")
    return max(stop - RADIAL_TRIM_BINS, 1)


def profile_samples(frac_r: np.ndarray, values: np.ndarray) -> dict[str, float]:
    mid_idx = nearest_index(frac_r, 0.5)
    return {
        "center_r": float(frac_r[0]),
        "center_value": float(values[0]),
        "mid_r": float(frac_r[mid_idx]),
        "mid_value": float(values[mid_idx]),
        "edge_r": float(frac_r[-1]),
        "edge_value": float(values[-1]),
    }


def load_theory() -> dict[str, np.ndarray]:
    theory_data, _ = parse_global_output(data_path("theory", "theory.txt"))
    temp_frac, _ = parse_global_output(data_path("theory", "temp_frac.txt"))
    press_frac, _ = parse_global_output(data_path("theory", "press_frac.txt"))

    theory_time_fs = np.asarray(theory_data["Time"], dtype=float)
    theory_radius_m = np.asarray(theory_data["Radius"], dtype=float)
    theory_tmin_idx = int(np.argmin(theory_radius_m))

    return {
        "time_rel_ns": (theory_time_fs - theory_time_fs[theory_tmin_idx]) * FS_TO_NS,
        "radius_um": theory_radius_m * 1.0e6,
        "velocity_m_per_s": np.asarray(theory_data["Velocity"], dtype=float),
        "temp_avg_k": np.asarray(theory_data["Temp_avg_uniform"], dtype=float),
        "tmin_idx": theory_tmin_idx,
        "radial_frac": np.asarray(temp_frac["frac_r_R"], dtype=float),
        "radial_temp_k": np.asarray(temp_frac["Temp_uniform"], dtype=float),
        "radial_press_gpa": np.asarray(press_frac["Press"], dtype=float) * ATM_TO_GPA,
    }


def load_md_global_1e8() -> dict[str, np.ndarray]:
    global_data, _ = parse_global_output(data_path("1e8", "run_1e8_full.lammps"))
    scale_factor = TOTAL_PHYSICAL_PARTICLES / 1.0e8

    time_fs = np.asarray(global_data["Time"], dtype=float)
    radius_a = np.asarray(global_data["v_SLradius"], dtype=float)
    center_temperature_k = np.asarray(global_data["c_coretemp"], dtype=float) / scale_factor
    tmin_idx = int(np.argmin(radius_a))
    tcenter_max_idx = int(np.argmax(center_temperature_k))

    return {
        "time_rel_ns": (time_fs - time_fs[tmin_idx]) * FS_TO_NS,
        "time_ns": time_fs * FS_TO_NS,
        "radius_um": radius_a * ANGSTROM_TO_UM,
        "wall_velocity_m_per_s": np.asarray(global_data["v_SLvradius"], dtype=float) * ANGSTROM_PER_FS_TO_M_PER_S,
        "avg_temperature_k": np.asarray(global_data["Temp"], dtype=float) / scale_factor,
        "center_temperature_k": center_temperature_k,
        "tmin_idx": tmin_idx,
        "tmin_ns": time_fs[tmin_idx] * FS_TO_NS,
        "tcenter_max_idx": tcenter_max_idx,
        "tcenter_max_ns": time_fs[tcenter_max_idx] * FS_TO_NS,
        "tcenter_max_minus_tmin_ps": (time_fs[tcenter_max_idx] - time_fs[tmin_idx]) * 1.0e-3,
    }


def load_md_radial_case(case: EmissionCase) -> dict[str, np.ndarray]:
    global_data, _ = parse_global_output(case.global_file)
    dist_steps, radii, temp_values, _ = parse_ave_chunk_output(case.distribution_file, "temp")
    _, _, density_mass_values, _ = parse_ave_chunk_output(case.distribution_file, "density/mass")
    _, _, density_number_values, _ = parse_ave_chunk_output(case.distribution_file, "density/number")
    _, _, virial_values, _ = parse_ave_chunk_output(case.distribution_file, "v_slvirial")

    global_steps = np.asarray(global_data["Step"], dtype=int)
    global_time_fs = np.asarray(global_data["Time"], dtype=float)
    radius_a = np.asarray(global_data["v_SLradius"], dtype=float)
    center_temperature_k = np.asarray(global_data["c_coretemp"], dtype=float) / case.scale_factor
    step_indices = align_steps(global_steps, dist_steps)

    scale_factor = case.scale_factor
    pressure_values_gpa = compute_pressure_from_virial_density(virial_values, density_number_values, 1.0) * ATM_TO_GPA
    tmin_global_idx = int(np.argmin(radius_a))
    tmin_dist_idx = nearest_index(global_time_fs[step_indices], global_time_fs[tmin_global_idx])
    tcenter_max_global_idx = int(np.argmax(center_temperature_k))
    tcenter_max_dist_idx = nearest_index(global_time_fs[step_indices], global_time_fs[tcenter_max_global_idx])
    rb_tcenter = radius_a[step_indices][tcenter_max_dist_idx]
    rb_tmin = radius_a[step_indices][tmin_dist_idx]
    temp_stop = trim_stop_index(radii, rb_tcenter)
    rmin_stop = trim_stop_index(radii, rb_tmin)

    temperature_frac_r = radii[:temp_stop] / rb_tcenter
    temperature_kilo_k = temp_values[tcenter_max_dist_idx, :temp_stop] / scale_factor * K_TO_KILO_K
    rmin_frac_r = radii[:rmin_stop] / rb_tmin
    density_1e3kgm3 = density_mass_values[tmin_dist_idx, :rmin_stop]
    pressure_gpa = pressure_values_gpa[tmin_dist_idx, :rmin_stop]

    return {
        "temperature_frac_r": temperature_frac_r,
        "temperature_kilo_k": temperature_kilo_k,
        "rmin_frac_r": rmin_frac_r,
        "density_1e3kgm3": density_1e3kgm3,
        "pressure_gpa": pressure_gpa,
        "tcenter_max_time_ns": global_time_fs[step_indices][tcenter_max_dist_idx] * FS_TO_NS,
        "tcenter_max_minus_tmin_ps": (global_time_fs[step_indices][tcenter_max_dist_idx] - global_time_fs[tmin_global_idx]) * 1.0e-3,
        "tcenter_max_time_match_fs": global_time_fs[step_indices][tcenter_max_dist_idx] - global_time_fs[tcenter_max_global_idx],
        "rb_tcenter_um": rb_tcenter * ANGSTROM_TO_UM,
        "tmin_time_ns": global_time_fs[step_indices][tmin_dist_idx] * FS_TO_NS,
        "tmin_time_match_fs": global_time_fs[step_indices][tmin_dist_idx] - global_time_fs[tmin_global_idx],
        "rb_tmin_um": rb_tmin * ANGSTROM_TO_UM,
        "temperature_samples": profile_samples(temperature_frac_r, temperature_kilo_k),
        "density_samples": profile_samples(rmin_frac_r, density_1e3kgm3),
        "pressure_samples": profile_samples(rmin_frac_r, pressure_gpa),
    }


def load_emission_case(case: EmissionCase) -> dict[str, np.ndarray]:
    global_data, _ = parse_global_output(case.global_file)
    dist_steps, radii, temp_values, _ = parse_ave_chunk_output(case.distribution_file, "temp")
    _, _, density_number_values, _ = parse_ave_chunk_output(case.distribution_file, "density/number")

    global_steps = np.asarray(global_data["Step"], dtype=int)
    global_time_fs = np.asarray(global_data["Time"], dtype=float)
    radius_a = np.asarray(global_data["v_SLradius"], dtype=float)
    step_indices = align_steps(global_steps, dist_steps)

    p_vals, _, _, _ = compute_emission_from_lammps_number(
        temp_vals=temp_values / case.scale_factor,
        number_density_lammps=density_number_values * case.scale_factor,
        radii=radii,
        ionization_energy_eV=IONIZATION_ENERGY_EV,
        prefactor=EMISSION_PREFACTOR,
        g_ratio=SAHA_G_RATIO,
    )
    p_band, _ = band_limited_emission_from_P_closed_form(
        p_vals,
        temp_values / case.scale_factor,
        lambda_c_nm=200.0,
    )

    total_emission_mw = np.sum(p_band * compute_shell_volumes(radii) * 1.0e-30, axis=1) * 1.0e3
    time_ns = global_time_fs[step_indices] * FS_TO_NS

    tmin_idx_global = int(np.argmin(radius_a))
    tmin_ns = global_time_fs[tmin_idx_global] * FS_TO_NS

    return {
        "time_ns": time_ns,
        "radius_um": radius_a[step_indices] * ANGSTROM_TO_UM,
        "radii": radii,
        "emission_mw_um3": p_band * EMISSION_TO_MW_PER_UM3,
        "total_emission_mw": total_emission_mw,
        "tmin_ns": tmin_ns,
    }


def plot_general_dynamics(md: dict[str, np.ndarray], theory: dict[str, np.ndarray]) -> None:
    mask_md = np.abs(md["time_rel_ns"]) <= GENERAL_DYNAMICS_X_HALFSPAN_NS
    mask_th = np.abs(theory["time_rel_ns"]) <= GENERAL_DYNAMICS_X_HALFSPAN_NS

    panels = [
        ("general_dynamics_radius.pdf", md["radius_um"], theory["radius_um"], r"$R_b$ ($\mu$m)"),
        ("general_dynamics_velocity.pdf", md["wall_velocity_m_per_s"], theory["velocity_m_per_s"], r"$U_b$ (m s$^{-1}$)"),
        ("general_dynamics_avgT.pdf", md["avg_temperature_k"] * K_TO_KILO_K, theory["temp_avg_k"] * K_TO_KILO_K, r"$\langle T \rangle$ ($10^3$ K)"),
    ]

    for filename, md_y, th_y, ylabel in panels:
        plt.figure(figsize=(4.4, 3.4))
        plt.plot(md["time_rel_ns"][mask_md], md_y[mask_md], color="#2ca02c", linewidth=2.0, label="MD")
        plt.plot(theory["time_rel_ns"][mask_th], th_y[mask_th], color="black", linestyle="--", linewidth=2.0, label="Kwak")
        plt.axvline(0.0, color="black", linestyle=":", linewidth=1.5)
        plt.xlim(-GENERAL_DYNAMICS_X_HALFSPAN_NS, GENERAL_DYNAMICS_X_HALFSPAN_NS)
        plt.xlabel(r"$t - t_{\min}$ (ns)")
        plt.ylabel(ylabel)
        if filename == "general_dynamics_radius.pdf":
            plt.legend(frameon=False, loc="upper right")
        save_current_figure(filename)


def plot_general_radial(radial_cases: list[EmissionCase], loaded_radial: dict[str, dict[str, np.ndarray]]) -> None:
    plt.figure(figsize=(4.4, 3.4))
    for case in radial_cases:
        case_data = loaded_radial[case.label]
        plt.plot(case_data["temperature_frac_r"], case_data["temperature_kilo_k"], color=case.color, linewidth=2.0, label=case.label)
    plt.xlim(0.0, 1.02)
    plt.xlabel(r"$r / R_b$")
    plt.ylabel(r"$T$ ($10^3$ K)")
    plt.legend(frameon=False, loc="upper right")
    save_current_figure("general_radial_T.pdf")

    plt.figure(figsize=(4.4, 3.4))
    for case in radial_cases:
        case_data = loaded_radial[case.label]
        plt.plot(case_data["rmin_frac_r"], case_data["density_1e3kgm3"], color=case.color, linewidth=2.0)
    plt.xlim(0.0, 1.02)
    plt.xlabel(r"$r / R_b$")
    plt.ylabel(r"$\rho$ ($10^3$ kg m$^{-3}$)")
    save_current_figure("general_radial_rho.pdf")

    plt.figure(figsize=(4.4, 3.4))
    for case in radial_cases:
        case_data = loaded_radial[case.label]
        plt.plot(case_data["rmin_frac_r"], case_data["pressure_gpa"], color=case.color, linewidth=2.0)
    plt.xlim(0.0, 1.02)
    plt.xlabel(r"$r / R_b$")
    plt.ylabel(r"$P$ (GPa)")
    save_current_figure("general_radial_press.pdf")


def plot_general_emission(cases: list[EmissionCase], loaded: dict[str, dict[str, np.ndarray]]) -> None:
    plt.figure(figsize=(4.4, 3.4))
    for case in cases:
        data = loaded[case.label]
        mask = (data["time_ns"] >= EMISSION_X_MIN_NS) & (data["time_ns"] <= EMISSION_X_MAX_NS)
        plt.plot(data["time_ns"][mask], data["total_emission_mw"][mask], color=case.color, linewidth=2.0, label=case.label)
        plt.axvline(data["tmin_ns"], color=case.color, linestyle=":", linewidth=1.2, alpha=0.8)
    plt.xlim(EMISSION_X_MIN_NS, EMISSION_X_MAX_NS)
    plt.xlabel("Time (ns)")
    plt.ylabel(r"$L_{\mathrm{tot}}$ (mW)")
    plt.legend(frameon=False, loc="upper right")
    save_current_figure("general_emission_total.pdf")

    plt.figure(figsize=(4.4, 3.4))
    for case in cases:
        data = loaded[case.label]
        mask = (data["time_ns"] >= EMISSION_X_MIN_NS) & (data["time_ns"] <= EMISSION_X_MAX_NS)
        plt.plot(
            data["time_ns"][mask],
            data["total_emission_mw"][mask] / np.max(data["total_emission_mw"]),
            color=case.color,
            linewidth=2.0,
        )
        plt.axvline(data["tmin_ns"], color=case.color, linestyle=":", linewidth=1.2, alpha=0.8)
    plt.xlim(EMISSION_X_MIN_NS, EMISSION_X_MAX_NS)
    plt.ylim(0.0, 1.05)
    plt.xlabel("Time (ns)")
    plt.ylabel(r"$L_{\mathrm{tot}} / L_{\mathrm{tot,max}}$")
    save_current_figure("general_emission_total_norm.pdf")

    data_1e8 = loaded[r"$N = 10^8$"]
    peak_idx = int(np.argmax(data_1e8["total_emission_mw"]))
    rb_peak = data_1e8["radius_um"][peak_idx] / ANGSTROM_TO_UM
    stop = trim_stop_index(data_1e8["radii"], rb_peak)
    frac_r = data_1e8["radii"][:stop] / rb_peak
    emission_profile = data_1e8["emission_mw_um3"][peak_idx, :stop]

    plt.figure(figsize=(4.4, 3.4))
    plt.plot(frac_r, emission_profile, color="#2ca02c", linewidth=2.0)
    plt.xlim(0.0, 1.02)
    plt.xlabel(r"$r / R_b$")
    plt.ylabel(r"Emission (mW/$\mu$m$^3$)")
    save_current_figure("general_emission_radial.pdf")


def write_summary(
    md: dict[str, np.ndarray],
    radial_cases: list[EmissionCase],
    loaded_radial: dict[str, dict[str, np.ndarray]],
    emission_cases: list[EmissionCase],
    loaded_emission: dict[str, dict[str, np.ndarray]],
) -> None:
    path = os.path.join(TODO_DIR, "section_7_5_results_summary.md")
    emission_lines = []
    radial_t_lines = []
    radial_collapse_lines = []
    for case in emission_cases:
        data = loaded_emission[case.label]
        peak_idx = int(np.argmax(data["total_emission_mw"]))
        peak_time = data["time_ns"][peak_idx]
        peak_val = data["total_emission_mw"][peak_idx]
        emission_lines.append(
            f"- {case.label}: peak emission = {peak_val:.6e} mW at {peak_time:.4f} ns; peak - t_min = {(peak_time - data['tmin_ns']) * 1.0e3:.1f} ps"
        )
    for case in radial_cases:
        data = loaded_radial[case.label]
        ts = data["temperature_samples"]
        ds = data["density_samples"]
        ps = data["pressure_samples"]
        radial_t_lines.append(
            f"- {case.label}: `t(T_center,max) = {data['tcenter_max_time_ns']:.4f} ns`, `t(T_center,max) - t_min = {data['tcenter_max_minus_tmin_ps']:.1f} ps`, `R_b = {data['rb_tcenter_um']:.4f} um`; `T(0) = {ts['center_value']:.2f}`, `T({ts['mid_r']:.2f} R_b) = {ts['mid_value']:.2f}`, `T({ts['edge_r']:.2f} R_b) = {ts['edge_value']:.2f}` in `10^3 K`."
        )
        radial_collapse_lines.append(
            f"- {case.label}: `t_min = {data['tmin_time_ns']:.4f} ns`, `R_b = {data['rb_tmin_um']:.4f} um`; `rho(0) = {ds['center_value']:.3f}`, `rho({ds['mid_r']:.2f} R_b) = {ds['mid_value']:.3f}`, `rho({ds['edge_r']:.2f} R_b) = {ds['edge_value']:.3f}` in `10^3 kg m^-3`; `P(0) = {ps['center_value']:.3f}`, `P({ps['mid_r']:.2f} R_b) = {ps['mid_value']:.3f}`, `P({ps['edge_r']:.2f} R_b) = {ps['edge_value']:.3f}` in `GPa`."
        )

    text = "\n".join(
        [
            "# Section 7.5 Results Summary",
            "",
            "## Overview",
            "",
            "- The general-dynamics plots compare the `N=1e8`, full-ionization MD run against the Kwak-style PDE theory.",
            "- The three radial-structure panels now compare the `N = 10^6`, `10^7`, and `10^8` MD runs directly, with no radial theory overlay.",
            "- Color convention without a legend: blue = `N = 10^6`, orange = `N = 10^7`, green = `N = 10^8`.",
            "- The dynamics panels are still plotted in `t - t_min`, so the vertical reference line is at `0` by construction.",
            "",
            "## Key Observations",
            "",
            f"- For the `N = 10^8` dynamics-comparison run, MD minimum radius occurs at {md['tmin_ns']:.4f} ns.",
            "- `general_radial_T.pdf` uses each run's own `t(T_center,max)` snapshot. `general_radial_rho.pdf` and `general_radial_press.pdf` use each run's own `t = t_min` snapshot, so the overlays are not tied to one shared absolute time.",
            "- The third radial panel is the pressure plot `figs/results/general_radial_press.pdf`, in `GPa`, using the `t = t_min` snapshot. The theory overlay is intentionally omitted for now because the magnitude gap makes the direct comparison visually unhelpful without additional normalization.",
            f"- The 1e8 emission pulse peaks at {(loaded_emission[r'$N = 10^8$']['time_ns'][int(np.argmax(loaded_emission[r'$N = 10^8$']['total_emission_mw']))]):.4f} ns, about {(loaded_emission[r'$N = 10^8$']['time_ns'][int(np.argmax(loaded_emission[r'$N = 10^8$']['total_emission_mw']))] - loaded_emission[r'$N = 10^8$']['tmin_ns']) * 1.0e3:.1f} ps relative to `t_min`.",
            "",
            "## Radial Temperature Snapshots",
            "",
            *radial_t_lines,
            "",
            "## Collapse Density And Pressure Snapshots",
            "",
            *radial_collapse_lines,
            "",
            "## Emission Peaks",
            "",
            *emission_lines,
            "",
        ]
    )
    with open(path, "w", encoding="ascii") as f:
        f.write(text)


def main() -> None:
    theory = load_theory()
    theory["velocity_m_per_s_at_tmin"] = theory["velocity_m_per_s"][theory["tmin_idx"]]

    md_global = load_md_global_1e8()
    radial_cases = [
        EmissionCase(
            label=r"$N = 10^6$",
            ensemble_particles=1.0e6,
            color="#1f77b4",
            global_file=data_path("1e6", "boundary", "run_1e6_01_08_alpha_1.lammps"),
            distribution_file=data_path("1e6", "boundary", "1e6_comdis_01_08_alpha_1_minN100_smooth.out"),
        ),
        EmissionCase(
            label=r"$N = 10^7$",
            ensemble_particles=1.0e7,
            color="#ff7f0e",
            global_file=data_path("1e7", "run_1e7_01_07.lammps"),
            distribution_file=data_path("1e7", "1e7_comdis_01_07_minN100_smooth.out"),
        ),
        EmissionCase(
            label=r"$N = 10^8$",
            ensemble_particles=1.0e8,
            color="#2ca02c",
            global_file=data_path("1e8", "run_1e8_full.lammps"),
            distribution_file=data_path("1e8", "1e8_comdis_full_minN100_smooth.out"),
        ),
    ]
    loaded_radial = {case.label: load_md_radial_case(case) for case in radial_cases}

    emission_cases = radial_cases
    loaded_emission = {case.label: load_emission_case(case) for case in emission_cases}

    plot_general_dynamics(md_global, theory)
    plot_general_radial(radial_cases, loaded_radial)
    plot_general_emission(emission_cases, loaded_emission)
    write_summary(md_global, radial_cases, loaded_radial, emission_cases, loaded_emission)


if __name__ == "__main__":
    main()
