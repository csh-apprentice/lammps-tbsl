import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


from ._config import RESULTS_DIR, data_path

FIG_DIR = RESULTS_DIR

from .utils.parse_data import parse_global_output


FS_TO_NS = 1.0e-6
K_TO_KILO_K = 1.0e-3
UNIFORM_COLOR = "#4C566A"
NONUNIFORM_COLOR = "#B04A5A"
JSPJ_EXTRACT_COLOR = "#2F2F2F"
VALIDATION_COMPARE_COLOR = "#7A6C9D"
JSPJ_X_MIN_US = 70.0
JSPJ_X_MAX_US = 160.0
JSPJ_Y_MIN_UM = 0.0
JSPJ_Y_MAX_UM = 30.0

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


def save_current_figure(filename: str) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), bbox_inches="tight")
    plt.close()


def load_validation_radius() -> dict[str, np.ndarray]:
    data, _ = parse_global_output(data_path("theory", "validation_1995_20ns.txt"))
    return {
        "time_us": np.asarray(data["Time"], dtype=float),
        "radius_um": np.asarray(data["Radius"], dtype=float),
    }


def load_jspj_extract() -> dict[str, np.ndarray]:
    data = np.loadtxt(data_path("theory", "JSPJ_compare.csv"), delimiter=",")
    return {
        "time_us": np.asarray(data[:, 0], dtype=float),
        "radius_um": np.asarray(data[:, 1], dtype=float),
    }


def load_theory_temperature_models() -> dict[str, np.ndarray]:
    data, _ = parse_global_output(data_path("theory", "theory.txt"))
    return {
        "time_ns": np.asarray(data["Time"], dtype=float) * FS_TO_NS,
        "temp_uniform_kilo_k": np.asarray(data["Temp_avg_uniform"], dtype=float) * K_TO_KILO_K,
        "temp_nonuniform_kilo_k": np.asarray(data["Temp_avg_nonuniform"], dtype=float) * K_TO_KILO_K,
    }


def plot_validation_radius(validation_data: dict[str, np.ndarray], jspj_extract: dict[str, np.ndarray]) -> None:
    validation_mask = (validation_data["time_us"] >= JSPJ_X_MIN_US) & (validation_data["time_us"] <= JSPJ_X_MAX_US)
    plt.figure(figsize=(4.4, 3.4))
    plt.plot(
        validation_data["time_us"][validation_mask],
        validation_data["radius_um"][validation_mask],
        color=VALIDATION_COMPARE_COLOR,
        linewidth=2.0,
        label="Ours",
    )
    plt.plot(
        jspj_extract["time_us"],
        jspj_extract["radius_um"],
        color=JSPJ_EXTRACT_COLOR,
        linewidth=1.6,
        linestyle="--",
        label="Kwak",
    )
    plt.xlim(JSPJ_X_MIN_US, JSPJ_X_MAX_US)
    plt.ylim(JSPJ_Y_MIN_UM, JSPJ_Y_MAX_UM)
    plt.xlabel(r"Time ($\mu$s)")
    plt.ylabel(r"$R_b$ ($\mu$m)")
    plt.legend(frameon=False, loc="upper right")
    save_current_figure("section3_validation_radius.pdf")


def plot_jspj_extract_only(jspj_extract: dict[str, np.ndarray]) -> None:
    plt.figure(figsize=(4.4, 3.4))
    plt.plot(
        jspj_extract["time_us"],
        jspj_extract["radius_um"],
        color=JSPJ_EXTRACT_COLOR,
        linewidth=1.2,
        marker="o",
        markersize=2.2,
        markerfacecolor=JSPJ_EXTRACT_COLOR,
        markeredgewidth=0.0,
    )
    plt.xlim(JSPJ_X_MIN_US, JSPJ_X_MAX_US)
    plt.ylim(JSPJ_Y_MIN_UM, JSPJ_Y_MAX_UM)
    plt.xlabel(r"Time ($\mu$s)")
    plt.ylabel(r"$R_b$ ($\mu$m)")
    save_current_figure("section3_validation_radius_kwak_only.pdf")


def plot_temperature_models(data: dict[str, np.ndarray]) -> None:
    plt.figure(figsize=(4.4, 3.4))
    plt.plot(
        data["time_ns"],
        data["temp_uniform_kilo_k"],
        color=UNIFORM_COLOR,
        linewidth=2.0,
        linestyle="-",
        label="Uniform",
    )
    plt.plot(
        data["time_ns"],
        data["temp_nonuniform_kilo_k"],
        color=NONUNIFORM_COLOR,
        linewidth=2.0,
        linestyle="--",
        label="Non-uniform",
    )
    plt.xlim(float(np.min(data["time_ns"])), float(np.max(data["time_ns"])))
    plt.xlabel("Time (ns)")
    plt.ylabel(r"$\langle T \rangle$ ($10^3$ K)")
    plt.legend(frameon=False, loc="upper right")
    save_current_figure("section3_temperature_model_comparison.pdf")


def print_summary(radius_data: dict[str, np.ndarray], jspj_extract: dict[str, np.ndarray], temperature_data: dict[str, np.ndarray]) -> None:
    uniform_peak_idx = int(np.argmax(temperature_data["temp_uniform_kilo_k"]))
    nonuniform_peak_idx = int(np.argmax(temperature_data["temp_nonuniform_kilo_k"]))
    validation_mask = (radius_data["time_us"] >= JSPJ_X_MIN_US) & (radius_data["time_us"] <= JSPJ_X_MAX_US)
    validation_peak_idx_local = int(np.argmax(radius_data["radius_um"][validation_mask]))
    jspj_peak_idx = int(np.argmax(jspj_extract["radius_um"]))
    validation_time_window = radius_data["time_us"][validation_mask]
    validation_radius_window = radius_data["radius_um"][validation_mask]
    print("Section 3 theory-comparison summary")
    print(
        "Validation radius overlay: "
        f"model window = {validation_time_window[0]:.4f} to {validation_time_window[-1]:.4f} us, "
        f"extract window = {np.min(jspj_extract['time_us']):.4f} to {np.max(jspj_extract['time_us']):.4f} us"
    )
    print(
        "Radius-peak comparison in overlay window: "
        f"model peak = {validation_radius_window[validation_peak_idx_local]:.4f} um at {validation_time_window[validation_peak_idx_local]:.4f} us, "
        f"JSPJ peak = {jspj_extract['radius_um'][jspj_peak_idx]:.4f} um at {jspj_extract['time_us'][jspj_peak_idx]:.4f} us"
    )
    print(
        "Uniform temperature: "
        f"T_max = {temperature_data['temp_uniform_kilo_k'][uniform_peak_idx] / K_TO_KILO_K:.2f} K at "
        f"{temperature_data['time_ns'][uniform_peak_idx]:.4f} ns"
    )
    print(
        "Non-uniform temperature: "
        f"T_max = {temperature_data['temp_nonuniform_kilo_k'][nonuniform_peak_idx] / K_TO_KILO_K:.2f} K at "
        f"{temperature_data['time_ns'][nonuniform_peak_idx]:.4f} ns"
    )


def main() -> None:
    radius_data = load_validation_radius()
    jspj_extract = load_jspj_extract()
    temperature_data = load_theory_temperature_models()
    plot_validation_radius(radius_data, jspj_extract)
    plot_jspj_extract_only(jspj_extract)
    plot_temperature_models(temperature_data)
    print_summary(radius_data, jspj_extract, temperature_data)


if __name__ == "__main__":
    main()
