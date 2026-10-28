import math
import os
from dataclasses import dataclass

import numpy as np


from ._config import REPORTS_DIR, data_path

TODO_DIR = REPORTS_DIR
OUTPUT_MARKDOWN = os.path.join(TODO_DIR, "section_7_4_table_values.md")


from .utils.parse_data import parse_ave_chunk_output, parse_global_output


PINFTY = 101325.0
INITIAL_RADIUS_M = 4.5e-6
KB = 1.380649e-23
TINF = 300.0
TOTAL_PHYSICAL_PARTICLES = PINFTY * (4.0 / 3.0) * math.pi * INITIAL_RADIUS_M**3 / (KB * TINF)

ANGSTROM_TO_UM = 1.0e-4
ANGSTROM_PER_FS_TO_M_PER_S = 1.0e5
PA_TO_GPA = 1.0e-9
PRESSURE_LEAD_FS = 200_000.0
ROBUST_TMAX_MIN_NCOUNT = 10.0


@dataclass(frozen=True)
class TableCase:
    nensem_label: str
    ensemble_particles: float
    alpha_t: str
    pair_label: str
    global_file: str
    distribution_file: str

    @property
    def scale_factor(self) -> float:
        return TOTAL_PHYSICAL_PARTICLES / self.ensemble_particles


def nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def compute_case_metrics(case: TableCase) -> dict[str, float | str]:
    global_data, _ = parse_global_output(case.global_file)
    _, _, temp_values, _ = parse_ave_chunk_output(case.distribution_file, "temp")
    _, _, ncount_values, _ = parse_ave_chunk_output(case.distribution_file, "Ncount")
    _, _, charge_values, _ = parse_ave_chunk_output(case.distribution_file, "v_slcharge")

    radius_a = np.asarray(global_data["v_SLradius"], dtype=float)
    radius_velocity_a_per_fs = np.asarray(global_data["v_SLvradius"], dtype=float)
    time_fs = np.asarray(global_data["Time"], dtype=float)
    avg_temperature_raw = np.asarray(global_data["Temp"], dtype=float)
    wall_pressure_pa = np.asarray(global_data["v_SLpb"], dtype=float)

    scale_factor = case.scale_factor
    literal_tmax = float(np.nanmax(temp_values / scale_factor))
    robust_t_field = np.where(ncount_values >= ROBUST_TMAX_MIN_NCOUNT, temp_values / scale_factor, np.nan)
    robust_tmax = float(np.nanmax(robust_t_field))
    ion_frac_percent = float(np.nanmax(np.sum(charge_values * ncount_values, axis=1) * 100.0 / case.ensemble_particles))

    radius_min_idx = int(np.argmin(radius_a))
    t_min_fs = time_fs[radius_min_idx]
    pwc_idx = nearest_index(time_fs, t_min_fs - PRESSURE_LEAD_FS)

    return {
        "nensem": case.nensem_label,
        "alpha_t": case.alpha_t,
        "pair": case.pair_label,
        "T_max_K": robust_tmax,
        "T_max_literal_K": literal_tmax,
        "f_ion_percent": ion_frac_percent,
        "T_av_max_K": float(np.nanmax(avg_temperature_raw / scale_factor)),
        "R_min_um": float(radius_a[radius_min_idx] * ANGSTROM_TO_UM),
        "v_W_max_m_per_s": float(np.nanmax(-radius_velocity_a_per_fs) * ANGSTROM_PER_FS_TO_M_PER_S),
        "p_W_c_GPa": float(wall_pressure_pa[pwc_idx] * PA_TO_GPA),
        "p_W_max_GPa": float(np.nanmax(wall_pressure_pa) * PA_TO_GPA),
        "t_min_ns": float(t_min_fs * 1.0e-6),
        "t_pwc_ns": float(time_fs[pwc_idx] * 1.0e-6),
        "p_W_c_Pa": float(wall_pressure_pa[pwc_idx]),
        "p_W_max_Pa": float(np.nanmax(wall_pressure_pa)),
    }


def render_markdown(rows: list[dict[str, float | str]]) -> str:
    lines: list[str] = []
    lines.append("# Section 7.4 Table 2 Values")
    lines.append("")
    lines.append("These are the 11 `This work` rows for Table 2 in the draft, using the current Section 7.1 to 7.3 data products and raw LAMMPS outputs.")
    lines.append("")
    lines.append("For the preprocessing pipeline behind the `comdis_*_minN100_smooth.out` inputs and the later use of `Ncount`, see [data_preprocessing_notes.md](../../docs/data_preprocessing_notes.md).")
    lines.append("")
    lines.append("## Main Table Values")
    lines.append("")
    lines.append("| Nensem | alpha_t | Pair | T_max (K) | T_av,max (K) | f_ion (%) | R_min (um) | v_W,max (m/s) | p_W,c (GPa) | p_W,max (GPa) |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            "| "
            f"{row['nensem']} | "
            f"{row['alpha_t']} | "
            f"{row['pair']} | "
            f"{row['T_max_K']:.1f} | "
            f"{row['T_av_max_K']:.1f} | "
            f"{row['f_ion_percent']:.4f} | "
            f"{row['R_min_um']:.4f} | "
            f"{row['v_W_max_m_per_s']:.1f} | "
            f"{row['p_W_c_GPa']:.4f} | "
            f"{row['p_W_max_GPa']:.4f} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(f"- `T_max` in the main table is a robust local maximum: the maximum temperature over bins with `Ncount >= {int(ROBUST_TMAX_MIN_NCOUNT)}` in the smoothed radial distribution output, after EP scaling to physical particle number.")
    lines.append("- This is intentionally stricter than the literal maximum, so singleton wall-bin spikes do not dominate the table statistic.")
    lines.append("- The heatmap plotting uses `vmax_q` only as a display cap; it does not define the scalar `T_max` used here.")
    lines.append("- `T_av,max` is the maximum of the global `Temp` signal after the same EP scaling.")
    lines.append("- `f_ion` is taken from the `comdis` charge field `v_slcharge` as `max_t sum_r(Ncount * v_slcharge) / N_ensem`, reported in percent.")
    lines.append("- In the current ionized runs this equals the retained total ionization fraction at the end of the run, and it is the same quantity plotted in Section 7.3 as the charge-sum percent.")
    lines.append("- A literal `q > 0` occupancy count is intentionally not used here, because in the smoothed `comdis` data it would overcount and drive most ionized cases to nearly `100%`.")
    lines.append("- Operationally, this column is the peak retained charge fraction in the current outputs. If a future model includes substantial recombination or non-monotonic charge loss, revisit the interpretation before calling it a cumulative ever-ionized fraction.")
    lines.append("- `R_min` is taken from the minimum of `v_SLradius`, converted from angstrom to micrometer.")
    lines.append("- `v_W,max` is the maximum inward wall speed from `-v_SLvradius`, converted from angstrom/fs to m/s.")
    lines.append("- `p_W,c` and `p_W,max` are taken from the global `v_SLpb` signal, interpreted as the gas pressure just inside the bubble wall.")
    lines.append("- This matches Eq. (3) in the draft, where `P_B` is the liquid pressure outside the bubble and `P_b` is the pressure inside the bubble wall.")
    lines.append("- `p_W,c` is sampled at the recorded time closest to `t_min - 200 ps`, where `t_min` is the time of minimum radius.")
    lines.append("- Pressure conversion uses `1 Pa = 1e-9 GPa`.")
    lines.append("")
    lines.append("## Literal-vs-Robust T_max Check")
    lines.append("")
    lines.append("| Nensem | alpha_t | Pair | T_max used in table (K) | literal T_max (K) |")
    lines.append("| --- | --- | --- | ---: | ---: |")
    for row in rows:
        lines.append(
            "| "
            f"{row['nensem']} | "
            f"{row['alpha_t']} | "
            f"{row['pair']} | "
            f"{row['T_max_K']:.1f} | "
            f"{row['T_max_literal_K']:.1f} |"
        )
    lines.append("")
    lines.append("## Timing Checks")
    lines.append("")
    lines.append("| Nensem | alpha_t | Pair | t_min (ns) | sampled time for p_W,c (ns) | p_W,c (Pa) | p_W,max (Pa) |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            "| "
            f"{row['nensem']} | "
            f"{row['alpha_t']} | "
            f"{row['pair']} | "
            f"{row['t_min_ns']:.4f} | "
            f"{row['t_pwc_ns']:.4f} | "
            f"{row['p_W_c_Pa']:.6e} | "
            f"{row['p_W_max_Pa']:.6e} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
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
    markdown = render_markdown(rows)
    os.makedirs(TODO_DIR, exist_ok=True)
    with open(OUTPUT_MARKDOWN, "w", encoding="ascii") as f:
        f.write(markdown)

    print(markdown)
    print()
    print(f"Wrote {OUTPUT_MARKDOWN}")


if __name__ == "__main__":
    main()
