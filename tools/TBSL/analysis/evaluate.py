#!/usr/bin/env python3
"""Evaluate one completed run with the paper's Saha--LTE emission convention.

Choose the temperature definition explicitly. This is postprocessing only.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.TBSL.analysis._config import align_steps
from tools.TBSL.analysis.run import sha256
from tools.TBSL.analysis.utils.parse_data import iter_ave_chunk_blocks, parse_global_output
from tools.TBSL.analysis.utils.postprocess.smooth_ave_chunk_minN_keepbins import smooth_ave_chunk_minN_keepbins


def emission_fields(temperature_k, ep_density_per_a3, ep_scale, prefactor=1.57e-40, cutoff_nm=200.0):
    """Return (Saha fraction, bolometric field, lambda>=cutoff field) in SI.

The coefficient and single-stage Ar parameters reproduce the paper convention;
the result remains a model diagnostic, distinct from MD retained charges.
"""
    from tools.TBSL.analysis.utils.sl_saha_emission import (
        band_limited_emission_from_P_closed_form,
        compute_emission_field_q2n2T,
        compute_saha_ionization_fraction,
    )

    temperature_k = np.asarray(temperature_k, dtype=float)
    density = np.asarray(ep_density_per_a3, dtype=float) * ep_scale * 1e30
    if not np.isfinite(ep_scale) or ep_scale <= 0:
        raise ValueError("EP scale must be finite and positive")
    if np.any(~np.isfinite(temperature_k)) or np.any(temperature_k < 0):
        raise ValueError("Temperature must be finite and nonnegative")
    if np.any(~np.isfinite(density)) or np.any(density < 0):
        raise ValueError("Density must be finite and nonnegative")
    q = compute_saha_ionization_fraction(temperature_k, density, 15.76, g_ratio=0.5)
    q = np.where(density > 0, q, 0.0)
    power = compute_emission_field_q2n2T(temperature_k, density, q, prefactor=prefactor)
    band, _ = band_limited_emission_from_P_closed_form(power, temperature_k, lambda_c_nm=cutoff_nm)
    return q, power, band


def shell_volumes_m3(radii_a, wall_radius_a=None):
    from tools.TBSL.analysis.utils.sl_saha_emission import compute_shell_edges

    edges = compute_shell_edges(radii_a)
    if np.any(np.diff(edges) <= 0) or np.any(edges < 0):
        raise ValueError("Shell edges must be nonnegative and strictly increasing")
    if wall_radius_a is not None:
        edges = np.minimum(edges, wall_radius_a)
    return 4 * np.pi / 3 * np.diff(edges ** 3) * 1e-30


def smooth_block(step, names, block, minimum):
    """Apply the historical center-outward keepbins algorithm to one block."""
    if minimum <= 0:
        return block
    lines = ["# " + " ".join(names) + "\n", f"{step} {len(block)} {block[:, 2].sum():.16g}\n"]
    lines += [" ".join(f"{v:.16g}" for v in row) + "\n" for row in block]
    result = smooth_ave_chunk_minN_keepbins(lines, minimum)
    return np.asarray([[float(v) for v in line.split()] for line in result[2:]])


def evaluate(global_file, radial_file, ep_scale, temperature, smooth_min_count=0, geometry="paper-shells", atomic_mass=39.95):
    thermo, _ = parse_global_output(global_file)
    steps = np.asarray(thermo["Step"], dtype=int)
    time = np.asarray(thermo["Time"])
    wall = np.asarray(thermo["v_SLradius"])
    records, fields = [], []
    radii = None
    for step, names, raw in iter_ave_chunk_blocks(radial_file):
        index = int(align_steps(steps, [step])[0])
        block = smooth_block(step, names, raw, smooth_min_count)
        if radii is None:
            radii = block[:, 1].copy()
        elif block.shape[0] != radii.size or not np.allclose(block[:, 1], radii):
            raise ValueError("Single-run heatmap evaluation requires a fixed radial grid")
        values = {name: block[:, i] for i, name in enumerate(names)}
        if temperature == "paper":
            if "temp" not in values:
                raise ValueError("Paper temperature requires the recorded ave/chunk 'temp' field")
            temp = values["temp"] / ep_scale
        elif temperature == "radial-flow":
            # Physical Ar mass; velocities are Angstrom/fs. No additional EP
            # temperature division: this moment formula already removes EP mass.
            variance = values["v_speed_squared"] - values["v_radial_velocity"] ** 2
            tolerance = 2e-5 * np.maximum(values["v_speed_squared"], 1e-30)
            if np.any(variance < -tolerance):
                raise ValueError("Velocity moments imply negative thermal variance")
            temp = atomic_mass * 1e7 / (3 * 8.31446261815324) * np.maximum(variance, 0)
        else:
            raise ValueError("Unknown temperature definition")
        density = values["density/number"]
        q_saha, bolometric, band = emission_fields(temp, density, ep_scale)
        volumes = shell_volumes_m3(radii, wall[index] if geometry == "clip-wall" else None)
        ncount = raw[:, 2]  # Keep original counts, including after smoothing.
        count = ncount.sum()
        if count <= 0:
            raise ValueError("No particles in radial output")
        record = {
            "Step": step, "time_ns": float(time[index] * 1e-6),
            "radius_um": float(wall[index] * 1e-4),
            "T_max_K": float(temp.max()),
            "T_particle_mean_K": float(np.dot(ncount, temp) / count),
            "saha_electron_count": float(np.dot(q_saha * density * ep_scale * 1e30, volumes)),
            "bolometric_mW": float(np.dot(bolometric, volumes) * 1e3),
            "lambda_ge_200nm_mW": float(np.dot(band, volumes) * 1e3),
        }
        if "v_slcharge" in values:
            record["MD_mean_charge"] = float(np.dot(ncount, values["v_slcharge"]) / count)
        if "v_ionized" in values:
            record["MD_ionized_EP_fraction"] = float(np.dot(ncount, values["v_ionized"]) / count)
        records.append(record)
        fields.append(band * 1e-15)  # W/m^3 to mW/um^3.
    if not records:
        raise ValueError("No completed radial frames")
    return records, radii, np.asarray(fields), float(time[np.argmin(wall)] * 1e-6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-file", required=True, type=Path)
    parser.add_argument("--radial-file", required=True, type=Path)
    parser.add_argument("--ep-scale", required=True, type=float)
    parser.add_argument("--temperature", required=True, choices=("paper", "radial-flow"))
    parser.add_argument("--smooth-min-count", type=int, default=0, help="0: already prepared/no smoothing; 100: paper keepbins smoothing")
    parser.add_argument("--geometry", choices=("paper-shells", "clip-wall"), default="paper-shells")
    parser.add_argument("--atomic-mass", type=float, default=39.95, help="physical mass in g/mol for velocity moments")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="Run")
    args = parser.parse_args()
    if args.ep_scale <= 0 or not np.isfinite(args.ep_scale) or args.smooth_min_count < 0 or args.atomic_mass <= 0:
        parser.error("EP scale/mass must be positive; smoothing count must be nonnegative")
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir.resolve() / ".cache/matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(args.output_dir.resolve() / ".cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records, radii, fields, tmin = evaluate(
        args.global_file, args.radial_file, args.ep_scale, args.temperature,
        args.smooth_min_count, args.geometry, args.atomic_mass,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "evaluation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    times = np.array([r["time_ns"] for r in records])
    band = np.array([r["lambda_ge_200nm_mW"] for r in records])
    peak = int(np.argmax(band))
    summary = {
        "label": args.label, "temperature_definition": args.temperature,
        "smoothing_min_count": args.smooth_min_count, "geometry": args.geometry,
        "ep_scale": args.ep_scale, "atomic_mass_g_mol": args.atomic_mass,
        "Ar_ionization_eV": 15.76, "saha_g_ratio": 0.5,
        "emission_prefactor": 1.57e-40, "lambda_min_nm": 200.0,
        "samples": len(records), "peak_band_mW": float(band[peak]),
        "peak_time_ns": float(times[peak]), "tmin_ns": tmin,
        "pulse_energy_band_J": float((getattr(np, "trapezoid", None) or np.trapz)(band * 1e-3, times * 1e-9)),
        "interpretation": "Single-stage Saha--LTE diagnostic; no MD recombination/radiation coupling. Time integral covers only the supplied sampling window.",
        "inputs": {str(p.resolve()): sha256(p) for p in (args.global_file, args.radial_file)},
        "sources": {str(p.relative_to(Path(__file__).parent)): sha256(p) for p in (Path(__file__), Path(__file__).parent / "utils/sl_saha_emission.py", Path(__file__).parent / "utils/parse_data.py", Path(__file__).parent / "utils/postprocess/smooth_ave_chunk_minN_keepbins.py")},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(args.output_dir / "emission_field.npz", time_ns=times, radii_um=radii * 1e-4, emission_mW_um3=fields)
    fig, ax = plt.subplots(figsize=(5, 3.4), constrained_layout=True)
    ax.plot(times - tmin, band)
    ax.axvline(0, color="0.5", ls=":")
    ax.set(xlabel=r"$t-t_{R\min}$ (ns)", ylabel=r"Saha--LTE diagnostic ($\lambda\geq200$ nm), mW", title=args.label)
    for ext in ("pdf", "png"):
        fig.savefig(args.output_dir / ("emission_curve." + ext), dpi=220)
    plt.close(fig)
    print(json.dumps({k: summary[k] for k in ("label", "samples", "peak_band_mW", "peak_time_ns", "temperature_definition")}, indent=2))


if __name__ == "__main__":
    main()
