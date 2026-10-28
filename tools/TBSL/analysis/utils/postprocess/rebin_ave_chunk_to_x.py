#!/usr/bin/env python3
"""
Rebin LAMMPS fix ave/chunk radial distribution from fixed-r bins to fixed x=r/R(t) bins,
and (optionally) append a new column:
    dq_pos = max(0, q_mean(t,j) - q_mean(t-1,j))
computed per new x-bin (row-wise in time).

Why this works:
- Rebinning into x=r/R(t) greatly reduces artifacts from bubble breathing.
- dq_pos highlights *increases* in per-particle charge (a proxy for ionization activity)
  without counting transport-out as “events”.

Assumptions:
- Input is a LAMMPS ave/chunk style file with blocks:
    <timestep> <num_chunks> <total_count>
    <Chunk> <Coord1> <Ncount> <prop1> <prop2> ...
- Coord1 is a radial coordinate (monotone increasing across chunks).
- Global file is a whitespace table with a header line containing at least:
    Step  v_SLradius
  (radius-key is configurable).

Rebin method:
- For each timestep, compute new physical edges r_new = x_edges * R(t).
- Redistribute old shells -> new shells via spherical-shell overlap volumes.
- Ncount is redistributed by overlap fraction (volume-based).
- Intensive properties are N-weighted averages.
- density/number and density/mass (if present) are recomputed from N and volume.

Output:
- Same format as ave/chunk, but num_chunks = resolution and Coord1 is x-center in [0,1].
- If --q-col is provided, append dq_pos column to header and each row.

Usage:
  python3 rebin_ave_chunk_to_x_dq.py \
    -i in.avechunk \
    -g global.table \
    -o out.x100.avechunk \
    --radius-key v_SLradius \
    --resolution 100 \
    --q-col qmean \
    --dq-name dq_pos
"""

import argparse
import math
from typing import List, Optional, Dict, Tuple

import numpy as np


# -----------------------------
# Helpers: parsing and geometry
# -----------------------------
def parse_global_table(path: str) -> Dict[str, np.ndarray]:
    """Parse a whitespace table with header in first non-empty line."""
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if not lines:
        raise ValueError(f"Global data file is empty: {path}")

    header = lines[0].split()
    data = []
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) != len(header):
            continue
        try:
            row = [float(x) for x in parts]
        except ValueError:
            continue
        data.append(row)

    if not data:
        raise ValueError(f"No numeric rows parsed from global file: {path}")

    arr = np.array(data, dtype=float)
    return {name: arr[:, i] for i, name in enumerate(header)}


def detect_density_columns(prop_names: Optional[List[str]]) -> Tuple[Optional[int], Optional[int]]:
    """
    Return (density_mass_idx, density_num_idx) as column indices in the FULL row array,
    matching ave/chunk convention:
      col0=Chunk, col1=Coord1, col2=Ncount, col3+=properties...
    """
    if prop_names is None:
        return None, None
    density_mass_idx = prop_names.index("density/mass") if "density/mass" in prop_names else None
    density_num_idx = prop_names.index("density/number") if "density/number" in prop_names else None
    return density_mass_idx, density_num_idx


def compute_old_edges_from_centers(r_centers: np.ndarray) -> np.ndarray:
    """Infer edges from centers (assume uniform spacing). Returns length M+1."""
    M = len(r_centers)
    if M == 1:
        dr = r_centers[0] if r_centers[0] > 0 else 1.0
    else:
        dr = (r_centers[-1] - r_centers[0]) / (M - 1)

    rmin = r_centers[0] - 0.5 * dr
    if rmin < 0 and abs(rmin) < 1e-9 * (abs(r_centers[-1]) + 1.0):
        rmin = 0.0

    return rmin + dr * np.arange(M + 1)


def shell_volume_edges(edges: np.ndarray) -> np.ndarray:
    """Volume of each spherical shell between edges[k] and edges[k+1]."""
    rin = edges[:-1]
    rout = edges[1:]
    return (4.0 / 3.0) * math.pi * (rout**3 - rin**3)


def overlap_shell_volume(a0: float, a1: float, b0: float, b1: float) -> float:
    """Overlap volume between radial intervals [a0,a1] and [b0,b1] in spherical shells."""
    lo = max(a0, b0)
    hi = min(a1, b1)
    if hi <= lo:
        return 0.0
    return (4.0 / 3.0) * math.pi * (hi**3 - lo**3)


def round_preserve_sum(counts_float: np.ndarray, target_sum: int) -> np.ndarray:
    """Round floats to ints while preserving total sum exactly."""
    floors = np.floor(counts_float).astype(int)
    frac = counts_float - floors
    cur = int(floors.sum())
    diff = target_sum - cur

    if diff > 0:
        idx = np.argsort(-frac)
        for k in range(min(diff, len(floors))):
            floors[idx[k]] += 1
    elif diff < 0:
        idx = np.argsort(frac)
        need = -diff
        for k in idx:
            if need == 0:
                break
            if floors[k] > 0:
                floors[k] -= 1
                need -= 1

    if floors.sum() != target_sum:
        floors[-1] += (target_sum - floors.sum())
    return floors


# -----------------------------
# Block rebinner
# -----------------------------
def rebin_block_to_x(
    chunk_rows: List[List[float]],
    prop_names: Optional[List[str]],
    R: float,
    x_edges: np.ndarray,
) -> List[List[float]]:
    """
    chunk_rows: list of rows [chunk, Coord1, Ncount, prop...]
    Returns new rows with:
      chunk index 1..M_new
      Coord1 = x_center (normalized)
      Ncount redistributed (integer, sum preserved)
      props redistributed (intensive N-weighted; densities recomputed if present)
    """
    rows = [r[:] for r in chunk_rows]
    ncols = max(len(r) for r in rows)
    for r in rows:
        if len(r) < ncols:
            r.extend([0.0] * (ncols - len(r)))

    rows.sort(key=lambda r: r[0])  # radial order
    r_centers = np.array([r[1] for r in rows], dtype=float)
    N_old = np.array([r[2] for r in rows], dtype=float)
    total_N = int(round(float(N_old.sum())))

    old_edges = compute_old_edges_from_centers(r_centers)
    V_old = np.maximum(shell_volume_edges(old_edges), 0.0)

    r_new_edges = x_edges * float(R)
    V_new = np.maximum(shell_volume_edges(r_new_edges), 0.0)

    M_old = len(rows)
    M_new = len(x_edges) - 1

    density_mass_idx, density_num_idx = detect_density_columns(prop_names)
    density_cols = set()
    if density_mass_idx is not None:
        density_cols.add(density_mass_idx)
    if density_num_idx is not None:
        density_cols.add(density_num_idx)

    N_new_float = np.zeros(M_new, dtype=float)
    num_intensive = np.zeros((M_new, ncols), dtype=float)
    mass_new = np.zeros(M_new, dtype=float)  # for density/mass

    for i in range(M_old):
        if V_old[i] <= 0 or N_old[i] <= 0:
            continue
        a0, a1 = float(old_edges[i]), float(old_edges[i + 1])

        for j in range(M_new):
            b0, b1 = float(r_new_edges[j]), float(r_new_edges[j + 1])
            Vov = overlap_shell_volume(a0, a1, b0, b1)
            if Vov <= 0:
                continue

            frac = Vov / V_old[i]
            Nij = N_old[i] * frac
            if Nij <= 0:
                continue

            N_new_float[j] += Nij

            # intensive props: N-weighted
            for col in range(3, ncols):
                if col in density_cols:
                    continue
                num_intensive[j, col] += rows[i][col] * Nij

            # density/mass: accumulate mass = rho_mass_old * V_overlap
            if density_mass_idx is not None:
                rho_m = float(rows[i][density_mass_idx])
                if np.isfinite(rho_m):
                    mass_new[j] += rho_m * Vov

    # integer Ncount preserving sum
    N_new_int = round_preserve_sum(N_new_float, total_N).astype(float)

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    out_rows: List[List[float]] = []

    for j in range(M_new):
        row = [0.0] * ncols
        row[0] = float(j + 1)
        row[1] = float(x_centers[j])   # Coord1 normalized
        row[2] = float(N_new_int[j])   # Ncount

        Nj = float(N_new_int[j])
        Vj = float(V_new[j])

        # densities recompute
        if density_num_idx is not None:
            row[density_num_idx] = (Nj / Vj) if (Vj > 0) else 0.0
        if density_mass_idx is not None:
            row[density_mass_idx] = (mass_new[j] / Vj) if (Vj > 0) else 0.0

        # other intensive props
        if Nj > 0:
            for col in range(3, ncols):
                if col in density_cols:
                    continue
                row[col] = num_intensive[j, col] / Nj
        else:
            for col in range(3, ncols):
                if col in density_cols:
                    continue
                row[col] = 0.0

        out_rows.append(row)

    return out_rows


# -----------------------------
# Whole-file rebinner + dq_pos
# -----------------------------
def rebin_ave_chunk_to_x(
    in_lines: List[str],
    step_to_R: Dict[int, float],
    radius_key: str,
    resolution: int,
    q_col: Optional[str] = None,
    dq_name: str = "dq_pos",
) -> List[str]:
    out_lines: List[str] = []

    prop_names: Optional[List[str]] = None
    q_fullcol_idx: Optional[int] = None
    prev_q_mean: Optional[np.ndarray] = None

    x_edges = np.linspace(0.0, 1.0, resolution + 1, dtype=float)

    i = 0
    n = len(in_lines)

    while i < n:
        line = in_lines[i]
        stripped = line.strip()

        # comments/blank lines
        if not stripped or stripped.startswith("#"):
            if stripped.startswith("#"):
                # capture header if present
                fields = stripped[1:].split()
                if len(fields) >= 3 and fields[0] == "Chunk" and fields[1] == "Coord1":
                    prop_names = fields
                    if q_col is not None:
                        if q_col not in prop_names:
                            raise ValueError(f"--q-col '{q_col}' not found in ave/chunk header.")
                        q_fullcol_idx = prop_names.index(q_col)

                        # write augmented header
                        if dq_name not in prop_names:
                            out_lines.append("# " + " ".join(prop_names + [dq_name]) + "\n")
                        else:
                            out_lines.append(line)
                    else:
                        out_lines.append(line)
                else:
                    out_lines.append(line)
            else:
                out_lines.append(line)
            i += 1
            continue

        parts = stripped.split()
        if len(parts) < 3:
            out_lines.append(line)
            i += 1
            continue

        # timestep header: "<timestep> <num_chunks> <total_count>"
        try:
            timestep = int(float(parts[0]))
            num_chunks = int(float(parts[1]))
        except ValueError:
            out_lines.append(line)
            i += 1
            continue

        # read chunk rows
        chunk_rows = []
        j = i + 1
        while j < n and len(chunk_rows) < num_chunks:
            l2 = in_lines[j]
            s2 = l2.strip()
            if not s2 or s2.startswith("#"):
                j += 1
                continue
            try:
                vals = [float(f) for f in s2.split()]
            except ValueError:
                break
            chunk_rows.append(vals)
            j += 1

        if len(chunk_rows) != num_chunks:
            # copy raw block if parsing failed
            out_lines.append(line)
            out_lines.extend(in_lines[i + 1:j])
            i = j
            continue

        if timestep not in step_to_R:
            raise ValueError(
                f"Step {timestep} not found in global data for '{radius_key}'. "
                "Ensure global file covers the same steps as ave/chunk."
            )

        R = float(step_to_R[timestep])
        if not np.isfinite(R) or R <= 0:
            raise ValueError(f"Invalid {radius_key} at step {timestep}: {R}")

        new_rows = rebin_block_to_x(
            chunk_rows=chunk_rows,
            prop_names=prop_names,
            R=R,
            x_edges=x_edges,
        )

        # append dq_pos = max(0, q_mean - prev_q_mean)
        if q_col is not None:
            if q_fullcol_idx is None:
                raise RuntimeError("Internal: q_fullcol_idx not set.")
            q_now = np.array([float(r[q_fullcol_idx]) for r in new_rows], dtype=float)

            if prev_q_mean is None:
                dq = np.zeros_like(q_now)
            else:
                dq = q_now - prev_q_mean
                dq = np.maximum(dq, 0.0)

            for k, r in enumerate(new_rows):
                r.append(float(dq[k]))

            prev_q_mean = q_now

        total_count_new = int(round(sum(r[2] for r in new_rows)))
        out_lines.append(f"{timestep} {resolution} {total_count_new}\n")
        for r in new_rows:
            out_lines.append("  " + " ".join(f"{v:.6g}" for v in r) + "\n")

        i = j

    return out_lines


def main():
    p = argparse.ArgumentParser(
        description="Rebin LAMMPS ave/chunk radial distribution to fixed x=r/R(t) bins; optionally add dq_pos."
    )
    p.add_argument("-i", "--input", required=True, help="Input ave/chunk file")
    p.add_argument("-g", "--global", dest="global_path", required=True,
                   help="Global thermo-like table file containing Step and radius key")
    p.add_argument("-o", "--output", required=True, help="Output ave/chunk file")
    p.add_argument("--radius-key", default="v_SLradius",
                   help="Column name in global file for bubble radius (default: v_SLradius)")
    p.add_argument("--resolution", type=int, default=100,
                   help="Number of x=r/R bins (default: 100)")

    # dq_pos options
    p.add_argument("--q-col", default=None,
                   help="Header name of per-particle mean charge column to compute dq_pos from. If None, no dq_pos.")
    p.add_argument("--dq-name", default="dq_pos",
                   help="Name of the appended max(0, delta q) column (default: dq_pos)")

    args = p.parse_args()

    if args.resolution < 1:
        raise ValueError("resolution must be >= 1")

    global_tbl = parse_global_table(args.global_path)
    if "Step" not in global_tbl:
        raise ValueError("Global file must contain a 'Step' column")
    if args.radius_key not in global_tbl:
        raise ValueError(f"Global file missing radius key '{args.radius_key}'")

    steps = global_tbl["Step"].astype(int)
    radii = global_tbl[args.radius_key].astype(float)
    step_to_R = {int(s): float(r) for s, r in zip(steps, radii)}

    with open(args.input, "r") as f:
        in_lines = f.readlines()

    out_lines = rebin_ave_chunk_to_x(
        in_lines=in_lines,
        step_to_R=step_to_R,
        radius_key=args.radius_key,
        resolution=args.resolution,
        q_col=args.q_col,
        dq_name=args.dq_name,
    )

    with open(args.output, "w") as f:
        f.writelines(out_lines)

    print(f"Wrote rebinned ave/chunk to: {args.output}")
    if args.q_col is not None:
        print(f"Appended column '{args.dq_name}' = max(0, delta({args.q_col})) per x-bin.")


if __name__ == "__main__":
    main()
