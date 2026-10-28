#!/usr/bin/env python3
"""
Auto-merge radial bins in LAMMPS fix ave/chunk output based on minimum
number of particles per bin.

Instead of specifying "merge N fine bins together", you specify a
minimum particle count per coarse bin, min_N. For each timestep, bins
are merged from the bubble center outward so that:

  - Each coarse bin (except possibly the last one with particles) has
    total Ncount >= min_N.
  - Bins beyond the last non-zero Ncount are treated as "invalid"
    (empty region) and merged into a single empty bin.

Density columns are handled volume-correctly:

  - density/number = total_number / total_volume
  - density/mass   = total_mass / total_volume
    with total_mass = sum( rho_mass_i * V_i ), V_i shell volumes.

Assumptions:
  - Output includes a header like:
      # Chunk Coord1 Ncount temp ... density/mass density/number
  - Chunk indices increase with radius and Coord1 is radial position
    from a uniform bin/sphere (constant Δr).

IMPORTANT:
  - Coord1 of each merged bin is set to the geometric center of the
    merged shell: (r_in + r_out)/2, where r_in and r_out are derived
    from the global rmin and Δr of the original fine binning.
"""

import argparse
import math
from typing import List, Optional, Tuple


def compute_shell_volumes(chunk_rows: List[List[float]]) -> Tuple[List[float], float, float]:
    """Compute spherical shell volumes for each radial bin.

    Returns:
      vols : list of shell volumes for each fine bin (aligned with chunk_rows)
      rmin : inner radius of the first fine bin
      dr   : radial bin width
    """
    M = len(chunk_rows)
    radii = [r[1] for r in chunk_rows]

    if M > 1:
        dr = (radii[-1] - radii[0]) / (M - 1)
    else:
        dr = radii[0] if radii[0] != 0 else 1.0

    rmin = radii[0] - 0.5 * dr
    # Clamp tiny negative to zero
    if rmin < 0 and abs(rmin) < 1e-6 * (radii[-1] or 1.0):
        rmin = 0.0

    vols = []
    for idx in range(M):
        r_in = rmin + idx * dr
        r_out = r_in + dr
        vol = 4.0 / 3.0 * math.pi * (r_out**3 - r_in**3)
        vols.append(vol)

    return vols, rmin, dr


def detect_density_columns(prop_names: Optional[List[str]]
                           ) -> Tuple[Optional[int], Optional[int]]:
    """Return (density_mass_idx, density_num_idx) based on header names."""
    density_mass_idx = None
    density_num_idx = None
    if prop_names is None:
        return None, None

    if "density/mass" in prop_names:
        density_mass_idx = prop_names.index("density/mass")
    if "density/number" in prop_names:
        density_num_idx = prop_names.index("density/number")

    return density_mass_idx, density_num_idx


def merge_group(
    chunk_rows: List[List[float]],
    vols_ordered: List[float],
    start: int,
    end: int,
    density_mass_idx: Optional[int],
    density_num_idx: Optional[int],
    ncols: int,
    new_chunk_index: int,
    rmin_global: float,
    dr_global: float,
) -> List[float]:
    """Merge fine bins [start, end) into one coarse bin."""
    merged = [0.0] * ncols
    merged[0] = float(new_chunk_index)

    group_rows = chunk_rows[start:end]
    group_vols = vols_ordered[start:end]

    counts = [r[2] for r in group_rows]
    N_tot = sum(counts)
    V_tot = sum(group_vols)

    # Coord1: geometric center of merged shell
    # r_in = rmin + start * dr, r_out = rmin + end * dr
    r_in = rmin_global + start * dr_global
    r_out = rmin_global + end * dr_global
    merged[1] = 0.5 * (r_in + r_out)

    # Ncount: total particles in merged shell
    merged[2] = N_tot

    # Non-density properties: Ncount-weighted averages
    if N_tot > 0:
        for col in range(3, ncols):
            if (density_mass_idx is not None and col == density_mass_idx) or \
               (density_num_idx is not None and col == density_num_idx):
                continue
            merged[col] = sum(r[col] * r[2] for r in group_rows) / N_tot
    else:
        for col in range(3, ncols):
            if (density_mass_idx is not None and col == density_mass_idx) or \
               (density_num_idx is not None and col == density_num_idx):
                continue
            merged[col] = 0.0

    # density/number: N_tot / V_tot
    if density_num_idx is not None:
        if V_tot > 0:
            merged[density_num_idx] = N_tot / V_tot
        else:
            merged[density_num_idx] = 0.0

    # density/mass: M_tot / V_tot, where M_tot = sum(rho_m_j * V_j)
    if density_mass_idx is not None:
        if V_tot > 0:
            M_tot = sum(r[density_mass_idx] * v
                        for r, v in zip(group_rows, group_vols))
            merged[density_mass_idx] = M_tot / V_tot
        else:
            merged[density_mass_idx] = 0.0

    return merged


def merge_ave_chunk_minN(lines: List[str], min_N: int) -> List[str]:
    """Auto-merge bins based on minimum particle count min_N per coarse bin."""
    out_lines: List[str] = []
    i = 0
    n = len(lines)

    # Parse header for property names
    prop_names = None
    for line in lines:
        s = line.strip()
        if s.startswith("#"):
            fields = s[1:].split()
            if len(fields) >= 3 and fields[0] == "Chunk" and fields[1] == "Coord1":
                prop_names = fields
                break

    density_mass_idx, density_num_idx = detect_density_columns(prop_names)

    volumes_cache = None  # list of shell volumes, aligned with chunk_rows index
    rmin_global = None
    dr_global = None

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # Copy comments and blank lines as-is
        if not stripped or stripped.startswith("#"):
            out_lines.append(line)
            i += 1
            continue

        parts = stripped.split()
        if len(parts) < 3:
            out_lines.append(line)
            i += 1
            continue

        # Expect timestep header: "<timestep> <num_chunks> <total_count>"
        try:
            timestep = int(float(parts[0]))
            num_chunks = int(float(parts[1]))
        except ValueError:
            out_lines.append(line)
            i += 1
            continue

        # Read chunk rows for this timestep
        chunk_rows = []
        j = i + 1
        while j < n and len(chunk_rows) < num_chunks:
            l2 = lines[j]
            s2 = l2.strip()
            if not s2 or s2.startswith("#"):
                j += 1
                continue
            fields = s2.split()
            try:
                vals = [float(f) for f in fields]
            except ValueError:
                break
            chunk_rows.append(vals)
            j += 1

        if len(chunk_rows) != num_chunks:
            # Inconsistent block; just copy as-is
            out_lines.append(line)
            out_lines.extend(lines[i + 1:j])
            i = j
            continue

        # Normalize row lengths
        ncols = max(len(r) for r in chunk_rows)
        for r in chunk_rows:
            if len(r) < ncols:
                r.extend([0.0] * (ncols - len(r)))

        # Sort by chunk index (col 0) so we go from center outward
        chunk_rows.sort(key=lambda r: r[0])

        # Cache volumes once, assuming binning doesn't change
        if volumes_cache is None:
            volumes_cache, rmin_global, dr_global = compute_shell_volumes(chunk_rows)

        # Volumes aligned with sorted chunk_rows by index
        vols_ordered = []
        for row in chunk_rows:
            idx_row = int(round(row[0])) - 1  # chunk index is 1-based
            if 0 <= idx_row < len(volumes_cache):
                vols_ordered.append(volumes_cache[idx_row])
            else:
                vols_ordered.append(0.0)

        # Find last bin with non-zero particles
        last_nonzero_idx = -1
        for idx_row, r in enumerate(chunk_rows):
            if r[2] > 0.0:
                last_nonzero_idx = idx_row

        total_count = sum(r[2] for r in chunk_rows)
        merged_rows = []

        if last_nonzero_idx == -1:
            # All bins empty: collapse to a single empty bin
            merged_rows.append(
                merge_group(
                    chunk_rows,
                    vols_ordered,
                    start=0,
                    end=num_chunks,
                    density_mass_idx=density_mass_idx,
                    density_num_idx=density_num_idx,
                    ncols=ncols,
                    new_chunk_index=1,
                    rmin_global=rmin_global,
                    dr_global=dr_global,
                )
            )
        else:
            # Valid region (bins 0..last_nonzero_idx) and invalid outer region
            idx = 0
            new_index = 1

            # Merge valid region based on min_N
            while idx <= last_nonzero_idx:
                group_start = idx
                N_acc = 0.0

                # Always include at least one row
                while idx <= last_nonzero_idx and (N_acc < min_N or idx == group_start):
                    N_acc += chunk_rows[idx][2]
                    idx += 1
                    if N_acc >= min_N:
                        break

                group_end = idx  # exclusive
                merged_rows.append(
                    merge_group(
                        chunk_rows,
                        vols_ordered,
                        start=group_start,
                        end=group_end,
                        density_mass_idx=density_mass_idx,
                        density_num_idx=density_num_idx,
                        ncols=ncols,
                        new_chunk_index=new_index,
                        rmin_global=rmin_global,
                        dr_global=dr_global,
                    )
                )
                new_index += 1

            # Merge invalid (outer empty) region into one empty bin
            if last_nonzero_idx + 1 < num_chunks:
                merged_rows.append(
                    merge_group(
                        chunk_rows,
                        vols_ordered,
                        start=last_nonzero_idx + 1,
                        end=num_chunks,
                        density_mass_idx=density_mass_idx,
                        density_num_idx=density_num_idx,
                        ncols=ncols,
                        new_chunk_index=new_index,
                        rmin_global=rmin_global,
                        dr_global=dr_global,
                    )
                )

        newM = len(merged_rows)

        # Write new header and merged rows for this timestep
        out_lines.append(f"{timestep} {newM} {int(total_count)}\n")
        for r in merged_rows:
            out_lines.append("  " + " ".join(f"{v:.6g}" for v in r) + "\n")

        i = j

    return out_lines


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Auto-merge radial bins in LAMMPS fix ave/chunk output so that "
            "each coarse bin has at least min_N particles (except possibly "
            "the last non-empty bin). Bins beyond the last non-zero Ncount "
            "are merged into one empty outer bin. Coord1 is set to the "
            "geometric center of the merged shell."
        )
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Input ave/chunk file"
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file (default: add _minN<min_N> before extension)",
    )
    parser.add_argument(
        "--min-N",
        dest="min_N",
        type=int,
        required=True,
        help="Minimum number of particles per coarse bin",
    )

    args = parser.parse_args()

    if args.min_N < 1:
        raise ValueError("min_N must be >= 1")

    in_path = args.input
    if args.output:
        out_path = args.output
    else:
        if "." in in_path:
            base, ext = in_path.rsplit(".", 1)
            out_path = f"{base}_minN{args.min_N}.{ext}"
        else:
            out_path = f"{in_path}_minN{args.min_N}"

    with open(in_path, "r") as f:
        lines = f.readlines()

    merged_lines = merge_ave_chunk_minN(lines, args.min_N)

    with open(out_path, "w") as f:
        f.writelines(merged_lines)

    print(f"Wrote merged file to {out_path}")


if __name__ == "__main__":
    main()
