#!/usr/bin/env python3
"""
Merge radial bins in LAMMPS fix ave/chunk output.

- Assumes output format like:

  # Chunk-averaged data for fix ...
  # Timestep Number-of-chunks Total-count
  # Chunk Coord1 Ncount temp v_slcharge v_slvelocity v_slvirial density/mass density/number
  0 100 10000
    1 r1  N1  ...
    2 r2  N2  ...
    ...

- Assumes radial bins come from `bin/sphere` with uniform Δr:
    Coord1 ~ bin-center radius.

Merging:
  - You pass --merge-bin-num N (N > 1).
  - For each timestep, we merge consecutive groups of N fine bins:
      new_num_bins = ceil(old_num_bins / N)

  - For each coarse bin:
      N_tot   = sum Ncount_j
      V_tot   = sum V_j          (shell volumes)
      # number density
      density/number = N_tot / V_tot
      # mass density
      M_tot          = sum( rho_m_j * V_j )
      density/mass   = M_tot / V_tot

    Other properties (temp, etc.) are Ncount-weighted averages.

  - Coord1 (radial position) for each merged bin is set to the geometric
    center of the merged spherical shell:
        Coord1 = 0.5 * (r_in + r_out),
    where r_in is the inner radius of the first fine bin in the group and
    r_out is the outer radius of the last fine bin in the group.
"""

import argparse
import math
from typing import List


def merge_ave_chunk_lines(lines: List[str], merge_bin_num: int) -> List[str]:
    out_lines: List[str] = []
    i = 0
    n = len(lines)

    # --- Detect column names from header line: "# Chunk Coord1 Ncount ..."
    prop_names = None
    density_mass_idx = None
    density_num_idx = None

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            fields = stripped[1:].split()
            # Look specifically for the "Chunk Coord1 Ncount ..." header
            if len(fields) >= 3 and fields[0] == "Chunk" and fields[1] == "Coord1":
                prop_names = fields
                if "density/mass" in prop_names:
                    density_mass_idx = prop_names.index("density/mass")
                if "density/number" in prop_names:
                    density_num_idx = prop_names.index("density/number")
                break

    volumes_cache = None  # dict: chunk_id -> volume
    rmin_global = None    # inner radius of first fine bin
    dr_global = None      # radial bin width

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # Copy comments/empty lines as-is
        if not stripped or stripped.startswith("#"):
            out_lines.append(line)
            i += 1
            continue

        parts = stripped.split()
        if len(parts) < 3:
            out_lines.append(line)
            i += 1
            continue

        # Timestep header: "<timestep> <num_chunks> <total_count>"
        try:
            timestep = int(float(parts[0]))
            num_chunks = int(float(parts[1]))
        except ValueError:
            out_lines.append(line)
            i += 1
            continue

        # Read the chunk rows for this timestep
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
            # Something weird; just copy original block
            out_lines.append(line)
            out_lines.extend(lines[i + 1:j])
            i = j
            continue

        # Normalize row lengths (defensive)
        ncols = max(len(r) for r in chunk_rows)
        for r in chunk_rows:
            if len(r) < ncols:
                r.extend([0.0] * (ncols - len(r)))

        # Sort rows by chunk ID (col 0) to ensure radial order (inner -> outer)
        chunk_rows.sort(key=lambda r: r[0])

        # --- Build / reuse spherical-shell volumes keyed by chunk id
        if volumes_cache is None:
            M = len(chunk_rows)
            radii = [r[1] for r in chunk_rows]  # Coord1 column

            if M > 1:
                dr = (radii[-1] - radii[0]) / (M - 1)
            else:
                dr = radii[0] if radii[0] != 0 else 1.0

            rmin = radii[0] - 0.5 * dr
            # Fix tiny negative rmin ~ -0
            if rmin < 0 and abs(rmin) < 1e-6 * (radii[-1] or 1.0):
                rmin = 0.0

            vols = []
            for idx_row in range(M):
                r_in = rmin + idx_row * dr
                r_out = r_in + dr
                vol = 4.0 / 3.0 * math.pi * (r_out**3 - r_in**3)
                vols.append(vol)

            volumes_cache = {int(round(chunk_rows[k][0])): vols[k] for k in range(M)}
            rmin_global = rmin
            dr_global = dr

        vols_ordered = [volumes_cache.get(int(round(r[0])), 0.0) for r in chunk_rows]

        M = num_chunks
        N = max(1, merge_bin_num)
        newM = math.ceil(M / N)

        total_count = sum(r[2] for r in chunk_rows)  # original total N

        merged_rows = []

        for coarse_idx in range(newM):
            start = coarse_idx * N          # first fine-bin index in this group (0-based)
            end = min((coarse_idx + 1) * N, M)  # exclusive end index

            fine_rows = chunk_rows[start:end]
            fine_vols = vols_ordered[start:end]

            new_chunk_index = coarse_idx + 1

            counts = [r[2] for r in fine_rows]  # Ncount
            N_tot = sum(counts)
            V_tot = sum(fine_vols)

            merged = [0.0] * ncols
            merged[0] = float(new_chunk_index)

            # --- Coord1: geometric center of merged spherical shell ---
            # r_in = rmin + start * dr
            # r_out = rmin + end * dr     (since each fine bin has width dr)
            if rmin_global is not None and dr_global is not None:
                r_in = rmin_global + start * dr_global
                r_out = rmin_global + end * dr_global
                merged[1] = 0.5 * (r_in + r_out)
            else:
                # Fallback: average original centers (shouldn't happen if cache is set)
                merged[1] = sum(r[1] for r in fine_rows) / len(fine_rows)

            # Ncount is still total number of particles in the merged shell
            merged[2] = N_tot

            # --- Non-density properties: Ncount-weighted averages ---
            if N_tot > 0:
                for col in range(3, ncols):
                    if (density_mass_idx is not None and col == density_mass_idx) or \
                       (density_num_idx is not None and col == density_num_idx):
                        continue
                    merged[col] = sum(r[col] * r[2] for r in fine_rows) / N_tot
            else:
                for col in range(3, ncols):
                    if (density_mass_idx is not None and col == density_mass_idx) or \
                       (density_num_idx is not None and col == density_num_idx):
                        continue
                    merged[col] = 0.0

            # --- Density/number: N_tot / V_tot ---
            if density_num_idx is not None:
                if V_tot > 0:
                    merged[density_num_idx] = N_tot / V_tot
                else:
                    merged[density_num_idx] = 0.0

            # --- Density/mass: M_tot / V_tot, M_tot = sum(rho_m_j * V_j) ---
            if density_mass_idx is not None:
                if V_tot > 0:
                    M_tot = sum(r[density_mass_idx] * v
                                for r, v in zip(fine_rows, fine_vols))
                    merged[density_mass_idx] = M_tot / V_tot
                else:
                    merged[density_mass_idx] = 0.0

            merged_rows.append(merged)

        # Write new timestep header and merged rows
        out_lines.append(f"{timestep} {newM} {int(total_count)}\n")
        for r in merged_rows:
            out_lines.append("  " + " ".join(f"{v:.6g}" for v in r) + "\n")

        i = j

    return out_lines


def main():
    parser = argparse.ArgumentParser(
        description="Merge radial bins in LAMMPS fix ave/chunk output, "
                    "handling density/mass and density/number correctly, "
                    "and setting Coord1 to the geometric shell center."
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Input ave/chunk file")
    parser.add_argument("-o", "--output",
                        help="Output file (default: add _mergedN before extension)")
    parser.add_argument("-n", "--merge-bin-num", type=int, required=True,
                        help="Number of fine bins to merge into one coarse bin (N > 1)")
    args = parser.parse_args()

    if args.merge_bin_num < 1:
        raise ValueError("merge_bin_num must be >= 1")

    in_path = args.input
    if args.output:
        out_path = args.output
    else:
        if "." in in_path:
            base, ext = in_path.rsplit(".", 1)
            out_path = f"{base}_merged{args.merge_bin_num}.{ext}"
        else:
            out_path = f"{in_path}_merged{args.merge_bin_num}"

    with open(in_path, "r") as f:
        lines = f.readlines()

    merged_lines = merge_ave_chunk_lines(lines, args.merge_bin_num)

    with open(out_path, "w") as f:
        f.writelines(merged_lines)

    print(f"Wrote merged file to {out_path}")


if __name__ == "__main__":
    main()
