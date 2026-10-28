#!/usr/bin/env python3
"""
Smooth LAMMPS fix ave/chunk radial output based on a minimum number of
particles per effective bin, while keeping the original bin grid.

- The number of bins, chunk indices, and radii (Coord1) are preserved.
- Ncount (column 2) is also preserved.
- Intensive properties (temp, v_*, density/number, density/mass, etc.)
  are overwritten with group-averaged values computed over neighboring
  bins so that each group has at least min_N particles, except possibly
  the final group with particles.

Density columns are handled volume-correctly:

  density/number = (sum Ncount_i in group) / (sum V_i in group)
  density/mass   = (sum (rho_mass_i * V_i) in group) / (sum V_i in group)

Bins beyond the last non-zero Ncount are treated as "invalid" and share
one empty group:
  - densities = 0
  - other properties = 0
  (but they remain separate bins in the output)
"""

import argparse
import math
from typing import List, Optional, Tuple


def compute_shell_volumes(chunk_rows: List[List[float]]) -> List[float]:
    """Compute spherical shell volumes for each radial bin.

    Uses Coord1 (column 1) and assumes uniform Δr from bin/sphere.
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

    return vols


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


def smooth_group_apply(
    new_rows: List[List[float]],
    group_indices: List[int],
    vols: List[float],
    density_mass_idx: Optional[int],
    density_num_idx: Optional[int],
    ncols: int,
) -> None:
    """
    Compute group-averaged properties and apply them to all bins in the group,
    modifying new_rows in-place.

    - new_rows: copy of original chunk rows to be overwritten.
    - group_indices: indices of bins in this group.
    - vols: list of volumes aligned with chunk_rows indices.
    """
    if not group_indices:
        return

    group_rows = [new_rows[i] for i in group_indices]
    group_vols = [vols[i] for i in group_indices]

    counts = [r[2] for r in group_rows]
    N_tot = sum(counts)
    V_tot = sum(group_vols)

    density_cols = set()
    if density_mass_idx is not None:
        density_cols.add(density_mass_idx)
    if density_num_idx is not None:
        density_cols.add(density_num_idx)

    # Precompute group averages for non-density columns
    mean_vals = [0.0] * ncols

    if N_tot > 0.0:
        for col in range(3, ncols):
            if col in density_cols:
                continue
            mean_vals[col] = sum(r[col] * r[2] for r in group_rows) / N_tot
    else:
        for col in range(3, ncols):
            if col in density_cols:
                continue
            mean_vals[col] = 0.0

    # Densities
    rho_num = 0.0
    if density_num_idx is not None:
        if V_tot > 0.0:
            rho_num = N_tot / V_tot
        else:
            rho_num = 0.0

    rho_mass = 0.0
    if density_mass_idx is not None:
        if V_tot > 0.0:
            M_tot = sum(r[density_mass_idx] * v
                        for r, v in zip(group_rows, group_vols))
            rho_mass = M_tot / V_tot
        else:
            rho_mass = 0.0

    # Apply to all bins in the group (keep index, coord, and Ncount)
    for idx in group_indices:
        row = new_rows[idx]
        for col in range(3, ncols):
            if col == density_num_idx:
                row[col] = rho_num
            elif col == density_mass_idx:
                row[col] = rho_mass
            else:
                row[col] = mean_vals[col]


def smooth_ave_chunk_minN_keepbins(lines: List[str], min_N: int) -> List[str]:
    out_lines: List[str] = []
    i = 0
    n = len(lines)

    # Parse header line with column names
    prop_names = None
    for line in lines:
        s = line.strip()
        if s.startswith("#"):
            fields = s[1:].split()
            if len(fields) >= 3 and fields[0] == "Chunk" and fields[1] == "Coord1":
                prop_names = fields
                break

    density_mass_idx, density_num_idx = detect_density_columns(prop_names)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # Copy comments/blank lines as-is
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
            # Inconsistent block; copy as-is
            out_lines.append(line)
            out_lines.extend(lines[i + 1:j])
            i = j
            continue

        # Normalize row lengths
        ncols = max(len(r) for r in chunk_rows)
        for r in chunk_rows:
            if len(r) < ncols:
                r.extend([0.0] * (ncols - len(r)))

        # Sort by chunk index (col 0) to ensure center→outward order
        chunk_rows.sort(key=lambda r: r[0])

        # Copy rows to modify in-place
        new_rows = [r[:] for r in chunk_rows]

        # Compute shell volumes for this timestep
        vols = compute_shell_volumes(chunk_rows)

        # Find last bin with non-zero Ncount
        last_nonzero_idx = -1
        for idx_row, r in enumerate(chunk_rows):
            if r[2] > 0.0:
                last_nonzero_idx = idx_row

        total_count = sum(r[2] for r in chunk_rows)

        if last_nonzero_idx == -1:
            # All bins empty: just zero intensive properties, keep grid
            group_indices = list(range(num_chunks))
            smooth_group_apply(
                new_rows,
                group_indices,
                vols,
                density_mass_idx,
                density_num_idx,
                ncols,
            )
        else:
            # Valid region: [0 .. last_nonzero_idx]
            idx = 0
            n_valid = last_nonzero_idx + 1

            while idx < n_valid:
                group_indices = []
                N_acc = 0.0

                # Always include at least one bin
                while idx < n_valid and (N_acc < min_N or len(group_indices) == 0):
                    group_indices.append(idx)
                    N_acc += chunk_rows[idx][2]
                    idx += 1
                    if N_acc >= min_N:
                        break

                # If we reached n_valid with N_acc < min_N, we still accept
                smooth_group_apply(
                    new_rows,
                    group_indices,
                    vols,
                    density_mass_idx,
                    density_num_idx,
                    ncols,
                )

            # Invalid outer region: merge as one empty group
            if last_nonzero_idx + 1 < num_chunks:
                group_indices = list(range(last_nonzero_idx + 1, num_chunks))
                smooth_group_apply(
                    new_rows,
                    group_indices,
                    vols,
                    density_mass_idx,
                    density_num_idx,
                    ncols,
                )

        # Write timestep header (same number of chunks as original)
        out_lines.append(f"{timestep} {num_chunks} {int(total_count)}\n")

        # Write smoothed rows, restoring original order by chunk index
        # (chunk_rows / new_rows are already sorted by index)
        for r in new_rows:
            out_lines.append("  " + " ".join(f"{v:.6g}" for v in r) + "\n")

        i = j

    return out_lines


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Smooth radial ave/chunk data so each effective bin has at least "
            "min_N particles, while keeping the original bin grid. "
            "Only intensive properties are overwritten."
        )
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Input ave/chunk file"
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file (default: add _smoothedMinN<min_N> before extension)",
    )
    parser.add_argument(
        "--min-N",
        dest="min_N",
        type=int,
        required=True,
        help="Minimum number of particles per effective bin",
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
            out_path = f"{base}_smoothedMinN{args.min_N}.{ext}"
        else:
            out_path = f"{in_path}_smoothedMinN{args.min_N}"

    with open(in_path, "r") as f:
        lines = f.readlines()

    smoothed_lines = smooth_ave_chunk_minN_keepbins(lines, args.min_N)

    with open(out_path, "w") as f:
        f.writelines(smoothed_lines)

    print(f"Wrote smoothed file to {out_path}")


if __name__ == "__main__":
    main()
