#!/usr/bin/env python3
import numpy as np
import argparse, math, time, os, shutil
from multiprocessing import Pool, cpu_count

# ---------------- PDF / CDF over s = r/Rb ----------------
def pdf_s(s, xa, xb):
    return np.clip(xa * s**4 + xb * s**2, 0.0, None)

def inv_cdf_s_trunc(xa, xb, s_max=1.0, table_n=131072):
    """Robust numeric inverse CDF Ft^{-1} on s in [0, s_max] with renormalization."""
    s = np.linspace(0.0, float(s_max), table_n, dtype=np.float64)
    pdf = pdf_s(s, xa, xb)
    if s.size < 2 or pdf.max() == 0.0:
        # fallback: uniform
        def inv(u): return np.array(u, dtype=np.float64) * float(s_max)
        return inv
    ds = s[1] - s[0]
    # trapezoid cumulative
    cdf = np.cumsum((pdf[:-1] + pdf[1:]) * 0.5 * ds)
    cdf = np.concatenate(([0.0], cdf))
    total = cdf[-1]
    if total <= 0.0:
        # fallback: uniform
        def inv(u): return np.array(u, dtype=np.float64) * float(s_max)
        return inv
    cdf /= total
    cdf = np.maximum.accumulate(cdf); cdf[-1] = 1.0

    def inv(u):
        u = np.clip(u, 0.0, 1.0)
        return np.interp(u, cdf, s).astype(np.float64)
    return inv

# ---------------- counts & shells ----------------
def shell_counts(N, M):
    """Equal mass per shell → allocate exactly N via largest remainder."""
    base = N // M
    counts = np.full(M, base, dtype=np.int64)
    counts[: (N % M)] += 1
    return counts  # sums to N

def shell_edges_and_mid(sinv, M):
    u_edges = np.linspace(0.0, 1.0, M + 1, dtype=np.float64)
    s_edges = sinv(u_edges)
    s_mid   = sinv((u_edges[:-1] + u_edges[1:]) * 0.5)
    return s_edges, s_mid

# ---------------- random directions ----------------
def random_unit_vectors(n, seed):
    """Uniform directions on S^2 with a specific seed (reproducible)."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(-1.0, 1.0, size=n)
    theta = rng.uniform(0.0, 2.0*np.pi, size=n)
    r_xy = np.sqrt(np.clip(1.0 - z*z, 0.0, 1.0))
    x = r_xy * np.cos(theta); y = r_xy * np.sin(theta)
    return np.stack([x, y, z], axis=1)

# ---------------- conservative within-shell cap ----------------
def cap_within_shell(Nk, r, dmin):
    """
    Conservative bound so typical chord spacing ≳ dmin:
    N_max ~ 16 * r^2 / dmin^2  (equal-area spacing heuristic).
    """
    if dmin <= 0 or Nk <= 0: return Nk
    nmax = int(math.floor(16.0 * (r*r) / (dmin*dmin)))
    return min(Nk, max(nmax, 0))

# ---------------- mass scaling (kept) ----------------
def compute_m_ensem(N_ensem):
    pi = np.pi
    boltzmann = 1.380649e-23
    m = 39.95
    initialR = 4.5e-6
    Tinfty = 300
    Pinfty = 101325
    V = (4/3) * pi * (initialR**3)
    N = (Pinfty * V) / (boltzmann * Tinfty)
    scale = N / N_ensem
    return scale * m

# ---------------- old R_in formula (kept) ----------------
def compute_max_radius_in(N_ensem, R_b):
    """
    R_in = R_b - d_ensem, with d_ensem scaled by ensemble size (your old script).
    """
    d = 3.66
    initialR = 4.5e-6
    Tinfty = 300
    Pinfty = 101325
    boltzmann = 1.380649e-23
    pi = np.pi
    V = (4/3) * pi * (initialR ** 3)
    N = (Pinfty * V) / (boltzmann * Tinfty)
    scale = N / N_ensem
    rscale = scale ** (1 / 3.0)
    d_ensem = rscale * d
    return R_b - d_ensem

# ---------------- seed mixer (no NumPy overflow warnings) ----------------
def splitmix64_seed(base_seed: int, k: int) -> int:
    """Deterministic 64-bit seed derived from (base_seed, k) with SplitMix64."""
    mask = (1 << 64) - 1
    z = (int(base_seed) + (k + 1) * 0x9E3779B97F4A7C15) & mask
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B & mask
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & mask
    z = z ^ (z >> 31)
    return int(z & mask)

# ---------------- worker ----------------
def worker_write_shells(args_tuple):
    (k_start, k_end, counts_use, Rb, xa, xb, M, icdf_points,
     out_dir, start_ids, base_seed, s_max) = args_tuple

    sinv = inv_cdf_s_trunc(xa, xb, s_max=s_max, table_n=icdf_points)
    part_path = os.path.join(out_dir, f"part_{k_start}_{k_end}.atoms")

    # large buffer for fewer syscalls
    with open(part_path, "w", buffering=1<<20) as f:
        for k in range(k_start, k_end):
            Nk = int(counts_use[k])
            if Nk <= 0:
                continue

            # exact quantiles within this shell's CDF interval (on truncated [0,s_max])
            u0 = k / M; u1 = (k + 1) / M
            m_idx = np.arange(Nk, dtype=np.float64)
            uq = u0 + (m_idx + 0.5) * (u1 - u0) / Nk
            r_frac = sinv(uq)              # in [0, s_max]
            r  = r_frac * Rb               # physical radius

            # random directions (unique, reproducible per shell)
            seed_k = splitmix64_seed(base_seed, k)
            dirs = random_unit_vectors(Nk, int(seed_k))  # (Nk,3)

            xyz = dirs * r[:, None]

            gid0 = int(start_ids[k])  # 0-based start index for shell k
            # write with 1-based IDs: gid = gid0 + i + 1
            for i in range(Nk):
                p = xyz[i]
                gid = gid0 + i + 1
                f.write(f"{gid} 0 1 0 {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

    return (k_start, part_path)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1000000000)
    ap.add_argument("--Rb", type=float, default=3.086969205287614e4)
    ap.add_argument("--xa", type=float, default=-1.5781)
    ap.add_argument("--xb", type=float, default=3.9468)
    ap.add_argument("--orbits", type=int, default=10000)
    ap.add_argument("--min_dist", type=float, default=2.0)
    ap.add_argument("--enforce_min_in_shell", action="store_true",
                    help="Cap Nk per shell for typical chord spacing ≥ min_dist (conservative; may reduce total).")
    ap.add_argument("--icdf_points", "--icdfsize", dest="icdf_points", type=int, default=131072)
    ap.add_argument("--output", type=str, default="initialize_satellite_1e9.lammpsdata")
    ap.add_argument("--nproc", type=int, default=cpu_count())
    ap.add_argument("--seed", type=int, default=12345, help="Base seed for per-shell RNG streams.")

    # Boundary-avoidance options
    ap.add_argument("--avoid_boundary", action="store_true",
                    help="Restrict radii to r <= R_in computed as Rb - d_ensem (from old script).")
    ap.add_argument("--rin_override", type=float, default=None,
                    help="If set, use this explicit R_in (in same units as Rb) instead of auto Rb - d_ensem.")

    # Header control (optional): keep full box or shrink to R_in
    ap.add_argument("--header_use_rin", action="store_true",
                    help="Write x/y/z bounds as ±R_in instead of ±Rb when boundary avoidance is active.")

    args = ap.parse_args()

    t0 = time.time()
    M = int(args.orbits)
    if M <= 0:
        raise ValueError("--orbits must be positive")

    # Decide effective outer radius for sampling
    Rb = float(args.Rb)
    if args.rin_override is not None:
        R_in = float(args.rin_override)
        if not (0.0 < R_in <= Rb):
            raise ValueError("--rin_override must satisfy 0 < R_in <= Rb")
        print(f"[Boundary] Using user-specified R_in = {R_in:.6f} (Rb = {Rb:.6f})")
    elif args.avoid_boundary:
        R_in = compute_max_radius_in(args.N, Rb)
        if R_in <= 0:
            raise ValueError("Computed R_in <= 0; check parameters.")
        print(f"[Boundary] Auto R_in from old formula: R_in = {R_in:.6f} (Rb = {Rb:.6f})")
    else:
        R_in = Rb
        print(f"[Boundary] No boundary avoidance: using full radius Rb = {Rb:.6f}")

    s_max = float(R_in / Rb)

    # Build inverse CDF on [0, s_max] for diagnostics & mid radii
    sinv = inv_cdf_s_trunc(args.xa, args.xb, s_max=s_max, table_n=args.icdf_points)
    s_edges, s_mid = shell_edges_and_mid(sinv, M)
    r_edges = s_edges * Rb
    r_mid   = s_mid   * Rb
    dr = np.diff(r_edges)
    if np.any(dr < args.min_dist):
        print(f"[Warn] Some shell gaps Δr < min_dist ({args.min_dist}). "
              f"min Δr = {dr.min():.6f}. Increase --orbits or min_dist if cross-shell spacing matters.")
    else:
        if dr.size > 0:
            print(f"[Info] min Δr across shells = {dr.min():.6f} (>= {args.min_dist})")

    # Counts per shell; optional within-shell cap
    counts = shell_counts(args.N, M)  # sums to N
    if args.enforce_min_in_shell:
        counts_use = np.array(
            [cap_within_shell(int(Nk), float(r_mid[k]), float(args.min_dist))
             for k, Nk in enumerate(counts)],
            dtype=np.int64
        )
        reduced = int(counts_use.sum())
        if reduced < args.N:
            print(f"[Warn] enforce_min_in_shell reduced total from {args.N:,} to {reduced:,}.")
    else:
        counts_use = counts
        reduced = int(counts_use.sum())

    total_out = reduced
    print(f"[Plan] total atoms to write: {total_out:,} across {M} shells; sampling truncated at R_in = {R_in:.6f}")

    # Prefix sums → unique, contiguous 0-based starts per shell
    start_ids = np.zeros(M+1, dtype=np.int64)
    start_ids[1:] = np.cumsum(counts_use)
    start_ids = start_ids[:-1]  # length M

    # Prepare output & tmp dir
    out_tmp_dir = args.output + ".tmp_parts"
    os.makedirs(out_tmp_dir, exist_ok=True)

    # Partition shells into ~equal contiguous chunks
    nproc = max(1, int(args.nproc))
    splits = np.linspace(0, M, nproc + 1, dtype=int)
    tasks = []
    for i in range(nproc):
        ks, ke = splits[i], splits[i+1]
        if ks == ke: continue
        tasks.append((
            ks, ke, counts_use, Rb, args.xa, args.xb, M, args.icdf_points,
            out_tmp_dir, start_ids, args.seed, s_max
        ))

    # Run workers
    parts = []
    if len(tasks) == 1:
        parts.append(worker_write_shells(tasks[0]))
    else:
        with Pool(processes=len(tasks)) as pool:
            for res in pool.imap_unordered(worker_write_shells, tasks):
                parts.append(res)

    # Stitch: sort by k_start, then concat
    parts.sort(key=lambda t: t[0])  # by k_start
    part_paths = [p[1] for p in parts]

    # Header & Atoms
    m_ensem = compute_m_ensem(total_out if total_out > 0 else args.N)
    masses = {1: m_ensem}

    header_R = R_in if (args.avoid_boundary or args.rin_override is not None) and args.header_use_rin else Rb

    with open(args.output, "w", buffering=1<<20) as fout:
        fout.write("LAMMPS data file via write_data, generated by shells+random (parallel, boundary-aware)\n\n")
        fout.write(f"{total_out} atoms\n")
        fout.write(f"{len(masses)} atom types\n\n")
        fout.write(f"{-header_R:.15f} {header_R:.15f} xlo xhi\n")
        fout.write(f"{-header_R:.15f} {header_R:.15f} ylo yhi\n")
        fout.write(f"{-header_R:.15f} {header_R:.15f} zlo zhi\n\n")

        fout.write("Masses\n\n")
        for t, m in masses.items():
            fout.write(f"{t} {m:.6f}\n")

        fout.write("\nAtoms # full\n\n")
        for pf in part_paths:
            with open(pf, "r", buffering=1<<20) as fin:
                shutil.copyfileobj(fin, fout, length=1<<20)

        fout.write("\nVelocities\n\n")
        for i in range(1, total_out + 1):
            fout.write(f"{i} 0.0 0.0 0.0\n")

    # Cleanup
    for pf in part_paths:
        try: os.remove(pf)
        except: pass
    try: os.rmdir(out_tmp_dir)
    except: pass

    dt = time.time() - t0
    print(f"[Done] Wrote {args.output} with {total_out:,} atoms in {dt/60:.2f} min using nproc={len(tasks)}")

if __name__ == "__main__":
    main()
