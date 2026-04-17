#!/usr/bin/env python3
import numpy as np
import argparse, math, time, sys

# ----------------------------
# Physics helpers (kept)
# ----------------------------
import random
from scipy.spatial import cKDTree

def random_fill_points(Rb, xa, xb, n_extra, min_dist, base_points, seed=12345, max_attempts=1000000):
    """
    Generate n_extra random points from the target radial PDF inside sphere,
    ensuring no overlaps with base_points or each other.
    """
    rng = np.random.default_rng(seed)
    Z = xa/5.0 + xb/3.0
    extras = []
    attempts = 0

    # Build KDTree for base FCC scaffold
    tree = cKDTree(base_points)

    # Will also maintain a small tree for extras as they accumulate
    extra_tree = None

    while len(extras) < n_extra and attempts < max_attempts:
        attempts += 1
        # sample r by rejection for s in [0,1]
        s = rng.random()
        u = rng.random()
        if u >= (xa*s**4 + xb*s**2) / Z:
            continue
        r = s * Rb
        vec = rng.normal(size=3)
        vec /= np.linalg.norm(vec)
        pt = vec * r

        # Check overlap against FCC + already accepted extras
        if tree.query_ball_point(pt, r=min_dist):
            continue
        if extra_tree is not None and extra_tree.query_ball_point(pt, r=min_dist):
            continue

        extras.append(pt)
        # Rebuild/initialize extras tree occasionally (simple & safe)
        if len(extras) == 1 or (len(extras) % 1024 == 0):
            extra_tree = cKDTree(np.array(extras))

    if len(extras) < n_extra:
        print(f"[Warn] Could only place {len(extras)} of {n_extra} requested extras (min_dist too strict).")

    return np.array(extras, dtype=np.float64)

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

def compute_max_radius_in(N_ensem, R_b):
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

# ----------------------------
# Target radial PDF & inverse CDF
# p(s) ∝ xa s^4 + xb s^2 on s∈[0,1]; we need T = Ft^{-1}∘F0 with F0(s)=s^3
# ----------------------------
def build_inverse_cdf(xa: float, xb: float, table_n: int = 4096):
    # Normalize p(s)
    Z = xa/5.0 + xb/3.0
    def Ft(s):
        return (xa/5.0 * s**5 + xb/3.0 * s**3) / Z
    s = np.linspace(0.0, 1.0, table_n, dtype=np.float64)
    y = Ft(s)
    y = np.clip(y, 0.0, 1.0)
    def inv(u: np.ndarray) -> np.ndarray:
        return np.interp(u, y, s).astype(np.float64)
    return inv

# ----------------------------
# FCC lattice utilities
# ----------------------------
FCC_BASIS = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.5, 0.5],
    [0.5, 0.0, 0.5],
    [0.5, 0.5, 0.0],
], dtype=np.float64)

def spacing_from_target_N(N: int, Rb: float) -> float:
    V = 4.0/3.0 * math.pi * (Rb**3)
    a = (4.0 * V / float(N)) ** (1.0/3.0)   # FCC: rho=4/a^3
    return a

def index_bounds(a: float, Rb: float):
    # Conservative integer bounds for i,j,k that cover the sphere
    imin = math.floor((-Rb - a) / a) - 1
    imax = math.ceil(( Rb + a) / a) + 1
    return imin, imax

# ----------------------------
# Generator for FCC points (inside sphere), yields warped points
# ----------------------------
def fcc_points_warped(Rb, a, inv_cdf):
    a64 = float(a)
    Rb2 = float(Rb) ** 2
    imin, imax = index_bounds(a, Rb)
    # Iterate i/j/k; generate 4 basis points per cell; keep inside sphere; apply radial warp; yield
    for k in range(imin, imax+1):
        for i in range(imin, imax+1):
            for j in range(imin, imax+1):
                cell = np.array([float(i), float(j), float(k)], dtype=np.float64)
                pts = (cell + FCC_BASIS) * a64  # (4,3)
                r2 = np.sum(pts**2, axis=1)
                mask = (r2 <= Rb2)
                if not np.any(mask):
                    continue
                sel = pts[mask]               # (M,3), M<=4
                # radial warp
                r = np.linalg.norm(sel, axis=1)  # float64
                s0 = r / Rb
                u = np.clip(s0**3, 0.0, 1.0)     # F0(s)=s^3
                # Inverse CDF of target
                s_target = inv_cdf(u)            # Ft^{-1}(u)
                scale = np.ones_like(s0)
                nz = s0 > 0
                scale[nz] = (s_target[nz] / s0[nz])
                sel = (sel * scale[:, None]).astype(np.float64)
                for p in sel:
                    yield p  # float64[3]

# ----------------------------
# Main (two-pass, with in-memory FCC scaffold)
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=int(1e4))
    ap.add_argument("--min_dist", type=float, default=0.4, help="Lower bound; actual NN distance from FCC spacing will be >= this.")
    ap.add_argument("--Rb", type=float, default=3.086969205287614e+04)
    ap.add_argument("--xa", type=float, default=-1.5781)
    ap.add_argument("--xb", type=float, default=3.9468)
    ap.add_argument("--output", type=str, default="initialize_fcc_1e6.lammpsdata")
    ap.add_argument("--exact_N", action="store_true", help="If set and total> N, thin to exactly N.")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    t0 = time.time()
    inv_cdf = build_inverse_cdf(args.xa, args.xb)

    # Pick FCC spacing from target N, then ensure d_nn >= min_dist
    a = spacing_from_target_N(args.N, args.Rb)
    d_nn = a / math.sqrt(2.0)
    if d_nn < args.min_dist:
        a = args.min_dist * math.sqrt(2.0)
        d_nn = a / math.sqrt(2.0)

    print(f"[Info] FCC a={a:.6f}, d_nn≈{d_nn:.6f} (>= min_dist {args.min_dist})")

    # -------- Pass 1: enumerate FCC, collect + bbox --------
    fcc_points_list = []
    glo = np.array([ np.inf,  np.inf,  np.inf], dtype=np.float64)
    ghi = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    for p in fcc_points_warped(args.Rb, a, inv_cdf):
        fcc_points_list.append(p)
        if p[0] < glo[0]: glo[0] = p[0]
        if p[1] < glo[1]: glo[1] = p[1]
        if p[2] < glo[2]: glo[2] = p[2]
        if p[0] > ghi[0]: ghi[0] = p[0]
        if p[1] > ghi[1]: ghi[1] = p[1]
        if p[2] > ghi[2]: ghi[2] = p[2]

    all_fcc_points = np.array(fcc_points_list, dtype=np.float64)
    total = all_fcc_points.shape[0]
    print(f"[Pass1] Found {total:,} lattice points inside sphere (after warp).")

    # Decide plan
    keepN = args.N
    need_extra = 0
    do_thin = False

    if total < args.N:
        need_extra = args.N - total
        print(f"[Pass1] FCC gave {total:,}, need {need_extra:,} more → will random-fill with min_dist={args.min_dist}.")
    elif total > args.N and args.exact_N:
        do_thin = True
        print(f"[Pass1] exact_N: will thin {total:,} → {args.N:,} using exact K/R.")
    else:
        keepN = total
        print(f"[Pass1] Will keep all FCC points (keepN={keepN:,}).")

    m_ensem = compute_m_ensem(keepN)
    masses = {1: m_ensem}

    # -------- Write header --------
    with open(args.output, "w") as f:
        f.write("LAMMPS data file via write_data, generated by FCC-warp (single-pass writer)\n\n")
        f.write(f"{keepN} atoms\n")
        f.write(f"{len(masses)} atom types\n\n")
        f.write(f"{glo[0]:.15f} {ghi[0]:.15f} xlo xhi\n")
        f.write(f"{glo[1]:.15f} {ghi[1]:.15f} ylo yhi\n")
        f.write(f"{glo[2]:.15f} {ghi[2]:.15f} zlo zhi\n\n")

        f.write("Masses\n\n")
        for t, m in masses.items():
            f.write(f"{t} {m:.6f}\n")

        f.write("\nAtoms # full\n\n")

        # -------- Pass 2: write atoms (stream) --------
        wrote = 0
        next_id = 1

        if do_thin:
            # Exact K/R selection over the FCC array (no duplicates, exactly N)
            rng = np.random.default_rng(args.seed)
            K = args.N
            R = total
            for p in all_fcc_points:
                if K <= 0: break
                # keep with prob K/R
                if rng.random() < (K / R):
                    f.write(f"{next_id} 0 1 0 {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
                    next_id += 1
                    wrote += 1
                    K -= 1
                R -= 1

        else:
            # Write all FCC points up to min(total, keepN)
            to_write = min(total, keepN)
            for i in range(to_write):
                p = all_fcc_points[i]
                f.write(f"{next_id} 0 1 0 {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
                next_id += 1
                wrote += 1

            # If we need random fill, enforce min_dist vs FCC + extras
            if need_extra > 0:
                extras = random_fill_points(args.Rb, args.xa, args.xb,
                                            need_extra, args.min_dist,
                                            base_points=all_fcc_points,
                                            seed=args.seed)
                # Debug output about extras
                if extras.shape[0] > 0:
                    radii_extras = np.linalg.norm(extras, axis=1) / args.Rb
                    print(f"[Debug] Extras: count={extras.shape[0]}, "
                          f"mean r/Rb={radii_extras.mean():.6f}, "
                          f"min={radii_extras.min():.6f}, max={radii_extras.max():.6f}")
                    if extras.shape[0] > 1:
                        dists = cKDTree(extras).query(extras, k=2)[0][:,1]  # NN within extras
                        print(f"[Debug] Extras min NN distance among themselves = {dists.min():.6f}")

                # Write extras
                for p in extras:
                    if wrote >= keepN:
                        break
                    f.write(f"{next_id} 0 1 0 {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
                    next_id += 1
                    wrote += 1

        # -------- Velocities (zeros) --------
        f.write("\nVelocities\n\n")
        for i in range(1, keepN+1):
            f.write(f"{i} 0.0 0.0 0.0\n")

    dt = time.time() - t0
    print(f"[Done] Wrote {args.output} with {keepN:,} atoms in {dt/60:.2f} min")
    print("[Note] No neighbor data during FCC pass; random-fill uses KDTree to respect min_dist.")
    if do_thin:
        print("[Note] Exact K/R thinning ensured exactly N atoms from FCC scaffold.")

if __name__ == "__main__":
    main()
