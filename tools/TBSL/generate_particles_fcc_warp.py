#!/usr/bin/env python3
import numpy as np
from multiprocessing import Pool, cpu_count
import argparse, os, time, math
from typing import Tuple

# ----------------------------
# Physics helpers (kept)
# ----------------------------
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
    # Build monotone table y=Ft(s) over s∈[0,1]
    s = np.linspace(0.0, 1.0, table_n, dtype=np.float64)
    y = Ft(s)
    # Ensure monotonic
    y = np.clip(y, 0.0, 1.0)
    # Return inverse via interpolation: given u in [0,1], find s ~ Ft^{-1}(u)
    def inv(u: np.ndarray) -> np.ndarray:
        return np.interp(u, y, s).astype(np.float32)
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
    # Sphere volume
    V = 4.0/3.0 * math.pi * (Rb**3)
    # FCC number density = 4 / a^3 -> N ≈ V * 4/a^3  => a ≈ (4V/N)^(1/3)
    a = (4.0 * V / float(N)) ** (1.0/3.0)
    return a

def slab_ranges_for_k(a: float, Rb: float) -> Tuple[int,int]:
    # k index bounds so that z in [-Rb, Rb]
    # lattice points are at positions (i,j,k)*a + basis*a
    # Conservative bound for integer k range:
    kmin = math.floor((-Rb - a) / a) - 1
    kmax = math.ceil(( Rb + a) / a) + 1
    return kmin, kmax

def split_int_range(kmin:int, kmax:int, nproc:int, rank:int) -> Tuple[int,int]:
    total = (kmax - kmin + 1)
    base = total // nproc
    extra = total % nproc
    start_off = rank * base + min(rank, extra)
    count = base + (1 if rank < extra else 0)
    if count == 0: return 1, 0
    ks = kmin + start_off
    ke = ks + count - 1
    return ks, ke

# ----------------------------
# Worker: generate one z-slab, warp, and save chunk as .npy (float32)
# ----------------------------
def worker(args):
    (rank, nproc, N, Rb, min_dist, xa, xb, outdir, oversample) = args

    a = spacing_from_target_N(int(N * oversample), Rb)
    d_nn = a / math.sqrt(2.0)  # FCC nearest-neighbor spacing

    # Sanity: ensure we exceed requested min_dist
    if d_nn < min_dist:
        # If too tight, bump a so d_nn >= min_dist
        a = min_dist * math.sqrt(2.0)
        d_nn = a / math.sqrt(2.0)

    inv_cdf = build_inverse_cdf(xa, xb)
    kmin, kmax = slab_ranges_for_k(a, Rb)
    ks, ke = split_int_range(kmin, kmax, nproc, rank)
    if ke < ks:
        np.save(os.path.join(outdir, f"chunk_{rank}.npy"), np.empty((0,3), dtype=np.float32))
        np.save(os.path.join(outdir, f"bbox_{rank}.npy"), np.array([[0,0,0],[0,0,0]], dtype=np.float32))
        return (rank, 0, d_nn)

    # Prepare integer mesh for z-slab
    coords = []
    a64 = np.float64(a)
    Rb2 = (np.float64(Rb))**2

    # i,j bounds (conservative)
    imin = math.floor((-Rb - a) / a) - 1
    imax = math.ceil(( Rb + a) / a) + 1
    jmin, jmax = imin, imax

    for k in range(ks, ke+1):
        K = np.float64(k)
        for i in range(imin, imax+1):
            I = np.float64(i)
            for j in range(jmin, jmax+1):
                J = np.float64(j)
                # 4 basis points per cell
                cell = np.array([I, J, K], dtype=np.float64)
                pts = (cell + FCC_BASIS) * a64  # shape (4,3)
                # filter inside sphere
                r2 = np.sum(pts**2, axis=1)
                mask = r2 <= Rb2
                if not np.any(mask): continue
                coords.append(pts[mask])

    if len(coords) == 0:
        points = np.empty((0,3), dtype=np.float32)
    else:
        points = np.vstack(coords).astype(np.float32)  # uniform-in-volume FCC

    # Radial warp: r <- Rb * T( (r/Rb)^3 ), where T = Ft^{-1} and F0(s)=s^3 ⇒ u = s0^3
    if points.shape[0] > 0:
        r = np.linalg.norm(points, axis=1).astype(np.float64)
        s0 = (r / Rb)
        # handle r=0 to avoid 0/0 scale; leave origin unchanged
        u = np.clip(s0**3, 0.0, 1.0)
        s_target = inv_cdf(u).astype(np.float64)
        scale = np.ones_like(s0, dtype=np.float64)
        nz = s0 > 0
        scale[nz] = (s_target[nz] / s0[nz])
        points = (points.astype(np.float64) * scale[:,None]).astype(np.float32)

    # Save chunk & bbox
    chunk_path = os.path.join(outdir, f"chunk_{rank}.npy")
    np.save(chunk_path, points)
    if points.shape[0] > 0:
        lo = points.min(axis=0)
        hi = points.max(axis=0)
    else:
        lo = np.array([0,0,0], dtype=np.float32)
        hi = np.array([0,0,0], dtype=np.float32)
    np.save(os.path.join(outdir, f"bbox_{rank}.npy"), np.stack([lo,hi], axis=0))

    return (rank, points.shape[0], d_nn)

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=int(1e9))
    ap.add_argument("--Rb", type=float, default=3.086969205287614e+04)
    ap.add_argument("--min_dist", type=float, default=0.5)
    ap.add_argument("--xa", type=float, default=-1.5781)
    ap.add_argument("--xb", type=float, default=3.9468)
    ap.add_argument("--nproc", type=int, default=cpu_count())
    ap.add_argument("--outdir", type=str, default="chunks_np")
    ap.add_argument("--oversample", type=float, default=1.00, help="Generate ~N*oversample then (optionally) downsample in merge.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"[Gen] Target N={args.N:,}, Rb={args.Rb}, min_dist≥{args.min_dist}, nproc={args.nproc}, oversample={args.oversample}")

    t0 = time.time()
    work = [(rank, args.nproc, args.N, args.Rb, args.min_dist, args.xa, args.xb, args.outdir, args.oversample)
            for rank in range(args.nproc)]
    with Pool(processes=args.nproc) as pool:
        stats = pool.map(worker, work)
    elapsed = time.time() - t0

    total = sum(n for _, n, _ in stats)
    d_nn_list = [d for _,_,d in stats if d>0]
    d_nn = d_nn_list[0] if d_nn_list else float("nan")
    print(f"[Gen] Total generated in chunks: {total:,} (d_nn≈{d_nn:.3f}) in {elapsed/60:.2f} min")
    print(f"[Gen] Chunks written to: {args.outdir}")

if __name__ == "__main__":
    main()
