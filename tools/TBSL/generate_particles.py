import numpy as np
from scipy.spatial import cKDTree
from multiprocessing import Pool, cpu_count
import time
import argparse
from tqdm import tqdm

# ----------------------------
# PDF definition and samplers
# ----------------------------
def p_r(r, xa, xb):
    return xa * r**4 + xb * r**2

def sample_r(p_func, p_max):
    while True:
        r = np.random.uniform(0, 1)
        y = np.random.uniform(0, p_max)
        if y < p_func(r):
            return r

def sample_point(p_func, p_max, Rb):
    r = sample_r(p_func, p_max)
    vec = np.random.normal(0, 1, 3)
    vec /= np.linalg.norm(vec)
    return r * Rb * vec

# ----------------------------
# Physical property calculations
# ----------------------------
def compute_m_ensem(N_ensem):
    pi = np.pi
    boltzmann = 1.380649e-23
    NA = 6.02214076e23
    m = 39.95
    initialR = 4.5e-6
    Tinfty = 300
    Pinfty = 101325

    V = (4/3) * pi * (initialR**3)
    N = (Pinfty * V) / (boltzmann * Tinfty)
    scale = N / N_ensem
    m_ensem = scale * m
    return m_ensem

def compute_max_radius_in(N_ensem, R_b):
    pi = np.pi
    boltzmann = 1.380649e-23
    NA = 6.02214076e23
    m = 39.95
    d = 3.66
    initialR = 4.5e-6
    Tinfty = 300
    Pinfty = 101325

    V = (4/3) * pi * (initialR ** 3)
    N = (Pinfty * V) / (boltzmann * Tinfty)
    scale = N / N_ensem
    rscale = scale ** (1 / 3.0)
    d_ensem = rscale * d
    return R_b - d_ensem

# ----------------------------
# Worker functions
# ----------------------------
def generate_batch_kdtree(seed_offset, args, target, R_in):
    np.random.seed(seed_offset)
    p_func = lambda r: p_r(r, args.xa, args.xb)
    p_max = max(p_func(np.linspace(0, 1, 1000))) * 1.05

    accepted = []
    positions = []
    attempts = 0

    show_bar = (seed_offset % args.nproc == 0)
    with tqdm(total=target, desc=f"Core {seed_offset % args.nproc}", position=0, leave=False, disable=not show_bar) as pbar:
        while len(accepted) < target and attempts < args.max_attempts:
            pt = sample_point(p_func, p_max, R_in)
            attempts += 1
            if positions:
                tree = cKDTree(positions)
                if tree.query_ball_point(pt, r=args.min_dist):
                    continue
            accepted.append(pt)
            positions.append(pt)
            pbar.update(1)

    return {"positions": np.array(positions), "attempts": attempts, "inserted": len(positions)}

def generate_batch_grid(seed_offset, args, target, R_in):
    np.random.seed(seed_offset)
    p_func = lambda r: p_r(r, args.xa, args.xb)
    p_max = max(p_func(np.linspace(0, 1, 1000))) * 1.05

    cell_size = args.min_dist
    grid = {}
    accepted = []
    attempts = 0

    def grid_index(pt):
        return tuple((pt // cell_size).astype(int))

    show_bar = (seed_offset % args.nproc == 0)
    with tqdm(total=target, desc=f"Core {seed_offset % args.nproc}", position=0, leave=False, disable=not show_bar) as pbar:
        while len(accepted) < target and attempts < args.max_attempts:
            pt = sample_point(p_func, p_max, R_in)
            attempts += 1
            idx = grid_index(pt)
            found = False
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        neighbor = (idx[0]+dx, idx[1]+dy, idx[2]+dz)
                        for other in grid.get(neighbor, []):
                            if np.linalg.norm(pt - other) < args.min_dist:
                                found = True
                                break
                        if found: break
                    if found: break
                if found: continue
            accepted.append(pt)
            grid.setdefault(idx, []).append(pt)
            pbar.update(1)

    return {"positions": np.array(accepted), "attempts": attempts, "inserted": len(accepted)}

# ----------------------------
# LAMMPS saving format
# ----------------------------
def save_lammps_data_format(filename, positions, velocities, masses, timestep=0, atom_type=1, mol_id=0):
    N = positions.shape[0]
    assert positions.shape == velocities.shape == (N, 3)

    xlo, ylo, zlo = np.min(positions, axis=0)
    xhi, yhi, zhi = np.max(positions, axis=0)

    with open(filename, "w") as f:
        f.write(f"LAMMPS data file via write_data, timestep = {timestep}\n\n")
        f.write(f"{N} atoms\n")
        f.write(f"{len(masses)} atom types\n\n")
        f.write(f"{xlo:.15f} {xhi:.15f} xlo xhi\n")
        f.write(f"{ylo:.15f} {yhi:.15f} ylo yhi\n")
        f.write(f"{zlo:.15f} {zhi:.15f} zlo zhi\n\n")

        f.write("Masses\n\n")
        for t, m in masses.items():
            f.write(f"{t} {m:.6f}\n")

        f.write("\nAtoms # full\n\n")
        for i, pos in enumerate(positions, 1):
            f.write(f"{i} {mol_id} {atom_type} 0 {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")

        f.write("\nVelocities\n\n")
        for i, vel in enumerate(velocities, 1):
            f.write(f"{i} {vel[0]:.6f} {vel[1]:.6f} {vel[2]:.6f}\n")

# ----------------------------
# Utility
# ----------------------------
def distribute_workload(N, nproc):
    per_core = [N // nproc] * nproc
    for i in range(N % nproc):
        per_core[i] += 1
    return per_core

# ----------------------------
# Main function
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=int(1e9))
    parser.add_argument("--min_dist", type=float, default=0.5)
    parser.add_argument("--Rb", type=float, default=3.086969205287614e+04)
    parser.add_argument("--xa", type=float, default=-1.5781)
    parser.add_argument("--xb", type=float, default=3.9468)
    parser.add_argument("--max_attempts", type=int, default=int(1e11))
    parser.add_argument("--nproc", type=int, default=cpu_count())
    parser.add_argument("--mode", type=str, choices=["kdtree", "grid"], default="grid")
    args = parser.parse_args()

    print(f"Using {args.nproc} cores to generate {args.N} particles with min_dist={args.min_dist}")
    targets = distribute_workload(args.N, args.nproc)
    seeds = [int(time.time()) + i for i in range(args.nproc)]

    m_ensem = compute_m_ensem(args.N)
    R_in = compute_max_radius_in(args.N, args.Rb)
    print(f"Computed m_ensem = {m_ensem:.6f} g/mol")
    print(f"Computed MAX_RADIUS_IN = {R_in:.2f} Angstroms")

    worker_func = generate_batch_grid if args.mode == "grid" else generate_batch_kdtree
    
    t0 = time.time()
    with Pool(processes=args.nproc) as pool:
        results = pool.starmap(worker_func, [
            (seeds[i], args, targets[i], R_in) for i in range(args.nproc)
        ])
    elapsed = time.time() - t0

    all_pos = np.vstack([r["positions"] for r in results])
    total_attempts = sum(r["attempts"] for r in results)
    total_inserted = sum(r["inserted"] for r in results)

    print("\n--- Summary ---")
    print(f"Inserted particles: {total_inserted}")
    print(f"Total attempts:     {total_attempts}")
    print(f"Success rate:       {total_inserted / total_attempts:.4f}")
    print(f"Total time (s):     {elapsed:.2f}")
    print(f"Particles/sec:      {total_inserted / elapsed:.2f}")

    velocities = np.zeros_like(all_pos)
    masses = {1: m_ensem}
    filename = f"initialize_{args.N}.lammpsdata"

    save_lammps_data_format(
        filename=filename,
        positions=all_pos,
        velocities=velocities,
        masses=masses,
        atom_type=1,
        mol_id=0,
    )

    print(f"Data file saved to: {filename}")

if __name__ == "__main__":
    main()
