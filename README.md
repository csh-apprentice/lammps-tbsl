# LAMMPS-TBSL

**LAMMPS-TBSL** is a fork of [LAMMPS](https://www.lammps.org) (August 2023 stable release) extended with custom components for **single-bubble sonoluminescence (SBSL)** simulations.  It couples a classical MD interior (ensemble of argon atoms) to a continuum Keller-Miksis bubble-wall ODE, enabling ab-initio-quality predictions of bubble dynamics, ionization, and thermal emission.

> This code was used to produce the results in:  
> *Shihan Cheng et al., "Molecular Dynamics Simulation of Single-Bubble Sonoluminescence", Journal of Chemical Physics (submitted)*

---

## New Features

### 1. Keller-Miksis Sphere Region — `region kmsphere`

The core of the simulation.  A spherical boundary whose radius evolves in real-time according to the **Keller-Miksis equation** (a Rayleigh-Plesset extension with acoustic radiation and compressibility):

$$\left(1 - \frac{\dot{R}}{c_l}\right) R\ddot{R} + \frac{3}{2}\dot{R}^2\left(1 - \frac{\dot{R}}{3c_l}\right) = \frac{1}{\rho_l}\left(1 + \frac{\dot{R}}{c_l}\right)\left[p_B - P_\infty(t)\right] + \frac{R}{\rho_l c_l}\dot{p}_B$$

The MD pressure $p_B$ is sampled from the atoms each timestep and fed back to the ODE integrator (4th-order Runge-Kutta).  A thermal boundary-layer ODE for the liquid temperature $T_{bl}$ and its thickness $\delta$ runs alongside.

**Thermo observables** (all accessible via `variable NAME equal func(all, km_sphere)`):

| Variable | Description |
|---|---|
| `slradius` | Bubble wall radius $R$ [Å] |
| `slvradius` | Wall velocity $\dot{R}$ [Å/ps] |
| `slcount` | Atom count inside bubble |
| `slpB` | MD bubble pressure $p_B$ [Pa·Å³/ensemble] |
| `slgastemp` | Gas temperature at the wall $T_b$ [K·scalefactor] |
| `slwalltemp` | Liquid temperature at wall $T_{bl}$ [K·scalefactor] |
| `sldelta` | Thermal boundary layer thickness $\delta$ [Å] |
| `slvdelta` | $\dot{\delta}$ [Å/ps] |
| `slnumatoms` | Atoms counted for pressure |
| `sldebug` | Internal debug flag |
| `slsteps` | KM integrator step counter (use with `halt` to stop at $10^7$ KM steps) |
| `thradius` | RK4 ODE state: $R$ [m] — needed for restart |
| `thvradius` | RK4 ODE state: $\dot{R}$ [m/s] — needed for restart |
| `tharadius` | RK4 ODE state: $\ddot{R}$ [m/s²] — needed for restart |
| `thTb0` | RK4 ODE state: $T_b$ [K] — needed for restart |
| `thdelta` | RK4 ODE state: $\delta$ [m] — needed for restart |

---

### 2. Molecular Dynamics Heat Bath Wall — `fix wall/mdhb`

A physically-motivated wall fix replacing the hard sphere boundary.  Atoms impinging on the bubble wall undergo **Maxwell–Boltzmann re-emission** with thermal accommodation coefficient $\alpha_t$:

- $\alpha_t = 1.0$: full thermalization (atom leaves at wall temperature $T_{bl}$)
- $\alpha_t = 0.0$: specular reflection
- $0 < \alpha_t < 1$: partial accommodation

Also applies a **Lennard-Jones 12-6 surface potential** for the liquid–gas interface.

```lammps
fix wall all wall/mdhb km_sphere lj126 <epsilon_w> <sigma_w> <cutoff> \
    <alpha_t> <scalefactor> <Cv_liquid> <cp_liquid> <T_inf> <T_blgas_init> <T_bl_init>
fix_modify wall virial yes
```

---

### 3. Ionization Pair Style — `pair lj/cutio/coul/dsf`

Extends standard LJ with a **sequential ionization ladder** for argon (up to Ar⁸⁺).  When two atoms approach within `cutoffio`, the kinetic energy is tested against ionization thresholds $I_1 \ldots I_8$.  Successfully ionized atoms gain charge and interact via **Damped Shifted Force (DSF)** Coulomb electrostatics—no Ewald summation needed.

```lammps
pair_style lj/cutio/coul/dsf <alpha_dsf> <cutoff_lj> <cutoff_coul> <cutoff_io>
pair_coeff * * 8 <I1> <I2> <I3> <I4> <I5> <I6> <I7> <I8> \
    <epsilon_lj> <sigma_lj> v_active <scalefactor>
```

`v_active` is an equal-style variable; set it to `step > restart_step` to suppress ionization at a restart's first step (see [Restart Tutorial](examples/TBSL/RESTART_TUTORIAL.md)).

A GPU-accelerated variant `pair lj/cutio/coul/dsf/gpu` is available; see [GPU Acceleration](#gpu-acceleration).

---

### 4. Restart Integrator — `run_style restartverlet`

Standard LAMMPS restart reads positions and velocities but not forces, making the first half-step of the Velocity-Verlet integrator incorrect.  `restartverlet` fixes this by **skipping `force_clear()`** at step 0, preserving forces loaded from the dump file.  Use it together with `read_sldump`.

```lammps
run_style restartverlet
```

---

### 5. Parallel Dump Loader — `read_sldump`

Reads the per-MPI-rank dump files written during a run—including forces `fx fy fz`—to fully restore the atomic state for restart.

```lammps
read_sldump ./dumpfiles/1e4_%.1000000.lammpstrj 1000000 \
    x y z vx vy vz fx fy fz q i_tlast i2_label d_mindist nfile 16
```

The `%` wildcard expands to the MPI rank index (0 … nfile-1).

---

### 6. Modified Initial Velocity Distribution — `velocity dist young`

Samples initial velocities from a spatially non-uniform Maxwell-Boltzmann distribution that matches the theoretical temperature profile inside the bubble at the start of the simulation ($T_b(r)$ decays from bulk gas temperature to the wall temperature over boundary layer thickness $\delta$).

```lammps
velocity all create <T_inf> <seed> dist young <Tb0> <Tbl> <A> <B> <eta> <R_in> <scalefactor>
```

---

## Ensemble Scaling

Real argon bubbles contain $N_\text{real} \sim 10^{10}$ atoms.  Direct simulation is intractable, so we simulate $N_\text{ensem} \in \{10^4, 10^6, 10^7, 10^8\}$ representative particles with scaled masses, interaction energies, and ionization thresholds:

| Quantity | Scaling | Rationale |
|---|---|---|
| Mass | $\times\, \text{sf}$ | preserve bulk density |
| LJ $\varepsilon$ | $\times\, \text{sf}$ | preserve pressure |
| Ionization thresholds | $\times\, \text{sf}$ | preserve energy per real atom |
| LJ $\sigma$, cutoff | $\times\, \text{sf}^{1/3}$ | preserve number density |
| Coulomb cutoff (fullrange) | $\times\, \text{sf}^{1/2}$ | preserve long-range screening |

where $\text{sf} = N_\text{real} / N_\text{ensem}$.

---

## Building

### CPU (MPI)

This repo follows standard LAMMPS CMake build conventions.  The custom files live entirely in `src/`; no changes to `CMakeLists.txt` are needed.

```bash
git clone https://github.com/csh-apprentice/lammps-tbsl.git
cd lammps-tbsl
mkdir build && cd build
cmake ../cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_MPI=yes \
      -DPKG_KSPACE=yes -DPKG_EXTRA-PAIR=yes
make -j$(nproc)
# binary: build/lmp
```

Tested on: GCC 11, OpenMPI 4.1, LAMMPS Aug 2023 base.

### GPU (CUDA)

```bash
mkdir build-gpu && cd build-gpu
cmake ../cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_MPI=yes \
      -DPKG_KSPACE=yes -DPKG_EXTRA-PAIR=yes \
      -DPKG_GPU=yes -DGPU_API=cuda -DGPU_ARCH=sm_86
make -j$(nproc)
# binary: build-gpu/lmp_gpu
```

Adjust `-DGPU_ARCH` to match your GPU (sm_80 for A100, sm_89 for RTX 4090, etc.).

**CUDA 12.4 + glibc 2.39 note** — if the build fails with `error: incomplete type is not allowed` on `sinpi`/`cospi`, this is handled automatically by `-Xcompiler -U_GNU_SOURCE` in the build system; no manual source edits are needed.

Tested on: NVIDIA RTX A6000, CUDA 12.4, GCC 11.

---

## Particle Initialization

Before running a simulation you need a LAMMPS data file with N atoms placed inside the bubble.
The script `tools/TBSL/generate_particles_lattice.py` generates this file by sampling positions
from a **radial probability distribution** that matches the theoretical density profile of the
gas inside the bubble at the start of the simulation:

$$p(r) \propto x_a \left(\frac{r}{R_b}\right)^4 + x_b \left(\frac{r}{R_b}\right)^2, \qquad x_a = -1.578,\; x_b = 3.947$$

Atoms are placed on a cubic lattice with spacing `h = 2 × d_ensem` (the hard-core diameter),
then N sites are drawn using the radial PDF weights.  A two-pass parallel threshold algorithm
handles large N efficiently on HPC clusters.

**Install dependencies** (once, into a conda environment named `TBSL`):

```bash
conda create -n TBSL python=3.10
conda activate TBSL
pip install -r tools/TBSL/requirements.txt
```

**Generate for any N:**

```bash
cd tools/TBSL

# 1e4 particles — fast, a few seconds on one node
python generate_particles_lattice.py \
  --N 10000 \
  --parallel --cpus 16 \
  --block 8 --tile 64 \
  --pilot_factor 2.0 --tau_slack 1.02 \
  --avoid_boundary \
  --output initialize_lattice_1e4.lammpsdata

# 1e6 particles — ~5 min on 128 cores
python generate_particles_lattice.py \
  --N 1000000 \
  --parallel --cpus 128 \
  --block 8 --tile 64 \
  --pilot_factor 2.0 --tau_slack 1.02 \
  --avoid_boundary \
  --output initialize_lattice_1e6.lammpsdata
```

On a SLURM cluster, use the provided script (adjust `--ntasks` to available cores):

```bash
sbatch tools/TBSL/run_intialize.sh
```

The `read_data` line in every `in.*` input file points to the corresponding `.lammpsdata` file,
so update that path after generating.

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--N` | 10000 | Number of ensemble particles |
| `--Rb` | 3.087×10⁴ Å | Initial bubble radius |
| `--avoid_boundary` | off | Exclude atoms within `d_ensem` of the wall |
| `--parallel --cpus K` | off | Use K parallel workers (recommended for N ≥ 10⁶) |
| `--output` | `initialize_lattice_radial_1e4.lammpsdata` | Output filename |

---

## Quick Start

A complete 1×10⁴-particle SBSL run (α_t = 1, short-range ionization, ~22 min on 16 cores):

```bash
cd examples/TBSL
# Edit the paths in run_lammps_1e6_alpha_0_shortio.sh to point to your binary
sbatch run_lammps_1e6_alpha_0_shortio.sh
```

Or for the validated 1e4 tutorial run:

```bash
cd test_1e4
sbatch run_1e4_alpha1_shortio.sh
```

Expected results (α_t = 1, N = 10⁴, Ar bubble, R₀ = 4.5 µm):

| Observable | Value |
|---|---|
| Minimum bubble radius | ~0.50 µm |
| Peak average temperature | ~11 500 K |
| Peak bubble pressure | ~4 GPa |
| Wall-clock time (16 MPI ranks) | ~22 min |

---

## GPU Acceleration

### Usage

Replace `pair lj/cutio/coul/dsf` with `pair lj/cutio/coul/dsf/gpu` and add `package gpu N neigh no` before it:

```lammps
package gpu 1 neigh no
pair_style lj/cutio/coul/dsf/gpu <alpha> <cutoff_lj> <cutoff_coul> <cutoff_io>
```

`neigh no` is mandatory: the ionization pipeline walks CPU-built neighbor lists, so GPU neighbor construction is not supported.

For multiple GPUs, set N in `package gpu N neigh no` to the number of physical GPUs, then launch with as many MPI ranks as CPU cores allow — not just one per GPU.  With `neigh no`, each MPI rank builds its own portion of the neighbor list on CPU; more ranks parallelise that work.  LAMMPS assigns GPUs to ranks round-robin:

```lammps
package gpu 2 neigh no          # 2 physical GPUs
```

```bash
mpirun -np 32 lmp_gpu -in in.simulation.lammps  # 32 ranks share 2 GPUs
```

Example inputs are in `test_1e4/in.simulation_1e4_alpha_1_shortio_gpu` and `test_1e4/in.restart_1e4_alpha_1_shortio_gpu`.  The restart procedure is identical to CPU — see [RESTART_TUTORIAL.md](examples/TBSL/RESTART_TUTORIAL.md).

### What differs from the CPU pair style

The GPU variant (`pair_lj_cutio_coul_dsf_gpu`) inherits from the CPU class.  Only the force loop is offloaded: LJ and DSF Coulomb forces are computed on the GPU via the LAMMPS Accelerator Library (LAL) kernel in `lib/gpu/lal_lj_cutio_dsf.cu`.  The three ionization phases — which require sequential per-atom state updates and MPI communication between phases — remain on CPU, unchanged.  `fix_gpu.cpp` was patched to recognise `run_style restartverlet` alongside the standard `verlet` integrator.

### Performance note

The GPU version is slower than CPU MPI for typical SBSL conditions.  The simulation spends most of its time in the dilute expansion phase, where ensemble scaling keeps the pair density constant (~0.04 neighbor pairs per atom) regardless of N.  At this occupancy, GPU force kernels are underutilised and the PCIe transfer overhead dominates.  Benchmarks on an RTX A6000 show the GPU is 1.7–14× slower than 32-rank CPU MPI depending on rank count.  GPU support is provided for environments where CPU parallelism is limited; CPU MPI is recommended for production runs.

---

## Restarting a Run

Long jobs are interrupted.  See [examples/TBSL/RESTART_TUTORIAL.md](examples/TBSL/RESTART_TUTORIAL.md) for the full step-by-step guide.  The short version:

```bash
# 1. Extract restart state from the thermo log at a dump checkpoint
python scaffold/read_restart.py test_1e4/run_1e4_alpha1_shortio.lammps 1000000

# 2. Paste output into test_1e4/in.restart_1e4_alpha_1_shortio (PASTE START/END block)

# 3. Submit
sbatch test_1e4/run_1e4_restart.sh

# 4. Verify (bit-for-bit for first 10k steps; peak quantities within ~5%)
python scaffold/compare_restart.py \
    test_1e4/run_1e4_alpha1_shortio.lammps \
    test_1e4/run_1e4_restart.lammps \
    1000000 933766.35
```

---

## Repository Layout

```
src/
  region_kmsphere.cpp/h         Keller-Miksis sphere boundary
  fix_wall_mdhb.cpp/h           MD heat bath wall
  pair_lj_cutio_coul_dsf.cpp/h  LJ + ionization + Coulomb DSF (CPU)
  GPU/
    lal_lj_cutio_dsf.cpp/h      GPU kernel wrapper for ionization pair style
    pair_lj_cutio_coul_dsf_gpu.cpp/h  GPU pair style (lj/cutio/coul/dsf/gpu)
    fix_gpu.cpp                 GPU package fix (patched for restartverlet)
  restartverlet.cpp/h           Restart-safe Verlet integrator
  read_sldump.cpp/h             Parallel dump loader for restart
  compute_reduce_slregion.cpp/h Per-region reduction compute
  variable.cpp                  sl*() / th*() observable functions
  velocity.cpp                  dist young initial conditions
  ... (standard LAMMPS source)

examples/TBSL/
  in.simulation_1e6_alpha_*     Production input scripts (1e6 particles)
  RESTART_TUTORIAL.md           Step-by-step restart guide
  history/                      All historical input scripts

test_1e4/
  in.simulation_1e4_alpha_1_shortio      Validated 1e4 benchmark input (CPU)
  in.simulation_1e4_alpha_1_shortio_gpu  Same benchmark, GPU variant
  in.restart_1e4_alpha_1_shortio         Restart input (CPU, step 1 000 000)
  in.restart_1e4_alpha_1_shortio_gpu     Restart input (GPU, step 1 000 000)
  run_1e4_alpha1_shortio.sh              SLURM CPU submission script
  run_1e4_alpha1_shortio_gpu.sh          SLURM GPU submission script
  run_1e4_restart.sh                     SLURM CPU restart script
  run_1e4_restart_gpu.sh                 SLURM GPU restart script

tools/TBSL/
  generate_particles_lattice.py  Radial-PDF lattice initializer
  run_intialize.sh               SLURM submission script for initialization
  requirements.txt               Python dependencies (numpy, tqdm)

scaffold/
  read_restart.py     Extract restart state from a thermo log
  compare_restart.py  Validate restart against original run (supports --gpu flag)
```

---

## Citation

If you use this code, please cite the LAMMPS paper and our work:

```bibtex
@article{thompson2022lammps,
  title={LAMMPS - A flexible simulation tool for particle-based materials modeling},
  author={Thompson, Aidan P and others},
  journal={Computer Physics Communications},
  volume={271},
  pages={108171},
  year={2022}
}
```

*(TBSL paper citation will be added upon publication)*
