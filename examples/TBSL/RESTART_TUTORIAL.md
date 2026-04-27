# Restarting a TBSL Simulation

Long SL simulations on HPC clusters are interrupted — job time limits, node failures, preemption.
This tutorial shows how to resume from a saved checkpoint so you lose at most one dump-file interval.

## What makes restart tricky in TBSL

Standard LAMMPS `read_restart` saves atom positions + velocities but **not forces**.
The Velocity-Verlet integrator needs the force at step *n* to correctly start step *n+1*:

```
v(n+½) = v(n) + F(n)·dt/2       ← needs F(n)
x(n+1) = x(n) + v(n+½)·dt
F(n+1) = eval_forces(x(n+1))
v(n+1) = v(n+½) + F(n+1)·dt/2
```

Without `F(n)`, the first half-step velocity is wrong, corrupting the energy.
TBSL addresses this with two mechanisms:

1. **`read_sldump`** — reads per-rank dump files that include `fx fy fz` columns.
   The forces written at the checkpoint are loaded back so the first half-step is exact.

2. **`run_style restartverlet`** — a modified Verlet integrator that skips `force_clear()` at
   step 0, so the loaded forces survive into the first half-step update.

The KM bubble-wall ODE integrator also carries state that must be re-seeded:
`th_radius`, `th_vradius`, `th_aradius`, `th_Tb0`, `th_delta` are the RK4 internal variables
stored in the `region kmsphere`.  These are logged at `%.31f` precision and re-injected via the
`rough* / format %.31g` pattern to avoid any precision loss.

---

## Step-by-step procedure (1e4 example)

### 1. Run the original simulation

```bash
cd /anvil/projects/x-phy250136/lammps-tbsl/test_1e4
sbatch run_1e4_alpha1_shortio.sh
```

This produces:
- `run_1e4_alpha1_shortio.lammps` — thermo log with TH columns at `%.31f`
- `dumpfiles/1e4_<rank>.<step>.lammpstrj` — per-rank atom snapshots every 500 000 steps

### 2. Choose a restart step

The restart step must satisfy **both** conditions:
- Present in the thermo log (every 1000 steps)
- Has matching dump files (every 500 000 steps for this run)

Valid choices: `500000`, `1000000`, `1500000`, `2000000`, `2500000`, `3000000`.

Pick a step **before** the collapse (step ~1 597 000) for the most interesting physics,
e.g. `restart_step = 1000000`.

### 3. Extract the restart state

```bash
cd /anvil/projects/x-phy250136/lammps-tbsl/test_1e4
python ../../scaffold/read_restart.py run_1e4_alpha1_shortio.lammps 1000000
```

Output (printed to stdout):
```
# Restart state extracted from step 1000000
variable roughslradius    equal 28547.123...
variable roughslradiusold equal 28547.123...
...
variable roughpB_old_init equal 234567.89...
variable slradius     format roughslradius    %.31g
...
```

### 4. Prepare the restart input

Open `in.restart_1e4_alpha_1_shortio` and:

a) Replace both occurrences of `RESTART_STEP` in the `read_sldump` line with your step number:
```lammps
read_sldump .../dumpfiles/1e4_%.1000000.lammpstrj 1000000 x y z vx vy vz fx fy fz q i_tlast i2_label d_mindist nfile 16
```

b) Paste the `read_restart.py` output between the `PASTE START / PASTE END` markers,
replacing all `PLACEHOLDER` values.

### 5. Submit the restart job

```bash
sbatch run_1e4_restart.sh
```

This produces `run_1e4_restart.lammps`.

### 6. Verify the restart

```bash
# The optional 4th argument is the scalefactor for converting raw Temp to Kelvin.
# For N=1e4 the scalefactor is ~933766.35 (printed at job startup).
python ../../scaffold/compare_restart.py \
    run_1e4_alpha1_shortio.lammps \
    run_1e4_restart.lammps \
    1000000 \
    933766.35
```

The script runs two validation phases:

**Phase 1 — near-term (first 10 000 steps after restart)**

Because LAMMPS MD is deterministic given identical atom forces and KM ODE state, the
restarted run should reproduce the original **bit-for-bit** for the first ~10 000 steps.
Relative differences below 1e-14 are last-bit rounding; anything above 1e-10 means the
restart state variables were not loaded correctly.

```
PHASE 1: PASS  (differences at last-bit floating-point level)
```

**Phase 2 — long-term physical observables**

MD is a chaotic system — last-bit rounding differences grow exponentially, and the two
trajectories will diverge at the atomic level over thousands of steps.  This is expected
and does **not** indicate a broken restart.  What matters is that macroscopic peak
quantities (minimum bubble radius, peak temperature) still agree within statistical
noise (~1–5% for N=1e4):

```
Observable                   Original          Restart    Rel diff
------------------------------------------------------------------------
v_SLradius (min)               5028 Å           5031 Å     6.0e-04
Temp/scalefactor (K) (max)    11505 K          11491 K     1.2e-03
```

---

## Key variables passed to `region kmsphere` at restart

| Variable | Source column | Physical meaning |
|---|---|---|
| `slradius` | `v_SLradius` | Bubble wall radius R(t) [Å] |
| `slradiusold` | `v_SLradiusold` | R(t-dt) — for finite-diff velocity |
| `slvraius` | `v_SLvradius` | dR/dt [Å/ps] |
| `sldelta` | `v_SLdelta` | Thermal boundary layer thickness δ [Å] |
| `sldeltaold` | `v_SLdeltaold` | δ(t-dt) |
| `slvdelta` | `v_SLvdelta` | dδ/dt |
| `slTblgas` | `v_SLTblgas` | Gas temperature at wall T_b [K] |
| `slTbl` | `v_SLTwall` | Liquid temperature just outside T_∞ᵢₙₜ [K] |
| `th_radius` | `v_THradius` | RK4 ODE state: R [m] |
| `th_vradius` | `v_THvradius` | RK4 ODE state: dR/dt [m/s] |
| `th_aradius` | `v_THaradius` | RK4 ODE state: d²R/dt² [m/s²] |
| `th_Tb0` | `v_THTb0` | RK4 ODE state: bubble gas temperature T_b [K] |
| `th_delta` | `v_THdelta` | RK4 ODE state: δ [m] |
| `pB_old_init` | `v_SLpB` | Bubble pressure at restart step [Pa·Å³] |
| `reset_time` | `Time` | Physical time at restart [fs] |
| `reset_dt` | `Dt` | Timestep at restart [fs] |

The `rough* / format %.31g` indirection preserves all 64-bit double precision bits
when the values travel through LAMMPS's variable string system.

---

## Why `variable active equal step>${sl_timestep}`

The ionization fix (`pair_lj/cutio`) must not fire at the restart step itself, because
the ion labels (`i2_label`) are already set in the dump.  Setting `active=0` at exactly
`step == sl_timestep` suppresses ionization for one step; from `sl_timestep+1` onward
`active=1` and ionization resumes normally.

---

## GPU restart

The procedure is identical to the CPU case.  Only three things differ:

1. **Use `lmp_gpu`** (the GPU-enabled binary).

2. **`nfile` must match the original GPU run** — a single-rank GPU run writes one dump file, so use `nfile 1`.

3. **Add `package gpu N neigh no`** at the top of the input (before `pair_style`).

```lammps
package gpu 1 neigh no
...
read_sldump .../dumpfiles/1e4_%.1000000.lammpstrj 1000000 \
    x y z vx vy vz fx fy fz q i_tlast i2_label d_mindist nfile 1
...
pair_style lj/cutio/coul/dsf/gpu <alpha> <cutoff_lj> <cutoff_coul> <cutoff_io>
run_style restartverlet
```

**Phase 1 tolerance for GPU:** Forces are computed in float32, so the restart is not bit-for-bit identical to the original run.  Relative differences of ~10⁻⁷ in KM ODE state variables (growing linearly with step count) are expected.  Pass `--gpu` to use the relaxed float32 threshold (1e-5 instead of 1e-10):

```bash
python ../../scaffold/compare_restart.py \
    full_gpu.log restart_gpu.log 1000000 933766.35 --gpu
```
