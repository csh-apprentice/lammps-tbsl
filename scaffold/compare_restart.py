#!/usr/bin/env python3
"""
compare_restart.py  —  Verify that a restarted run matches the original.

Usage:
    python compare_restart.py <original_log> <restart_log> <restart_step> [scalefactor] [--gpu]

Two-phase validation:
  PHASE 1 — Near-term (first ~10 000 steps after restart):
    CPU (double precision): the restarted run should reproduce the original
    bit-for-bit.  Relative differences below ~1e-14 are expected; anything
    above 1e-10 suggests the restart state was not loaded correctly.

    GPU (float32 forces): force calculations use float32, so bit-for-bit
    agreement is not possible.  Pass --gpu to use a relaxed threshold of
    1e-5.  Differences of ~1e-7 growing linearly are normal; values above
    1e-5 indicate a restart state problem.

  PHASE 2 — Long-term (full overlap):
    MD is a chaotic system — tiny rounding differences grow exponentially over
    time.  The restarted run will diverge from the original trajectory, but key
    physical observables (minimum bubble radius, peak temperature) should agree
    to within statistical noise (~1–5% for N=1e4).  This validates that the
    KM ODE and thermodynamic state were correctly restored even after the runs
    diverge at the atomic level.

Arguments:
    original_log   SLURM .lammps output from the uninterrupted run.
    restart_log    SLURM .lammps output from the restart run.
    restart_step   LAMMPS timestep at which the restart began.
    scalefactor    Optional: divide Temp by this for physical K (e.g. 933766.35).
    --gpu          Use relaxed float32 tolerance (1e-5) for Phase 1.

Example (CPU):
    python compare_restart.py \\
        test_1e4/run_1e4_alpha1_shortio.lammps \\
        test_1e4/run_1e4_restart.lammps \\
        1000000

Example (GPU):
    python compare_restart.py \\
        test_local/full_gpu.log \\
        test_local/restart_gpu_full.log \\
        1000000 933766.35 --gpu
"""

import sys
import os

# KM ODE state columns — govern restart correctness; must match closely.
# CPU: threshold 1e-10 (double precision). GPU: threshold 1e-5 (float32 accumulation).
NEARTERM_ODE_COLS = [
    "v_SLradius",
    "v_SLvradius",
    "v_SLdelta",
    "v_THradius",
    "v_THaradius",
]

# Instantaneous ensemble observables — NOT used for pass/fail in GPU mode.
# GPU force accumulation is non-deterministic (CUDA atomic ops + float32), so
# these differ by ~1e-4 even between two fresh GPU runs from identical initial
# conditions.  They are shown for information only.
NEARTERM_INSTANT_COLS = [
    "v_SLpB",
    "Temp",
]

NEARTERM_COLS = NEARTERM_ODE_COLS + NEARTERM_INSTANT_COLS

# Scalar quantities for long-term physical validation
# Each entry: (column, aggregation)  where aggregation is 'min' or 'max'
LONGTERM_CHECKS = [
    ("v_SLradius",  "min"),   # minimum bubble radius → collapse
    ("Temp",        "max"),   # peak temperature
    ("v_SLpB",      "max"),   # peak bubble pressure
    ("v_SLTblgas",  "max"),   # peak gas temperature at wall
]

NEARTERM_STEPS = 10_000   # steps to consider "short-term"


def parse_lammps_log(filename):
    headers = None
    rows = []
    with open(filename) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("Step"):
                headers = line.split()
                continue
            if headers is None:
                continue
            parts = line.split()
            if len(parts) != len(headers):
                headers = None
                continue
            try:
                row = {h: float(v) for h, v in zip(headers, parts)}
                rows.append(row)
            except ValueError:
                headers = None
    return rows


def rows_from_step(rows, step):
    return {int(r["Step"]): r for r in rows if int(r["Step"]) >= step}


def rel_diff(a, b):
    denom = max(abs(a), abs(b), 1e-300)
    return abs(a - b) / denom


def phase1_nearterm(orig_map, restart_map, restart_step, gpu_mode=False):
    limit = restart_step + NEARTERM_STEPS
    # Skip the exact restart step: at step 0 of the restart, v_SLpB reports
    # the injected pB_old_init rather than the freshly-computed pressure,
    # so it will legitimately differ from the original by construction.
    near_steps = sorted(s for s in set(orig_map) & set(restart_map)
                        if restart_step < s <= limit)
    if not near_steps:
        print("  (no common steps found in the near-term window)")
        return

    # GPU uses float32 forces; bit-for-bit agreement is impossible.
    # Differences of ~1e-7 growing linearly are expected and correct.
    threshold = 1e-5 if gpu_mode else 1e-10

    col_w = 14
    header = f"{'Step':>12}  " + "  ".join(f"{c:>{col_w}}" for c in NEARTERM_COLS)
    print(header)
    print("-" * len(header))

    max_diffs = {c: 0.0 for c in NEARTERM_COLS}
    for step in near_steps:
        o, r = orig_map[step], restart_map[step]
        diffs = []
        for c in NEARTERM_COLS:
            if c in o and c in r:
                d = rel_diff(o[c], r[c])
                max_diffs[c] = max(max_diffs[c], d)
                diffs.append(f"{d:{col_w}.3e}")
            else:
                diffs.append(f"{'N/A':>{col_w}}")
        print(f"{step:>12}  " + "  ".join(diffs))

    print()
    # Pass/fail is based on KM ODE state columns only; instantaneous observables
    # are non-deterministic on GPU due to float32 atomic accumulation.
    ode_worst = max((max_diffs.get(c, 0.0) for c in NEARTERM_ODE_COLS), default=0.0)
    print(f"  Max rel-diff over first {NEARTERM_STEPS} steps:")
    for c in NEARTERM_COLS:
        if gpu_mode and c in NEARTERM_INSTANT_COLS:
            note = "  (GPU non-deterministic, not used for pass/fail)"
        else:
            note = f"  <-- PROBLEM (>{threshold:.0e})" if max_diffs[c] > threshold else ""
        print(f"    {c:<22} {max_diffs[c]:.3e}{note}")
    print()
    if gpu_mode:
        if ode_worst < threshold:
            print("  PHASE 1: PASS (GPU float32 — KM ODE state differences consistent with float32 arithmetic)")
        else:
            print(f"  PHASE 1: FAIL  — KM ODE state differences above {threshold:.0e} (GPU float32 threshold).")
            print("           Check that rough* values were pasted correctly and that")
            print("           run_style restartverlet is used.")
    else:
        worst = max(max_diffs.values())
        if worst < 1e-14:
            print("  PHASE 1: BIT-FOR-BIT IDENTICAL  (restart is exact)")
        elif worst < threshold:
            print("  PHASE 1: PASS  (differences at last-bit floating-point level)")
        else:
            print(f"  PHASE 1: FAIL  — differences above {threshold:.0e} in the first 10 000 steps.")
            print("           Check that rough* values were pasted correctly and that")
            print("           run_style restartverlet is used.")


def phase2_longterm(orig_rows, restart_rows, restart_step, scalefactor=None):
    # Only rows after restart_step
    orig_tail    = [r for r in orig_rows    if int(r["Step"]) >= restart_step]
    restart_tail = [r for r in restart_rows if int(r["Step"]) >= restart_step]

    if not orig_tail or not restart_tail:
        print("  (not enough rows for long-term comparison)")
        return

    print(f"  {'Observable':<28} {'Original':>16}  {'Restart':>16}  {'Rel diff':>10}")
    print("  " + "-" * 76)
    for col, agg in LONGTERM_CHECKS:
        if col not in orig_tail[0] or col not in restart_tail[0]:
            continue
        fn = min if agg == "min" else max
        orig_val    = fn(r[col] for r in orig_tail    if col in r)
        restart_val = fn(r[col] for r in restart_tail if col in r)

        # Convert Temp to physical K using scalefactor if provided
        label = col
        if col == "Temp" and scalefactor:
            orig_val    /= scalefactor
            restart_val /= scalefactor
            label = "Temp/scalefactor (K)"

        d = rel_diff(orig_val, restart_val)
        flag = "  <-- LARGE" if d > 0.10 else ""
        print(f"  {label:<28} {orig_val:>16.4g}  {restart_val:>16.4g}  {d:>10.3e}{flag}")

    print()
    print("  Long-term divergence is expected for MD (chaotic system).")
    print("  Peak quantities should agree within ~5% for N=1e4.")


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    args = sys.argv[1:]
    gpu_mode = "--gpu" in args
    args = [a for a in args if a != "--gpu"]

    orig_file    = args[0]
    restart_file = args[1]
    restart_step = int(args[2])
    # Optional: scalefactor for Temp conversion (e.g. 933766.35 for N=1e4)
    scalefactor = float(args[3]) if len(args) > 3 else None

    for f in (orig_file, restart_file):
        if not os.path.isfile(f):
            print(f"ERROR: file not found: {f}")
            sys.exit(1)

    print(f"Parsing {orig_file} ...", file=sys.stderr)
    orig_rows = parse_lammps_log(orig_file)
    print(f"Parsing {restart_file} ...", file=sys.stderr)
    restart_rows = parse_lammps_log(restart_file)

    orig_map    = rows_from_step(orig_rows,    restart_step)
    restart_map = rows_from_step(restart_rows, restart_step)

    common_steps = sorted(set(orig_map) & set(restart_map))
    if not common_steps:
        print("ERROR: no common steps found at or after the restart step.")
        sys.exit(1)

    print(f"\nOriginal:  {len(orig_rows)} thermo rows, "
          f"steps {int(orig_rows[0]['Step'])} – {int(orig_rows[-1]['Step'])}")
    print(f"Restart:   {len(restart_rows)} thermo rows, "
          f"steps {int(restart_rows[0]['Step'])} – {int(restart_rows[-1]['Step'])}")
    print(f"Overlap:   {len(common_steps)} common steps, "
          f"{common_steps[0]} – {common_steps[-1]}")

    if gpu_mode:
        print("  [GPU mode: using relaxed float32 tolerance for Phase 1]")

    print(f"\n{'='*70}")
    print(f"PHASE 1 — Near-term {'float32' if gpu_mode else 'bit-for-bit'} check (first {NEARTERM_STEPS} steps)")
    print(f"{'='*70}")
    phase1_nearterm(orig_map, restart_map, restart_step, gpu_mode=gpu_mode)

    print(f"{'='*70}")
    print(f"PHASE 2 — Long-term physical observables (all overlapping steps)")
    print(f"{'='*70}")
    phase2_longterm(orig_rows, restart_rows, restart_step, scalefactor)


if __name__ == "__main__":
    main()
