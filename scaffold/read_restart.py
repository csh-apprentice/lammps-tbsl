#!/usr/bin/env python3
"""
read_restart.py  —  Extract LAMMPS restart state from a thermo log.

Usage:
    python read_restart.py <log_file> <restart_step>

Arguments:
    log_file      Path to the LAMMPS thermo log (the .lammps SLURM output or
                  any file whose thermo lines start with the "Step" header).
    restart_step  The LAMMPS timestep to restart from.  Must be a step that
                  appears in the thermo output AND has a matching set of
                  per-rank dump files (i.e. a multiple of your dump frequency).

Output:
    Prints a LAMMPS-ready "rough* / formatted" variable block that you can
    paste directly into an in.restart_* input file.

Prerequisites for a valid restart step
---------------------------------------
  1. The step must be in the thermo log (printed every `thermo N` steps).
  2. Dump files must exist for that step — for a run with Nstep=500000 these
     are steps 0, 500000, 1000000, 1500000, ...
  3. The thermo log must include TH variables (v_THradius, v_THvradius,
     v_THaradius, v_THTb0, v_THdelta) and v_SLdebug — added in the updated
     in.simulation_* input files.

Example:
    python read_restart.py test_1e4/run_1e4_alpha1_shortio.lammps 1000000
"""

import sys
import os


def parse_lammps_log(filename):
    """Return (headers, rows) where rows is a list of dicts."""
    headers = None
    rows = []
    with open(filename) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # Detect a thermo header line (starts with "Step")
            if line.startswith("Step"):
                headers = line.split()
                continue
            if headers is None:
                continue
            parts = line.split()
            if len(parts) != len(headers):
                # End of thermo block (e.g. "Loop time..." line)
                headers = None
                continue
            try:
                row = {h: float(v) for h, v in zip(headers, parts)}
                rows.append(row)
            except ValueError:
                headers = None  # malformed line, reset
    return rows


def find_step(rows, step):
    """Return the row dict for the requested step, or None."""
    for row in rows:
        if int(row["Step"]) == int(step):
            return row
    return None


REQUIRED_COLS = [
    "Step", "Dt", "Time",
    "v_SLradius", "v_SLradiusold", "v_SLvradius",
    "v_SLdelta", "v_SLdeltaold", "v_SLvdelta",
    "v_SLTwall", "v_SLTblgas",
    "v_SLpB",
    "v_THradius", "v_THvradius", "v_THaradius", "v_THTb0", "v_THdelta",
]


def check_columns(row):
    missing = [c for c in REQUIRED_COLS if c not in row]
    if missing:
        print("ERROR: The following required columns are missing from the log:")
        for c in missing:
            print(f"  {c}")
        print()
        print("Make sure your simulation input uses the updated thermo_style that")
        print("includes v_THradius, v_THvradius, v_THaradius, v_THTb0, v_THdelta,")
        print("and thermo_modify format float \"%.31f\".")
        sys.exit(1)


def emit_block(row):
    sl_timestep   = int(row["Step"])
    reset_time    = row["Time"]
    reset_dt      = row["Dt"]
    slradius      = row["v_SLradius"]
    slradiusold   = row["v_SLradiusold"]
    slvraius      = row["v_SLvradius"]
    sldelta       = row["v_SLdelta"]
    sldeltaold    = row["v_SLdeltaold"]
    slvdelta      = row["v_SLvdelta"]
    slTbl         = row["v_SLTwall"]
    slTblgas      = row["v_SLTblgas"]
    pB_old_init   = row["v_SLpB"]
    th_radius     = row["v_THradius"]
    th_vradius    = row["v_THvradius"]
    th_aradius    = row["v_THaradius"]
    th_Tb0        = row["v_THTb0"]
    th_delta      = row["v_THdelta"]

    # sl_time is in seconds (LAMMPS Time column is in fs; *1e-15 → s)
    sl_time_expr  = f"${{reset_time}}*1e-15"

    print("# ============================================================")
    print(f"# Restart state extracted from step {sl_timestep}")
    print("# Paste this block into your in.restart_* input BEFORE the")
    print("# 'variable slradius format ...' lines.")
    print("# ============================================================")
    print()
    print(f"variable roughslradius    equal {slradius:.31f}")
    print(f"variable roughslradiusold equal {slradiusold:.31f}")
    print(f"variable roughslvraius    equal {slvraius:.31f}")
    print(f"variable roughsldelta     equal {sldelta:.31f}")
    print(f"variable roughsldeltaold  equal {sldeltaold:.31f}")
    print(f"variable roughslvdelta    equal {slvdelta:.31f}")
    print(f"variable roughslTbl       equal {slTbl:.31f}")
    print(f"variable roughslTblgas    equal {slTblgas:.31f}")
    print()
    print(f"variable roughth_radius   equal {th_radius:.31f}")
    print(f"variable roughth_vradius  equal {th_vradius:.31f}")
    print(f"variable roughth_aradius  equal {th_aradius:.31f}")
    print(f"variable roughth_Tb0      equal {th_Tb0:.31f}")
    print(f"variable roughth_delta    equal {th_delta:.31f}")
    print()
    print(f"variable sl_timestep      equal {sl_timestep:.1f}")
    print(f"variable reset_time       equal {reset_time:.31f}")
    print(f"variable sl_time          equal ${{reset_time}}*1e-15")
    print(f"variable reset_timestep   equal ${{sl_timestep}}")
    print(f"variable reset_dt         equal {reset_dt:.31f}")
    print(f"variable roughpB_old_init equal {pB_old_init:.31f}")
    print()
    print("# --- keep full precision by routing through 'format' ---")
    print("variable slradius     format roughslradius    %.31g")
    print("variable slradiusold  format roughslradiusold %.31g")
    print("variable slvraius     format roughslvraius    %.31g")
    print("variable sldelta      format roughsldelta     %.31g")
    print("variable sldeltaold   format roughsldeltaold  %.31g")
    print("variable slvdelta     format roughslvdelta    %.31g")
    print("variable slTbl        format roughslTbl       %.31g")
    print("variable slTblgas     format roughslTblgas    %.31g")
    print()
    print("variable th_radius    format roughth_radius   %.31g")
    print("variable th_vradius   format roughth_vradius  %.31g")
    print("variable th_aradius   format roughth_aradius  %.31g")
    print("variable th_Tb0       format roughth_Tb0      %.31g")
    print("variable th_delta     format roughth_delta    %.31g")
    print("variable pB_old_init  format roughpB_old_init %.31g")
    print()


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    log_file     = sys.argv[1]
    restart_step = int(sys.argv[2])

    if not os.path.isfile(log_file):
        print(f"ERROR: file not found: {log_file}")
        sys.exit(1)

    print(f"Parsing {log_file} ...", file=sys.stderr)
    rows = parse_lammps_log(log_file)
    if not rows:
        print("ERROR: no thermo data found in log file.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(rows)} thermo rows, steps "
          f"{int(rows[0]['Step'])} – {int(rows[-1]['Step'])}", file=sys.stderr)

    row = find_step(rows, restart_step)
    if row is None:
        available = sorted({int(r["Step"]) for r in rows})
        print(f"ERROR: step {restart_step} not found in log.", file=sys.stderr)
        print("Available steps (first 20):", available[:20], file=sys.stderr)
        sys.exit(1)

    check_columns(row)
    emit_block(row)


if __name__ == "__main__":
    main()
