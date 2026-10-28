# COM ionization: CPU deployment and validation

This revision contains the COM-frame ionization correction, owner/ghost
communication updates, CPU/GPU shared reaction pipeline, and portable synthetic
tests. The analysis tools from the earlier revision are also present under
`tools/TBSL/analysis`. Raw bubble output and historical regression fixtures are
not included in this source deployment.

## Build on Anvil

Use the matching C++ compiler, MPI and CMake environment for both build and run.
Configure a fresh directory to avoid mixing an old executable or GPU build into
this CPU check. Run from the repository root:

```sh
cmake -S unittest/tbsl -B build-anvil-com \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_MPI=ON -DBUILD_OMP=OFF \
  -DPKG_GPU=OFF -DPKG_MOLECULE=ON
cmake --build build-anvil-com --parallel 16 \
  --target lmp test_ionization_pair tbsl_custom_restart_protocol tbsl_review_probes
```

The simulation executable is **`build-anvil-com/lammps/lmp`**. MOLECULE is needed
for the production template's `atom_style full`; the unit fixtures use charge
atoms. The test executables are directly under `build-anvil-com/`.

The existing SLURM scripts still reference `src/lmp_mpi`. Change the executable
path in the job you actually submit to the absolute path of the newly built
`build-anvil-com/lammps/lmp`. Compilation does not update `src/lmp_mpi`.

Run the tiny MPI checks inside an allocation using the site's normal launcher;
for a compatible mpiexec environment:

```sh
ctest --test-dir build-anvil-com --output-on-failure
mpiexec -n 8 build-anvil-com/test_ionization_pair
mpiexec -n 16 build-anvil-com/test_ionization_pair

mkdir -p build-anvil-com/restart-check
mpiexec -n 16 build-anvil-com/tbsl_custom_restart_protocol \
  build-anvil-com/restart-check

mpiexec -n 16 build-anvil-com/tbsl_review_probes integrated
mpiexec -n 16 build-anvil-com/tbsl_review_probes charge_q8
```

The restart fixture requires an existing writable output directory, unique to
that execution. It records both the guarded positive control and an intentionally
unguarded negative control; only the guarded path determines its exit status.
No command above submits a batch job or starts a coupled bubble trajectory.

Record `git rev-parse HEAD`, the input SHA256 values and the executable SHA256
with each job. [source_sha256.json](source_sha256.json) identifies the checked
production files. A remote rebuild will generally have a different binary hash
from a local build with another toolchain.

## Physical and numerical checks

With masses in g/mol and velocities in Angstrom/fs, the shared collision helper
returns `E_rel = 0.5e4 * mu * |vi-vj|^2` in kJ/mol, with
`mu = mi*mj/(mi+mj)`. Both selection and execution require `E_rel > chi`.
An accepted event updates `v -> V + s*(v-V)`, where V is the pair COM velocity
and `s = sqrt((2/3)*(1-chi/E_rel))`. Momentum is unchanged and the resolved
kinetic loss is `chi + (E_rel-chi)/3`, including the approximate electron
thermalization in Ruuth et al., PRE 66, 036310 (2002), Sec. II.G.

The October 28 local validation completed:

- 19 cases, 202 checks reported by rank zero, zero failures on each of
  1/2/4/8/16 MPI ranks. Cases cover boost invariance, thresholds, unequal mass,
  EP scaling, competing candidates, local index zero, charge saturation,
  migration, half/full lists and Newton on/off.
- Release-library tests passed on 1/16 ranks. The same source passed ASan+UBSan
  on 1/2/4/8/16 ranks, with the pair and fixture instrumented and the remainder
  of LAMMPS uninstrumented; MPI leak detection was disabled.
- Actual NVE cross-rank updates and owner migration passed on 16 ranks:
  444 summed rank-local assertions. q=8 cases passed with 192 summed rank-local
  assertions. Counts include repeated rank-local checks, not independent samples.
- The author's guarded `read_sldump + restartverlet + active=step>checkpoint`
  two-particle continuation passed 30 state checks on each of 1/2/4/8/16 ranks;
  recorded final x/v/f/q/tlast/label/mindist/mass fields match the continuous
  reference exactly. Ionization continues after the checkpoint.

The index-zero test now scales positions with the subdomain width to keep both
endpoints owned by rank zero. The stale-partner test relocates in bounded
increments between ghost rebuilds. Their old fixed-geometry assumptions were
invalid for 8/16 ranks; production code was unchanged when correcting them.

The coupled-bubble restart was checked separately with the author's protocol
and CPU near-term tolerance of 1e-10. That is a numerical tolerance, not a
promise of bitwise identity of a coupled MPI trajectory. Restart initial native
virial/Press output must not be treated as an independently recomputed pressure.

## Audited input contract and remaining findings

The current CPU templates under `test_1e4/` use one run per input, four explicit
pair-style arguments, the property order below, and the author's custom dump
restart. Preserve that contract for the first short Anvil validation:

```lammps
fix mass all property/atom rmass ghost yes
fix timecollide all property/atom i_tlast
fix ionlabel all property/atom i_label ghost yes
fix ion_mindist all property/atom d_mindist ghost yes
```

Restore the full x/v/f/q/custom-property checkpoint, physical time, dt and KM/TH
state with `scaffold/read_restart.py`. Match the read_sldump step, extracted
step and dump file count. Keep `active = step > checkpoint` and
`run_style restartverlet`; redeclare the complete pair_style/pair_coeff.

The following conditional defects remain open; this revision does not claim to
fix them or certify unrestricted production/GPU use:

| Path | Remaining issue |
| --- | --- |
| Ordinary `run 1; run 1` | Repeated setup can add an ionization event. |
| Native binary `read_restart` | Custom pair parameter/activation restoration is incomplete; writing binary files does not validate reading them. |
| Reordered custom properties | The text loader and wall still rely on fixed property indices; keep the audited declaration order. |
| Three-argument pair_style | Reads a nonexistent fourth logical argument; use the explicit four-argument form. |
| `scaffold/compare_restart.py` | May print FAIL yet exit 0, and labels differences below 1e-14 as bitwise identical. Inspect actual differences and complete overlap, not just its exit code or wording. |
| GPU | Shared reaction logic and wrapper syntax checked; GPU force/continuation equivalence still requires working hardware. |

The standard two-argument pair-style form is also supported by the source,
but was not the production-template configuration audited here. The inactive
pair switch skips forces as well as reactions and must not be used as a generic
no-ionization model. A no-ionization comparison needs an appropriate force-only
pair setup.

For intentionally failing reproductions, see [review/README.md](review/README.md).
Start with short CPU checks on Anvil before committing to large paired runs;
the local evidence does not establish timestep/EP convergence or revised paper
conclusions at larger N.
