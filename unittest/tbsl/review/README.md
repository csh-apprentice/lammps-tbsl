# Reproduce the conditional review findings

These small synthetic probes reuse the isolated LAMMPS fixture. The complete
`tbsl_review_probes` executable intentionally fails for known issues outside
the audited production input contract. Those failures do not prove a defect in
the author's guarded custom restart. See [the deployment notes](../DEPLOYMENT.md)
for the validation scope and remaining issues.

Build against the same library as the simulation and main test suite:

```sh
cmake --build build-anvil-com --parallel 16 \
  --target tbsl_review_probes tbsl_custom_restart_protocol

# Passing communication, integration, terminal-charge and standard-order checks:
mpiexec -n 16 build-anvil-com/tbsl_review_probes integrated
mpiexec -n 16 build-anvil-com/tbsl_review_probes charge_q8
mpiexec -n 16 build-anvil-com/tbsl_review_probes dump_standard

# Intentionally failing conditional cases:
mpiexec -n 1 build-anvil-com/tbsl_review_probes split_run
mpiexec -n 1 build-anvil-com/tbsl_review_probes binary_restart_parameters
mpiexec -n 1 build-anvil-com/tbsl_review_probes dump_reordered
mpiexec -n 1 build-anvil-com/tbsl_review_probes three_settings

# Synthetic 50% mismatch: this currently prints FAIL yet exits zero.
python3 scaffold/compare_restart.py \
  unittest/tbsl/review/fixtures/restart_original.log \
  unittest/tbsl/review/fixtures/restart_broken.log 0
```

The actual guarded restart is tested separately:

```sh
mkdir -p build-anvil-com/restart-check
mpiexec -n 16 build-anvil-com/tbsl_custom_restart_protocol \
  build-anvil-com/restart-check
```

It compares a continuous four-step reactive trajectory with a two-step
checkpoint followed by two continued steps. All 30 guarded final-state checks
must pass. It also writes an unguarded negative control, which is not the
author's production configuration. Use a distinct output directory per run.

The optional `custom_restart/analyze.py` needs NumPy and expects three runs in
`custom_restart/results/mpi1`, `mpi2`, and `mpi4`; it writes a detailed comparison
of both guarded and unguarded states. These result directories are generated
locally and are ignored by git.

The loader probes use temporary files named
`/tmp/tbsl-review-dump-<ranks>-<case>.dump` and remove them on completion. Do not
run identical loader cases concurrently with the same MPI rank count on the
same filesystem. All fixture inputs here are synthetic; no paper data are
required.
