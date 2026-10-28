# TBSL center-of-mass ionization tests

This focused suite addresses JCP Reviewer 2, Comment 1. It calls the production
`PairLJCutIOCoulDSF` implementation through the LAMMPS C++ library. Fixtures
contain two or three particles, with no time integration, bubble wall, continuum
coupling, or production input/data files. `run 0` initializes neighbors and, in
the migration tests, redistributes atoms; it does not advance a trajectory.

The suite has no googletest/libyaml download dependency. From the repository root:

```sh
cmake -S unittest/tbsl -B build-tbsl-tests -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug -DBUILD_MPI=ON -DBUILD_OMP=OFF
cmake --build build-tbsl-tests --target test_ionization_pair -j 8

mpiexec -n 1 build-tbsl-tests/test_ionization_pair
mpiexec -n 2 build-tbsl-tests/test_ionization_pair
mpiexec -n 4 build-tbsl-tests/test_ionization_pair
```

Alternatively, `ctest --test-dir build-tbsl-tests --output-on-failure` runs the
registered 1/2/4-rank tests, each with a 30-second timeout. This timeout starts
when the test runs; it does not bound a client's pending sandbox approval.
MPI launchers need local process communication sockets. In a restricted agent
environment, the launcher can require approval to run outside the sandbox.

An optional positional substring selects a case, for example:

```sh
mpiexec -n 2 build-tbsl-tests/test_ionization_pair boost
```

The deterministic assertions use closed-form expected velocities, pair momentum,
and kinetic loss. They read owned atoms, so a modified ghost cannot conceal a
missing owner update. Two- and four-rank decompositions put selected pairs on
different ranks; the four-rank tests also include ranks without owned atoms.
The default numeric tolerance is `2e-12 * max(1, abs(expected))`.

Coverage includes COM boosts, equal velocities, strict threshold boundaries,
stationary particles, unequal masses, EP mass/threshold scaling, terminal charge
states, an explicit local-index-zero partner, competing and tied candidates,
rechecking energy after labeling, one event per particle per round, atom
migration, and partners leaving the ghost halo. Half and full neighbor lists
are exercised with Newton pair both on and off. The full-list fixture tests the
shared CPU ionization pipeline's list semantics, not GPU force arithmetic.

See [DEPLOYMENT.md](DEPLOYMENT.md) for the reference equations, observed results,
Anvil build commands, and remaining validation boundaries. The checked source
hashes are in [source_sha256.json](source_sha256.json). Generated test output is
kept local; the reproducible synthetic fixtures and test programs are included.

The October 28 final check additionally runs the full suite on 8 and 16 ranks,
with 19 cases / 202 checks reported by rank zero. The index-zero fixture keeps
both endpoints owned by rank zero for every tested rank count; the stale-partner
fixture uses bounded moves between ghost rebuilds. This removes two assumptions
that were only valid in the original smaller-rank matrix. See
[DEPLOYMENT.md](DEPLOYMENT.md) for the sanitizer, Release, custom-restart results
and remaining conditional defects. The standalone review programs can now be
built as CMake targets `tbsl_review_probes` and `tbsl_custom_restart_protocol`.
