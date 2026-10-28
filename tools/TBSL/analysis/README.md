# Sonoluminescence analysis and paper figures

Portable versions of the author's analysis tools from `hyde03`, retrieved on
2026-10-27. The original file hashes are in [source_manifest.json](source_manifest.json).
The scripts evaluate existing output; they do not launch LAMMPS or submit jobs.

## Setup and quick checks

Python 3.10 or newer:

```sh
python3 -m pip install -r tools/TBSL/analysis/requirements.txt
python3 -m unittest discover -s tools/TBSL/analysis/tests -v
python3 tools/TBSL/analysis/run.py --list
python3 tools/TBSL/analysis/run.py cutoff --output-dir /tmp/tbsl-cutoff
```

Only NumPy and Matplotlib are required. Public tests use synthetic inputs and
analytic checks. Optional local historical fixtures add regression checks
against the unchanged original scripts; those research data are not published
with this code. Tests do not need server access or a LAMMPS build.

## Regenerate the paper panels

Run from this checkout on a machine with access to the historical data tree.
On Hyde the data root is
`/home/shihan/projects/sonoluminescence/lammps/analysis` (a symlink to
`/data/shihan/sonoluminescence/analysis`). The checkout itself can be elsewhere:

```sh
python3 tools/TBSL/analysis/run.py pair \
  --data-root /data/shihan/sonoluminescence/analysis \
  --output-dir /tmp/tbsl-figures/pair --check

# Omit --check after all inputs are available.
python3 tools/TBSL/analysis/run.py pair \
  --data-root /data/shihan/sonoluminescence/analysis \
  --output-dir /tmp/tbsl-figures/pair
```

| Recipe | Outputs |
| --- | --- |
| `theory` | Radius validation and uniform/nonuniform temperature comparison |
| `scaling` | Global and radial comparisons for 10^6, 10^7, 10^8 EPs |
| `boundary` | Accommodation sweep temperatures and temperature heatmaps |
| `pair` | Interaction sweep temperatures, retained charge, emission heatmaps |
| `table` | Eleven historical Table 2 rows and diagnostic values |
| `general` | Collapse dynamics, radial profiles, total/radial emission |
| `cutoff` | Historical cutoff convergence sequences and LJ potential |
| `performance` | Timing shares, ionization overhead, throughput and timestep |
| `appendix` | Heatmaps, radial sweeps, scalar plots and appendix summary |

[figure_map.json](figure_map.json) maps **all 55 PDF references** in the current
manuscript sections to these recipes. Most output paths match the manuscript.
The two section-3 figures are generated under `figs/results/`, whereas the
manuscript references them directly under `figs/`; the map records both paths.
Some recipes also generate supplemental panels that the manuscript does not use.

The runner creates a separate `figs/` tree and `reports/` directory, plus a
`RECIPE_provenance.json` recording input/source/output hashes, package versions,
command and completion status. It does not copy figures into Overleaf. Review
the regenerated panels and their inputs before replacing manuscript assets.

Defaults are the ignored `data/` and `artifacts/` directories beside this file.
Raw outputs, notebook outputs, caches and generated figures are not committed.
The source-only integration excludes old duplicate emission code and ad hoc
restart/CPU-column rewriting scripts.

### Point the historical recipes at revised runs

Use `--input-map replacements.json` to replace particular historical inputs:

```json
{
  "1e6/pair/run_1e6_01_13_alpha_1_shortio.lammps": "revised/1e6_short/global.table",
  "1e6/pair/1e6_comdis_01_13_alpha_1_shortio_minN100_smooth.out": "revised/1e6_short/radial_minN100.out"
}
```

Replacement paths are absolute or relative to `--data-root`. The **historical
recipes retain their original case definitions**, including EP counts, gas,
R0=4.5 micrometers, alpha values, plotting windows and performance core counts
(128, 2048, 2048). Use matching revised cases; mapping a 10^4 run into a 10^6
recipe would apply the wrong scaling. Use the single-run evaluator below for
other EP counts. The performance recipe's historical core counts must be
updated explicitly if the job allocation changes.

The original preprocessing programs are available under
[utils/postprocess/](utils/postprocess/). For example:

```sh
python3 tools/TBSL/analysis/utils/postprocess/smooth_ave_chunk_minN_keepbins.py --help
```

Always write a distinct output file. The paper's keepbins smoother groups from
the center outward, retains original `Ncount`, averages intensive quantities
with particle weights, and averages densities with shell volumes. A final
wall group can contain fewer than 100 EPs. See
[data_preprocessing_notes.md](docs/data_preprocessing_notes.md).

## Evaluate a new run, including the COM pilot

The evaluator writes CSV, a compressed spatial emission field, PDF/PNG light
curves and a JSON summary. EP scaling and temperature definition are explicit:

```sh
python3 tools/TBSL/analysis/evaluate.py \
  --global-file experiments/r2c1_com_1e4/runs/com/run.log \
  --radial-file experiments/r2c1_com_1e4/runs/com/radial.out \
  --ep-scale 933766.347775394 \
  --temperature radial-flow --smooth-min-count 100 --geometry clip-wall \
  --output-dir tools/TBSL/analysis/artifacts/r2c1_com \
  --label 'COM pilot, N=10000'
```

- `--temperature paper` requires a recorded `temp` field and divides it by the
  EP scale. This reproduces the historical emission input. It must not be
  assumed to remove coherent motion unless the originating LAMMPS compute did.
- `--temperature radial-flow` requires `<v_radial>` and `<v^2>` fields named
  `v_radial_velocity` and `v_speed_squared`. It uses the microscopic Ar mass
  and subtracts the squared mean radial flow. It does not remove arbitrary
  nonspherical bulk motion. These are the moments saved by the COM pilot.
- `--smooth-min-count 100` applies the historical keepbins smoother to raw
  fields. Leave it at zero for already smoothed inputs. This differs from the
  pilot's earlier wall-inward merging diagnostic; their maxima are different
  statistics.
- `--geometry paper-shells` uses complete shells inferred from centers, matching
  the historical emission sum. `clip-wall` clips shell volumes at the current
  bubble radius. Record the choice when comparing curves.

The inherited single-stage Saha model uses chi=15.76 eV, g_i/g_0=0.5 and
`P = 1.57e-40 q_Saha^2 n_SI^2 sqrt(T)`. The lambda>=200 nm field multiplies
this by `1-exp[-hc/(200 nm k_B T)]`. Total power uses shell volumes in m^3;
the paper's adopted coefficient gives W, reported as mW. This remains a
Saha--LTE **diagnostic**, without opacity, recombination dynamics or radiation
feedback; this port does not independently establish absolute calibration.
MD retained charge and occupied ionized-EP fraction are kept separately from
the equilibrium Saha reconstruction. The legacy table's `f_ion_percent` is
actually `100*sum(Ncount*q_MD)/N_EP`, a mean charge, which equals occupancy
only while charges are 0/1.

## Port changes and validation

- Relative package imports replace machine-specific `sys.path` changes.
- Strict readers support historical tables and complete LAMMPS logs. They
  reject malformed/truncated blocks, conflicting restart steps and changing
  grids in the fixed-grid API. An iterator is provided for adaptive grids.
- Sampling alignment checks bounds before indexing; scaling profiles correctly
  map global extrema onto the distribution sampling times.
- The Saha quadratic uses an algebraically equivalent stable root; the band
  fraction uses `expm1`. The standalone electron-count and normalized-spectrum
  helpers now convert Angstrom^3 to m^3 and honor their length-unit argument.
  These SI helper fixes do not change the paper recipes, which already convert
  their spatial emission sums explicitly. The older generic
  `compute_light_curve_from_emission` still returns native volume units, as its
  arbitrary-units docstring specifies; use the evaluator for physical totals.
- Historical smoothing, robust display caps, physical parameters and the table
  estimator are preserved. Display caps do not define scalar maxima.

The numerical tests verify local historical table and emission regression, Saha
equilibrium and limits, density/volume/spectrum units, EP scaling, radial-flow
invariance, smoothing conservation, sampling alignment and malformed input.
See [VALIDATION.md](VALIDATION.md) for the executed checks and scope.
