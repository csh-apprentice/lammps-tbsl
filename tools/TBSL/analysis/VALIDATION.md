# Analysis integration validation — 2026-10-27

No simulations were run for this integration.

## Numerical regression

`python3 -m unittest discover -s tools/TBSL/analysis/tests -v`

19 tests pass. The 12-frame historical fixture comes from the author's
N_EP=10^6, alpha_t=1, full-ionization outputs. Source hashes and sampled steps
are recorded in `tests/fixtures/historical_sample.json`. Expected table metrics
and the emission curve were calculated with the unchanged Hyde scripts before
porting. Table metrics agree at rtol=2e-14; the emission curve agrees at
rtol=2e-12. The fixture is sparse: its extrema are not historical full-run
extrema or new scientific results.

Other checks exercise the quadratic Saha relation, nearly complete ionization,
band integration, SI density and volume conversions, explicit spectrum length
units, particle/charge/density conservation under smoothing, radial boost
invariance of thermal moments, and input/sampling error handling.

## Plot and reporting checks

- Theory recipe: regenerated from the actual historical continuum/reference
  tables. Radius and temperature panels completed.
- Cutoff recipe: regenerated the inherited convergence sequences and potential
  panel. These are stored reference values, not a new convergence experiment.
- Scaling, boundary, pair, table, general and appendix recipes: complete figure
  and report generation checked using a sparse historical fixture mapped to
  their inputs. These smoke outputs are software checks; the repeated fixture
  does not represent different physical cases or EP-count convergence.
- Performance recipe: checked separately with the actual historical global
  logs, timing snippets and Anvil segment filenames. Historical core-count
  assumptions remain explicit in the recipe and README.
- `figure_map.json` accounts for all 55 PDF references in the current manuscript
  sections. It also records the section-3 output-path difference.

Generated checks and input copies are under ignored `artifacts/` and `data/`.
Historical regression fixtures and expected values are also ignored and remain
local. GitHub Actions runs synthetic/analytic checks; the two private historical
regressions skip when their data are absent. The workflow covers Python 3.10
and 3.12; remote CI status is distinct from the local test result.

## COM pilot postprocessing

The evaluator was run on all three previously completed N_EP=10^4 arms with
the same EP scale, center-outward min-count-100 smoothing, radial-flow thermal
temperature, and wall-clipped volumes. Those curves are a new, explicitly
defined low-resolution diagnostic; they do not reproduce the historical paper
temperature estimator. Raw simulation inputs and output files were unchanged.
The emitted-light estimates must not be inferred directly from the retained
MD ionization fractions.
