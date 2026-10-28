# Optional private regression inputs

This directory is ignored except for its README and ignore rules. Historical
simulation output and its expected results remain local. They are not required
for the public synthetic/numerical test suite.

If the following private inputs are present, the test suite additionally checks
the original paper's table metrics and emitted-light curve:

- `historical_global.table`
- `historical_radial.out`
- `historical_expected.json`

`historical_sample.json` records their source hashes and sampled steps locally.
Do not add these data to a public commit without explicit publication approval.
