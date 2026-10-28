# Data Preprocessing Notes

This note records how the `comdis` files used in Sections 7.1 to 7.4 are produced and how `Ncount` is used afterward.

## Pipeline

1. Start from the raw LAMMPS `fix ave/chunk` radial output.
2. Run the postprocessing script [smooth_ave_chunk_minN_keepbins.py](../utils/postprocess/smooth_ave_chunk_minN_keepbins.py) with `min_N = 100`.
3. Use the resulting smoothed `comdis_*_minN100_smooth.out` files for plotting and scalar extraction.

## What The Smoothing Step Does

- The original radial bin grid is preserved.
- The original `Ncount` column is also preserved.
- Intensive properties such as `temp`, `v_*`, `density/number`, and `density/mass` are overwritten by group-averaged values computed over neighboring bins.
- Bins are grouped from the center outward until the accumulated particle count reaches at least `min_N`.
- The final occupied group is still accepted even if it does not reach `min_N`.
- Bins beyond the last non-zero `Ncount` are treated as invalid and written as zeros for intensive quantities.

## Why Boundary Outliers Can Still Survive

- Because grouping proceeds from the center outward, the last occupied group near the bubble wall can remain underfilled.
- That means the smoothed file is much better than the raw file, but it can still contain rare low-support spikes near the boundary.
- This is why a large local `T_max` can still come from the last occupied wall-adjacent bin even after `min_N = 100` smoothing.

## What `Ncount` Means After Smoothing

- `Ncount` is not the count of a smoothed super-bin.
- It is the original per-bin particle count kept from the raw `ave/chunk` output.
- So `Ncount` remains a useful measure of how much direct particle support a plotted bin has, even though the intensive properties have already been smoothed.

## How We Use `Ncount` In Later Analysis

- Heatmaps:
  - `Ncount` is used for robust display scaling through `vmax_q`.
  - This does not redefine the data values.
  - It only prevents a few low-support extreme bins from setting the whole colorbar range.
- Table 2 `T_max`:
  - We use the same smoothed temperature field.
  - But the scalar `T_max` is taken only over bins with `Ncount >= 10`.
  - This is a robust reporting rule so singleton wall-bin spikes do not define the table statistic.

## Practical Interpretation

- `min_N = 100` smoothing fixes most of the raw bin noise.
- `vmax_q` is a plotting-only robustness trick.
- `Ncount >= 10` for Table 2 is a scalar-extraction robustness trick.
- These are complementary steps, not duplicates.
