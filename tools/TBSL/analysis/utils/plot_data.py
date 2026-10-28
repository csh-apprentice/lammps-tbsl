import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# --- physical constants for Saha equation (SI units) ---
E_CHARGE = 1.602176634e-19   # Coulomb
K_B      = 1.380649e-23      # J/K
H_PLANCK = 6.62607015e-34    # J*s
M_E      = 9.10938356e-31    # kg
PI       = np.pi

def plot_global_multi_lines(
    data_list,
    x_attr,
    y_attr,
    labels=None,
    uselabel=True,
    filename="plot",
    x_scale=1.0,
    y_scale=1.0,
    save=False,
    xlabel=None,
    ylabel=None,
    title=None,
    colors=None,
    output_dir="output",
    # --- X range options ---
    x_min=None,
    x_max=None,
    # --- NEW: per-line style controls ---
    line_styles=None,     # e.g. ["-", "--", ":", "-."] or "-" for all
    markers=None,         # e.g. ["o", None, "s", None] or "o" for all
    line_widths=None,     # e.g. [1.0, 2.0, 1.5, 1.0] or 2.0 for all
    marker_sizes=None,    # e.g. [2, 2, 3, 2] or 2 for all
    # --- THEORY CURVE ---
    theory_data=None,
    enable_theory=False,
    theory_xattr="Time",
    theory_yattr=None,
    theory_label="theory",
    theory_scale=1.0,
    theory_linestyle="--",
    theory_marker=None,
    theory_linewidth=2.0,
):
    """
    Generalization of plot_global_two_lines:
    - data_list: list of dicts, each with keys x_attr and y_attr.
    - labels: list of labels (same length as data_list) or None -> "line1", "line2", ...
    - x_scale, y_scale: float or list/tuple of length N (N = len(data_list)).
    - colors: optional list of colors; reused cyclically if fewer than N.
    - x_min/x_max: if provided, only plot points where x in [x_min, x_max] (after scaling).
    - line_styles/markers/line_widths/marker_sizes: per-line styling, each can be:
        * None (use defaults)
        * a single value applied to all
        * a list/tuple of length N
    - theory_* control theory curve styling (if enabled).
    """

    # ------------------------------
    # Basic checks
    # ------------------------------
    if not isinstance(data_list, (list, tuple)) or len(data_list) == 0:
        raise ValueError("data_list must be a non-empty list/tuple of data dicts")

    N = len(data_list)

    # Labels
    if labels is None:
        labels = [f"line{i+1}" for i in range(N)]
    if len(labels) != N:
        raise ValueError("labels must be None or have the same length as data_list")

    # ------------------------------
    # Normalize helper for per-line args
    # ------------------------------
    def _norm_per_line(v, name, default=None):
        if v is None:
            return [default] * N
        if isinstance(v, (list, tuple)):
            if len(v) != N:
                raise ValueError(f"{name} must be None, a scalar, or a list/tuple of length {N}")
            return list(v)
        # scalar
        return [v] * N

    # x_scale
    if not isinstance(x_scale, (list, tuple)):
        x_scale = [x_scale] * N
    if len(x_scale) != N:
        raise ValueError("x_scale must be a float or a list/tuple of length equal to len(data_list)")

    # y_scale
    if not isinstance(y_scale, (list, tuple)):
        y_scale = [y_scale] * N
    if len(y_scale) != N:
        raise ValueError("y_scale must be a float or a list/tuple of length equal to len(data_list)")

    # per-line styles (defaults chosen to match your existing behavior)
    line_styles = _norm_per_line(line_styles, "line_styles", default="-")
    line_widths = _norm_per_line(line_widths, "line_widths", default=(1.0 if uselabel else 2.0))
    markers = _norm_per_line(markers, "markers", default=("o" if uselabel else None))
    marker_sizes = _norm_per_line(marker_sizes, "marker_sizes", default=2)

    # ------------------------------
    # Validate existence of keys
    # ------------------------------
    for idx, d in enumerate(data_list):
        if x_attr not in d or y_attr not in d:
            raise ValueError(f"Attribute {x_attr} or {y_attr} missing in data_list[{idx}]")

    # Theory checks
    if enable_theory:
        if theory_yattr is None:
            raise ValueError("enable_theory=True but theory_yattr is None")
        if theory_data is None:
            if theory_yattr not in data_list[0]:
                raise ValueError(f"theory_yattr '{theory_yattr}' not found in data_list[0]")
        else:
            if theory_xattr not in theory_data or theory_yattr not in theory_data:
                raise ValueError(f"theory_data must contain '{theory_xattr}' and '{theory_yattr}'")

    # ------------------------------
    # Colors
    # ------------------------------
    if colors is None:
        colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
            "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173"
        ]

    # Serif font
    plt.rcParams["font.family"] = "serif"

    # ------------------------------
    # Plot lines
    # ------------------------------
    plt.figure(figsize=(8, 6))

    any_x_scaled = False
    any_y_scaled = False

    for i, d in enumerate(data_list):
        x = np.array(d[x_attr]) * x_scale[i]
        y = np.array(d[y_attr]) * y_scale[i]

        any_x_scaled |= (x_scale[i] != 1.0)
        any_y_scaled |= (y_scale[i] != 1.0)

        # range filtering
        mask = np.ones_like(x, dtype=bool)
        if x_min is not None:
            mask &= (x >= x_min)
        if x_max is not None:
            mask &= (x <= x_max)

        x_plot = x[mask]
        y_plot = y[mask]
        if x_plot.size == 0:
            continue

        color = colors[i % len(colors)]

        plot_kwargs = dict(
            color=color,
            linestyle=line_styles[i],
            linewidth=line_widths[i],
        )

        # marker control
        if markers[i] is not None:
            plot_kwargs.update(marker=markers[i], markersize=marker_sizes[i])

        if uselabel:
            plot_kwargs["label"] = labels[i]

        plt.plot(x_plot, y_plot, **plot_kwargs)

    # ------------------------------
    # Theory curve
    # ------------------------------
    if enable_theory:
        if theory_data is None:
            d0 = data_list[0]
            x_th = np.array(d0[x_attr]) * x_scale[0]
        else:
            d0 = theory_data
            x_th = np.array(d0[theory_xattr]) * x_scale[0]

        y_th = np.array(d0[theory_yattr]) * theory_scale

        mask_th = np.ones_like(x_th, dtype=bool)
        if x_min is not None:
            mask_th &= (x_th >= x_min)
        if x_max is not None:
            mask_th &= (x_th <= x_max)

        x_th_plot = x_th[mask_th]
        y_th_plot = y_th[mask_th]

        if x_th_plot.size != 0:
            th_kwargs = dict(
                color="black",
                linestyle=theory_linestyle,
                linewidth=theory_linewidth,
            )
            if theory_marker is not None:
                th_kwargs.update(marker=theory_marker)

            if uselabel:
                th_kwargs["label"] = theory_label

            plt.plot(x_th_plot, y_th_plot, **th_kwargs)

    # ------------------------------
    # Labels & title
    # ------------------------------
    if xlabel is None:
        xlabel = f"{x_attr} (scaled)" if any_x_scaled else x_attr
    if ylabel is None:
        ylabel = f"{y_attr} (scaled)" if any_y_scaled else y_attr
    if title is None:
        title = f"{y_attr} vs {x_attr} (Global)"

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()

    # ------------------------------
    # Save / show
    # ------------------------------
    if save:
        base = os.path.splitext(filename)[0]
        out_name = f"{base}_{y_attr}_vs_{x_attr}_multi.pdf"
        savepath = os.path.join(output_dir, out_name)
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
        print(f"✅ Saved multi-line global plot as {savepath}")

    plt.show()


def plot_global(data, x_attr, y_attr, filename, x_scale=1.0, y_scale=1.0,save=False,output_dir="output",):
    if x_attr not in data or y_attr not in data:
        raise ValueError("Attributes not found in global data")

    x = np.array(data[x_attr]) * x_scale
    y = np.array(data[y_attr]) * y_scale

    plt.figure(figsize=(8,6))
    plt.plot(x, y, marker="o", markersize=2, linewidth=1)
    plt.xlabel(f"{x_attr} (scaled x{ x_scale })" if x_scale!=1 else x_attr)
    plt.ylabel(f"{y_attr} (scaled x{ y_scale })" if y_scale!=1 else y_attr)
    plt.title(f"{y_attr} vs {x_attr}")
    plt.grid(True)
    plt.show()

    if save:
        out_name = os.path.splitext(filename)[0] + f"_{y_attr}_vs_{x_attr}.png"
        savepath=os.path.join(output_dir, out_name)
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches="tight") 
        print(f"✅ Saved global plot as {out_name}")


def compute_shell_edges(radii):
    edges = [0.0]
    for i in range(len(radii)-1):
        edges.append(0.5 * (radii[i] + radii[i+1]))
    edges.append(radii[-1] + (radii[-1] - radii[-2]) / 2)
    return np.array(edges)


# ============================================================
# NEW: light–emission helpers
# ============================================================

def compute_shell_volumes(radii):
    """
    Compute spherical shell volumes from bin centers.

    Parameters
    ----------
    radii : (N_r,) array
        Bin-center radii (same as returned from parse_ave_chunk_output).

    Returns
    -------
    volumes : (N_r,) array
        Volume of each shell: 4/3*pi*(r_{i+1}^3 - r_i^3)
    """
    edges = compute_shell_edges(radii)
    r_in = edges[:-1]
    r_out = edges[1:]
    volumes = (4.0 / 3.0) * np.pi * (r_out**3 - r_in**3)
    return volumes


def compute_emission_field_q2n2T(temp_vals, density_vals, ion_frac_vals,
                                 temp_exponent=0.5, prefactor=1.0):
    """
    Compute a continuum emission proxy field:

        P(r,t) ∝ q(r,t)^2 n(r,t)^2 T(r,t)^{temp_exponent}

    where:
    - temp_vals      : (N_t, N_r) array of shell temperatures
    - density_vals   : (N_t, N_r) array of number densities
    - ion_frac_vals  : (N_t, N_r) array of ionization fraction or
                       average charge per particle (dimensionless)

    This does *not* try to get absolute units; it's a relative
    intensity field suitable for visualization.
    """
    T = np.asarray(temp_vals)
    n = np.asarray(density_vals)
    q = np.asarray(ion_frac_vals)

    if T.shape != n.shape or T.shape != q.shape:
        raise ValueError("temp_vals, density_vals, and ion_frac_vals must have the same shape")

    # Avoid negative / zero temperatures
    T_clipped = np.clip(T, 1e-12, None)

    P = prefactor * (q**2) * (n**2) * (T_clipped**temp_exponent)
    return P  # shape (N_t, N_r)


def compute_light_curve_from_emission(emission_vals, radii):
    """
    Integrate P(r,t) over the bubble volume to get a global light curve I(t).

    Parameters
    ----------
    emission_vals : (N_t, N_r) array
        Emission field P(r,t) from compute_emission_field_q2n2T.
    radii : (N_r,) array
        Bin-center radii.

    Returns
    -------
    I_t : (N_t,) array
        Global intensity vs time (arbitrary units).
    """
    P = np.asarray(emission_vals)
    shell_volumes = compute_shell_volumes(radii)  # (N_r,)

    if P.shape[1] != shell_volumes.shape[0]:
        raise ValueError("emission_vals.shape[1] must match length of radii")

    # For each time: sum_j P(t,j) * ΔV_j
    I_t = np.dot(P, shell_volumes)
    return I_t


def compute_emission_spectrum(temp_vals, density_vals, ion_frac_vals,
                              radii, lambda_grid,
                              time_indices=None,
                              prefactor=1.0):
    """
    Compute an approximate continuum emission spectrum I_lambda
    using a simple LTE-like kernel:

        j_lambda(r,t) ∝ q^2 n^2 T^{-1/2} exp(-hc / (λ k_B T))

    Parameters
    ----------
    temp_vals      : (N_t, N_r) array, temperatures
    density_vals   : (N_t, N_r) array, number densities
    ion_frac_vals  : (N_t, N_r) array, ionization fractions
    radii          : (N_r,) array, bin-center radii
    lambda_grid    : (N_λ,) array, wavelengths in meters
    time_indices   : 1D array of indices into time dimension to include;
                     if None, uses all time steps.
    prefactor      : overall multiplicative constant (for relative spectra).

    Returns
    -------
    I_lambda : (N_λ,) array
        Emission spectrum (arbitrary units).
    """
    T = np.asarray(temp_vals)
    n = np.asarray(density_vals)
    q = np.asarray(ion_frac_vals)

    if T.shape != n.shape or T.shape != q.shape:
        raise ValueError("temp_vals, density_vals, ion_frac_vals must have same shape")

    N_t, N_r = T.shape

    if time_indices is None:
        time_indices = np.arange(N_t)
    else:
        time_indices = np.asarray(time_indices, dtype=int)

    # Physical constants (SI)
    h = 6.62607015e-34       # Planck [J s]
    c = 2.99792458e8         # speed of light [m/s]
    kB = 1.380649e-23        # Boltzmann [J/K]

    # Geometry
    shell_volumes = compute_shell_volumes(radii)  # (N_r,)

    # Clip T to avoid divide-by-zero / overflow
    T_sel = np.clip(T[time_indices, :], 1e-6, None)   # (N_t_sel, N_r)
    n_sel = n[time_indices, :]
    q_sel = q[time_indices, :]

    # Precompute q^2 n^2 T^{-1/2}
    base = (q_sel**2) * (n_sel**2) * (T_sel**(-0.5))  # (N_t_sel, N_r)

    I_lambda = np.zeros_like(lambda_grid, dtype=float)

    for i, lam in enumerate(lambda_grid):
        # Exponential factor: exp(-hc / (λ k_B T))
        exp_factor = np.exp(- (h * c) / (lam * kB * T_sel))  # (N_t_sel, N_r)

        j = base * exp_factor  # (N_t_sel, N_r)

        # Integrate over volume and time (Δt omitted => relative units)
        I_lambda[i] = np.sum(j * shell_volumes[None, :])

    return prefactor * I_lambda


def plot_light_curve(timesteps, intensity,
                     x_scale=1.0,
                     xlabel=None,
                     ylabel="Intensity (arb. units)",
                     title="Light curve",
                     save=False,
                     filename="light_curve",
                     output_dir="output"):
    """
    Simple convenience wrapper to plot I(t) with your standard style.
    """
    t = np.asarray(timesteps) * x_scale
    I = np.asarray(intensity)

    plt.rcParams["font.family"] = "serif"
    plt.figure(figsize=(8, 4))

    plt.plot(t, I, linewidth=2)
    if xlabel is None:
        xlabel = f"Time (scaled x{x_scale})" if x_scale != 1.0 else "Time"
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

    if save:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(filename)[0]
        out_name = os.path.join(output_dir, f"{base}.pdf")
        plt.savefig(out_name, dpi=300, bbox_inches="tight")
        print(f"✅ Saved light curve as {out_name}")

    plt.show()
    
    
def _auto_vmin_smallest_positive(z_vals):
    """
    Return smallest strictly-positive finite value in z_vals.
    Works for ndarray or masked arrays.
    Falls back to 0.0 if no positive value exists.
    """
    if np.ma.isMaskedArray(z_vals):
        data = z_vals.compressed()
    else:
        data = np.asarray(z_vals).ravel()

    # keep finite and > 0
    data = data[np.isfinite(data)]
    data = data[data > 0]

    if data.size == 0:
        return 0.0
    return float(data.min())

import numpy as np

def _weighted_quantile(values, weights, q):
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)

    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v = v[m]
    w = w[m]
    if v.size == 0:
        return np.nan

    idx = np.argsort(v)
    v = v[idx]
    w = w[idx]

    cw = np.cumsum(w)
    total = cw[-1]
    if total <= 0:
        return np.nan

    target = q * total
    j = np.searchsorted(cw, target, side="left")
    j = min(max(j, 0), v.size - 1)
    return float(v[j])

def _auto_vmax_row_weighted_percentile(z_vals, ncount, q=0.99, ignore_zeros=True):
    Z = z_vals
    C = np.asarray(ncount)

    if C.shape != np.shape(Z):
        raise ValueError(f"ncount shape {C.shape} must match z_vals shape {np.shape(Z)}")

    row_qs = []
    T = C.shape[0]
    for t in range(T):
        row_v = Z[t, :]
        row_c = C[t, :]

        # respect masks if z_vals is masked
        if np.ma.isMaskedArray(row_v):
            keep = ~row_v.mask
            row_v = np.asarray(row_v.data)[keep]
            row_c = np.asarray(row_c)[keep]
        else:
            row_v = np.asarray(row_v)

        row_c = np.asarray(row_c)

        if ignore_zeros:
            m = np.isfinite(row_v) & (row_v > 0) & np.isfinite(row_c) & (row_c > 0)
        else:
            m = np.isfinite(row_v) & np.isfinite(row_c) & (row_c > 0)

        row_v = row_v[m]
        row_c = row_c[m]

        if row_v.size == 0:
            continue

        qv = _weighted_quantile(row_v, row_c, q)
        if np.isfinite(qv):
            row_qs.append(qv)

    if len(row_qs) == 0:
        return None
    return float(np.max(row_qs))



def plot_distribution(
    timesteps,
    radii,
    values,
    attribute,
    global_x=None,
    global_data=None,
    x_scale=1.0,
    y_scale=1.0,
    c_scale=1.0,
    cmap="hot",
    save=False,
    output_dir="output",
    vmin=None,
    vmax=None,
    mask_zeros=False,
    # --- NEW: x_min/x_max ---
    x_min=None,
    x_max=None,
    # --- NEW: particle-level vmax control ---
    ncount=None,  # shape (n_time, n_rbins), aligned with values
    vmax_q=0.99,
    auto_vmax_mode="row_weighted_percentile",  # or None/"none"
    # --- overlay + void masking ---
    global_ref=None,
    ref_color="lightgray",
    ref_line_kwargs=None,
    xlabel=None,
    ylabel=None,
    title=None,
    ylim=None,
):
    """Heatmap of distribution data with scaling options, x-range slicing,
    auto-vmin (smallest non-zero), and robust auto-vmax using particle-level
    per-row weighted percentiles (if ncount is provided).

    auto_vmax_mode="row_weighted_percentile":
      For each time row, compute the weighted q-quantile across bins using ncount
      as weights (particle-level). Then set vmax to the max of those row quantiles.
    """

    # ------------------------------
    # Basic sanity
    # ------------------------------
    n_time = len(timesteps)
    if len(values) != n_time:
        raise ValueError("values must have the same number of time samples as timesteps")

    # ------------------------------
    # Radial edges
    # ------------------------------
    edges = compute_shell_edges(radii)

    # ------------------------------
    # X axis selection
    # ------------------------------
    if global_x is None:
        x_vals_raw = np.array(timesteps)
        default_xlabel = "Timestep"
    else:
        if global_data is None or global_x not in global_data:
            raise ValueError(f"Global attribute '{global_x}' not found in global_data")
        x_vals_raw = np.array(global_data[global_x][:n_time])
        default_xlabel = global_x

    x_vals_all = x_vals_raw * x_scale

    # ------------------------------
    # x_min/x_max mask (time)
    # ------------------------------
    tmask = np.ones_like(x_vals_all, dtype=bool)
    if x_min is not None:
        tmask &= (x_vals_all >= x_min)
    if x_max is not None:
        tmask &= (x_vals_all <= x_max)

    if not np.any(tmask):
        raise ValueError("No timesteps fall within [x_min, x_max] after scaling")

    x_vals = x_vals_all[tmask]
    z_vals = np.array(values)[tmask, :] * c_scale  # (T_slice, R)

    # Slice ncount to match plotted time range (if provided)
    ncount_slice = None
    if ncount is not None:
        ncount_arr = np.asarray(ncount)
        if ncount_arr.shape[0] != n_time:
            raise ValueError("ncount must have the same time dimension as timesteps/values")
        ncount_slice = ncount_arr[tmask, :]

    y_edges = edges * y_scale

    # ------------------------------
    # Mask zeros if requested
    # ------------------------------
    if mask_zeros:
        z_vals = np.ma.masked_where(z_vals <= 0, z_vals)

    # ------------------------------
    # Auto vmin: smallest non-zero in plotted range
    # ------------------------------
    if vmin is None:
        vmin = _auto_vmin_smallest_positive(z_vals)

    # ------------------------------
    # Auto vmax: per-row particle-level percentile, then max across rows
    # Only if user didn't pass vmax.
    # ------------------------------
    if vmax is None and auto_vmax_mode == "row_weighted_percentile" and ncount_slice is not None:
        vmax_auto = _auto_vmax_row_weighted_percentile(
            z_vals=z_vals,
            ncount=ncount_slice,
            q=vmax_q,
            ignore_zeros=True
        )
        if vmax_auto is not None and np.isfinite(vmax_auto):
            vmax = vmax_auto

    # ------------------------------
    # Build x edges for pcolormesh (monotone x)
    # ------------------------------
    if x_vals.size == 1:
        dx = 1.0
        x_edges = np.array([x_vals[0] - 0.5 * dx, x_vals[0] + 0.5 * dx])
    else:
        x_mid = 0.5 * (x_vals[:-1] + x_vals[1:])
        left = x_vals[0] - (x_mid[0] - x_vals[0])
        right = x_vals[-1] + (x_vals[-1] - x_mid[-1])
        x_edges = np.concatenate(([left], x_mid, [right]))

    X, Y = np.meshgrid(x_edges, y_edges)

    # ------------------------------
    # Serif font
    # ------------------------------
    plt.rcParams["font.family"] = "serif"

    # ------------------------------
    # Plot heatmap
    # ------------------------------
    plt.figure(figsize=(8, 6))
    mesh = plt.pcolormesh(
        X, Y, z_vals.T,
        cmap=cmap,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
        label="_nolegend_"
    )
    plt.colorbar(mesh)

    # ------------------------------
    # Optional: overlay global_ref line + fill void above it
    # ------------------------------
    if global_ref is not None:
        if global_data is None or global_ref not in global_data:
            raise ValueError(f"global_ref '{global_ref}' not found in global_data")

        ref_all = np.array(global_data[global_ref][:n_time]) * y_scale
        ref = ref_all[tmask]

        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        void_mask = (y_centers[None, :] > ref[:, None])

        void_overlay = np.ma.masked_where(~void_mask, np.ones_like(void_mask, dtype=float))
        void_cmap = mcolors.ListedColormap([ref_color])

        plt.pcolormesh(
            X, Y, void_overlay.T,
            cmap=void_cmap,
            shading="auto",
            vmin=0, vmax=1,
            rasterized=True,
            label="_nolegend_"
        )

        if ref_line_kwargs is None:
            ref_line_kwargs = {}
        ref_line_kwargs = {
            "color": ref_line_kwargs.get("color", "black"),
            "linewidth": ref_line_kwargs.get("linewidth", 2),
            "linestyle": ref_line_kwargs.get("linestyle", "-"),
            **ref_line_kwargs,
        }
        plt.plot(x_vals, ref, label="_nolegend_", **ref_line_kwargs)

    # ------------------------------
    # Y limits
    # ------------------------------
    if ylim is not None:
        if len(ylim) != 2:
            raise ValueError("ylim must be a tuple/list: (ymin, ymax)")
        plt.ylim(ylim[0], ylim[1])

    # ------------------------------
    # Labels & title
    # ------------------------------
    if xlabel is None:
        label_x = default_xlabel
        if x_scale != 1:
            label_x = f"{default_xlabel} (scaled x{x_scale})"
    else:
        label_x = xlabel

    if ylabel is None:
        label_y = "Radius"
        if y_scale != 1:
            label_y = f"Radius (scaled x{y_scale})"
    else:
        label_y = ylabel

    plt.xlabel(label_x)
    plt.ylabel(label_y)

    plot_title = f"Radial Evolution of {attribute}" if title is None else title
    plt.title(plot_title)
    plt.grid(False)

    # ------------------------------
    # Save / show
    # ------------------------------
    if save:
        base_x = (default_xlabel if xlabel is None else xlabel).replace(" ", "_")
        base_attr = attribute.replace(" ", "_")
        out_name = f"{base_x}_vs_{base_attr}.pdf"
        savepath = os.path.join(output_dir, out_name)
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
        print(f"✅ Saved distribution plot as {savepath}")

    plt.show()

# # ============================================================
# #  Saha-based ionization fraction (LTE, first ionization)
# # ============================================================

# def compute_saha_ionization_fraction(
#     temp_vals,
#     density_vals,
#     ionization_energy_eV,
#     g_ratio=2.0,
#     min_temperature=300.0,
#     min_density=1e5,
# ):
#     """
#     Compute LTE ionization fraction q(T,n) from the Saha equation
#     for a single species with first ionization energy chi.

#     We assume:
#       - single ionization stage (neutral <-> singly ionized)
#       - charge neutrality: n_e = n_i
#       - total number density n = n_0 + n_i
#       - ionization fraction q = n_i / n

#     Saha gives:
#         (n_e n_i) / n_0 = S(T)
#       with
#         S(T) = 2 * g_ratio * (2*pi*m_e*kB*T / h^2)^(3/2) * exp(-chi/(kB*T))

#     Using n_e = n_i = q n, n_0 = (1-q)n:

#         (q^2 n^2) / ( (1-q) n ) = S(T)
#         => n q^2 / (1-q) = S(T)
#         => q^2 / (1-q) = S(T) / n

#     We solve this algebraically for q in [0,1].
#     """
#     T = np.asarray(temp_vals, dtype=float)
#     n = np.asarray(density_vals, dtype=float)

#     if T.shape != n.shape:
#         raise ValueError("temp_vals and density_vals must have the same shape")

#     # Avoid division by zero / underflow:
#     T_clipped = np.clip(T, min_temperature, None)
#     n_clipped = np.clip(n, min_density, None)

#     # Convert ionization energy from eV to Joules
#     chi_J = ionization_energy_eV * E_CHARGE

#     # Saha prefactor:
#     #   S(T) = 2 * g_ratio * (2*pi*m_e*kB*T / h^2)^(3/2) * exp(-chi / (kB*T))
#     saha_pref = 2.0 * g_ratio
#     saha_core = (2.0 * PI * M_E * K_B * T_clipped) / (H_PLANCK**2)
#     S_T = saha_pref * (saha_core**1.5) * np.exp(-chi_J / (K_B * T_clipped))

#     # A = S(T)/n:
#     A = S_T / n_clipped  # same shape as T

#     # Solve q^2 + A q - A = 0  (from q^2 = A(1-q))
#     # Positive root:
#     #   q = [-A + sqrt(A^2 + 4A)] / 2
#     disc = A**2 + 4.0 * A
#     sqrt_disc = np.sqrt(disc)
#     q = (-A + sqrt_disc) * 0.5

#     # Clamp to [0,1] for safety
#     q = np.clip(q, 0.0, 1.0)
#     return q


# def compute_emission_field_q2n2T(temp_vals, density_vals, ion_frac_vals,
#                                  temp_exponent=0.5, prefactor=1.0):
#     T = np.asarray(temp_vals)
#     n = np.asarray(density_vals)
#     q = np.asarray(ion_frac_vals)

#     if T.shape != n.shape or T.shape != q.shape:
#         raise ValueError("temp_vals, density_vals, and ion_frac_vals must have the same shape")

#     T_clipped = np.clip(T, 1e-12, None)
#     P = prefactor * (q**2) * (n**2) * (T_clipped**temp_exponent)
#     return P


# def compute_emission_field_saha(
#     temp_vals,
#     density_vals,
#     ionization_energy_eV,
#     g_ratio=2.0,
#     temp_exponent=0.5,
#     prefactor=1.0,
#     **saha_kwargs,
# ):
#     """
#     Convenience wrapper:
#       1) compute q_Saha(T,n)
#       2) compute P(r,t) ∝ q_Saha^2 n^2 T^{temp_exponent}

#     Returns
#     -------
#     P_vals : (N_t, N_r) array
#         Emission field.
#     q_saha : (N_t, N_r) array
#         Saha ionization fraction used.
#     """
#     q_saha = compute_saha_ionization_fraction(
#         temp_vals,
#         density_vals,
#         ionization_energy_eV=ionization_energy_eV,
#         g_ratio=g_ratio,
#         **saha_kwargs,
#     )
#     P_vals = compute_emission_field_q2n2T(
#         temp_vals, density_vals, q_saha,
#         temp_exponent=temp_exponent,
#         prefactor=prefactor,
#     )
#     return P_vals, q_saha


# def compute_saha_charge_distribution(
#     temp_vals,
#     density_vals,
#     radii,
#     ionization_energy_eV,
#     g_ratio=2.0,
#     saha_kwargs=None,
# ):
#     """
#     Compute Saha-based charge distribution:

#     Given T(r,t), n(r,t) and radii, we compute:
#         q_saha(r,t)   : LTE ionization fraction
#         n_e(r,t)      : electron number density = q_saha * n
#         N_e(t)        : total number of electrons in bubble vs time

#     Parameters
#     ----------
#     temp_vals : (N_t, N_r) array
#         Temperature field T(r,t).
#     density_vals : (N_t, N_r) array
#         Number density field n(r,t).
#     radii : (N_r,) array
#         Bin-center radii.
#     ionization_energy_eV : float
#         First ionization energy in eV (e.g. 15.76 for Ar).
#     g_ratio : float, optional
#         Degeneracy factor ratio g_i/g_0; default 2.0.
#     saha_kwargs : dict or None
#         Extra keyword args forwarded to compute_saha_ionization_fraction,
#         e.g. min_temperature, min_density.

#     Returns
#     -------
#     q_saha : (N_t, N_r) array
#         Saha ionization fraction q(r,t).
#     n_e : (N_t, N_r) array
#         Electron number density field.
#     N_e_total : (N_t,) array
#         Total number of electrons vs time (in units of "number of electrons").
#         Multiply by E_CHARGE to get Coulombs if desired.
#     """
#     if saha_kwargs is None:
#         saha_kwargs = {}

#     # 1) Ionization fraction from Saha
#     q_saha = compute_saha_ionization_fraction(
#         temp_vals,
#         density_vals,
#         ionization_energy_eV=ionization_energy_eV,
#         g_ratio=g_ratio,
#         **saha_kwargs,
#     )

#     # 2) Electron density
#     n = np.asarray(density_vals)
#     n_e = q_saha * n  # q_saha * n(r,t)

#     # 3) Total electrons vs time
#     shell_volumes = compute_shell_volumes(radii)  # (N_r,)
#     # For each time k: sum_j n_e[k,j] * ΔV_j
#     N_e_total = np.dot(n_e, shell_volumes)

#     return q_saha, n_e, N_e_total


# def plot_saha_total_charge(
#     timesteps,
#     N_e_total,
#     x_scale=1.0,
#     xlabel=None,
#     ylabel="Total electrons (arb. units)",
#     title="Saha-predicted total charge vs time",
#     save=False,
#     filename="saha_total_charge",
#     output_dir="output",
# ):
#     """
#     Plot Saha-based total electron count vs time as a 1D curve.
#     """
#     t = np.asarray(timesteps) * x_scale
#     N_e = np.asarray(N_e_total)

#     plt.rcParams["font.family"] = "serif"
#     plt.figure(figsize=(8, 4))
#     plt.plot(t, N_e, linewidth=2)

#     if xlabel is None:
#         xlabel = "Time" if x_scale == 1.0 else f"Time (scaled x{x_scale})"
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.grid(True)

#     if save:
#         os.makedirs(output_dir, exist_ok=True)
#         base = os.path.splitext(filename)[0]
#         out_name = os.path.join(output_dir, f"{base}.pdf")
#         plt.savefig(out_name, dpi=300, bbox_inches="tight")
#         print(f"✅ Saved Saha total charge plot as {out_name}")

#     plt.show()