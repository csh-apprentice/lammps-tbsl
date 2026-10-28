"""
sl_saha_emission.py

Utilities for computing Saha-LTE ionization, charge distribution,
and continuum light emission from LAMMPS real-unit outputs of
fix ave/chunk (temperature + density).

Assumptions:
- LAMMPS 'units real':
    * temperature: Kelvin
    * time: fs (we do NOT convert time here; you can scale in plotting)
    * length: Å
- Density from fix ave/chunk:
    * either 'density/number' (1/Å^3), OR
    * 'density/mass' (g/cm^3)
- This module converts those densities to SI number density [1/m^3]
  before applying the Saha equation.

Main user-facing entry points you’ll likely use:
  - number_density_from_lammps_number_real
  - number_density_from_lammps_mass_real
  - compute_saha_ionization_fraction
  - compute_saha_charge_distribution
  - compute_emission_field_saha
  - compute_emission_from_lammps_number
  - compute_light_curve_from_emission
  - compute_emission_spectrum_saha
  - plot_emission_heatmap
  - plot_light_curve
  - plot_total_charge_saha
"""

import os
import numpy as np
import matplotlib.pyplot as plt

_trapezoid = getattr(np, "trapezoid", None) or np.trapz

# -------------------------
# Physical constants (SI)
# -------------------------
E_CHARGE = 1.602176634e-19   # C
K_B      = 1.380649e-23      # J/K
H_PLANCK = 6.62607015e-34    # J*s
M_E      = 9.10938356e-31    # kg
PI       = np.pi
AMU_KG   = 1.66053906660e-27 # 1 amu in kg

# Speed of light (SI) used in spectrum kernel
C_LIGHT  = 2.99792458e8      # m/s
# ============================================================
# Unit conversion helpers for LAMMPS real units
# ============================================================

def number_density_from_lammps_number_real(n_lammps):
    """
    Convert LAMMPS real-units 'density/number' (1/Å^3) to SI
    number density (1/m^3).

    n_lammps : array-like
        Number density reported by fix ave/chunk with density/number,
        in units of 1/Å^3.

    Returns
    -------
    n_SI : ndarray
        Number density in 1/m^3.
    """
    n_lammps = np.asarray(n_lammps, dtype=float)
    # 1 Å = 1e-10 m => 1 / Å^3 = 1e30 / m^3
    return n_lammps * 1.0e30


def number_density_from_lammps_mass_real(rho_lammps, atomic_mass_amu):
    """
    Convert LAMMPS real-units 'density/mass' (g/cm^3) to SI
    number density (1/m^3), given atomic mass in amu.

    rho_lammps : array-like
        Mass density from fix ave/chunk density/mass in g/cm^3.
    atomic_mass_amu : float
        Atomic/molecular mass in amu (e.g. 39.948 for Argon).

    Returns
    -------
    n_SI : ndarray
        Number density in 1/m^3.
    """
    rho_lammps = np.asarray(rho_lammps, dtype=float)
    # g/cm^3 -> kg/m^3
    rho_SI = rho_lammps * 1000.0
    m_atom = atomic_mass_amu * AMU_KG
    n_SI = rho_SI / m_atom
    return n_SI


# ============================================================
# Geometry: shells and volumes
# ============================================================

def compute_shell_edges(radii):
    """
    Compute shell edges from radial bin centers.

    radii : (N_r,) array
        Bin-center radii (e.g. from ave/chunk).
    Returns
    -------
    edges : (N_r+1,) array
        Shell edges, with edges[0] = 0.
    """
    radii = np.asarray(radii, dtype=float)
    if radii.ndim != 1 or radii.size < 2:
        raise ValueError("radii must be a 1D array with at least 2 elements")

    edges = [0.0]
    for i in range(len(radii) - 1):
        edges.append(0.5 * (radii[i] + radii[i+1]))
    edges.append(radii[-1] + (radii[-1] - radii[-2]) / 2.0)
    return np.array(edges)


def compute_shell_volumes(radii):
    """
    Compute spherical shell volumes from bin centers.

    radii : (N_r,) array
        Bin-center radii (consistent with parse_ave_chunk_output).

    Returns
    -------
    volumes : (N_r,) array
        Shell volumes: 4/3*pi*(r_{out}^3 - r_{in}^3)
        with units of (length)^3 in whatever units radii are given.
    """
    edges = compute_shell_edges(radii)
    r_in = edges[:-1]
    r_out = edges[1:]
    volumes = (4.0 / 3.0) * PI * (r_out**3 - r_in**3)
    return volumes


def compute_shell_volumes_SI(radii, length_unit="angstrom"):
    """Return shell volumes in m^3 with an explicit input length unit."""
    if length_unit == "angstrom":
        factor = 1e-30
    elif length_unit == "m":
        factor = 1.0
    else:
        raise ValueError("length_unit must be 'angstrom' or 'm'")
    return compute_shell_volumes(radii) * factor


# ============================================================
# Saha LTE ionization fraction
# ============================================================

def compute_saha_ionization_fraction(
    temp_vals,
    density_vals_SI,
    ionization_energy_eV,
    g_ratio=0.5,
    min_temperature=300.0,
    min_density=1e5,
):
    """
    Compute LTE ionization fraction q(T,n) from the Saha equation
    for a single species with first ionization energy chi, under
    the simplifying assumptions:

      - single ionization stage (neutral <-> singly ionized),
      - charge neutrality: n_e = n_i,
      - total number density n = n_0 + n_i,
      - ionization fraction q = n_i / n.

    Saha equation (in one common form):
        (n_e n_i) / n_0 = S(T)

      with
        S(T) = 2 * g_ratio
               * (2*pi*m_e*kB*T / h^2)^(3/2)
               * exp(-chi / (kB*T))

    Using n_e = n_i = q n, n_0 = (1-q)n:

        (q^2 n^2) / ((1-q) n) = S(T)
        => n q^2 / (1-q) = S(T)
        => q^2 / (1-q) = S(T) / n = A

    We then solve the quadratic:
        q^2 + A q - A = 0
      => q = [-A + sqrt(A^2 + 4A)] / 2  (physical root in [0,1])

    Parameters
    ----------
    temp_vals : array-like, shape (N_t, N_r)
        Temperature field T(r,t) in Kelvin.
    density_vals_SI : array-like, shape (N_t, N_r)
        Number density field n(r,t) in 1/m^3.
    ionization_energy_eV : float
        First ionization energy in eV (e.g. 15.76 for Argon).
    g_ratio : float, optional
        Degeneracy factor ratio g_i / g_0; default 0.5.
    min_temperature : float, optional
        Minimum temperature (K) for numerical stability.
    min_density : float, optional
        Minimum number density (1/m^3) for numerical stability.

    Returns
    -------
    q : ndarray, shape (N_t, N_r)
        Ionization fraction q_Saha in [0,1].
    """
    T = np.asarray(temp_vals, dtype=float)
    n = np.asarray(density_vals_SI, dtype=float)

    if T.shape != n.shape:
        raise ValueError("temp_vals and density_vals_SI must have the same shape")

    # Clip to avoid division by zero and underflow
    T_clipped = np.clip(T, min_temperature, None)
    n_clipped = np.clip(n, min_density, None)

    # Convert ionization energy from eV to Joules
    chi_J = ionization_energy_eV * E_CHARGE

    # Saha prefactor:
    #   S(T) = 2 * g_ratio * (2*pi*m_e*kB*T / h^2)^(3/2) * exp(-chi/(kB*T))
    saha_pref = 2.0 * g_ratio
    saha_core = (2.0 * PI * M_E * K_B * T_clipped) / (H_PLANCK**2)
    S_T = saha_pref * (saha_core**1.5) * np.exp(-chi_J / (K_B * T_clipped))

    # A = S(T)/n
    A = S_T / n_clipped

    # Algebraically identical physical root without large-A cancellation.
    # Evaluate each branch on a bounded argument, including A=0 and A=inf.
    small = np.minimum(A, 1.0)
    root = np.sqrt(small)
    q_small = 2.0 * root / (root + np.sqrt(small + 4.0))
    q_large = 2.0 / (1.0 + np.sqrt(1.0 + 4.0 / np.maximum(A, 1.0)))
    q = np.where(A <= 1.0, q_small, q_large)

    q = np.clip(q, 0.0, 1.0)
    return q


# ============================================================
# Emission field (q^2 n^2 T^{1/2}) and light curve
# ============================================================

def compute_emission_field_q2n2T(temp_vals, density_vals_SI, ion_frac_vals,
                                 temp_exponent=0.5, prefactor=1.0):
    """
    Compute continuum emission proxy:

        P(r,t) ∝ q(r,t)^2 n(r,t)^2 T(r,t)^{temp_exponent}

    where q is an ionization fraction (e.g., Saha result),
    n is number density (SI, 1/m^3), and T in K.

    Returns
    -------
    P : ndarray, shape (N_t, N_r)
        Emission field P(r,t) in arbitrary units.
    """
    T = np.asarray(temp_vals, dtype=float)
    n = np.asarray(density_vals_SI, dtype=float)
    q = np.asarray(ion_frac_vals, dtype=float)

    if T.shape != n.shape or T.shape != q.shape:
        raise ValueError("temp_vals, density_vals_SI, ion_frac_vals must share shape")

    T_clipped = np.clip(T, 1e-12, None)
    P = prefactor * (q**2) * (n**2) * (T_clipped**temp_exponent)
    return P


def compute_emission_field_saha(
    temp_vals,
    density_vals_SI,
    radii,
    ionization_energy_eV,
    g_ratio=0.5,
    temp_exponent=0.5,
    prefactor=1.0,
    saha_kwargs=None,
):
    """
    Convenience wrapper:
      1) compute q_Saha(T,n),
      2) compute P(r,t) ∝ q_Saha^2 n^2 T^{temp_exponent},
      3) compute global light curve I(t) = ∫ P dV.

    Parameters
    ----------
    temp_vals : (N_t, N_r) array, in Kelvin.
    density_vals_SI : (N_t, N_r) array, in 1/m^3.
    radii : (N_r,) array of bin centers (same units as used in LAMMPS).
    ionization_energy_eV : float
        First ionization energy in eV.
    g_ratio, temp_exponent, prefactor : see above.
    saha_kwargs : dict or None
        Extra keyword args for compute_saha_ionization_fraction, e.g.
        min_temperature, min_density.

    Returns
    -------
    P_vals : (N_t, N_r) array
        Emission field.
    q_saha : (N_t, N_r) array
        Saha ionization fraction used.
    I_t : (N_t,) array
        Global intensity vs time, I(t) = ∑_j P(t,j) ΔV_j.
    """
    if saha_kwargs is None:
        saha_kwargs = {}

    q_saha = compute_saha_ionization_fraction(
        temp_vals,
        density_vals_SI,
        ionization_energy_eV=ionization_energy_eV,
        g_ratio=g_ratio,
        **saha_kwargs,
    )

    P_vals = compute_emission_field_q2n2T(
        temp_vals, density_vals_SI, q_saha,
        temp_exponent=temp_exponent,
        prefactor=prefactor,
    )

    I_t = compute_light_curve_from_emission(P_vals, radii)
    return P_vals, q_saha, I_t


def compute_light_curve_from_emission(emission_vals, radii):
    """
    Integrate P(r,t) over volume to get a global light curve I(t).

    emission_vals : (N_t, N_r) array
        Emission field P(r,t).
    radii : (N_r,) array
        Bin-center radii.

    Returns
    -------
    I_t : (N_t,) array
        Global intensity vs time in arbitrary units.
    """
    P = np.asarray(emission_vals, dtype=float)
    shell_volumes = compute_shell_volumes(radii)

    if P.shape[1] != shell_volumes.shape[0]:
        raise ValueError("emission_vals.shape[1] must match length of radii")

    I_t = np.dot(P, shell_volumes)
    return I_t


# ============================================================
# Saha-based charge distribution
# ============================================================

def compute_saha_charge_distribution(
    temp_vals,
    density_vals_SI,
    radii,
    ionization_energy_eV,
    g_ratio=0.5,
    saha_kwargs=None,
    length_unit_for_volume="angstrom",
):
    """
    Compute Saha-based charge distribution.

    Given T(r,t), n(r,t) and radii, we compute:
        q_saha(r,t)   : LTE ionization fraction
        n_e(r,t)      : electron number density = q_saha * n
        N_e_total(t)  : total number of electrons in bubble vs time

    Parameters
    ----------
    temp_vals : (N_t, N_r) array
        Temperature field T(r,t) in K.
    density_vals_SI : (N_t, N_r) array
        Number density field n(r,t) in 1/m^3.
    radii : (N_r,) array
        Bin-center radii.
    ionization_energy_eV : float
        First ionization energy in eV.
    g_ratio : float
        Degeneracy ratio g_i / g_0.
    saha_kwargs : dict or None
        Extra arguments for compute_saha_ionization_fraction.

    Returns
    -------
    q_saha : (N_t, N_r) array
        Saha ionization fraction.
    n_e : (N_t, N_r) array
        Electron number density (1/m^3).
    N_e_total : (N_t,) array
        Total number of electrons vs time (unit: "number of electrons").
        Multiply by E_CHARGE to get Coulombs if desired.
    """
    if saha_kwargs is None:
        saha_kwargs = {}

    q_saha = compute_saha_ionization_fraction(
        temp_vals,
        density_vals_SI,
        ionization_energy_eV=ionization_energy_eV,
        g_ratio=g_ratio,
        **saha_kwargs,
    )

    n = np.asarray(density_vals_SI, dtype=float)
    n_e = q_saha * n  # electron number density

    shell_volumes = compute_shell_volumes_SI(radii, length_unit_for_volume)
    N_e_total = np.dot(n_e, shell_volumes)

    return q_saha, n_e, N_e_total


# ============================================================
# Wrapper: from LAMMPS number density directly
# ============================================================

def compute_emission_from_lammps_number(
    temp_vals,
    number_density_lammps,
    radii,
    ionization_energy_eV,
    g_ratio=0.5,
    temp_exponent=0.5,
    prefactor=1.0,
    saha_kwargs=None,
):
    """
    High-level convenience function:

    Starting from:
      - temp_vals             : T(r,t) from ave/chunk in K
      - number_density_lammps : number density from ave/chunk in 1/Å^3
      - radii                 : bin-center radii (Å or other length unit)

    We:
      1) convert density to SI number density,
      2) compute q_Saha(T,n),
      3) compute emission field P(r,t),
      4) compute global light curve I(t).

    Returns
    -------
    P_vals : (N_t, N_r) array
        Emission field.
    q_saha : (N_t, N_r) array
        Saha ionization fraction.
    I_t : (N_t,) array
        Light curve.
    n_SI : (N_t, N_r) array
        Number density in SI, in case you want to reuse.
    """
    if saha_kwargs is None:
        saha_kwargs = {}

    n_SI = number_density_from_lammps_number_real(number_density_lammps)

    P_vals, q_saha, I_t = compute_emission_field_saha(
        temp_vals=temp_vals,
        density_vals_SI=n_SI,
        radii=radii,
        ionization_energy_eV=ionization_energy_eV,
        g_ratio=g_ratio,
        temp_exponent=temp_exponent,
        prefactor=prefactor,
        saha_kwargs=saha_kwargs,
    )

    return P_vals, q_saha, I_t, n_SI


# ============================================================
# Emission spectrum (Saha LTE)
# ============================================================

def compute_emission_spectrum_saha(
    temp_vals,
    density_vals_SI,
    radii,
    ionization_energy_eV,
    lambda_grid,
    time_indices=None,
    g_ratio=0.5,
    saha_kwargs=None,
    prefactor=1.0,
):
    """
    Compute an approximate continuum emission spectrum I_lambda using
    Saha ionization and a simple LTE free-free-like kernel:

        j_lambda(r,t) ∝ q^2 n^2 T^{-1/2} exp(-hc / (λ k_B T))

    Then we integrate over radial shells and selected time steps.

    Parameters
    ----------
    temp_vals : (N_t, N_r) array, K
    density_vals_SI : (N_t, N_r) array, 1/m^3
    radii : (N_r,) array
    ionization_energy_eV : float
        First ionization energy (eV).
    lambda_grid : (N_λ,) array
        Wavelength grid in meters, e.g. np.linspace(200e-9, 800e-9, 300).
    time_indices : array-like or None
        Indices of time steps to include (e.g., flash window). If None,
        use all time steps.
    g_ratio : float
        Degeneracy ratio for Saha.
    saha_kwargs : dict or None
        Extra args forwarded to compute_saha_ionization_fraction.
    prefactor : float
        Overall factor for units (we treat I_lambda in arbitrary units).

    Returns
    -------
    I_lambda : (N_λ,) array
        Spectral emission in arbitrary units.
    q_saha_sel : (N_t_sel, N_r) array
        Saha ionization fraction for the selected time window (for inspection).
    """
    if saha_kwargs is None:
        saha_kwargs = {}

    T = np.asarray(temp_vals, dtype=float)
    n = np.asarray(density_vals_SI, dtype=float)

    if T.shape != n.shape:
        raise ValueError("temp_vals and density_vals_SI must share shape")

    N_t, N_r = T.shape

    if time_indices is None:
        time_indices = np.arange(N_t)
    else:
        time_indices = np.asarray(time_indices, dtype=int)

    # Saha ionization
    q_saha_full = compute_saha_ionization_fraction(
        T,
        n,
        ionization_energy_eV=ionization_energy_eV,
        g_ratio=g_ratio,
        **saha_kwargs,
    )
    T_sel = T[time_indices, :]
    n_sel = n[time_indices, :]
    q_sel = q_saha_full[time_indices, :]

    # Constants for spectral kernel
    h = H_PLANCK
    c = 2.99792458e8
    kB = K_B

    # Clip T to avoid overflow
    T_sel_clipped = np.clip(T_sel, 1e-6, None)
    # Base factor: q^2 n^2 T^{-1/2}
    base = (q_sel**2) * (n_sel**2) * (T_sel_clipped**(-0.5))

    # Geometry
    shell_volumes = compute_shell_volumes(radii)  # (N_r,)

    I_lambda = np.zeros_like(lambda_grid, dtype=float)

    for i, lam in enumerate(lambda_grid):
        exp_factor = np.exp(- (h * c) / (lam * kB * T_sel_clipped))  # (N_t_sel, N_r)
        j = base * exp_factor
        I_lambda[i] = np.sum(j * shell_volumes[None, :])

    return prefactor * I_lambda, q_sel



# ============================================================
# Spectral post-processing (normalized to bolometric P_br)
# ============================================================

def _spectral_shape_kernel(lambda_m, T, include_lambda_power=True):
    r"""Return a *shape-only* continuum kernel K(\lambda;T).

    This is NOT a calibrated absolute emissivity. It is only used to
    distribute a known bolometric emissivity P_br across wavelength.

    K(\lambda;T) \propto exp(-hc/(\lambda k_B T)) * (\lambda^{-2} optional)

    Parameters
    ----------
    lambda_m : float
        Wavelength in meters.
    T : ndarray
        Temperature in Kelvin.
    include_lambda_power : bool
        If True, include an extra \lambda^{-2} factor (often used when
        converting between frequency/wavelength forms in simple models).

    Returns
    -------
    K : ndarray
        Same shape as T.
    """
    T_clip = np.clip(np.asarray(T, dtype=float), 1e-6, None)
    lam = float(lambda_m)
    expo = np.exp(-(H_PLANCK * C_LIGHT) / (lam * K_B * T_clip))
    if include_lambda_power:
        expo = expo / (lam**2)
    return expo


def compute_band_fraction_from_T(
    temp_vals,
    wavelength_grid,
    wavelength_unit: str = "nm",
    wavelength_min: float = 200.0,
    include_lambda_power: bool = True,
    eps: float = 1e-300,
):
    r"""Compute the fraction of bolometric emission that lies above a cutoff.

    We assume a normalized spectral shape W(\lambda;T) \propto K(\lambda;T),
    with normalization over the provided wavelength_grid.

    f_{>=min}(T) = \int_{\lambda>=min} K d\lambda / \int K d\lambda

    Parameters
    ----------
    temp_vals : (N_t, N_r) array
        Temperature in Kelvin.
    wavelength_grid : (N_\lambda,) array
        Wavelength samples (nm by default).
    wavelength_unit : str
        "nm" or "m".
    wavelength_min : float
        Cutoff wavelength (same unit as wavelength_unit).
    include_lambda_power : bool
        Whether to include \lambda^{-2} in the shape kernel.

    Returns
    -------
    frac : (N_t, N_r) ndarray
        Fraction in [0,1].
    """
    T = np.asarray(temp_vals, dtype=float)

    lam = np.asarray(wavelength_grid, dtype=float)
    unit = (wavelength_unit or "nm").lower()
    if unit in ["nm", "nanometer", "nanometers"]:
        lam_m = lam * 1e-9
        cutoff_m = float(wavelength_min) * 1e-9
    elif unit in ["m", "meter", "meters"]:
        lam_m = lam
        cutoff_m = float(wavelength_min)
    else:
        raise ValueError("wavelength_unit must be 'nm' or 'm'")

    # Build K over grid: (Nlam, Nt, Nr)
    K_all = []
    for l in lam_m:
        K_all.append(_spectral_shape_kernel(l, T, include_lambda_power=include_lambda_power))
    K_all = np.stack(K_all, axis=0)

    denom = _trapezoid(K_all, x=lam_m, axis=0)
    denom = np.maximum(denom, eps)

    mask = lam_m >= cutoff_m
    if not np.any(mask):
        raise ValueError("No wavelengths >= cutoff within wavelength_grid")

    num = _trapezoid(K_all[mask, ...], x=lam_m[mask], axis=0)
    frac = num / denom
    return np.clip(frac, 0.0, 1.0)


def compute_normalized_spectrum_from_P(
    P_vals,
    temp_vals,
    radii,
    wavelength_grid,
    wavelength_unit: str = "nm",
    include_lambda_power: bool = True,
    length_unit_for_volume: str = "angstrom",
    eps: float = 1e-300,
):
    r"""Compute time-resolved spectrum I_\lambda(t) normalized to bolometric P.

    You already have a bolometric emissivity field P(r,t) (e.g. Schanz Eq. 12).
    This function *decomposes* that P into a wavelength-resolved spectrum via
    a shape kernel K(\lambda;T), normalized so that integrating over wavelength
    (over the provided grid) recovers the original P.

        j_\lambda(r,t) = P(r,t) * W(\lambda;T(r,t))
        W = K / \int K d\lambda

    Then:
        I_\lambda(t) = \int j_\lambda(r,t) dV

    Parameters
    ----------
    P_vals : (N_t, N_r) array
        Bolometric emissivity field.
    temp_vals : (N_t, N_r) array
        Temperature field in Kelvin.
    radii : (N_r,) array
        Radial bin centers.
    wavelength_grid : (N_\lambda,) array
        Wavelength samples.
    wavelength_unit : str
        "nm" or "m".

    Returns
    -------
    I_lambda_t : (N_t, N_\lambda) ndarray
        Spectrum vs time (arb units consistent with P * volume).
    lam_native : ndarray
        Wavelength grid in the same unit as input (returned for convenience).
    """
    P = np.asarray(P_vals, dtype=float)
    T = np.asarray(temp_vals, dtype=float)
    if P.shape != T.shape:
        raise ValueError("P_vals and temp_vals must share shape (N_t, N_r)")

    lam = np.asarray(wavelength_grid, dtype=float)
    unit = (wavelength_unit or "nm").lower()
    if unit in ["nm", "nanometer", "nanometers"]:
        lam_m = lam * 1e-9
        lam_native = lam
    elif unit in ["m", "meter", "meters"]:
        lam_m = lam
        lam_native = lam
    else:
        raise ValueError("wavelength_unit must be 'nm' or 'm'")

    dV = compute_shell_volumes_SI(radii, length_unit_for_volume)

    Nt, Nr = P.shape
    Nlam = lam_m.size

    # Build K(lam;T) for each wavelength
    K_all = np.zeros((Nlam, Nt, Nr), dtype=float)
    for i, l in enumerate(lam_m):
        K_all[i] = _spectral_shape_kernel(l, T, include_lambda_power=include_lambda_power)

    denom = _trapezoid(K_all, x=lam_m, axis=0)
    denom = np.maximum(denom, eps)

    # weights W: (Nlam, Nt, Nr)
    W = K_all / denom[None, :, :]

    # integrate over volume for each lambda
    I_lambda_t = np.zeros((Nt, Nlam), dtype=float)
    for i in range(Nlam):
        I_lambda_t[:, i] = np.sum(P * W[i] * dV[None, :], axis=1)

    return I_lambda_t, lam_native


def integrate_spectrum_over_wavelength(
    I_lambda_t,
    wavelength_grid,
    wavelength_unit: str = "nm",
    wavelength_min: float = 200.0,
):
    r"""Integrate a time-resolved spectrum over wavelength >= cutoff.

    Parameters
    ----------
    I_lambda_t : (N_t, N_\lambda) array
    wavelength_grid : (N_\lambda,) array
    wavelength_unit : str
        "nm" or "m".
    wavelength_min : float
        Cutoff wavelength.

    Returns
    -------
    I_band_t : (N_t,) array
    """
    lam = np.asarray(wavelength_grid, dtype=float)
    unit = (wavelength_unit or "nm").lower()
    if unit in ["nm", "nanometer", "nanometers"]:
        cutoff = float(wavelength_min)
    elif unit in ["m", "meter", "meters"]:
        cutoff = float(wavelength_min)
    else:
        raise ValueError("wavelength_unit must be 'nm' or 'm'")

    mask = lam >= cutoff
    if not np.any(mask):
        raise ValueError("No wavelengths >= cutoff within wavelength_grid")

    return _trapezoid(np.asarray(I_lambda_t)[:, mask], x=lam[mask], axis=1)


def compute_band_limited_emission_from_P(
    P_vals,
    temp_vals,
    wavelength_grid,
    wavelength_unit: str = "nm",
    wavelength_min: float = 200.0,
    include_lambda_power: bool = True,
):
    """Compute band-limited emissivity field P_{>=min}(r,t) from bolometric P.

    Uses the fast shortcut:
        P_band(r,t) = P(r,t) * f_{>=min}(T(r,t))

    Returns
    -------
    P_band : (N_t, N_r) ndarray
    frac : (N_t, N_r) ndarray
    """
    frac = compute_band_fraction_from_T(
        temp_vals,
        wavelength_grid=wavelength_grid,
        wavelength_unit=wavelength_unit,
        wavelength_min=wavelength_min,
        include_lambda_power=include_lambda_power,
    )
    return np.asarray(P_vals, dtype=float) * frac, frac


def time_integrate_spectrum(I_lambda_t, timesteps, start_idx=None, end_idx=None):
    r"""Integrate I_\lambda(t) over time to get a time-integrated spectrum."""
    I = np.asarray(I_lambda_t, dtype=float)
    t = np.asarray(timesteps, dtype=float)
    Nt = I.shape[0]
    if start_idx is None:
        start_idx = 0
    if end_idx is None or end_idx > Nt:
        end_idx = Nt
    sl = slice(start_idx, end_idx)
    return _trapezoid(I[sl, :], x=t[sl], axis=0)


def plot_spectrum(
    wavelength_grid,
    spectrum,
    xlabel="Wavelength (nm)",
    ylabel="Emission (arb. units)",
    title="Spectrum",
    save=False,
    filename="spectrum",
    output_dir="output",
):
    """Plot a wavelength spectrum with serif fonts and PDF saving."""
    lam = np.asarray(wavelength_grid, dtype=float)
    S = np.asarray(spectrum, dtype=float)

    plt.rcParams["font.family"] = "serif"
    plt.figure(figsize=(7, 4))
    plt.plot(lam, S, linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

    if save:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(filename)[0]
        out_name = os.path.join(output_dir, f"{base}.pdf")
        plt.savefig(out_name, dpi=300, bbox_inches="tight")
        print(f"✅ Saved spectrum plot as {out_name}")

    plt.show()

# ============================================================
# Plotting helpers (serif + PDF)
# ============================================================

def plot_emission_heatmap(
    timesteps,
    radii,
    emission_vals,
    xlabel="Time",
    ylabel="Radius",
    title="Emission distribution",
    cmap="hot",
    log_scale=True,
    vmin=None,
    vmax=None,
    strong_contrast=False,
    contrast_percentiles=(5, 99.5),
    save=False,
    filename="emission_heatmap",
    output_dir="output",
):
    """
    Quick heatmap for P(r,t) with serif fonts and PDF saving.

    Parameters
    ----------
    timesteps : (N_t,) array
        Time grid (centers).
    radii : (N_r,) array
        Radial bin centers.
    emission_vals : (N_t, N_r) array
        Emission field P(r,t) in arbitrary units.
    """
    t = np.asarray(timesteps, dtype=float)
    r = np.asarray(radii, dtype=float)
    Z = np.asarray(emission_vals, dtype=float)

    if Z.shape != (t.size, r.size):
        raise ValueError(
            f"emission_vals shape {Z.shape} must be (len(timesteps), len(radii)) "
            f"= ({t.size}, {r.size})"
        )

    plt.rcParams["font.family"] = "serif"
    plt.figure(figsize=(8, 6))

    # ----- determine vmin/vmax and norm -----
    import matplotlib.colors as mcolors

    Z_plot = Z.T  # so y-axis = radius, x-axis = time

    norm = None
    if log_scale:
        # Only consider positive values for log scaling
        Z_pos = np.where(Z_plot > 0, Z_plot, np.nan)
        if strong_contrast and vmin is None and vmax is None:
            flat = Z_pos[~np.isnan(Z_pos)]
            if flat.size > 0:
                vmin = np.nanpercentile(flat, contrast_percentiles[0])
                vmax = np.nanpercentile(flat, contrast_percentiles[1])
        if vmin is None:
            vmin = np.nanmin(Z_pos) if np.isfinite(Z_pos).any() else 1e-30
        if vmax is None:
            vmax = np.nanmax(Z_pos) if np.isfinite(Z_pos).any() else 1.0
        vmin = max(vmin, 1e-30)
        vmax = max(vmax, vmin * 10.0)
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        if strong_contrast and vmin is None and vmax is None:
            flat = Z_plot.ravel()
            vmin = np.nanpercentile(flat, contrast_percentiles[0])
            vmax = np.nanpercentile(flat, contrast_percentiles[1])

    # ----- plot with imshow -----
    # extent = [x_min, x_max, y_min, y_max]
    extent = [t[0], t[-1], r[0], r[-1]]

    im = plt.imshow(
        Z_plot,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
    )

    cbar = plt.colorbar(im)
    cbar.set_label("Emission (arb. units)")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if save:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(filename)[0]
        out_name = os.path.join(output_dir, f"{base}.pdf")
        plt.savefig(out_name, dpi=300, bbox_inches="tight")
        print(f"✅ Saved emission heatmap as {out_name}")

    plt.show()



def plot_light_curve(
    timesteps,
    intensity,
    x_scale=1.0,
    xlabel=None,
    ylabel="Intensity (arb. units)",
    title="Light curve",
    save=False,
    filename="light_curve",
    output_dir="output",
):
    """
    Plot global intensity vs time with serif fonts and PDF output.
    """
    t = np.asarray(timesteps, dtype=float) * x_scale
    I = np.asarray(intensity, dtype=float)

    plt.rcParams["font.family"] = "serif"
    plt.figure(figsize=(8, 4))

    plt.plot(t, I, linewidth=2)
    if xlabel is None:
        xlabel = "Time" if x_scale == 1.0 else f"Time (scaled x{x_scale})"
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


def plot_total_charge_saha(
    timesteps,
    N_e_total,
    x_scale=1.0,
    xlabel=None,
    ylabel="Total electrons (arb. units)",
    title="Saha total charge vs time",
    save=False,
    filename="saha_total_charge",
    output_dir="output",
):
    """
    Plot Saha-based total electron count vs time.
    """
    t = np.asarray(timesteps, dtype=float) * x_scale
    N_e = np.asarray(N_e_total, dtype=float)

    plt.rcParams["font.family"] = "serif"
    plt.figure(figsize=(8, 4))
    plt.plot(t, N_e, linewidth=2)

    if xlabel is None:
        xlabel = "Time" if x_scale == 1.0 else f"Time (scaled x{x_scale})"
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

    if save:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(filename)[0]
        out_name = os.path.join(output_dir, f"{base}.pdf")
        plt.savefig(out_name, dpi=300, bbox_inches="tight")
        print(f"✅ Saved Saha total charge plot as {out_name}")

    plt.show()


def band_fraction_ge_lambda_closed_form(temp_vals_K, lambda_c_nm=200.0):
    """
    Closed-form fraction of emission in wavelengths >= lambda_c
    for kernel K(λ;T) = λ^{-2} exp(-a/λ), a = hc/(k_B T),
    normalized over λ in (0, ∞).

    Returns f_ge: same shape as temp_vals_K, in [0,1].
    """
    T = np.asarray(temp_vals_K, dtype=float)
    T = np.clip(T, 1e-6, None)

    lambda_c_m = float(lambda_c_nm) * 1e-9
    a = (H_PLANCK * C_LIGHT) / (K_B * T)  # meters
    x = a / lambda_c_m

    # f_ge = 1 - exp(-a/lambda_c)
    f_ge = -np.expm1(-x)
    return f_ge


def band_limited_emission_from_P_closed_form(P_vals, temp_vals_K, lambda_c_nm=200.0):
    """
    P_band(r,t) = P_bolometric(r,t) * f_{>=lambda_c}(T(r,t))
    using closed-form fraction over λ∈(0,∞).
    """
    f = band_fraction_ge_lambda_closed_form(temp_vals_K, lambda_c_nm=lambda_c_nm)
    return np.asarray(P_vals, float) * f, f


def normalized_spectrum_from_P_closed_form(
    P_vals, temp_vals_K, radii, wavelength_grid_nm,
    integrate_over_volume=True,
    length_unit_for_volume="angstrom",
):
    """
    Builds I_lambda(t) (or j_lambda(t,r,λ)) using closed-form normalization over λ∈(0,∞):

      j_λ(r,t) = P(r,t) * a * λ^{-2} exp(-a/λ),  a=hc/(kT)

    wavelength_grid_nm is only for sampling/plotting (normalization is analytic).
    """
    P = np.asarray(P_vals, dtype=float)
    T = np.asarray(temp_vals_K, dtype=float)
    if P.shape != T.shape:
        raise ValueError("P_vals and temp_vals_K must have shape (Nt, Nr)")

    lam_m = np.asarray(wavelength_grid_nm, dtype=float) * 1e-9
    lam_m = np.clip(lam_m, 1e-30, None)  # avoid divide-by-zero

    # a(t,r) in meters
    T_clip = np.clip(T, 1e-6, None)
    a = (H_PLANCK * C_LIGHT) / (K_B * T_clip)  # (Nt, Nr)

    # W(t,r,λ) = a * λ^-2 * exp(-a/λ)
    # compute as (Nt, Nr, Nlam) but you can avoid full 3D if only want I_lambda(t)
    expo = np.exp(-a[:, :, None] / lam_m[None, None, :])
    W = a[:, :, None] * (lam_m[None, None, :] ** -2) * expo
    j_lambda = P[:, :, None] * W  # (Nt, Nr, Nlam)

    if not integrate_over_volume:
        return j_lambda, wavelength_grid_nm

    dV = compute_shell_volumes_SI(radii, length_unit_for_volume)
    I_lambda_t = np.sum(j_lambda * dV[None, :, None], axis=1)  # (Nt, Nlam)
    return I_lambda_t, wavelength_grid_nm
