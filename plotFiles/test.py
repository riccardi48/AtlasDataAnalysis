"""
Error propagation for total charge and frequency above threshold.

Pipeline:
  fit params (V_0, t_epi, edl)
      -> f(depth): piecewise charge collection efficiency  (base = 0 fixed)
      -> Q_total = integral of f over sensor depth
      -> P(Q > threshold) via Landau CDF

Analytical propagation via Jacobian at each step.

With base = 0 the piecewise function simplifies to:
    f(depth) = V_0                                   for depth < t_epi
    f(depth) = V_0 * exp(-(depth - t_epi) / edl)    for depth >= t_epi

And the analytic integral becomes:
    Q = V_0 * t_epi + V_0 * edl * [1 - exp(-L/edl)]
      = V_0 * [t_epi + edl * (1 - exp(-L/edl))]
where L = D - t_epi  (substrate thickness).

Usage:
    Set your fit parameters, covariance matrix, sensor geometry,
    Landau MPV/width, and threshold below, then run.
"""

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# 1.  Landau CDF
#     scipy has no native Landau distribution.
#     We use the moyal distribution as the standard analytic approximation.
#     Replace with your own CDF if you have a more accurate implementation.
# ---------------------------------------------------------------------------
landau = stats.moyal


def landau_cdf(q, mpv, width):
    """P(Q <= q) for a Landau(mpv, width) distribution."""
    return landau.cdf(q, loc=mpv, scale=width)


def landau_pdf(q, mpv, width):
    """Landau probability density at q."""
    return landau.pdf(q, loc=mpv, scale=width)


# ---------------------------------------------------------------------------
# 2.  Piecewise charge collection efficiency  f(depth; V_0, t_epi, edl)
#     base = 0 throughout
# ---------------------------------------------------------------------------

def f(depth, V_0, t_epi, edl):
    """
    Charge collection efficiency at each depth (base = 0).

    Parameters
    ----------
    depth  : array_like  depths at which to evaluate [same units as t_epi]
    V_0    : float       plateau value in epitaxial region
    t_epi  : float       epitaxial layer thickness
    edl    : float       exponential decay length beyond t_epi
    """
    depth = np.asarray(depth, dtype=float)
    result = np.empty_like(depth)
    mask = depth < t_epi
    result[mask] = V_0
    d_sub = depth[~mask]
    result[~mask] = V_0 * np.exp(-(d_sub - t_epi) / edl)
    return result


# ---------------------------------------------------------------------------
# 3.  Total charge  Q = integral_{0}^{D} f(depth) d(depth)
#
#     Q = V_0 * t_epi  +  V_0 * edl * [1 - exp(-L/edl)]
#       = V_0 * [t_epi + edl * (1 - exp(-L/edl))]
#
#     where L = D - t_epi
# ---------------------------------------------------------------------------

def Q_total(V_0, t_epi, edl, D):
    """
    Analytic total charge integrated from 0 to D (base = 0).

    D : float   total sensor depth
    """
    L = D - t_epi
    return V_0 * (t_epi + edl * (1.0 - np.exp(-L / edl)))


# ---------------------------------------------------------------------------
# 4.  Jacobian  dQ/d(params),  params = [V_0, t_epi, edl]
#
#     dQ/dV_0   = t_epi + edl*(1 - exp(-L/edl))
#               = Q / V_0                            [convenient cross-check]
#
#     dQ/dt_epi = V_0 * (1 - exp(-L/edl))
#               [flat region gains +1, substrate lower limit shifts by -1]
#
#     dQ/dedl   = V_0 * [1 - exp(-L/edl) - (L/edl)*exp(-L/edl)]
# ---------------------------------------------------------------------------

def dQ_dparams(V_0, t_epi, edl, D):
    """
    Returns gradient array [dQ/dV_0, dQ/dt_epi, dQ/dedl].
    """
    L = D - t_epi
    exp_term = np.exp(-L / edl)

    dQ_dV0  = t_epi + edl * (1.0 - exp_term)
    dQ_dtep = V_0 * (1.0 - exp_term)
    dQ_dedl = V_0 * (1.0 - exp_term - (L / edl) * exp_term)

    return np.array([dQ_dV0, dQ_dtep, dQ_dedl])


# ---------------------------------------------------------------------------
# 5.  Propagation to sigma_Q
#     sigma_Q^2 = J_Q^T . Cov . J_Q
# ---------------------------------------------------------------------------

def sigma_Q(V_0, t_epi, edl, D, cov):
    """
    Uncertainty on total charge via first-order error propagation.

    cov : (3,3) covariance matrix for [V_0, t_epi, edl]
    """
    J = dQ_dparams(V_0, t_epi, edl, D)
    return np.sqrt(J @ cov @ J)


# ---------------------------------------------------------------------------
# 6.  Frequency above threshold  P_above = 1 - CDF(threshold; Q_total, xi)
#
#     Q_total is used as the Landau location parameter (MPV).
#     xi (Landau width) is treated as fixed and known separately.
#
#     dP_above/dQ_mpv = +pdf(threshold; Q_mpv, xi)   [chain rule on 1-CDF]
#     => sigma_P = pdf(threshold; Q_mpv, xi) * sigma_Q
# ---------------------------------------------------------------------------

def frequency_above_threshold(Q_mpv, xi, threshold):
    """P(charge > threshold) where charge ~ Landau(Q_mpv, xi)."""
    return 1.0 - landau_cdf(threshold, Q_mpv, xi)


def sigma_P_above(Q_mpv, xi, threshold, sig_Q):
    """
    Uncertainty on P(charge > threshold) propagated from sigma_Q.

    dP/dQ_mpv = pdf(threshold; Q_mpv, xi)
    """
    return landau_pdf(threshold, Q_mpv, xi) * sig_Q


# ---------------------------------------------------------------------------
# 7.  Full uncertainty budget  (per-param contribution to sigma_P)
# ---------------------------------------------------------------------------

def uncertainty_budget(V_0, t_epi, edl, D, cov, threshold):
    """
    Compute P_above, sigma_P, and the breakdown by fit parameter.

    Parameters
    ----------
    V_0, t_epi, edl : fit parameter values
    D               : total sensor depth
    cov             : (3,3) covariance matrix, order [V_0, t_epi, edl]
    xi              : Landau width (fixed, not propagated)
    threshold       : charge threshold

    Returns
    -------
    dict with keys: V_0, t_epi, edl, total, Q_mpv, sigma_Q, P_above, dP_dQ
    """
    J_Q   = dQ_dparams(V_0, t_epi, edl, D)       # shape (3,)
    Q_mpv = Q_total(V_0, t_epi, edl, D)
    xi = Q_mpv/4
    dpdf  = landau_pdf(threshold*50, Q_mpv, xi)      # dP/dQ_mpv

    J_P = dpdf * J_Q                              # dP/d(params), shape (3,)

    var_P_total = J_P @ cov @ J_P                 # full propagation (with correlations)
    var_P_diag  = J_P**2 * np.diag(cov)           # diagonal only (for budget display)

    names  = ["V_0", "t_epi", "edl"]
    budget = {name: np.sqrt(v) for name, v in zip(names, var_P_diag)}
    budget["total"]   = np.sqrt(var_P_total)
    budget["Q_mpv"]   = Q_mpv
    budget["sigma_Q"] = np.sqrt(J_Q @ cov @ J_Q)
    budget["P_above"] = frequency_above_threshold(Q_mpv, xi, threshold*50)
    budget["dP_dQ"]   = dpdf
    return budget


# ===========================================================================
# EXAMPLE  ---  replace with your actual values
# ===========================================================================

if __name__ == "__main__":

    # --- Fit results --------------------------------------------------------
    V_0   = 1.0     # plateau efficiency (or voltage)
    t_epi = 25.0    # epitaxial thickness [um]
    edl   = 15.0    # exponential decay length [um]

    # Covariance matrix from your fit, order: [V_0, t_epi, edl]
    # e.g. from scipy.optimize.curve_fit:  popt, pcov = curve_fit(...)
    # or from iminuit:  cov = np.array(m.covariance)
    cov = np.array([
        [1e-4,  0.0,   0.0 ],
        [0.0,   0.25,  0.0 ],
        [0.0,   0.0,   0.25],
    ])

    # --- Sensor geometry ----------------------------------------------------
    D = 100.0       # total sensor depth [um]

    # --- Landau parameters --------------------------------------------------
    # xi is the Landau width --- from a separate calibration or literature.
    xi = 5.0        # [same units as Q]

    # --- Threshold ----------------------------------------------------------
    threshold = 45.0

    # --- Run ----------------------------------------------------------------
    budget = uncertainty_budget(V_0, t_epi, edl, D, cov, threshold)

    print("=" * 55)
    print("  Charge collection --- error propagation summary")
    print("=" * 55)
    print(f"  Q_total (Landau MPV)  : {budget['Q_mpv']:.4f}")
    print(f"  sigma_Q               : {budget['sigma_Q']:.4f}")
    print(f"  dP/dQ at threshold    : {budget['dP_dQ']:.6f}")
    print()
    print(f"  P(Q > {threshold})    : {budget['P_above']:.6f}")
    print(f"  sigma_P (total)       : {budget['total']:.6f}")
    print()
    print("  Uncertainty budget (sigma_P from each param):")
    print("  [diagonal contributions --- excludes cross-correlations]")
    for name in ["V_0", "t_epi", "edl"]:
        frac = (budget[name] / budget["total"]) * 100 if budget["total"] > 0 else 0
        print(f"    {name:8s}  sigma_P = {budget[name]:.6f}  ({frac:.1f}%)")
    print("=" * 55)

    # --- Optional: scan over threshold --------------------------------------
    print("\nThreshold scan:")
    print(f"  {'Threshold':>12}  {'P_above':>10}  {'sigma_P':>10}  {'rel err %':>10}")
    for thr in np.linspace(threshold * 0.5, threshold * 1.5, 9):
        Q   = budget["Q_mpv"]
        P   = frequency_above_threshold(Q, xi, thr)
        sP  = sigma_P_above(Q, xi, thr, budget["sigma_Q"])
        rel = (sP / P * 100) if P > 0 else float("inf")
        print(f"  {thr:12.2f}  {P:10.6f}  {sP:10.6f}  {rel:10.2f}")