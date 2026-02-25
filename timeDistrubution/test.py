"""
Solver for the charge accumulation equation:

    q(x0, t) = (N/4) * [2*cosh((x0-w2)/Ln)
                         - exp((x0-w2)/Ln) * erf((x0-w2)/(2*Ln) * sqrt(tau_n/t) + sqrt(t/tau_n))
                         - exp(-(x0-w2)/Ln) * erf((x0-w2)/(2*Ln) * sqrt(tau_n/t) - sqrt(t/tau_n))]

Steps:
  1. q_func(t, x0, N, ...) — evaluates q at a given t
  2. solve_t(T, x0, N, ...) — finds t such that q(x0, t) = T  (scalar N)
  3. sample_t_distribution(T, x0, n_samples, ...) — draws N from a Landau
     distribution and returns the resulting distribution of t values
"""
import sys
sys.path.append("..")
import numpy as np
from scipy.special import erf
from scipy.optimize import brentq
from scipy.stats import moyal          # moyal ≈ Landau distribution
import matplotlib.pyplot as plt
from scipy.stats.sampling import NumericalInversePolynomial
from landau import landau

class LandauDist:
    def __init__(self,mu,sig):
        self.mu = mu
        self.sig = sig
    def pdf(self, x):
       return landau.pdf(x,self.mu,self.sig)
    def cdf(self, x):
       return landau.cdf(x,self.mu,self.sig)



# ---------------------------------------------------------------------------
# 1.  Core equation
# ---------------------------------------------------------------------------

def q_func(t, x0, N, w2, Ln, tau_n):
    """
    Evaluate q(x0, t) for scalar or array t.

    Parameters
    ----------
    t     : float or array — time (must be > 0)
    x0    : float — fixed position
    N     : float — amplitude (charge scale factor)
    w2    : float — boundary/reference position
    Ln    : float — diffusion length
    tau_n : float — carrier lifetime
    """
    t = np.asarray(t, dtype=float)
    xi = (x0 - w2) / Ln                     # dimensionless position

    sqrt_ratio  = np.sqrt(tau_n / t)         # sqrt(tau_n / t)
    sqrt_ratio2 = np.sqrt(t / tau_n)         # sqrt(t / tau_n)

    arg_plus  = xi / 2 * sqrt_ratio + sqrt_ratio2
    arg_minus = xi / 2 * sqrt_ratio - sqrt_ratio2

    q = (N / 4) * (
        2 * np.cosh(xi)
        - np.exp( xi) * erf(arg_plus)
        - np.exp(-xi) * erf(arg_minus)
    )
    return q


# ---------------------------------------------------------------------------
# 2.  Root-finder: solve q(x0, t) = T for t  (scalar N)
# ---------------------------------------------------------------------------

def solve_t(T, x0, N, w2, Ln, tau_n,
            t_min=1e-12, t_max=1e3, xtol=1e-12):
    """
    Find t > 0 such that q(x0, t) = T.

    Uses Brent's method on the interval [t_min, t_max].
    Returns NaN if no root is found in that interval.

    Parameters
    ----------
    T         : float — target charge value
    x0, N, w2, Ln, tau_n : floats — equation parameters
    t_min, t_max : bracket for the root search (adjust to your time scale)
    xtol      : absolute tolerance for t
    """
    def residual(t):
        return q_func(t, x0, N, w2, Ln, tau_n) - T

    fa = residual(t_min)
    fb = residual(t_max)

    if fa * fb > 0:
        # No sign change — T may be outside the range of q over [t_min, t_max]
        return np.nan

    t_sol = brentq(residual, t_min, t_max, xtol=xtol)
    return t_sol


# ---------------------------------------------------------------------------
# 3.  Distribution of t when N ~ Landau
# ---------------------------------------------------------------------------

def sample_t_distribution(T, x0, w2, Ln, tau_n,rng,
                           n_samples=5000,
                           landau_loc=0.0, landau_scale=1.0,
                           t_min=1e-12, t_max=1e3,
                           verbose=True):
    """
    Draw N from a Landau distribution, then for each realisation find the
    t that satisfies q(x0, t) = T.  Returns an array of valid t values.

    The Landau distribution is approximated by the Moyal distribution
    (scipy.stats.moyal), which is the standard analytic approximation.

    Parameters
    ----------
    T             : float — target charge value
    x0            : float — fixed position
    w2, Ln, tau_n : floats — equation parameters
    n_samples     : int   — number of Monte-Carlo draws
    landau_loc    : float — location parameter of Landau (peak shift)
    landau_scale  : float — scale parameter of Landau (width)
    t_min, t_max  : floats — time bracket for root search
    verbose       : bool  — print progress summary
    """
    # Draw N values from Landau (Moyal approximation)
    N_samples = rng.rvs(n_samples)

    # N must be positive (it's a charge scale); keep only positive draws
    N_samples = N_samples[N_samples > 0]

    t_values = []
    n_failed = 0

    for N in N_samples:
        t = solve_t(T, x0, N, w2, Ln, tau_n, t_min=t_min, t_max=t_max)
        if np.isnan(t):
            n_failed += 1
        else:
            t_values.append(t)

    t_values = np.array(t_values)

    if verbose:
        print(f"Samples attempted : {len(N_samples)}")
        print(f"Successful solves : {len(t_values)}")
        print(f"Failed (no root)  : {n_failed}")
        if len(t_values):
            print(f"t  mean  = {t_values.mean():.4g}")
            print(f"t  std   = {t_values.std():.4g}")
            print(f"t  range = [{t_values.min():.4g}, {t_values.max():.4g}]")

    return t_values


# ---------------------------------------------------------------------------
# 4.  Quick demo / usage example
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # --- Physical parameters (adjust to your system) ---
    x0    = 52*10**(-6)
    w2    = 37*10**(-6)
    Ln    = 42*10**(-6)
    D     = 36*10**(-4)
    tau_n   = (Ln**2) / D
    T     = 38     # target charge value

    # ---- Part A: solve for t with a single deterministic N ----
    N_det = 100
    t_sol = solve_t(T, x0, N_det, w2, Ln, tau_n)
    print(f"[Part A] q(x0, t*) = T = {T}  =>  t* = {t_sol:.6g} s")

    # Verify
    q_check = q_func(t_sol, x0, N_det, w2, Ln, tau_n)
    print(f"         Verification: q(x0, t*) = {q_check:.6g}  (should be {T})\n")

    # ---- Part B: distribution of t when N ~ Landau ----
    print("[Part B] Sampling t distribution with Landau-distributed N ...\n")
    dist = LandauDist(100,100/4)
    urng = np.random.default_rng()
    rng = NumericalInversePolynomial(dist, random_state=urng)
    N = rng.rvs(100000)

    t_dist = sample_t_distribution(
        T, x0, w2, Ln, tau_n,rng,
        n_samples    = 50000,
        t_min        = 1e-12,
        t_max        = 1e3,
    )

    # ---- Plot ----
    if len(t_dist) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].hist(t_dist, bins=200, color="steelblue", edgecolor="white", linewidth=0.4,range=(0,500*10**(-9)))
        axes[0].set_xlabel("t  [s]")
        axes[0].set_ylabel("Count")
        axes[0].set_title(f"Distribution of t  (q = T = {T},  x0 = {x0})")
        axes[0].set_xlim(0,500*10**(-9))

        axes[1].hist(np.log10(t_dist), bins=60, color="darkorange", edgecolor="white", linewidth=0.4)
        axes[1].set_xlabel("log₁₀(t)  [s]")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Distribution of log₁₀(t)")

        plt.tight_layout()
        plt.savefig("/home/atlas/rballard/AtlasDataAnalysis/timeDistrubution/test.pdf")
