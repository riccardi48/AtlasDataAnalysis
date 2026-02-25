import numpy as np
from scipy.special import erf
from scipy.stats import landau

def electron_diffusion_length_p_type(
    N_A,
    T=300.0,
    # Mobility parameters (300 K typical values)
    mu_n0=1350.0,      # cm^2/Vs
    mu_n_min=65.0,     # cm^2/Vs
    N_ref=1e17,        # cm^-3
    alpha=0.7,
    # Recombination parameters
    sigma_n=1e-15,     # cm^2 (SRH capture cross section)
    v_th=1e7,          # cm/s (thermal velocity)
    C_n=2.8e-31        # cm^6/s (Auger coefficient)
):
    """
    Returns minority electron diffusion length L_n (cm)
    for p-type silicon with acceptor concentration N_A (cm^-3).
    """

    # Physical constants
    k = 1.380649e-23      # J/K
    q = 1.602176634e-19   # C

    # Einstein relation factor (kT/q) in volts
    kT_q = (k * T) / q

    # Caughey–Thomas mobility model
    mu_n = mu_n_min + (mu_n0 - mu_n_min) / (
        1.0 + (N_A / N_ref)**alpha
    )

    # Diffusion coefficient (cm^2/s)
    D_n = kT_q * mu_n

    # Lifetime including SRH + Auger
    tau_n = 1.0 / (sigma_n * v_th * N_A + C_n * N_A**2)

    # Diffusion length (cm)
    L_n = np.sqrt(D_n * tau_n)

    return L_n

def depletion_width_scaled(
    NA,                 # New acceptor concentration (cm^-3)
    VR,                 # New reverse bias voltage (V)
    NA0=2e14,           # Reference concentration (cm^-3)
    VR0=48.6,           # Reference reverse bias (V)
    Wp0_um=37.0*10**(-6),        # Reference depletion width (micrometers)
    Vbi=0.7             # Built-in voltage estimate (V)
):
    """
    Returns depletion width (micrometers) for a one-sided p-n junction.

    Scaling is relative to a known reference point.
    Assumes abrupt junction and ND >> NA.
    """

    numerator = (Vbi + VR) / NA
    denominator = (Vbi + VR0) / NA0

    Wp_um = Wp0_um * np.sqrt(numerator / denominator)

    return Wp_um


if __name__ == "__main__":
    N_A = 2e14  # cm^-3
    L = electron_diffusion_length_p_type(N_A)
    print("Diffusion length (µm):", L * 1e4)

