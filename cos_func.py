import numpy as np
import matplotlib.pyplot as plt
from get_data import load_sn_arrays
from scipy.optimize import minimize
from scipy.constants import c
from scipy.integrate import quad

# --- 1. Load SN Ia data (shared cleaning + err + sig_int in quadrature) ---
z, mu, mu_err = load_sn_arrays()


def adv_h_calc(z_values, OmegaM, OmegaA, H0):
    '''
    This function takes redshift, OmegaM, OmegaA, and H0
    and returns the Hubble parameter as a function of redshift.
    It assumes flat curvature of spacetime (OmegaK = 0),
    but allows other values to test alternatives.
    '''
    OmegaK = 1 - (OmegaM + OmegaA)
    inside = (OmegaM * (1 + z_values) ** 3) + (OmegaK * (1 + z_values) ** 2) + OmegaA
    Hz = H0 * np.sqrt(inside)
    return Hz

C_KM_S = c / 1e3

def mu_model(z_values, OmegaM, OmegaA, H0):
    """Distance-modulus prediction from (OmegaM, OmegaA, H0)."""
    OmegaK = 1 - (OmegaM + OmegaA)

    z_sorted_idx = np.argsort(z_values)
    z_sorted = z_values[z_sorted_idx]
    # Integrate from z=0 (prepend 0). Using only data z_min as the lower limit
    # makes the first trapezoid start at z_min, leaves χ(z_min)=0 and d_L≈0 → invalid μ.
    z_grid = np.concatenate(([0.0], z_sorted))
    e2 = (OmegaM * (1 + z_grid) ** 3) + (OmegaK * (1 + z_grid) ** 2) + OmegaA
    if np.any(e2 <= 0) or H0 <= 0:
        return None

    inv_e = 1.0 / np.sqrt(e2)
    dz = np.diff(z_grid)
    # Trapezoid cumulative integral of dz/E(z) from 0 to z_grid[j].
    cum_int = np.concatenate(([0.0], np.cumsum(0.5 * (inv_e[1:] + inv_e[:-1]) * dz)))
    chi_at_data = cum_int[1:]  # χ(z) at each z_sorted

    if np.isclose(OmegaK, 0.0):
        d_m = chi_at_data
    elif OmegaK > 0:
        sqrt_ok = np.sqrt(OmegaK)
        d_m = np.sinh(sqrt_ok * chi_at_data) / sqrt_ok
    else:
        sqrt_abs_ok = np.sqrt(-OmegaK)
        d_m = np.sin(sqrt_abs_ok * chi_at_data) / sqrt_abs_ok

    d_l_sorted = (C_KM_S / H0) * (1 + z_sorted) * d_m
    if np.any(d_l_sorted <= 0):
        return None

    mu_sorted = 5 * np.log10(d_l_sorted) + 25
    mu_pred = np.empty_like(mu_sorted)
    mu_pred[z_sorted_idx] = mu_sorted
    return mu_pred


def chi_squared_flat(params, z_values, mu_obs, mu_unc):
    """
    Flat LCDM with OmegaLambda = 1 - OmegaM.
    Fit both OmegaM and H0 directly to the Union2.1 distance-modulus data.
    """
    OmegaM, H0 = params
    OmegaA = 1.0 - OmegaM
    if OmegaM < 0 or OmegaA < 0 or H0 <= 0:
        return 1e30
    mu_pred = mu_model(z_values, OmegaM, OmegaA, H0)
    if mu_pred is None or not np.all(np.isfinite(mu_pred)):
        return 1e30
    residuals = (mu_obs - mu_pred) / mu_unc
    return np.sum(residuals ** 2)


# --- 3. Fit flat LCDM parameters using all valid redshifts from load_sn_arrays ---
z_fit = z
mu_fit = mu
mu_err_fit = mu_err
print(f"Using {len(z_fit)} supernovae across all valid z > 0 values")

result = minimize(
    chi_squared_flat,
    x0=np.array([0.30, 70.0]),
    args=(z_fit, mu_fit, mu_err_fit),
    method="L-BFGS-B",
    bounds=[(0.05, 0.95), (40.0, 100.0)],
)

if not result.success:
    raise RuntimeError(f"Parameter fit failed: {result.message}")

best_OmegaM, best_H0 = float(result.x[0]), float(result.x[1])
best_OmegaA = 1.0 - best_OmegaM
best_chi2 = result.fun
ndof = len(z_fit) - 2
reduced_chi2 = best_chi2 / ndof if ndof > 0 else np.nan

print(
    f"Best fit (flat LCDM, free H0): OmegaM={best_OmegaM:.4f}, "
    f"OmegaA={best_OmegaA:.4f}, H0={best_H0:.3f} km/s/Mpc"
)
print(f"chi^2 = {best_chi2:.2f}, chi^2_red = {reduced_chi2:.3f}")

Hz_array = adv_h_calc(z_fit, best_OmegaM, best_OmegaA, best_H0)
Hz_at_z0 = adv_h_calc(0.0, best_OmegaM, best_OmegaA, best_H0)


# --- 4. Plot H(z) using the best-fit cosmology ---
fig, ax1 = plt.subplots(1, figsize=(8, 8))
fig.suptitle("Best-Fit H(z) vs Redshift (all valid z)", fontsize=13)
ax1.scatter(z_fit, Hz_array, s = 2, color="navy")
ax1.annotate(f"H(z=0) = {Hz_at_z0:.2f} km/s/Mpc", xy=(0.05, 0.92),
             xycoords='axes fraction', fontsize=11, color='darkgreen')
ax1.annotate(r"$H_0$" + f"(fit from SN data) = {best_H0:.2f}", xy=(0.05, 0.86),
             xycoords='axes fraction', fontsize=10, color='darkred')
ax1.annotate(r"$\chi^2_{\text{red}}$" + f" = {reduced_chi2:.3f}", xy=(0.05, 0.75),
             xycoords='axes fraction', fontsize=10, color='black')
ax1.set_ylabel("H(z) (km/s/Mpc)")
ax1.set_xlabel("Redshift")
ax1.grid(True, alpha=0.3)
plt.show()


#5. Hubble Diagram: predicted distance modulus vs redshift
#   E(z) = sqrt( Omega_m*(1+z)^3 + Omega_k*(1+z)^2 + Omega_Lambda )
#   d_H  = c / H0
#   d_C  = d_H * integral_0^z  dz' / E(z')
#   d_L  = (1+z) * d_C
#   mu   = 5 * log10(d_L [Mpc]) + 25

def E_z(z_prime, OmegaM, OmegaA):
    """Dimensionless Hubble rate E(z) = H(z)/H0; the integrand's denominator."""
    OmegaK = 1.0 - OmegaM - OmegaA
    return np.sqrt(OmegaM * (1.0 + z_prime)**3 + OmegaK * (1.0 + z_prime)**2 + OmegaA)


def comoving_distance(z_obs, OmegaM, OmegaA, H0):
    """Comoving distance d_C = (c/H0) * integral_0^z dz'/E(z'), in Mpc."""
    d_H = C_KM_S / H0                                          # Hubble distance in Mpc
    integrand = lambda z_prime: 1.0 / E_z(z_prime, OmegaM, OmegaA)
    integral_value, _ = quad(integrand, 0.0, z_obs)            # quad integrates from 0 to z_obs
    return d_H * integral_value


def distance_modulus(z_obs, OmegaM, OmegaA, H0):
    """
    Predicted distance modulus at redshift z_obs.
      d_C = d_H * integral_0^z dz'/E(z')     [comoving distance]
      d_L = (1+z) * d_C                       [luminosity distance, flat space]
      mu  = 5 * log10(d_L [Mpc]) + 25         [distance modulus]
    """
    d_C = comoving_distance(z_obs, OmegaM, OmegaA, H0)
    d_L = (1.0 + z_obs) * d_C                 # luminosity distance in Mpc
    return 5.0 * np.log10(d_L) + 25.0         # +25 converts Mpc to the standard 10 pc baseline


#Compute predicted mu at each observed supernova redshift
mu_predicted_data = np.array([
    distance_modulus(z_i, best_OmegaM, best_OmegaA, best_H0)
    for z_i in z_fit
])

#Chi^2: sum of ((observed - predicted) / uncertainty)^2; chi^2_red ~ 1 means good fit
residuals       = (mu_fit - mu_predicted_data) / mu_err_fit
chi2_hubble     = np.sum(residuals**2)
ndof_hubble     = len(z_fit) - 2
chi2_red_hubble = chi2_hubble / ndof_hubble

print(f"Hubble diagram chi^2 = {chi2_hubble:.2f},  chi^2_red = {chi2_red_hubble:.3f}")

#Smooth model curve: 300 log-spaced z values so the plotted line looks continuous
z_smooth  = np.logspace(np.log10(z_fit.min()), np.log10(z_fit.max()), 300)
mu_smooth = np.array([
    distance_modulus(z_i, best_OmegaM, best_OmegaA, best_H0)
    for z_i in z_smooth
])

# Plot: two panels sharing the same x-axis
fig2, (ax_hub, ax_res) = plt.subplots(2, 1, figsize=(10, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
fig2.suptitle("Type Ia Supernova Hubble Diagram", fontsize=14, fontweight='bold')

# Top panel: data points with error bars, and the model curve on top
ax_hub.errorbar(z_fit, mu_fit, yerr=mu_err_fit, fmt='o', markersize=2.5,color='steelblue', ecolor='lightsteelblue', alpha=0.75, label="Union2.1 SNe Ia")
ax_hub.plot(z_smooth, mu_smooth, color='crimson', linewidth=2, label=rf"Flat $\Lambda$CDM: $\Omega_M$={best_OmegaM:.3f}, $\Omega_\Lambda$={best_OmegaA:.3f}, $H_0$={best_H0:.1f} km/s/Mpc")
ax_hub.annotate(rf"$\chi^2$={chi2_hubble:.1f}, $\chi^2_{{red}}$={chi2_red_hubble:.3f}", xy=(0.97, 0.05), xycoords='axes fraction', fontsize=9, ha='right')
ax_hub.set_ylabel(r"Distance Modulus $\mu$ (mag)")
ax_hub.legend(fontsize=8)
ax_hub.grid(True, alpha=0.3)

# Bottom panel: residuals in units of sigma, should scatter around 0
ax_res.scatter(z_fit, residuals, s=4, color='steelblue', alpha=0.6)
ax_res.axhline(0,  color='crimson', linestyle='--')   # zero line
ax_res.axhline(+1, color='gray',    linestyle=':')    # ±1 sigma guides
ax_res.axhline(-1, color='gray',    linestyle=':')
ax_res.set_ylabel(r"Residual ($\sigma$)")
ax_res.set_xlabel(r"Redshift $z$ (log scale)")
ax_res.set_xscale('log')
ax_res.set_ylim(-5, 5)
ax_res.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()