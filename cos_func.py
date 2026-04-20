import numpy as np
import matplotlib.pyplot as plt
from get_data import load_sn_arrays
from scipy.optimize import minimize


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


def mu_model(z_values, OmegaM, OmegaA, H0):
    """Distance-modulus prediction from (OmegaM, OmegaA, H0)."""
    c_light = 3e5  # km/s
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

    d_l_sorted = (c_light / H0) * (1 + z_sorted) * d_m
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
ax1.annotate(f"H0 (fit from SN data) = {best_H0:.2f}", xy=(0.05, 0.86),
             xycoords='axes fraction', fontsize=10, color='darkred')
ax1.annotate(f"chi^2_red = {reduced_chi2:.3f}", xy=(0.05, 0.75),
             xycoords='axes fraction', fontsize=10, color='black')
ax1.set_ylabel("H(z) (km/s/Mpc)")
ax1.set_xlabel("Redshift")
ax1.grid(True, alpha=0.3)
plt.show()
