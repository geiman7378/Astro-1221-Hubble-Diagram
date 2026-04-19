from Plot_Data import HubbleFitter
import numpy as np
import matplotlib.pyplot as plt
from get_data import SNData, TEX_FILE


# --- 1. Load data (parsed once in get_data.py) ---
sn = SNData(TEX_FILE)
z = sn["z"]
valid = np.isfinite(z)
z = z[valid]


# --- 2. Import final H0 fit by running Plot_Data fitter ---
fitter = HubbleFitter().fit()
final_h0 = fitter.H0
print(f"Using fitted H0 = {final_h0:.3f} km/s/Mpc")


def adv_h_calc(z_values, OmegaM=0.28, OmegaA=0.72, H0=final_h0):
    '''
    This function takes redshift, OmegaM, OmegaA, and H0
    and returns the Hubble parameter as a function of redshift.
    It assumes flat curvature of spacetime (OmegaK = 0),
    but allows other values to test alternatives.
    '''
    OmegaK = 1 - (OmegaM + OmegaA)
    Hz = H0 * np.sqrt((OmegaM * (1 + z_values) ** 3) + (OmegaK * (1 + z_values) ** 2) + OmegaA)
    return Hz


# --- 3. Filter to large redshifts and compute H(z) ---
mask = z > 0.1
z_fit = z[mask]
Hz_array = adv_h_calc(z_fit)
Hz_at_z0 = adv_h_calc(0.0)
print(f"Using {mask.sum()} supernovae with z > 0.1")


# --- 4. Plotting ---
fig, ax1 = plt.subplots(1, figsize=(8, 8))
fig.suptitle("H(z) vs Redshift (z > 0.1)", fontsize=13)
ax1.scatter(z_fit, Hz_array, s = 2, color="navy")
ax1.annotate(f"H(z=0) = {Hz_at_z0:.1f} km/s/Mpc", xy=(0.05, 0.87),
             xycoords='axes fraction', fontsize=11, color='darkgreen')
ax1.set_ylabel("H(z) (km/s/Mpc)")
ax1.set_xlabel("Redshift")
ax1.grid(True, alpha=0.3)
plt.show()
