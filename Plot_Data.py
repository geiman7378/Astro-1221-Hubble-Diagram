import numpy as np
import matplotlib.pyplot as plt
import re

# --- 1. Parse the .tex file ---
data = []
with open("SCPUnion2.1_AllSNe.tex", "r") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("%"):
            continue

        tokens = re.findall(r'-?\d+\.\d+(?:\(\d+\.?\d*\))?', line)
        if len(tokens) < 6:
            continue

        def parse_val_err(token):
            m = re.match(r'(-?\d+\.\d+)(?:\((\d+\.?\d*)\))?', token)
            if not m:
                return None, None
            val = float(m.group(1))
            err = None
            if m.group(2):
                raw_err = m.group(2)
                val_decimals = len(m.group(1).split('.')[1]) if '.' in m.group(1) else 0
                if "." in raw_err:
                    err = float(raw_err)
                else:
                    err = float(raw_err) * 10**(-val_decimals)
            return val, err

        try:
            z            = float(tokens[0])
            #_m,  _dm     = parse_val_err(tokens[1]) these try things aren't needed for this code to work rn
            #_x1, _dx1    = parse_val_err(tokens[2]) if you need it then uncomment them.
            #_c,  _dc     = parse_val_err(tokens[3])
            mu,  dmu     = parse_val_err(tokens[4])
            if z > 0 and mu is not None and dmu is not None and dmu > 0:
                data.append((z, mu, dmu))
        except (ValueError, IndexError):
            continue

data = np.array(data)
z, mu, dmu = data[:, 0], data[:, 1], data[:, 2]

# --- 2. Filter to small redshifts ---
mask = z < 0.1
z_fit, mu_fit, dmu_fit = z[mask], mu[mask], dmu[mask]
print(f"Using {mask.sum()} supernovae with z < 0.1")

# --- 3. Fit mu = 5*log10(z) + C ---
log_z_fit = 5 * np.log10(z_fit)
C_values = mu_fit - log_z_fit
weights = 1.0 / dmu_fit**2
C_best = np.sum(weights * C_values) / np.sum(weights)
C_err = 1.0 / np.sqrt(np.sum(weights))
print(f"Best-fit constant C = {C_best:.3f} ± {C_err:.3f}")

# --- 4. Extract H0 from C ---
c_light = 3e5  # km/s
H0 = c_light / (10**((C_best - 25) / 5))
H0_err_upper = c_light * 10**((C_best + C_err - 25) / 5) - H0
H0_err_lower = H0 - c_light * 10**((C_best - C_err - 25) / 5)
print(f"H₀ = {H0:.1f} + {H0_err_upper:.1f} / - {H0_err_lower:.1f}  km/s/Mpc")

# --- 5. Plot ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8),
                                gridspec_kw={'height_ratios': [3, 1]})
fig.suptitle("Hubble Diagram — Linear Approximation (z < 0.1)", fontsize=13)

# Top panel: data + fit
log_z_all = 5 * np.log10(z_fit)
z_line = np.linspace(z_fit.min() * 0.9, z_fit.max() * 1.1, 200)
mu_line = 5 * np.log10(z_line) + C_best

ax1.errorbar(log_z_all, mu_fit, yerr=dmu_fit,
             fmt='o', markersize=4, color='steelblue',
             ecolor='lightsteelblue', elinewidth=1, capsize=2,
             label='SN Ia data')
ax1.plot(5 * np.log10(z_line), mu_line, 'r-', linewidth=2,
         label=f'Fit: μ = 5log₁₀(z) + {C_best:.2f}')
ax1.set_ylabel("Distance Modulus μ")
ax1.set_xlabel("5 log₁₀(z)")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.annotate(f"H₀ = {H0:.1f} km/s/Mpc", xy=(0.05, 0.9),
             xycoords='axes fraction', fontsize=11, color='darkred')

# Bottom panel: residuals
residuals = mu_fit - (log_z_fit + C_best)
ax2.errorbar(log_z_all, residuals, yerr=dmu_fit,
             fmt='o', markersize=4, color='steelblue',
             ecolor='lightsteelblue', elinewidth=1, capsize=2)
ax2.axhline(0, color='red', linewidth=1.5)
ax2.set_ylabel("Residuals (mag)")
ax2.set_xlabel("5 log₁₀(z)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("hubble_fit.png", dpi=150)
plt.show()