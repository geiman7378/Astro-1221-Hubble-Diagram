import numpy as np
import matplotlib.pyplot as plt
import os
from get_data import TEX_FILE, TEX_URL, load_sn_arrays


class HubbleFitter:
    C_LIGHT = 3e5  # km/s

    def __init__(self, tex_file=TEX_FILE, z_max=0.1):
        if not os.path.exists(tex_file):
            print(f"{tex_file} not found. Download from: {TEX_URL}")
            raise SystemExit(1)

        self.z_max = z_max
        self.z, self.mu, self.dmu = load_sn_arrays(tex_file)

        self.C_best = self.C_err = self.H0 = self.H0_err = None
        self.z_fit = self.mu_fit = self.dmu_fit = None

    def fit(self):
        mask = self.z < self.z_max
        self.z_fit, self.mu_fit, self.dmu_fit = self.z[mask], self.mu[mask], self.dmu[mask]

        log_z = 5 * np.log10(self.z_fit)
        weights = 1.0 / self.dmu_fit**2
        self.C_best = np.sum(weights * (self.mu_fit - log_z)) / np.sum(weights)
        self.C_err  = 1.0 / np.sqrt(np.sum(weights))

        self.H0     = self.C_LIGHT / (10 ** ((self.C_best - 25) / 5))
        self.H0_err = self.H0 - self.C_LIGHT / 10 ** ((self.C_best + self.C_err - 25) / 5)
        return self

    def plot(self, save_path="hubble_fit.png", show=True):
        log_z = 5 * np.log10(self.z_fit)
        z_line = np.linspace(self.z_fit.min() * 0.9, self.z_fit.max() * 1.1, 200)
        mu_line = 5 * np.log10(z_line) + self.C_best

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 1]})
        fig.suptitle(f"Hubble Diagram Linear Approximation (z < {self.z_max})", fontsize=13)

        ax1.errorbar(log_z, self.mu_fit, yerr=self.dmu_fit,
                     fmt='o', markersize=4, color='steelblue',
                     ecolor='lightsteelblue', elinewidth=1, capsize=2, label='SN Ia data')
        ax1.plot(5 * np.log10(z_line), mu_line, 'r-', linewidth=2,
                 label=f'Fit: μ = 5log₁₀(z) + {self.C_best:.2f}')
        tick_positions = [-13, -12, -11, -10, -9, -8, -7, -6, -5]

        ax1.set_xticks(tick_positions)
        ax1.set_xticklabels([f"{10**(t/5):.3f}" for t in tick_positions])
        ax1.set_ylabel(f"Distance Modulus" + r" $\mu = 5\log_{10}(z)$")
        ax1.set_xlabel("Redshift (z)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.annotate(f"H₀ = {self.H0:.1f} ± {self.H0_err:.2f} km/s/Mpc",
                     xy=(0.05, 0.9), xycoords='axes fraction', fontsize=11, color='darkred')

        residuals = self.mu_fit - (log_z + self.C_best)
        ax2.errorbar(log_z, residuals, yerr=self.dmu_fit,
                     fmt='o', markersize=4, color='steelblue',
                     ecolor='lightsteelblue', elinewidth=1, capsize=2)
        ax2.axhline(0, color='red', linewidth=1.5)
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels([f"{10**(t/5):.3f}" for t in tick_positions])
        ax2.set_ylabel("Residuals (mag)")
        ax2.set_xlabel("Redshift (z)")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        if show:
            plt.show()
        return fig, (ax1, ax2)


if __name__ == "__main__":
    fitter = HubbleFitter()
    fitter.fit()
    fitter.plot()