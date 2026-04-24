import numpy as np
import matplotlib.pyplot as plt
import os
from get_data import TEX_FILE, TEX_URL, load_sn_arrays


class HubbleFitter:
    C_LIGHT = 3e5  # km/s
    
    def __init__(self, tex_file=TEX_FILE, z_max=0.1):
        '''
            First thing this function does is check if the data exists. if it doesn't, it returns an error.
            Secondly, it defines the maximum redshift (z_max) to be the value you add into the function (0.1 in this case).
            It also grabs the data from christans class in get_data and sets z (redshift), mu (distance_modulus), and dmu (uncertianty of the modulus) to the respective values in the data
            It then just sets all the constents you need (C_best, C_err, H0, H0_err, z_fit, mu_fit, dmu_fit) to 0 just incase they had a value already for some reason.
        '''
        if not os.path.exists(tex_file):
            print(f"{tex_file} not found. Download from: {TEX_URL}")
            raise SystemExit(1)

        self.z_max = z_max
        self.z, self.mu, self.dmu = load_sn_arrays(tex_file)

        self.C_best = self.C_err = self.H0 = self.H0_err = None
        self.z_fit = self.mu_fit = self.dmu_fit = None

    def fit(self):
        ''' 
            This function handles the fitting of the data and prepares it to be plotted
            Firstly, it filters out all of the objects that have a redshift of 0.1 or higher (this is because values above that start needing other parameters like dark energy)
            It then applies that mask to the fit of the redshift, distance modulus and the error of that modulus.
            Next, the code makes it so that the scale is put through a log based scale to make it more evident that a linear approach works
            It then finds the weights of those values because if a point has a greater uncertianty it shouldn't be affecting the line as much
            Next, it solves for the hubble constant (H0) via the formula shown above and then creates the error using the same formula except adding the error to C_best
        '''
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
        '''
            This code handles the plotting of the data so you can visually see it
            firstly, it redifnes log_z because the code needs to use it again and it can't grab it from the fit function
            It then defines the line for redshift using linspace and then mu_line just applies that to the distance modulus formula.
            Right after that, it makes the default graph and gives it a name.
            Then it just creates the key so that why the user knows what the points are signifying along side the best fit line
            The code uses tick_positions to overrights the default pyplot values so its more obvious the graph is on a log based scale to the user
            The rest of the code down to the line that says "residuals" basically just handles the visuals of the first graph
            The residiuals part figures out how well of a fit the line was. when you run the code, you'll notice no distnct patterns in the points which means this fit was pretty good.

        '''
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

# this basically just runs the class if you're just running this part of the module
if __name__ == "__main__":
    fitter = HubbleFitter()
    fitter.fit()
    fitter.plot()