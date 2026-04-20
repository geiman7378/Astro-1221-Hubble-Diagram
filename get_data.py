import os
import numpy as np

# Download the file if it doesn't exist
TEX_FILE = 'SCPUnion2.1_AllSNe.tex'
TEX_URL = 'https://supernova.lbl.gov/Union/figures/SCPUnion2.1_AllSNe.tex'

# Creates an array with each row being a supernova, and each column being a different property (all are unitless).
class SNData:
    def __init__(self, filepath):
        SN        = [] # Name
        z         = [] # Redshift
        m_b       = [] # Magnitude in the B band
        x1        = [] # Stretch
        color     = [] # Color (how red or blue it is)
        mu        = [] # Distance modulus
        sig_int   = [] # Intrinsic dispersion
        m_b_err   = [] # Error in magnitude in the B band
        x1_err    = [] # Error in stretch
        color_err = [] # Error in color
        mu_err    = [] # Error in distance modulus

        with open(filepath, 'r') as f:
            for line in f:
                # Remove the \\ at the end and split by &
                line = line.strip().replace('\\\\', '')
                cols = line.split('&')
                if len(cols) < 7:
                    continue

                SN.append(cols[0].strip())
                z.append(float(cols[1].strip()))

                # Columns 3-6 look like "16.86(0.19)" where 16.86 is the value and 0.19 is the error.
                # We loop over all four of these columns at once. Each iteration gives us:
                # col_list - the list we want to add the main value to (e.g. m_b)
                # err_list - the list we want to add the error to (e.g. m_b_err)
                # raw - the raw text from that column (e.g. "16.86(0.19)")
                for col_list, err_list, raw in [
                    (m_b,   m_b_err,   cols[2].strip()),
                    (x1,    x1_err,    cols[3].strip()),
                    (color, color_err, cols[4].strip()),
                    (mu,    mu_err,    cols[5].strip()),
                ]:
                    if raw == r'\nodata':
                        col_list.append(np.nan)
                        err_list.append(np.nan)
                    else:
                        # Split "16.86(0.19)" into ["16.86", "0.19)"]
                        parts = raw.split('(')
                        value = float(parts[0])
                        # Strip the closing ) before converting the error to a float
                        error = float(parts[1].strip(')'))
                        col_list.append(value)
                        err_list.append(error)

                raw_sig = cols[6].strip()
                sig_int.append(np.nan if raw_sig == r'\nodata' else float(raw_sig))

        # Store everything as a numpy structured array so you can do data['m_b'] etc.
        self.array = np.array(
            list(zip(SN, z, m_b, x1, color, mu, sig_int,
                     m_b_err, x1_err, color_err, mu_err)),
            dtype=[
                ('SN','U20'),
                ('z','f8'),
                ('m_b','f8'),
                ('x1','f8'),
                ('color','f8'),
                ('mu','f8'),
                ('sig_int','f8'),
                ('m_b_err','f8'),
                ('x1_err','f8'),
                ('color_err','f8'),
                ('mu_err','f8'),
            ]
        )

    def __getitem__(self, col):
        return self.array[col]

# Austin - added this here so that way Joseph and I don't have to manually format this data ourselves each time
def load_sn_arrays(tex_file=TEX_FILE):
    """
    Return SN Ia arrays ready for fitting: finite z, mu, errors; z > 0.
    Uncertainty is sqrt(mu_err^2 + sig_int^2) where intrinsic scatter is
    available (Union2.1 provides both).
    """
    sn = SNData(tex_file)
    z = sn["z"]
    mu = sn["mu"]
    mu_err = sn["mu_err"]
    sig_int = sn["sig_int"]
    valid = (
        np.isfinite(z)
        & np.isfinite(mu)
        & np.isfinite(mu_err)
        & (z > 0)
        & (mu_err > 0)
    )
    z = z[valid]
    mu = mu[valid]
    mu_err = mu_err[valid]
    sig_int = sig_int[valid]
    extra = np.where(np.isfinite(sig_int), sig_int, 0.0)
    sigma = np.sqrt(mu_err**2 + extra**2)
    return z, mu, sigma


data = SNData(TEX_FILE)