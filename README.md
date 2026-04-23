# Astro 1221 - Hubble Diagram Project

This project uses Union2.1 Type Ia supernova data to build a Hubble diagram and estimate cosmological parameters.

## Project Goals 

**Project 7: Type Ia Supernova Hubble Diagram (2011 Nobel Prize; discovery 1998)**

This project creates a fit for the relationship between supernova distance and redshift to measure the universe's expansion.

-  Load date from 20+ supernovae with measured mu and z
-  Fit for Hubble Constant
-  Creating a Hubble diagram
-  Plotting

## Files

- `get_data.py`: parses `SCPUnion2.1_AllSNe.tex` and returns cleaned arrays `(z, mu, mu_err)`.
- `Plot_Data.py`: low-redshift fit (`z < 0.1`) using `mu = 5 log10(z) + C`, then estimates `H0`.
- `cos_func.py`: flat LambdaCDM fit over all valid redshifts (`OmegaM`, `H0`, with `OmegaLambda = 1 - OmegaM`).
- `SCPUnion2.1_AllSNe.tex`: input dataset.

## Requirements

- Python 3.9+
- `numpy`, `matplotlib`, `scipy`

```bash
pip install numpy matplotlib scipy
```

## Run

From the project root:

```bash
python Plot_Data.py
python cos_func.py
```

### What each script outputs

- `Plot_Data.py`:
  - low-`z` linear fit and `H0` estimate
  - Hubble diagram + residuals
  - saved figure `hubble_fit.png`
- `cos_func.py`:
  - best-fit `OmegaM`, `OmegaLambda`, `H0`, and reduced chi-squared
  - `H(z)` vs redshift plot
  - full Hubble diagram comparing LambdaCDM and low-`z` linear approximation

## Expected Results (Sanity Checks)

- `H0` is often around `~60-80 km/s/Mpc`.
- `OmegaM` is typically positive and below 1 (often `~0.2-0.4`).
- `OmegaLambda = 1 - OmegaM` is usually positive.
- Reduced chi-squared near `~1` indicates a good fit.
- Residuals should generally scatter around 0 without strong trends.

These are practical checks, not strict grading cutoffs.

## Troubleshooting

- Missing packages (`ModuleNotFoundError`):
  - run `python -m pip install numpy matplotlib scipy`
- Missing dataset file:
  - keep `SCPUnion2.1_AllSNe.tex` in the project root, or update `TEX_FILE` in `get_data.py`
- `cos_func.py` fit failure:
  - verify valid rows are still present and `mu_err > 0`
- Plot window not showing:
  - run in a normal local terminal; `Plot_Data.py` still writes `hubble_fit.png`
 
## AI Usage

**AI Tools Used**

- Cursor
- Claude

**What AI Helped with**

- Help create plotting code
- Debugging
- Identify mathematical errors
- Drafting Documentation 

## Notes

- `\nodata` entries are converted to `NaN` during parsing.
- Both analyses use `mu_err` as the observational uncertainty.

## What deviations reveal about dark energy

To put it simply, the deviations reveal that the universe is in fact expanding, thus "dark energy" does exist. 
Source: https://science.nasa.gov/dark-energy/
