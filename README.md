# Monte Carlo option pricing starter

Companion code for **From Random Samples to Option Prices: Monte Carlo in Practice**, Manupriya Gupta's IIT Delhi MathSoc alumni session, November 10, 2026.

Price a European call with NumPy, measure sampling uncertainty, compare against Black–Scholes, and investigate common implementation mistakes. All inputs are synthetic. No market data, account, API key or paid service is needed.

[Open the guided notebook in Google Colab](https://colab.research.google.com/github/manupriya-gupta-giga/monte-carlo-starter/blob/main/monte_carlo_workshop.ipynb)

## Run locally

Use Python **3.11 or newer**. NumPy is pinned to the version used for the slide results.

```bash
git clone https://github.com/manupriya-gupta-giga/monte-carlo-starter.git
cd monte-carlo-starter
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python monte_carlo.py
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell instead.

To save the computed results as JSON:

```bash
python monte_carlo.py results.json
```

The notebook can also be opened in a local Jupyter installation. In Colab, run the setup cell first and then run cells in order. Colab is optional and may require a Google sign-in. Local execution uses no remote service after dependencies are installed.

## Expected output

For S0 = K = 100, r = 0.05, volatility = 0.20, T = 1 year, no dividends, seed 42 and NumPy 2.3.5:

```text
All deterministic checks passed
100,000 draws: price 10.420541, SE 0.046770
Approximate 95% CI [10.328872, 10.512210]
Black-Scholes 10.450584
```

The full run prints a convergence table from 100 through 1,000,000 samples. These are nested prefixes of one random stream. Another seed changes the simulation results. Software and random-generator versions can also affect reproducibility.

## What the example does

1. Generate independent standard-normal draws.
2. Sample the exact terminal geometric Brownian motion law under the risk-neutral pricing measure.
3. Apply `max(S_T - K, 0)` and discount by `exp(-r*T)`.
4. Estimate the price, standard error and approximate 95% confidence interval.
5. Compare with a same-model Black–Scholes benchmark.
6. Check maturity-zero and volatility-zero cases, reproducibility, nonnegative payoffs, strike monotonicity and invalid inputs.

The script also computes two deliberately incorrect examples, omitting discounting and substituting a 12% forecast drift, plus an antithetic-pair extension. Optional JSON output includes those results. Antithetic SE is computed from independent **pair averages**, not from dependent individual members.

## Try these changes

- Change N to 10,000 and 1,000,000. Compare SE rather than expecting the point estimate to improve on every run.
- Change the seed. Which outputs change? Why is the benchmark unchanged?
- Set volatility to zero. Predict the result `max(S0 - K*exp(-r*T), 0)` first.
- Increase the strike using identical random draws. Explain why the call payoffs cannot increase.
- Compare 100,000 ordinary payoffs with 50,000 antithetic pairs at the same payoff-evaluation budget.

## Model and interpretation

This is a no-dividend, constant-rate, constant-volatility Black–Scholes model with frictionless continuous trading assumptions. It uses risk-neutral drift `r`, not a prediction of stock returns. Terminal-only sampling is sufficient for this European payoff and avoids time-grid discretization error under this model.

The confidence interval is a large-sample normal approximation for independent finite-variance draws. Across repeated runs, approximately 95% of intervals produced by this procedure would cover the fixed model price. A realized interval is not a guarantee. It does not account for model error, calibration error or coding bugs. No live trading, recommendations, employer systems or proprietary data are included.

## Files

- `monte_carlo.py`: runnable reference, deterministic checks and computed experiments.
- `monte_carlo_workshop.ipynb`: guided, self-contained walkthrough with exercises.
- `requirements.txt`: tested NumPy dependency.

## References

- [NumPy standard-normal sampling](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.standard_normal.html)
- [MIT OCW: Black–Scholes and risk-neutral valuation](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/resources/mit18_s096f13_lecnote19/)
- [MIT OCW: derivatives and Monte Carlo](https://www.ocw.mit.edu/courses/15-450-analytics-of-finance-fall-2010/ab2510b021bee94c384057c138c343d4_MIT15_450F10_handout02.pdf)

Author: [Manupriya Gupta](https://www.linkedin.com/in/manupriya-gupta/).
