"""MathSoc IIT Delhi, 10 November 2026. Educational European call pricer.

Run: python monte_carlo.py
Requires Python 3.11+ and NumPy 2.3.5. Prices use the no-dividend Black-Scholes model.
No historical, employer, client or live market data is used.
"""
import json
import math
import sys
from pathlib import Path
import numpy as np


def validate(s0, k, r, sigma, t, n):
    if not all(math.isfinite(x) for x in (s0, k, r, sigma, t)):
        raise ValueError("Inputs must be finite")
    if s0 <= 0 or k <= 0 or sigma < 0 or t < 0:
        raise ValueError("Require S0,K > 0 and sigma,T >= 0")
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError("N must be an integer >= 2")


def discounted_payoffs(s0=100., k=100., r=.05, sigma=.20,
                       t=1., n=100_000, seed=42):
    validate(s0, k, r, sigma, t, n)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    st = s0 * np.exp((r - .5 * sigma**2) * t + sigma * np.sqrt(t) * z)
    return np.exp(-r * t) * np.maximum(st - k, 0.)


def summarize(x):
    n = len(x)
    price = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(n))
    return {"n": n, "price": price, "se": se,
            "lo": price - 1.96 * se, "hi": price + 1.96 * se}


def black_scholes_call(s0=100., k=100., r=.05, sigma=.20, t=1.):
    validate(s0, k, r, sigma, t, 2)
    if t == 0:
        return max(s0 - k, 0.)
    if sigma == 0:
        return max(s0 - k * math.exp(-r * t), 0.)
    phi = lambda x: .5 * (1. + math.erf(x / math.sqrt(2.)))
    d1 = (math.log(s0 / k) + (r + .5 * sigma**2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    return s0 * phi(d1) - k * math.exp(-r * t) * phi(d2)


def checks():
    assert math.isclose(black_scholes_call(), 10.450583572185565, abs_tol=1e-12)
    assert np.allclose(discounted_payoffs(t=0, s0=110), 10.)
    assert np.allclose(discounted_payoffs(sigma=0), black_scholes_call(sigma=0))
    assert np.array_equal(discounted_payoffs(n=100), discounted_payoffs(n=100))
    assert np.all(discounted_payoffs(n=100) >= 0)
    # Same random draws isolate the effect of changing the strike.
    assert np.all(discounted_payoffs(k=110, n=100) <= discounted_payoffs(k=100, n=100))
    for kwargs in ({"n": 1}, {"sigma": -1}, {"t": -1}, {"s0": 0}, {"r": float("nan")}):
        try:
            discounted_payoffs(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid inputs accepted: {kwargs}")


def main():
    checks()
    x = discounted_payoffs(n=1_000_000)
    result = summarize(x[:100_000])
    convergence = [summarize(x[:n]) for n in (100, 1_000, 10_000, 100_000, 1_000_000)]
    benchmark = black_scholes_call()
    # Controlled error examples on the same random draws.
    rng = np.random.default_rng(42)
    z = rng.standard_normal(100_000)
    wrong_st = 100 * np.exp((.12 - .5 * .2**2) + .2 * z)
    bugs = {"no_discount": result["price"] * math.exp(.05),
            "forecast_drift_12pct": float((math.exp(-.05) * np.maximum(wrong_st - 100, 0.)).mean())}
    # Each antithetic pair is one independent observation for its SE.
    z_pair = np.random.default_rng(42).standard_normal(50_000)
    st_plus = 100 * np.exp(.03 + .2 * z_pair)
    st_minus = 100 * np.exp(.03 - .2 * z_pair)
    pair_means = math.exp(-.05) * (np.maximum(st_plus-100, 0) + np.maximum(st_minus-100, 0)) / 2
    data = {"numpy_version": np.__version__, "seed": 42,
            "inputs": {"S0": 100, "K": 100, "r": .05, "sigma": .2, "T": 1, "q": 0},
            "benchmark": benchmark, "result": result, "convergence": convergence,
            "bugs": bugs, "antithetic": summarize(pair_means)}
    print("All deterministic checks passed")
    print(f"NumPy {np.__version__}, seed 42")
    print(f"100,000 draws: price {result['price']:.6f}, SE {result['se']:.6f}")
    print(f"Approximate 95% CI [{result['lo']:.6f}, {result['hi']:.6f}]")
    print(f"Black-Scholes {benchmark:.6f}")
    print(" N           price          SE       approx 95% CI")
    for row in convergence:
        print(f"{row['n']:>9,}  {row['price']:>11.6f}  {row['se']:>10.6f}  [{row['lo']:.6f}, {row['hi']:.6f}]")
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
