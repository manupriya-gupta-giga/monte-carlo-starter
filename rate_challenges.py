"""Synthetic rate challenge: exact Vasicek transitions, approximate integrals.
Run: python rate_challenges.py. Custom coupon paid in arrears, not a term-rate FRN.
"""
import math
import numpy as np
from monte_carlo import summarize


def simulate_period_integrals(n=20_000, substeps=20, seed=42,
                              r0=.04, a=.7, b=.04, eta=.015,
                              periods=8, delta=.25):
    if n < 2 or substeps < 1 or periods < 1 or a <= 0 or eta < 0 or delta <= 0:
        raise ValueError("Invalid simulation inputs")
    rng = np.random.default_rng(seed)
    dt = delta / substeps
    decay = math.exp(-a * dt)
    sd = eta * math.sqrt(-math.expm1(-2 * a * dt) / (2 * a))
    r = np.full(n, r0, dtype=float)
    integrals = np.zeros((n, periods))
    for j in range(periods):
        for _ in range(substeps):
            nxt = b + (r - b) * decay + sd * rng.standard_normal(n)
            integrals[:, j] += .5 * (r + nxt) * dt
            r = nxt
    return integrals


def price_paths(integrals, *, face=100., delta=.25, spread=.002,
                initial_coupon=.04, floor=0., cap=.06, memory=True):
    """Baseline: memory=False, spread=0, floor=-inf, cap=inf."""
    if floor > cap or face <= 0 or delta <= 0:
        raise ValueError("Invalid contract inputs")
    discount = np.exp(-np.cumsum(integrals, axis=1))
    accrued = np.expm1(integrals) / delta
    previous = np.full(len(integrals), initial_coupon, dtype=float)
    coupons = np.empty_like(integrals)
    for j in range(integrals.shape[1]):
        raw = (.5 * previous + .5 * accrued[:, j] if memory else accrued[:, j])
        previous = np.clip(raw + spread, floor, cap)
        coupons[:, j] = previous
    return (discount * face * delta * coupons).sum(axis=1) + face * discount[:, -1]


def vasicek_zero_coupon(t=2., r0=.04, a=.7, b=.04, eta=.015):
    B = -math.expm1(-a * t) / a
    A = math.exp((b - eta**2 / (2*a*a)) * (B-t) - eta**2 * B*B / (4*a))
    return A * math.exp(-B*r0)


def checks():
    integrals = simulate_period_integrals(n=2000)
    plain = price_paths(integrals, memory=False, spread=0., floor=-np.inf, cap=np.inf)
    assert np.allclose(plain, 100., atol=1e-10, rtol=0)
    assert np.all(price_paths(integrals, cap=.08) >= price_paths(integrals, cap=.04)-1e-12)
    assert np.array_equal(integrals, simulate_period_integrals(n=2000))
    deterministic = simulate_period_integrals(n=10, eta=0., r0=.04, b=.04)
    assert np.allclose(deterministic, .04*.25)
    c, expected = .04, 0.
    L = math.expm1(.04*.25)/.25
    for j in range(1, 9):
        c = min(.06, max(0., .5*c + .5*L + .002))
        expected += 100.*.25*c*math.exp(-.04*.25*j)
    expected += 100.*math.exp(-.04*2.)
    assert np.allclose(price_paths(deterministic), expected, atol=1e-10)
    assert math.isclose(vasicek_zero_coupon(eta=0.), math.exp(-.04*2.), abs_tol=1e-12)
    return expected


def main():
    deterministic = checks()
    integrals = simulate_period_integrals()
    print("Contract checks passed")
    print("Custom memory-coupon note:", summarize(price_paths(integrals)))
    print("Constant-rate reference:", deterministic)
    print("Zero-coupon MC:", summarize(np.exp(-integrals.sum(axis=1))))
    print("Vasicek zero-coupon benchmark:", vasicek_zero_coupon())
    print("CI covers sampling error only. Refine substeps to study integration bias.")


if __name__ == '__main__':
    main()
