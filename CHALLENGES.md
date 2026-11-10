# Audience challenges: beyond Black–Scholes

All contracts and inputs are synthetic. Work under the risk-neutral measure Q. Deliver a price, SE, contract definition and checks. Sampling intervals exclude model and integration error.

## 1. Arithmetic-average Asian call

Use S0=K=100, r=5%, sigma=20%, T=1, no dividends. Observe monthly at 1/12,...,1; exclude S0. Pay max(mean(observations)-K,0) at maturity.

- Simulate GBM at every observation date under Q.
- Discount each payoff by exp(-r*T), then compute mean and SE.
- Explain why terminal-only sampling is insufficient.
- Check: one observation at maturity recovers the European call on identical draws.
- Compare ordinary samples with antithetic pair averages. Do not substitute the geometric-average formula for this arithmetic-average contract.

## 2. A floating-rate note with coupon memory

A plain floating-rate note is not automatically a no-closed-form problem. Standard floaters can be valued using forecast and discount curves. We add coupon memory for a path-dependent simulation exercise, without asserting that no analytical method could ever exist.

### Synthetic contract

Face F=100, maturity 2 years, eight quarterly periods delta=0.25. No default, settlement lag or amortization. Pay coupons at each quarter end and principal at maturity. Initial coupon c0=4%, spread s=0.2%, floor=0%, cap=6%.

For quarter j:

```
I_j = integral of r(t) over that quarter
L_j = expm1(I_j) / delta
c_j = clip(0.5*c_(j-1) + 0.5*L_j + s, floor, cap)
D_j = exp(-sum(I_1,...,I_j))
path_PV = sum(D_j * F * delta * c_j) + D_8 * F
price = mean(path_PV)
```

This custom compounded-in-arrears coupon is not an advance-reset term-rate FRN and does not reproduce SOFR conventions. Use the SAME rate path for accrual and discounting. A constant average discount rate cannot replace pathwise discounting when cash flows and rates are correlated.

### Model and starter

Under Q, use Vasicek dr=a*(b-r)*dt+eta*dW, with r0=4%, a=0.7/year, b=4%, eta=0.015/sqrt(year). Negative rates are possible. Parameters are synthetic and not calibrated.

Run `python rate_challenges.py`. Implement the coupon recursion yourself before reading `price_paths`. The reference uses exact Gaussian rate transitions but APPROXIMATE trapezoidal integrated rates, initially 20 substeps per quarter. Exact rate transitions do not imply exact discounted cash flows.

Vary memory weights, spread, cap and floor separately. Explain how two coupon histories can give different next coupons even with the same current L_j.

## 3. Trust a price without its own closed form

1. Remove memory, spread, cap and floor: coupon_j=F*expm1(I_j). Discounted coupons and principal telescope to F on EVERY path. This validates timing/discount consistency, not the rate dynamics. The identity applies to this matched single-curve synthetic accrual convention, not every real floater.
2. Set eta=0 and r0=b. Calculate coupon recursion and PV independently with constant rates.
3. Raise the cap on identical paths. Positive discount factors and the increasing coupon recursion imply PV cannot decrease.
4. Compare mean(exp(-sum(I_j))) against the analytic Vasicek zero-coupon price printed by the script. Finite samples and time integration prevent exact agreement.
5. Increase N at fixed grid to study sampling error; refine substeps at large N to study integration bias. Use repeated seeds or coupled fine/coarse paths. The basic script changes its random stream with substeps, so one price difference alone cannot identify bias.
6. Stretch: an issuer-callable note needs an exercise policy or backward induction. Do not assume automatic redemption on every path.

Report N, substeps, seed, price and SE in a convergence table. Explain remaining limitations.

## References

- [QuantLib native bond example](https://github.com/lballabio/QuantLib/blob/master/Examples/Bonds/Bonds.cpp), with floating coupons and discount-curve valuation.
- [Jones and Jacobs, History Dependent Financial Claims: A Monte Carlo Approach](https://www.sfu.ca/~rjones/econ811/readings/monte.pdf).
