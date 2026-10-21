"""Sensitivity of the dense-attempt worst case to the positivity bound eta (T=60, D=12)."""
import json, numpy as np
from fractions import Fraction as Fr
from core import *
T, D = 60, 12
nom = risk_dense(example_chain(0.4, 0.9), T, D)
rows = []
for eta in [0.01, 0.02, 0.05, 0.1]:
    lo, hi, _ = rho_interval({w: float(v) for w, v in C_EXAMPLE.items()}, 1, eta)
    # b irrelevant to first order; use middle of its eta-range
    blo, bhi = 9 * eta, min(1 - eta, 9 * (1 - eta))
    b = 0.5 * (blo + bhi)
    rs = np.linspace(lo, hi, 1501)
    R = [risk_dense(example_chain(1 - r, b), T, D) for r in rs]
    i = int(np.argmax(R))
    rows.append(dict(eta=eta, rho_lo=lo, rho_hi=hi, rho_worst=float(rs[i]), R_plus=float(R[i]), R_minus=float(min(R)), ratio_nom=float(R[i] / nom)))
    print(rows[-1])
json.dump(dict(R_nom=nom, rows=rows), open("results/eta.json", "w"), indent=1)
