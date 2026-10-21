"""Dense-attempt regime (B >= T): verify Theorem 1 (two-sided bound) and Theorem 2
(extremal persistence, interior worst case, phase transition) on the M=2, L=1 family."""
import json, numpy as np
from fractions import Fraction as Fr
from core import *

y = float(C_EXAMPLE[(0,0)])
a_lo, a_hi = map(float, A_RANGE)
rho_lo, rho_hi = 1-a_hi, 1-a_lo
out = {"rho_range":[rho_lo,rho_hi], "y":y, "cases":[]}
A = np.linspace(a_lo, a_hi, 1201)
worst_bound_violation = 0.0
for (T,D) in [(6,6),(12,6),(30,6),(60,6),(120,6),(30,10),(60,10),(200,10),(100,15),(300,15),(100,24),(400,24)]:
    for b in [0.45, 0.7, 0.95]:
        R = []; EN=[]; LO=[]
        for a in A:
            ch = example_chain(a,b)
            r = risk_dense(ch,T,D); s = dense_stats(ch,T,D)
            R.append(r); EN.append(s["EN"]); LO.append(s["lower"])
            worst_bound_violation = max(worst_bound_violation, r - s["EN"], s["lower"] - r)
        R=np.array(R); EN=np.array(EN)
        i = int(R.argmax()); j=int(EN.argmax()); m=int(R.argmin())
        rs = rho_star(T,D,2)
        pred = rho_hi if rs is None else min(max(float(rs),rho_lo),rho_hi)
        nom = risk_dense(example_chain(0.4,0.9),T,D)
        out["cases"].append(dict(T=T,D=D,b=b, rho_argmax_R=float(1-A[i]), rho_argmax_EN=float(1-A[j]),
            rho_pred=pred, rho_star=None if rs is None else float(rs), Rmax=float(R[i]), Rmin=float(R[m]),
            rho_argmin_R=float(1-A[m]), R_nominal=float(nom), span=float(R[i]/R[m]), nominal_ratio=float(R[i]/nom)))
out["max_bound_violation"] = worst_bound_violation
for c in out["cases"]:
    if c["b"]==0.7:
        print(f"T={c['T']:4d} D={c['D']:2d}: argmax R rho={c['rho_argmax_R']:.4f}  argmax E[N]={c['rho_argmax_EN']:.4f}  pred={c['rho_pred']:.4f}  "
              f"Rmax={c['Rmax']:.3e} Rmin={c['Rmin']:.3e} (at {c['rho_argmin_R']:.3f}) Rnom={c['R_nominal']:.3e} span={c['span']:.1f} Rmax/Rnom={c['nominal_ratio']:.1f}")
print("max bound violation (should be <= ~1e-16):", worst_bound_violation)
json.dump(out, open("results/dense_M2.json","w"), indent=1)
