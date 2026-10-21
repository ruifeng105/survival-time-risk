"""Phase diagram of the worst-case persistence and exponential risk span (dense regime)."""
import json, numpy as np
from core import *
y = 0.12; rho_lo, rho_hi = 11/30, 0.95; rho_nom = 0.6; M = 2
def g(rho, k, n): return rho**k*(1+n*(1-rho))
def rho_w(k, n):
    if n <= k: return rho_hi
    return min(max(k*(n+1)/((k+1)*n), rho_lo), rho_hi)
out = {"closed_form":{}, "exact":[], "span":[]}
Ds = list(range(3, 33))
for n in [12, 48, 200, 10**9]:
    out["closed_form"][str(n)] = [rho_w(D-M, n) for D in Ds]
A = np.linspace(1-rho_hi, 1-rho_lo, 501)
for n in [12, 48, 200]:
    for D in range(4, 31, 2):
        T = D + n
        R = np.array([risk_dense(example_chain(a, 0.7), T, D) for a in A])
        out["exact"].append(dict(n=n, D=D, rho_argmax=float(1-A[R.argmax()]), rho_pred=rho_w(D-M,n)))
    print("n", n, "done")
n = 48
for D in range(3, 33):
    T = D + n
    Rw = max(risk_dense(example_chain(1-r, 0.7), T, D) for r in np.linspace(rho_lo, rho_hi, 201))
    Rlo = risk_dense(example_chain(1-rho_lo, 0.7), T, D)
    Rhi = risk_dense(example_chain(1-rho_hi, 0.7), T, D)
    Rnom = risk_dense(example_chain(0.4, 0.9), T, D)
    Rmin = min(Rlo, Rhi)
    k = D - M
    out["span"].append(dict(D=D, Rmax=Rw, Rmin=Rmin, Rnom=Rnom,
        pred_span=g(rho_w(k,n),k,n)/min(g(rho_lo,k,n), g(rho_hi,k,n)),
        pred_nom=g(rho_w(k,n),k,n)/g(rho_nom,k,n)))
for e in out["exact"]:
    if abs(e["rho_argmax"]-e["rho_pred"])>0.02: print("gap", e)
for s in out["span"][::3]:
    print(f"D={s['D']}: R+/R- exact {s['Rmax']/s['Rmin']:.3g} (E[N] pred {s['pred_span']:.3g}); R+/Rnom exact {s['Rmax']/s['Rnom']:.3g} (pred {s['pred_nom']:.3g}); Rmax={s['Rmax']:.3g}")
json.dump(out, open("results/phase.json","w"), indent=1)
