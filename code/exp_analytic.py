"""Analytic reversal criterion: sigma_1 = rho, and the per-slot decay exponents
(1/g) log sigma_g of the risk under periodic attempts with spacing g."""
import json, math, numpy as np
from fractions import Fraction as Fr
from core import *
out = {}
C1 = example_chain(0.56, 0.95); C2 = example_chain(19/30, 0.45)
# sigma_1 equals rho on random channels (M=2 and M=3)
import random; random.seed(5); err = 0
for _ in range(200):
    ch = example_chain(random.uniform(0.05, 19/30), random.uniform(0.45, 0.95))
    err = max(err, abs(sigma(ch, 1) - float(ch.rho())))
p2 = {(0,0):0.35,(0,1):0.85,(1,0):0.55,(1,1):0.92}
c3 = chain_from_p(2, p2).marginal(3); R3 = box_ranges(c3, 2, 0.05)
for _ in range(200):
    ch = box_chain(c3, 2, {m: random.uniform(*R3[m]) for m in words(2)})
    err = max(err, abs(sigma(ch, 1) - float(ch.rho())))
out["max_err_sigma1_eq_rho"] = err
print("max |sigma_1 - rho| =", err)
for name, ch in (("C1", C1), ("C2", C2)):
    Q, W0 = loss_matrix(ch, 2)
    out[name] = dict(rho=float(ch.rho()), sigma2=sigma(ch, 2), Q2=Q.tolist(), W0=[''.join(map(str,w)) for w in W0])
    print(name, "rho=%.4f sigma2=%.4f" % (ch.rho(), sigma(ch, 2)), "Q2=", np.round(Q, 4).tolist())
# closed form for M=2: Q = [[rho^2, (1-rho)(1-b)], [(1-p10) rho, p10 (1-b)]]
for name, (a, b) in (("C1", (0.56, 0.95)), ("C2", (19/30, 0.45))):
    rho = 1 - a; p10 = 1 - 1.5 * a
    tr = rho**2 + p10 * (1 - b); det = rho * (1 - b) * (rho * p10 - (1 - rho) * (1 - p10))
    s = (tr + math.sqrt(tr**2 - 4 * det)) / 2
    print(name, "closed-form sigma2 = %.6f" % s)
    out[name]["sigma2_closed_form"] = s
# decay of risk ratios with D (n fixed, even D)
n = 48; rows = []
for D in range(4, 41, 2):
    T = D + n
    rd = (risk_dense(C1, T, D), risk_dense(C2, T, D))
    S = periodic_slots(T, 2)
    rp = (risk_schedule(C1, T, D, S), risk_schedule(C2, T, D, S))
    rows.append(dict(D=D, dense_C1=rd[0], dense_C2=rd[1], g2_C1=rp[0], g2_C2=rp[1]))
out["by_D"] = rows
for r in rows[::3]:
    print(f"D={r['D']:2d}: dense R1/R2={r['dense_C1']/r['dense_C2']:.3g}  spaced R2/R1={r['g2_C2']/r['g2_C1']:.3g}")
Ds = np.array([r["D"] for r in rows if r["D"] >= 20])
sl_d = np.polyfit(Ds, [math.log(r["dense_C1"]/r["dense_C2"]) for r in rows if r["D"] >= 20], 1)[0]
sl_g = np.polyfit(Ds, [math.log(r["g2_C2"]/r["g2_C1"]) for r in rows if r["D"] >= 20], 1)[0]
pd = math.log(out["C1"]["rho"]/out["C2"]["rho"]); pg = 0.5*math.log(out["C2"]["sigma2"]/out["C1"]["sigma2"])
print(f"slope dense {sl_d:.4f} vs log(rho1/rho2) {pd:.4f};  slope spaced {sl_g:.4f} vs 0.5 log(s2/s1) {pg:.4f}")
first_rev = min(r["D"] for r in rows if r["dense_C1"] > r["dense_C2"] and r["g2_C2"] > r["g2_C1"])
all_rev = all(r["dense_C1"] > r["dense_C2"] and r["g2_C2"] > r["g2_C1"] for r in rows if r["D"] >= first_rev)
print("reversal holds for all even D >=", first_rev, ":", all_rev)
out.update(slope_dense=sl_d, pred_dense=pd, slope_spaced=sl_g, pred_spaced=pg, first_reversal_D=first_rev, reversal_all_D_after=all_rev)
json.dump(out, open("results/analytic.json", "w"), indent=1)
