"""Recheck the key numerical claims of the paper from the stored results (run: python3 verify_paper_claims.py)."""
import os, random, math
from fractions import Fraction as Fr
from core import *
random.seed(3)
ok = True
def check(cond, msg):
    global ok
    print(("PASS " if cond else "FAIL ") + msg); ok &= bool(cond)
# Lemma 1 / rho interval for the example
R = box_ranges(C_EXAMPLE, 1, ETA_EXAMPLE)
check(R[(0,)] == (Fr(11,250), Fr(57,500)) and R[(1,)] == (Fr(1,250), Fr(11,250)), "Lemma 1 box for the example")
lo, hi, nomr = rho_interval(C_EXAMPLE, 1, ETA_EXAMPLE)
check((lo, hi, nomr) == (Fr(11,30), Fr(19,20), Fr(3,5)), "rho interval and nominal persistence")
nomt = box_nominal(C_EXAMPLE, 1)
ch_nom = box_chain(C_EXAMPLE, 1, nomt)
H_nom = ch_nom.entropy_rate()
Hmax = max(box_chain(C_EXAMPLE, 1, {(0,): random.uniform(*map(float,R[(0,)])), (1,): random.uniform(*map(float,R[(1,)]))}).entropy_rate() for _ in range(2000))
check(H_nom >= Hmax - 1e-12, f"max-entropy point is the L=1 model (H_nom={H_nom:.6f} >= sampled max {Hmax:.6f})")
# dense closed form (Eq. (12)) and the constant theta_M of Sec. III and S3.2 (C below)
worst_ratio = 0; ok_C = True
for _ in range(200):
    a = random.uniform(0.05, 19/30); b = random.uniform(0.45, 0.95)
    ch = example_chain(a, b)
    T = random.randint(13, 120); D = random.randint(3, 12)
    if T < D: continue
    st = dense_stats(ch, T, D)
    cf = EN_closed_form(0.12, 1 - a, T, D, 2)
    worst_ratio = max(worst_ratio, abs(st["EN"] - cf) / cf)
    C = 12.5 * max(1 - b, b / 9)
    J = max(T - 2 * D, 0)
    ok_C &= J * st["kappa"] <= C * st["EN"] + 1e-15
check(worst_ratio < 1e-12, f"closed form of Lambda (max rel. err {worst_ratio:.1e})")
check(ok_C, "J*kappa <= C*Lambda with C = 12.5 max(1-b, b/9) on 200 random cases")
# Theorem 2 items
def g(r,k,n): return r**k*(1+n*(1-r))
ok_peak = True
for k in range(1, 25):
    for n in [0, 1, 5, 20, 100, 1000]:
        grid = [i/20000 for i in range(1, 20000)]
        am = max(grid, key=lambda r: g(r,k,n))
        pred = 1.0 if n <= k else k*(n+1)/((k+1)*n)
        ok_peak &= abs(am - min(pred, 0.99995)) <= 1e-3
check(ok_peak, "Theorem 2, items 1) and 2): peak location on a fine grid")
check(2 + Fr(19,20)/(1-Fr(19,20)) == 21, "D* = M + rho_hi/(1-rho_hi) = 21 for rho_hi = 19/20")
# Proposition 1, second part, on random M=3 boxes
p2 = {(0,0):0.35,(0,1):0.85,(1,0):0.55,(1,1):0.92}
c3 = chain_from_p(2, p2).marginal(3)
R3 = box_ranges(c3, 2, 0.05); maxerr = 0
for _ in range(50):
    t = {m: random.uniform(*R3[m]) for m in words(2)}
    ch = box_chain(c3, 2, t)
    lag = sum(ch.block((0,) + m + (0,)) for m in words(2))
    maxerr = max(maxerr, abs(lag - sum(t.values())))
check(maxerr < 1e-15, "Proposition 1 (M=L+1): P(gamma_t=0, gamma_{t+M}=0) = sum_m x_m (M=3)")
import json, numpy as np
# decay rate of spaced attempts (Eq. (14), Prop. S2): sigma_1 = rho, sigma_g >= rho^g, Q_g positive for g >= M; closed-form Q_2 (M=2)
err1 = 0; ok_low = True; ok_pos = True; err_q2 = 0
for _ in range(100):
    a = random.uniform(0.05, 19/30); b = random.uniform(0.45, 0.95)
    ch = example_chain(a, b); rho = 1 - a
    err1 = max(err1, abs(sigma(ch, 1) - rho))
    for g_ in (2, 3, 4):
        ok_low &= sigma(ch, g_) >= rho**g_ - 1e-12
        ok_pos &= bool((loss_matrix(ch, g_)[0] > 0).all())
    Q, W0 = loss_matrix(ch, 2); p10 = 1 - 1.5 * a
    Qcf = np.array([[rho**2, (1 - rho) * (1 - b)], [(1 - p10) * rho, p10 * (1 - b)]])
    err_q2 = max(err_q2, np.abs(Q - Qcf).max())
check(err1 < 1e-12, "decay rate (Prop. S2): sigma_1 = rho")
check(ok_low, "decay rate (Prop. S2): sigma_g >= rho^g (g = 2,3,4)")
check(ok_pos, "decay rate (Prop. S2): Q_g entrywise positive for g >= M")
check(err_q2 < 1e-12, "closed-form Q_2 for M = 2 matches the loss-to-loss matrix")
an = json.load(open("results/analytic.json"))
check(round(an["C1"]["sigma2"], 3) == 0.238 and round(an["C2"]["sigma2"], 3) == 0.433, "sigma_2(C1) = 0.238, sigma_2(C2) = 0.433")
check(abs(an["slope_dense"] - an["pred_dense"]) < 1e-4 and abs(an["slope_spaced"] - an["pred_spaced"]) < 1e-4, "supplement S3.5: spectral slopes match to four digits")
check(an["first_reversal_D"] == 4 and an["reversal_all_D_after"] and max(r["D"] for r in an["by_D"]) == 40, "Sec. IV and supplement S3.5: reversal for every even D from 4 to 40 (n = 48)")
# certificates and baselines (Sec. V, Table S4)
bl = json.load(open("results/baselines.json"))
exp = [(1.46e-2, 0.242, 0.253, 0.973), (2.07e-4, 0.142, 0.143, 0.795), (5.97e-2, 0.286, None, 0.985)]
good = True
for row, (rn, rp, ce, iv) in zip(bl, exp):
    good &= abs(row["R_nom"] / rn - 1) < 0.005 and abs(row["R_plus"] - rp) < 6e-4 and abs(row["interval_robust"] - iv) < 6e-4
    good &= (ce is None and row["box_certificate"] is None) or (ce is not None and abs(row["box_certificate"] - ce) < 6e-4)
    good &= row["box_certificate"] is None or row["box_certificate"] >= row["R_plus"]
check(good, "baseline numbers (Sec. V) and box certificate >= grid worst case")
# Gilbert-Elliott channels (Sec. V, Table S7)
ge = json.load(open("results/ge.json"))
dense0 = [r["R_dense"] for r in ge if r["eG"] == 0.0]
check(max(dense0) / min(dense0) - 1 < 0.003, "GE with e_G = 0: dense risks agree within 0.3%")
b15 = {(r["eG"], r["eB"]): r["R_B15"] for r in ge}
check(abs(b15[(0.0, 0.65)] / b15[(0.0, 1.0)] - 2.13) < 0.01, "GE budgeted risk grows by 2.1x")
check(all(r["in_box"] for r in ge), "GE second-order projections inside the box")
check(max(abs(r["R_dense_proj2"] - r["R_dense"]) / r["R_dense"] for r in ge) < 0.004
      and max(abs(r["R_B15_proj2"] - r["R_B15"]) / r["R_B15"] for r in ge) < 0.08, "projection error within 0.4% (dense) and 8% (B=15)")
# eta sensitivity (Table S5) and Remark S1
et = json.load(open("results/eta.json"))
check(all(15.5 < r["ratio_nom"] < 16.6 for r in et["rows"] if r["eta"] <= 0.05), "supplement Table S5: R+/R_nom about 16 for eta in [0.01, 0.05]")
et_ = {r["eta"]: r for r in et["rows"]}
check(round(et_[0.05]["R_plus"], 3) == 0.240 and abs(et_[0.1]["rho_worst"] - et_[0.1]["rho_hi"]) < 1e-12 and abs(et_[0.1]["rho_hi"] - 0.9) < 1e-12
      and round(float(rho_star(60, 12, 2)), 3) == 0.928 and all(et_[e]["rho_lo"] < et_[e]["rho_worst"] < et_[e]["rho_hi"] for e in (0.01, 0.02, 0.05)),
      "supplement S4.2: Table S5 gives 0.240 at eta = 0.05 (b at mid-range); interior worst case for eta <= 0.05, rho_hi = 0.9 < rho* = 0.928 for eta = 0.1")
check(round({r["B"]: r for r in json.load(open("results/budget.json"))["closed_loop_sweep"]}[60]["Rmax"], 3) == 0.241 and round(json.load(open("results/baselines.json"))[0]["R_plus"], 3) == 0.242,
      "supplement S4.2: the dense worst case at D = 12 is 0.241 on the 47x21 grid (Table S3) and 0.242 with the 2001-point refinement in rho (Table S4)")
m3 = json.load(open("results/M3L1.json"))
check(abs(m3["Lambda_max"] - 0.254) < 5e-4 and abs(json.load(open("results/baselines.json"))[0]["box_certificate"] - 0.253) < 5e-4,
      "supplement Remark S1: max Lambda 0.254 for M=3 and 0.253 for M=2 (T=60, D=12)")
dm_ = json.load(open("results/dense_M2.json"))["cases"]
check(abs(max(c["Rmax"] for c in dm_ if c["T"] == 60 and c["D"] == 10) - 0.289) < 5e-4, "Fig. 2 text: R+ = 0.289 at T=60, D=10 (max over b)")
# further numbers quoted in the text
check(rho_star(8, 3, 2) == Fr(3, 5), "rho* = rho_nom = 3/5 at D = 3, T = 8: the fitted model then maximizes Lambda (S3.4)")
g8_ = max((risk_dense(example_chain(1 - r_, b_), 8, 3), b_) for r_ in np.linspace(11 / 30, 0.95, 117) for b_ in np.linspace(0.45, 0.95, 11))
check(g8_[1] == 0.95 and g8_[0] > risk_dense(example_chain(0.4, 0.9), 8, 3),
      "S3.4: even at D = 3, T = 8 the exact dense risk over the class is largest at b = 0.95, above that of the fitted model")
corner = [r["worst_corner_stationary"] for r in bl]; inter = [r["interval_robust"] for r in bl]
check(0.775 < min(corner) and max(corner) < 0.985 and 0.795 <= min(inter) and max(inter) < 0.99,
      "time-invariant interval corner 0.78-0.98, interval relaxation 0.80-0.98")
swb = {r["B"]: r for r in json.load(open("results/budget.json"))["closed_loop_sweep"]}
check(abs(swb[60]["rho_max"] - 0.925) < 1e-3 and swb[60]["b_max"] == 0.95 and abs(swb[15]["rho_max"] - 0.899) < 1e-3 and swb[15]["b_max"] == 0.45
      and abs(swb[15]["Rmax"] / swb[15]["modelled"] - 4.8) < 0.05,
      "closed loop: grid worst channel (0.925, 0.95) at B=60 -> (0.899, 0.45) at B=15, where R+/R_nom = 4.8")
cv_ = json.load(open("results/budget.json"))["curves"]["60"]
dev_ = max(abs(a / b - 1) for a, b in zip(cv_["b095"], cv_["b045"]))
check(round(dev_, 2) == 0.02, f"dense (T=60, D=12): b from 0.45 to 0.95 at fixed rho changes the risk by at most {100 * dev_:.1f}%")
bs_ = json.load(open("results/budget.json"))
check(round(bs_["b_sensitivity"][0]["ratio_periodic_g2"], 1) == 34.6 and round(bs_["b_sensitivity"][0]["ratio_closed"], 2) == 9.05
      and bs_["thm1_schedule_max_rel_gap"] < 0.16,
      "rho = 11/30, b 0.95 -> 0.45: two-slot periodic risk 34.6x, pi_L (B=15) 9.05x; Theorem 1 gap below 16% on sampled schedules")
# Proposition 1 (periodic attempts and decimated loss runs) and the 2-decimated profiles of C1, C2
dc = json.load(open("results/decimation.json"))
check(dc["identity_max_rel_err"] < 1e-12, "Proposition 1: periodic crossing count = dense count of the decimated process (incl. GE)")
r1, r2 = dc["C1"]["rho2"], dc["C2"]["rho2"]
deep = {}
for name_, (a_, b_) in {"C1": (0.56, 0.95), "C2": (19 / 30, 0.45)}.items():   # depths 1..60 through Q_2 (supplement S3.5)
    ch_ = example_chain(a_, b_); Q_, W0_ = loss_matrix(ch_, 2); mu_ = np.array([float(ch_.mu[w]) for w in W0_])
    p_ = [mu_ @ np.linalg.matrix_power(Q_, j - 1) @ np.ones(len(W0_)) for j in range(1, 62)]
    deep[name_] = [p_[j] / p_[j - 1] for j in range(1, 61)]
check(np.allclose(deep["C1"][:len(r1)], r1, atol=1e-12) and np.allclose(deep["C2"][:len(r2)], r2, atol=1e-12)
      and round(r1[0], 3) == 0.284 and round(r2[0], 2) == 0.44
      and round(min(deep["C1"]), 4) == 0.2326 and round(max(deep["C1"]), 3) == 0.284
      and round(min(deep["C2"]), 4) == 0.4293 and round(max(deep["C2"]), 2) == 0.44 and max(deep["C1"]) < min(deep["C2"]),
      "2-decimated persistences: rho_1^(2) = 0.284 (C1) and 0.44 (C2), within [0.2326, 0.284] and [0.4293, 0.44] for j <= 60, C1 below C2 at every depth")
check(abs(deep["C1"][-1] - dc["C1"]["sigma2"]) < 5e-9 and abs(deep["C2"][-1] - dc["C2"]["sigma2"]) < 5e-9,
      "decimated persistences converge to sigma_2 (eight digits at j = 60)")
check(abs(10 / 0.6 - 48 / (1 + 48 * 0.4) - 14.3) < 0.01, "elasticity d log Lambda / d rho = 14.3 at rho_nom (T=60, D=12)")
# Fig. 4 sweep
sw = json.load(open("results/sweep.json"))["rows"]
dr = [r["dense"]["R_plus"] / r["dense"]["R_nom"] for r in sw]; br = [r["B15"]["R_plus"] / r["B15"]["R_nom"] for r in sw]
check(all(r["dense"]["R_nom"] <= 1e-4 for r in sw), "sweep: every fitted model certifies R_nom <= 1e-4 (dense)")
check(45 < min(dr) and max(dr) < 5400 and 4 < min(br) and max(br) < 650, f"sweep: dense ratio {min(dr):.0f}-{max(dr):.0f}, B=15 ratio {min(br):.1f}-{max(br):.0f}")
check(min(d for r, d in zip(sw, dr) if r["eps"] == 0.005) > 45, "sweep: dense ratio exceeds 45 even at 0.5% loss")
gil_ = lambda e, l: {(0, 0): e - e / l, (0, 1): e / l, (1, 0): e / l, (1, 1): 1 - e - e / l}
empty_ = sum(any(a > b for a, b in box_ranges(gil_(Fr(str(r["eps"])), Fr(str(r["ell"]))), 1, Fr(1, 20)).values()) for r in sw)
check(len(sw) == 36 and empty_ == 27 and min(r["D"] for r in sw) == 6 and max(r["D"] for r in sw) == 42,
      "sweep: D from 6 to 42; eta = 0.05 would leave the class empty for 27 of the 36 fits (exact arithmetic)")
check(all(r["dense"]["R_minus"] <= 1e-4 < r["dense"]["R_plus"] for r in sw),
      "sweep (Sec. V-A): R^- <= eps = 1e-4 < R^+ for every fit with dense attempts")
def b_bounds_(r, e=1e-4):   # range of b = p(01) over the M=2 box of a Gilbert fit (Lemma 1, m = 1)
    c01 = r["eps"] / r["ell"]; c11 = 1 - r["eps"] - c01
    return 1 - min((1 - e) * c01, c01 - e * c11) / c01, 1 - max(e * c01, c01 - (1 - e) * c11) / c01
nk_ = [r for r in sw if 60 - r["D"] <= r["D"] - 2]
onb_ = sum(any(abs(r["B15"]["b_worst"] - x) < 1e-6 for x in b_bounds_(r)) for r in sw)
check(sorted((r["eps"], r["ell"]) for r in nk_) == [(e, 5.0) for e in (0.01, 0.02, 0.05, 0.1, 0.2)]
      and all(abs(r["dense"]["rho_worst"] - (1 - 1e-4)) < 1e-6 for r in nk_) and onb_ >= 18,
      f"sweep: n <= k exactly for mean burst 5 and loss >= 1%, dense worst case at rho_hi; {onb_} of 36 B=15 worst cases on a bound of b")
# fading channels (Table I, Table S10)
fr = json.load(open("results/fading.json"))["rows"]; fs = json.load(open("results/fading_shadow.json"))["rows"]
row = lambda rows, eps, fd, D: next(r for r in rows if r["eps"] == eps and r["fd"] == fd and r["D"] == D)
s16 = row(fs, 0.01, 0.03, 16)["dense"]
check(abs(s16["truth"] / s16["M1"] - 68) < 1 and abs(s16["truth"] / s16["M2"] - 19) < 0.5,
      "shadowed 1%, D=16: Gilbert fit 68x too low, second-order fit 19x too low")
r10 = row(fr, 0.05, 0.03, 10)["dense"]
check(abs(r10["truth"] / r10["M2"] - 1.85) < 0.03 and abs(r10["M1"] / r10["truth"] - 1) < 0.1,
      "Rayleigh fd=0.03, D=10: second-order fit 1.8x too low, Gilbert fit within 10%")
check(all(r["dense"]["M1"] > r["dense"]["truth"] for r in fr if r["eps"] == 0.05),
      "Rayleigh at 5% loss: Gilbert fit conservative for dense attempts")
check(all(r["dense"]["M1"] < r["dense"]["truth"] for r in fs if r["D"] >= 12),
      "shadowed: Gilbert fit underestimates dense risk for D >= 12")
prof = row(fs, 0.05, 0.03, 10)["profile"]["dense"]
check(abs(prof[0] - 0.76) < 0.01 and abs(prof[-1] - 0.92) < 0.01 and all(b >= a - 1e-3 for a, b in zip(prof, prof[1:])),
      "shadowed 5%: persistence grows with burst depth from 0.76 to 0.92")
check(max(r["p0_01"] for r in fr + fs) <= 0.011, "fading channels rarely relapse: 1 - b <= 0.011")
out_ = [r for r in fr + fs if not r["dense"]["class_min"] <= r["dense"]["truth"] <= r["dense"]["class_max"]]
check(json.load(open("results/fading.json"))["eta"] == 1e-4 and len(fr + fs) == 15 and len(out_) == 3
      and all(r in fr and r["fd"] <= 0.01 and r["D"] >= 14 and r["dense"]["truth"] < r["dense"]["class_min"] for r in out_)
      and all(r["dense"]["truth"] <= r["dense"]["class_max"] for r in fr + fs),
      "Table I footnote and S5.4: the M=2 class (eta = 1e-4) bounds the dense trace risk from above in all 15 cases; the 3 exceptions, slow Rayleigh at D >= 14, lie below R^-")
o15_ = {(r["fd"], r["D"], "below" if r["piL_B15"]["truth"] < r["piL_B15"]["class_min"] else "above")
        for r in fr + fs if not r["piL_B15"]["class_min"] <= r["piL_B15"]["truth"] <= r["piL_B15"]["class_max"]}
check(o15_ == {(0.01, 16, "below"), (0.01, 20, "below"), (0.005, 8, "above")},
      "supplement S5.4: for pi_L (B=15) the trace risk lies below the class range at f_D = 0.01, D >= 16, and above R^+ at f_D = 0.005, D = 8")
# closed-loop decomposition (Eq. (15), Prop. S3), Proposition 2, and the redesign (supplement S5.2)
cl = json.load(open("results/closedloop.json"))
check(cl["decomposition_err"] < 1e-12 and cl["max_rel_diff"] < 1e-14 and all(abs(r["Lambda"] / r["R"] - 1) < 1e-14 for r in cl["grid"] + cl["rows"]),
      "closed-loop decomposition (Eq. (15), Prop. S3): Lambda_pi = R to a relative 1e-14 on the 9x5 grid, matching the direct crossing count")
check(cl["patterns"]["0"] == [5, 7, 9, 10, 11, 12], "pi_L retry pattern at the mission start (Sec. IV-B)")
rv_ = json.load(open("results/reversal_exact.json"))
pr_ = [rv_[k][c]["float"] for k in ("dense", "periodic_g2", "closed_loop_B15") for c in ("C1", "C2")]
rb_ = {k: (Fr(v[0]) / Fr(3, 25), 1 - Fr(v[1]) / Fr(2, 25)) for k, v in rv_["box_t"].items()}   # (rho, b) = (x0/c(00), 1 - x1/c(01))
check(rv_["box_t"] == {"C1": ["33/625", "1/250"], "C2": ["11/250", "11/250"]}
      and rb_ == {"C1": (Fr(11, 25), Fr(19, 20)), "C2": (Fr(11, 30), Fr(9, 20))}
      and [f"{x:.2e}" for x in pr_] == ["9.10e-04", "1.66e-04", "3.46e-03", "4.45e-02", "1.30e-02", "6.04e-02"]
      and pr_[0] > pr_[1] and pr_[2] < pr_[3] and pr_[4] < pr_[5]
      and round(rv_["pi_L_B15_retry_gap_distribution_nominal"]["2"], 2) == 0.70,
      "Proposition 2: (rho, b) = (0.44, 0.95) and (11/30, 0.45) for C1 and C2, the six exact risks and their orders; 70% of pi_L's retries spaced two slots")
rd = json.load(open("results/redesign.json"))
ch_ = {r["name"]: r for r in rd["chains"]}; ge65 = next(r for r in rd["ge"] if r["eB"] == 0.65)
check(abs(ch_["C2"]["R_piL"] - 6.04e-2) < 5e-5 and abs(ch_["C2"]["R_rec"] - 2.68e-2) < 5e-5
      and abs(ch_["C1"]["R_piL"] - 1.30e-2) < 5e-5 and abs(ch_["C1"]["R_rec"] - 1.67e-2) < 5e-5,
      "supplement S5.2: the redesign lowers C2 from 6.04e-2 to 2.68e-2 but raises C1 from 1.30e-2 to 1.67e-2")
check(abs(ge65["R_piL"] - 0.127) < 5e-4 and abs(ge65["R_rec"] - 0.093) < 1e-3 and round(ge65["R_rec"] / ge65["R_genie"] - 1, 2) == 0.06,
      "supplement S5.2: GE e_B=0.65 0.127 -> 0.093, 6% above the state-observing genie")
grid_ = rd["grid"]; win_ = [g["R_rec"] < g["R_piL"] for g in grid_]
check(sum(win_) == 22 and len(grid_) - sum(win_) == 23 and round(max(g["R_rec"] / g["R_piL"] for g in grid_), 2) == 1.47
      and all(w for g, w in zip(grid_, win_) if g["b"] <= 0.575 + 1e-9) and all(g["b"] >= 0.7 - 1e-9 for g, w in zip(grid_, win_) if not w),
      "supplement S5.2: redesign wins on all b <= 0.575, loses on 23 of 45 (all with b >= 0.7) by up to 1.47x")
check(all(r["R_genie"] <= min(r["R_piL"], r["R_rec"]) + 1e-12 for r in rd["chains"] + rd["ge"] + rd["grid"]), "genie is a lower bound on both policies")
# Measured links (FlockLab)
fl = json.load(open("results/flocklab.json")); fr_ = fl["rows"]
rd_ = np.array([r["dense"]["truth"] / r["dense"]["M1"] for r in fr_]); rm2 = np.array([r["dense"]["truth"] / r["dense"]["M2"] for r in fr_])
rb_ = np.array([r["B15"]["truth"] / r["B15"]["M1"] for r in fr_])
tests_ = [r["tests"] for r in fr_]
check(len(fr_) == 171 and min(tests_) == 567 and max(tests_) == 805 and all(0.01 <= r["loss"] <= 0.25 for r in fr_),
      "FlockLab: 171 links, loss 1-25%, 567-805 distinct tests each (deduplicated)")
check(round(np.mean(rd_ > 1), 2) == 0.64 and round(rd_.max()) == 12,
      "FlockLab dense: measured risk above Gilbert on 64%, up to 12x")
ex_ = np.mean([r["dense"]["truth"] > 1e-2 for r in fr_]); exs_ = np.mean([r["dense"]["truth"] - 2 * r["dense"]["se"] > 1e-2 for r in fr_])
check(max(r["dense"]["M1"] for r in fr_) <= 1e-2 and round(ex_, 2) == 0.55 and round(exs_, 2) == 0.42,
      "FlockLab dense: the fit certifies 1e-2 on every link, yet the measured risk exceeds 1e-2 on 55% (42% beyond two standard errors)")
sig_ = np.mean([r["dense"]["truth"] - 2 * r["dense"]["se"] > r["dense"]["M1"] for r in fr_])
check(round(sig_, 2) == 0.47, "FlockLab dense: above Gilbert by more than two standard errors on 47%")
sig2_ = np.mean([r["dense"]["truth"] - 2 * r["dense"]["se"] > r["dense"]["M2"] for r in fr_])
check(round(np.mean(rm2 > 1), 2) == 0.85 and round(sig2_, 2) == 0.49, "FlockLab dense: second-order fit too low on 85%, by more than two standard errors on 49%")
lo15_ = [r["B15"]["truth"] - 2 * r["B15"]["se"] > r["B15"]["M1"] for r in fr_]
hi15_ = [r["B15"]["truth"] + 2 * r["B15"]["se"] < r["B15"]["M1"] for r in fr_]
check(round(np.mean(lo15_), 2) == 0.33 and round(np.mean(hi15_), 2) == 0.36 and abs(rb_.max() - 5.4) < 0.05 and lo15_[int(rb_.argmax())],
      "FlockLab B=15: modelled risk of pi_L significantly too low on 33% (up to 5.4x), significantly too high on 36%")
rep_ = {tuple(r["link"]): r for r in fr_ if r["link"] in fl["representative"]}
a_, b_ = rep_[("3", "32")], rep_[("15", "33")]
check(a_["D"] == 15 and abs(a_["dense"]["truth"] - 5.0e-2) < 1e-3 and round(a_["dense"]["M1"] / a_["dense"]["truth"], 2) == 0.17
      and abs(a_["B15"]["truth"] - 7.5e-2) < 1e-3 and round(a_["B15"]["M1"] / a_["B15"]["truth"], 2) == 0.26
      and b_["D"] == 18 and abs(b_["dense"]["truth"] - 8.0e-2) < 1e-3 and round(b_["dense"]["M1"] / b_["dense"]["truth"], 2) == 0.10
      and abs(b_["B15"]["truth"] - 1.1e-1) < 5e-3 and round(b_["B15"]["M1"] / b_["B15"]["truth"], 2) == 0.18,
      "Table I measured rows (links 3->32 and 15->33)")
from scipy.stats import spearmanr
low_ = [r for r in fr_ if r["dense"]["truth"] + 2 * r["dense"]["se"] < r["dense"]["M1"]]
check(round(len(low_) / len(fr_), 2) == 0.20 and max(r["mean_burst"] for r in low_) < 1.9,
      "FlockLab dense: significantly lower than Gilbert on 20%, all with mean burst < 1.9")
z_ = [r for r in fr_ if r["dense"]["truth"] == 0]; z3_ = sum(3 / r["tests"] < r["dense"]["M1"] for r in z_)
check(len(z_) == 7 and all(r in low_ for r in z_) and z3_ == 4 and round((len(low_) - len(z_) + z3_) / len(fr_), 2) == 0.19,
      "supplement S5.5: 7 links without a failing mission count as significantly lower; by the rule of three only 4 do (19% instead of 20%)")
rho_s = spearmanr(rd_, [r["mean_burst"] for r in fr_]).correlation
check(round(rho_s, 2) == 0.92, f"FlockLab: rank correlation of the error with mean burst length {rho_s:.3f}")
rho_l = spearmanr(rd_, [r["loss"] for r in fr_]).correlation
check(round(rho_l, 2) == 0.93, f"FlockLab: rank correlation of the error with loss rate {rho_l:.3f}")
gain_ = np.array([r["B15"]["truth"] / r["B15"]["rec"] for r in fr_])
check(abs(np.median(gain_) - 1.015) < 0.003 and round(gain_.max(), 2) == 1.27 and int(np.sum(gain_ < 1)) == 52,
      f"supplement S5.2: median redesign gain {100 * (np.median(gain_) - 1):.1f}% on the measured links, at most 1.27x, worse on 52 of 171")
check(len(rep_) == 4 and all(r["dense"]["class_min"] <= r["dense"]["truth"] <= r["dense"]["class_max"] for r in rep_.values()),
      "FlockLab: the M=2 class brackets the measured dense risk on the burstiest link of each of the four loss bands")
# Table I as printed (Loss, D, R, dense L=1, dense L=2, pi_L L=1), cell by cell against the stored results
TABLE_I = [("5", 20, "0.018", "1.7", "1.5", "1.7"), ("5", 10, "0.025", "1.1", "0.54", "1.4"), ("5", 18, "0.016", "0.30", "0.32", "0.40"),
           ("1", 12, "0.0025", "0.086", "0.21", "0.19"), ("1", 16, "0.0010", "0.015", "0.053", "0.034"),
           ("12", 15, "0.050", "0.17", "0.39", "0.26"), ("19", 18, "0.080", "0.10", "0.26", "0.18")]
src_ = [(row(fr, 0.05, 0.01, 20), "piL_B15"), (row(fr, 0.05, 0.03, 10), "piL_B15"), (row(fs, 0.05, 0.03, 18), "piL_B15"),
        (row(fs, 0.01, 0.03, 12), "piL_B15"), (row(fs, 0.01, 0.03, 16), "piL_B15"), (a_, "B15"), (b_, "B15")]
def same_(s, v):   # printed value s equals v rounded to the significant digits that s shows
    return abs(float(s) - float(f"{v:.{len(s.replace('.', '').lstrip('0'))}g}")) < 1e-12
okT_ = True
for (loss_, D_, *cells_), (r, k15) in zip(TABLE_I, src_):
    d = r["dense"]
    okT_ &= round(100 * r.get("eps", r["loss"])) == int(loss_) and r["D"] == D_ and same_(cells_[0], d["truth"])
    okT_ &= same_(cells_[1], d["M1"] / d["truth"]) and same_(cells_[2], d["M2"] / d["truth"]) and same_(cells_[3], r[k15]["M1"] / r[k15]["truth"])
    okT_ &= (r is src_[0][0]) == (not d["class_min"] <= d["truth"] <= d["class_max"])   # the dagger marks the first row only
tex_ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "main.tex")
if os.path.exists(tex_):   # private repository: TABLE_I must match the table in main.tex
    body_ = open(tex_, encoding="utf-8").read().split(r"\label{tab:fading}")[1].split(r"\bottomrule")[0]
    rows_ = [[c.strip().replace("$", "").replace(r"^\dagger", "").replace(r"\%", "") for c in l.split("&")]
             for l in body_.split(r"\\") if l.count("&") == 6 and "Channel" not in l]
    okT_ &= [tuple(c[1:]) for c in rows_] == [(l, str(D), *c) for l, D, *c in TABLE_I]
check(okT_, "Table I: every Loss, D, R and ratio cell matches the stored results; the dagger marks the only case outside the M=2 class")
print("ALL PASS" if ok else "SOME CHECKS FAILED")
