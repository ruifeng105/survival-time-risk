"""Recheck the key numerical claims of the paper from the stored results (run: python3 verify_paper_claims.py)."""
import random, math
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
check(abs(an["C1"]["sigma2"] - 0.2385) < 5e-4 and abs(an["C2"]["sigma2"] - 0.4334) < 5e-4, "sigma_2(C1) = 0.239, sigma_2(C2) = 0.433")
check(abs(an["C1"]["sigma2"] / 0.44**2 - 1.23) < 0.005 and abs(an["C2"]["sigma2"] / (11/30)**2 - 3.22) < 0.005, "relapse penalties 1.23 and 3.22")
check(abs(an["slope_dense"] - an["pred_dense"]) < 1e-4 and abs(an["slope_spaced"] - an["pred_spaced"]) < 1e-4, "supplement S3.5: spectral slopes match to four digits")
check(an["first_reversal_D"] == 4 and an["reversal_all_D_after"] and max(r["D"] for r in an["by_D"]) == 40, "Sec. IV and supplement S3.5: reversal for every even D from 4 to 40 (n = 48)")
# certificates and baselines (Sec. V, Table S5)
bl = json.load(open("results/baselines.json"))
exp = [(1.46e-2, 0.242, 0.253, 0.973), (2.07e-4, 0.142, 0.143, 0.795), (5.97e-2, 0.286, None, 0.985)]
good = True
for row, (rn, rp, ce, iv) in zip(bl, exp):
    good &= abs(row["R_nom"] / rn - 1) < 0.005 and abs(row["R_plus"] - rp) < 6e-4 and abs(row["interval_robust"] - iv) < 6e-4
    good &= (ce is None and row["box_certificate"] is None) or (ce is not None and abs(row["box_certificate"] - ce) < 6e-4)
    good &= row["box_certificate"] is None or row["box_certificate"] >= row["R_plus"]
check(good, "baseline numbers (Sec. V) and box certificate >= grid worst case")
# Gilbert-Elliott channels (Sec. V, Table S8)
ge = json.load(open("results/ge.json"))
dense0 = [r["R_dense"] for r in ge if r["eG"] == 0.0]
check(max(dense0) / min(dense0) - 1 < 0.003, "GE with e_G = 0: dense risks agree within 0.3%")
b15 = {(r["eG"], r["eB"]): r["R_B15"] for r in ge}
check(abs(b15[(0.0, 0.65)] / b15[(0.0, 1.0)] - 2.13) < 0.01, "GE budgeted risk grows by 2.1x")
check(all(r["in_box"] for r in ge), "GE second-order projections inside the box")
check(max(abs(r["R_dense_proj2"] - r["R_dense"]) / r["R_dense"] for r in ge) < 0.004
      and max(abs(r["R_B15_proj2"] - r["R_B15"]) / r["R_B15"] for r in ge) < 0.08, "projection error within 0.4% (dense) and 8% (B=15)")
# eta sensitivity (Table S6) and Remark S1
et = json.load(open("results/eta.json"))
check(all(15.5 < r["ratio_nom"] < 16.6 for r in et["rows"] if r["eta"] <= 0.05), "supplement Table S6: R+/R_nom about 16 for eta in [0.01, 0.05]")
m3 = json.load(open("results/M3L1.json"))
check(abs(m3["R_at_max"] - 0.240) < 5e-4, "supplement Remark S1: risk 0.240 at the Lambda-maximizer for M=3, L=1")
check(abs(m3["Lambda_max"] - 0.254) < 5e-4 and abs(json.load(open("results/baselines.json"))[0]["box_certificate"] - 0.253) < 5e-4,
      "supplement Remark S1: max Lambda 0.254 for M=3 and 0.253 for M=2 (T=60, D=12)")
dm_ = json.load(open("results/dense_M2.json"))["cases"]
check(abs(max(c["Rmax"] for c in dm_ if c["T"] == 60 and c["D"] == 10) - 0.289) < 5e-4, "Fig. 2 text: R+ = 0.289 at T=60, D=10 (max over b)")
# further numbers quoted in the text
check(rho_star(8, 3, 2) == Fr(3, 5), "rho* = rho_nom = 3/5 at D = 3, T = 8: the fitted model then maximizes Lambda (S3.3)")
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
check(0.23 <= min(r1) and max(r1) <= 0.285 and 0.425 <= min(r2) and max(r2) <= 0.445 and max(r1) < min(r2),
      "2-decimated persistences in [0.23,0.28] (C1) and [0.43,0.44] (C2), C1 below C2 at every depth")
check(abs(r1[-1] - dc["C1"]["sigma2"]) < 2e-3 and abs(r2[-1] - dc["C2"]["sigma2"]) < 2e-3, "decimated persistences converge to sigma_2")
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
# fading channels (Table I, Table S11)
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
check(max(r["p0_01"] for r in fr + fs) <= 0.011, "fading channels rarely relapse: P(0|01) <= 0.011")
out_ = [r for r in fr + fs if not r["dense"]["class_min"] <= r["dense"]["truth"] <= r["dense"]["class_max"]]
check(json.load(open("results/fading.json"))["eta"] == 1e-4 and len(fr + fs) == 15 and len(out_) == 3
      and all(r in fr and r["fd"] <= 0.01 and r["D"] >= 14 for r in out_)
      and all(r["dense"]["class_min"] <= r["dense"]["truth"] for r in fr + fs if r["fd"] > 0.01 or r["D"] < 14),
      "M=2 class (eta = 1e-4) brackets the trace-driven dense risk except in 3 of 15 cases, all slow Rayleigh at D >= 14")
# closed-loop bound (Eq. (15), Prop. S3) and the redesign paragraph
cl = json.load(open("results/closedloop.json"))
check(cl["decomposition_err"] < 1e-12, "closed-loop bound (Prop. S3): retry-pattern decomposition equals the direct crossing count")
check(cl["patterns"]["10"] == [15, 17, 19, 20, 21, 22], "pi_L retry pattern after a success at s=10 with r=12")
rv_ = json.load(open("results/reversal_exact.json"))
pr_ = [rv_[k][c]["float"] for k in ("dense", "periodic_g2", "closed_loop_B15") for c in ("C1", "C2")]
check(rv_["box_t"] == {"C1": ["33/625", "1/250"], "C2": ["11/250", "11/250"]}
      and [f"{x:.2e}" for x in pr_] == ["9.10e-04", "1.66e-04", "3.46e-03", "4.45e-02", "1.30e-02", "6.04e-02"]
      and pr_[0] > pr_[1] and pr_[2] < pr_[3] and pr_[4] < pr_[5]
      and round(rv_["pi_L_B15_retry_gap_distribution_nominal"]["2"], 2) == 0.70,
      "Proposition 2: (x0, x1) of C1 and C2, the six exact risks and their orders; 70% of pi_L's retries spaced two slots")
c2 = {r["name"]: r["Lambda"] / r["R"] for r in cl["rows"]}
check(cl["max_gap"] <= 0.305 and round(c2["C2"] - 1, 2) == 0.16 and round(c2["C2_D16"] - 1, 2) == 0.07 and round(c2["C2_D20"] - 1, 2) == 0.02,
      "Lambda_pi exceeds R by at most 30% over the class; by 16%, 7%, 2% on C2 for D = 12, 16, 20")
check(all(r["Lambda"] >= r["R"] for r in cl["grid"] + cl["rows"]), "closed-loop bound (Prop. S3): R <= Lambda_pi on every evaluated channel")
rd = json.load(open("results/redesign.json"))
ch_ = {r["name"]: r for r in rd["chains"]}; ge65 = next(r for r in rd["ge"] if r["eB"] == 0.65)
check(abs(ch_["C2"]["R_piL"] - 6.04e-2) < 1e-4 and abs(ch_["C2"]["R_rec"] - 2.68e-2) < 1e-4, "redesign: C2 6.0e-2 -> 2.7e-2")
check(abs(ge65["R_piL"] - 0.127) < 5e-4 and abs(ge65["R_rec"] - 0.093) < 1e-3 and round(ge65["R_rec"] / ge65["R_genie"] - 1, 2) == 0.06,
      "redesign: GE e_B=0.65 0.127 -> 0.093, 6% above the genie")
grid_ = rd["grid"]; win_ = [g["R_rec"] < g["R_piL"] for g in grid_]
check(sum(win_) == 22 and len(grid_) - sum(win_) == 23 and round(max(g["R_rec"] / g["R_piL"] for g in grid_), 2) == 1.47
      and all(w for g, w in zip(grid_, win_) if g["b"] <= 0.575 + 1e-9) and ch_["C1"]["R_piL"] < ch_["C1"]["R_rec"],
      "redesign grid: wins on all b <= 0.575, loses on 23 of 45 by up to 1.47x, including C1")
check(all(r["R_genie"] <= min(r["R_piL"], r["R_rec"]) + 1e-12 for r in rd["chains"] + rd["ge"] + rd["grid"]), "genie is a lower bound on both policies")
# Measured links (FlockLab)
fl = json.load(open("results/flocklab.json")); fr_ = fl["rows"]
rd_ = np.array([r["dense"]["truth"] / r["dense"]["M1"] for r in fr_]); rm2 = np.array([r["dense"]["truth"] / r["dense"]["M2"] for r in fr_])
rb_ = np.array([r["B15"]["truth"] / r["B15"]["M1"] for r in fr_])
tests_ = [r["tests"] for r in fr_]
check(len(fr_) == 171 and min(tests_) == 567 and max(tests_) == 805 and all(0.01 <= r["loss"] <= 0.25 for r in fr_),
      "FlockLab: 171 links, loss 1-25%, 567-805 distinct tests each (deduplicated)")
check(round(np.mean(rd_ > 1), 2) == 0.64 and round(np.mean(rd_ >= 2), 2) == 0.49 and round(rd_.max()) == 12,
      "FlockLab dense: measured risk above Gilbert on 64%, >= 2x on 49%, up to 12x")
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
rho_s = spearmanr(rd_, [r["mean_burst"] for r in fr_]).correlation
check(round(rho_s, 2) == 0.92, f"FlockLab: rank correlation of the error with mean burst length {rho_s:.3f}")
rho_l = spearmanr(rd_, [r["loss"] for r in fr_]).correlation
check(round(rho_l, 2) == 0.93, f"FlockLab: rank correlation of the error with loss rate {rho_l:.3f}")
gain_ = np.array([r["B15"]["truth"] / r["B15"]["rec"] for r in fr_])
check(abs(np.median(gain_) - 1.015) < 0.003, f"FlockLab: median redesign gain {100 * (np.median(gain_) - 1):.1f}%")
check(len(rep_) == 4 and all(r["dense"]["class_min"] <= r["dense"]["truth"] <= r["dense"]["class_max"] for r in rep_.values()),
      "FlockLab: the M=2 class brackets the measured risk on the burstiest link of each of the four loss bands")
print("ALL PASS" if ok else "SOME CHECKS FAILED")
