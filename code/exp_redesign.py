"""What does knowing the right statistic buy? pi_L is designed with the recovery curve
P(gamma_{t+k} = 1 | last observed outcome o, k slots ago) of the Gilbert fit. pi_rec runs the
same dynamic program with the channel's true recovery curve, a statistic that can be
estimated from a trace. Both are evaluated exactly on second-order channels of the class
and on the Gilbert-Elliott channels of Section V and Table S8 (T = 60, D = 12, B = 15)."""
import json
import numpy as np
from core import words, example_chain, A_RANGE, B_RANGE
from policy import design_pi_L, design_from_pred, evaluate_policy
from ge import ge_matching, evaluate_policy_ge

T, D, B = 60, 12, 15
tab_L, _ = design_pi_L(0.4, 0.9, T, D, B)


def curve_chain(ch):
    """Recovery curve of a stationary M-th order chain."""
    W = words(ch.M); succ = sum(ch.mu[w] * ch.p[w] for w in W)
    def pred(o, k):
        if o is None: return succ
        dist = {w: ch.mu[w] * (ch.p[w] if o == 1 else 1 - ch.p[w]) for w in W}   # context at t, gamma_t = o
        dist = {w[1:] + (o,): v for w, v in dist.items()}
        tot = sum(dist.values()); dist = {w: v / tot for w, v in dist.items()}   # context at t+1
        for _ in range(k - 1):
            nd = {}
            for w, v in dist.items():
                for g, pg in ((1, ch.p[w]), (0, 1 - ch.p[w])):
                    nd[w[1:] + (g,)] = nd.get(w[1:] + (g,), 0) + v * pg
            dist = nd
        return sum(v * ch.p[w] for w, v in dist.items())
    return pred


def curve_ge(ge):
    S = ["G", "B"]
    P = np.array([[ge.P[a][b] for b in S] for a in S]); e = np.array([ge.e[s] for s in S]); pi = np.array([ge.pi[s] for s in S])
    def pred(o, k):
        if o is None: return float(pi @ (1 - e))
        post = pi * ((1 - e) if o == 1 else e); post /= post.sum()                # state at t given gamma_t = o
        return float(post @ np.linalg.matrix_power(P, k) @ (1 - e))
    return pred


def genie_chain(ch):
    """Optimal risk of a scheduler that observes the channel context in every slot: a lower
    bound on the risk of every policy under ACK/NACK feedback."""
    from functools import lru_cache
    @lru_cache(maxsize=None)
    def V(t, d, r, w):
        if t > T: return 0.0
        q = ch.p[w]; w1, w0 = w[1:] + (1,), w[1:] + (0,)
        nxt = lambda dd, rr, ww: 1.0 if dd >= D else V(t + 1, dd, rr, ww)
        v_idle = q * nxt(d + 1, r, w1) + (1 - q) * nxt(d + 1, r, w0)
        if r == 0: return v_idle
        return min(v_idle, q * V(t + 1, 0, r - 1, w1) + (1 - q) * nxt(d + 1, r - 1, w0))
    return sum(ch.mu[w] * V(1, 0, B, w) for w in words(ch.M))


def genie_ge(ge):
    from functools import lru_cache
    @lru_cache(maxsize=None)
    def V(t, d, r, s):
        if t > T: return 0.0
        nxt = lambda dd, rr, ss: 1.0 if dd >= D else V(t + 1, dd, rr, ss)
        v_idle = sum(pp * nxt(d + 1, r, s2) for g, pp, s2 in ge.trans(s))
        if r == 0: return v_idle
        v_tx = sum(pp * (V(t + 1, 0, r - 1, s2) if g == 1 else nxt(d + 1, r - 1, s2)) for g, pp, s2 in ge.trans(s))
        return min(v_idle, v_tx)
    return sum(p * V(1, 0, B, s) for s, p in ge.pi.items())


out = {"T": T, "D": D, "B": B, "chains": [], "ge": []}
for name, (a, b) in {"nominal": (0.4, 0.9), "C1": (0.56, 0.95), "C2": (19 / 30, 0.45)}.items():
    ch = example_chain(a, b)
    tab_r, mod_r = design_from_pred(curve_chain(ch), T, D, B)
    rL, rR = float(evaluate_policy(ch, tab_L, T, D, B)[0]), float(evaluate_policy(ch, tab_r, T, D, B)[0])
    rG = genie_chain(ch)
    out["chains"].append(dict(name=name, rho=1 - a, b=b, R_piL=rL, R_rec=rR, R_genie=rG))
    print(f"{name:8s} rho={1 - a:.3f} b={b:.2f}: pi_L {rL:.4e}  pi_rec {rR:.4e}  genie {rG:.4e}  pi_L/min {rL / min(rL, rR):.2f}")
grid = []
for a in np.linspace(float(A_RANGE[0]), float(A_RANGE[1]), 9):
    for b in np.linspace(float(B_RANGE[0]), float(B_RANGE[1]), 5):
        ch = example_chain(a, b)
        tab_r, _ = design_from_pred(curve_chain(ch), T, D, B)
        rL, rR = float(evaluate_policy(ch, tab_L, T, D, B)[0]), float(evaluate_policy(ch, tab_r, T, D, B)[0])
        grid.append(dict(rho=1 - a, b=b, R_piL=rL, R_rec=rR, R_genie=genie_chain(ch)))
ratios = [g["R_piL"] / g["R_rec"] for g in grid]
gains = [g["R_piL"] / min(g["R_piL"], g["R_rec"]) for g in grid]
print(f"class grid: best-of(pi_L, pi_rec) improves on pi_L by up to {max(gains):.2f}x; "
      f"pi_L / genie from {min(g['R_piL'] / g['R_genie'] for g in grid):.2f} to {max(g['R_piL'] / g['R_genie'] for g in grid):.2f}")
print(f"class grid (45 channels): pi_L / pi_rec from {min(ratios):.2f} to {max(ratios):.2f}; "
      f"worst-case risk {max(g['R_piL'] for g in grid):.3f} -> {max(g['R_rec'] for g in grid):.3f}")
for eG, eB in [(0.0, 1.0), (0.0, 0.8), (0.0, 0.7), (0.0, 0.65), (0.01, 0.9)]:
    ge, _ = ge_matching(eG, eB)
    tab_r, _ = design_from_pred(curve_ge(ge), T, D, B)
    rL, rR = evaluate_policy_ge(ge, tab_L, T, D, B), evaluate_policy_ge(ge, tab_r, T, D, B)
    rG = genie_ge(ge)
    out["ge"].append(dict(eG=eG, eB=eB, R_piL=rL, R_rec=rR, R_genie=rG))
    print(f"GE ({eG}, {eB}): pi_L {rL:.4e}  pi_rec {rR:.4e}  genie {rG:.4e}  pi_L/min {rL / min(rL, rR):.2f}")
out["grid"] = grid
json.dump(out, open("results/redesign.json", "w"), indent=1)
