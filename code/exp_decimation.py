"""Periodic attempts and decimated loss runs: for g | D, the crossing count of the periodic schedule S_g on any
stationary channel equals the dense crossing count of the g-decimated channel gamma_{ig}
with survival time D/g (Proposition 1 of the paper). Checked on random M=2, M=3 chains and on Gilbert-Elliott channels;
also the run-persistence profiles of C1/C2 and of their 2-decimations (reversal mechanism)."""
import json, random
import numpy as np
from fractions import Fraction as Fr
from core import *
from ge import GE, ge_matching

random.seed(7)


def gapped(ch_block, pattern, g):
    """P(gamma_g = v_1, gamma_2g = v_2, ...) for a stationary channel with block(.)"""
    tot = 0.0
    free = [i for i in range(len(pattern) * g - g + 1) if i % g != 0]
    L = len(pattern) * g - g + 1
    for bits in range(2 ** len(free)):
        w = [0] * L
        for j, v in enumerate(pattern):
            w[j * g] = v
        for j, i in enumerate(free):
            w[i] = (bits >> j) & 1
        tot += ch_block(tuple(w))
    return tot


def decimated_lambda(ch_block, T, D, g):
    m, n_g = D // g, (T - D) // g
    return gapped(ch_block, (0,) * m, g) + n_g * gapped(ch_block, (1,) + (0,) * m, g)


def run_profile(ch_block, g, jmax):
    p = [gapped(ch_block, (0,) * j, g) for j in range(1, jmax + 2)]
    return [p[j] / p[j - 1] for j in range(1, jmax + 1)]   # rho^{(g)}_j, j = 1..jmax


out = {"identity": []}
cases = []
for _ in range(6):
    cases.append(("M2", example_chain(random.uniform(0.05, 19 / 30), random.uniform(0.45, 0.95))))
p2 = {(0, 0): 0.35, (0, 1): 0.85, (1, 0): 0.55, (1, 1): 0.92}
c3 = chain_from_p(2, p2).marginal(3)
R3 = box_ranges(c3, 2, 0.05)
for _ in range(4):
    cases.append(("M3", box_chain(c3, 2, {m: random.uniform(*R3[m]) for m in words(2)})))
worst = 0.0
for name, ch in cases:
    for (T, D, g) in [(40, 8, 2), (40, 9, 3), (30, 8, 4)]:
        lhs = float(schedule_stats(ch, T, D, periodic_slots(T, g))["EN"])
        rhs = decimated_lambda(lambda w: float(ch.block(w)), T, D, g)
        worst = max(worst, abs(lhs - rhs) / lhs)
def crossings_ge(ge, T, D, S):
    """E[#{t <= T: d_t = D}] for the age process continued past D (forward recursion)."""
    S = set(S)
    dist = {(s, 0): p for s, p in ge.pi.items()}
    EN = 0.0
    for t in range(1, T + 1):
        nd = {}
        for (s, d), pr in dist.items():
            for g_, pp, s2 in ge.trans(s):
                dd = 0 if (t in S and g_ == 1) else min(d + 1, D + 1)
                nd[(s2, dd)] = nd.get((s2, dd), 0) + pr * pp
        dist = nd
        EN += sum(pr for (s, d), pr in dist.items() if d == D)
    return EN


for eG, eB in [(0.0, 0.7), (0.01, 0.9)]:
    ge, _ = ge_matching(eG, eB)
    for (T, D, g) in [(30, 8, 2), (30, 9, 3)]:
        lam = crossings_ge(ge, T, D, periodic_slots(T, g))
        rhs = decimated_lambda(ge.block, T, D, g)
        worst = max(worst, abs(lam - rhs) / lam)
out["identity_max_rel_err"] = worst
print(f"decimation identity: max relative error {worst:.1e}")

# run profiles of C1, C2 (dense and 2-decimated)
C1, C2 = example_chain(0.56, 0.95), example_chain(19 / 30, 0.45)
for name, ch in (("C1", C1), ("C2", C2)):
    out[name] = {"rho1": run_profile(lambda w: float(ch.block(w)), 1, 8),
                 "rho2": run_profile(lambda w: float(ch.block(w)), 2, 8),
                 "sigma2": sigma(ch, 2)}
    print(name, "dense profile", np.round(out[name]["rho1"], 3), "| 2-decimated", np.round(out[name]["rho2"], 3),
          "| sigma2", round(out[name]["sigma2"], 4))
json.dump(out, open("results/decimation.json", "w"), indent=1)
