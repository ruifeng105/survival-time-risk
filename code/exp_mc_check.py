"""Independent Monte Carlo cross-check of the exact evaluators (channels C1, C2 of Proposition 2).
Simulates the second-order chain directly from p(w) with an independently solved stationary law;
only the policy table is taken from policy.py."""
import json
import numpy as np
from policy import design_pi_L

T, D, B = 60, 12, 15
rng = np.random.default_rng(12345)


def chain(a, b):
    p = np.array([a, b, 1 - 1.5 * a, 1 - b / 9])      # contexts 00, 01, 10, 11
    P = np.zeros((4, 4))
    for w in range(4):
        P[w, 2 * (w & 1) + 1] += p[w]; P[w, 2 * (w & 1)] += 1 - p[w]
    ev, evec = np.linalg.eig(P.T)
    mu = np.real(evec[:, np.argmin(abs(ev - 1))]); return p, mu / mu.sum()


def policy_array(table):
    A = np.zeros((T + 2, D + 1, B + 1, 3, T + 2), dtype=np.int8)
    for (t, d, r, o, k), u in table.items():
        if u: A[t, d, r, 2 if o is None else o, k] = 1
    return A


def simulate(p, mu, pol, N):
    ctx = rng.choice(4, size=N, p=mu)
    d = np.zeros(N, np.int64); r = np.full(N, B if pol is not None else T); o = np.full(N, 2); k = np.zeros(N, np.int64)
    failed = np.zeros(N, bool)
    for t in range(1, T + 1):
        g = (rng.random(N) < p[ctx]).astype(np.int64)
        u = np.ones(N, bool) if pol is None else (r > 0) & (pol[t, np.minimum(d, D), np.minimum(r, B), o, k] == 1)
        d = np.where(u & (g == 1), 0, d + 1); failed |= d >= D
        k = np.where(u, 1, np.where(o == 2, 0, k + 1)); o = np.where(u, g, o); r = np.where(u, r - 1, r)
        ctx = 2 * (ctx & 1) + g
    return float(failed.mean()), float(failed.std() / np.sqrt(N))


exact = json.load(open("results/reversal_exact.json"))
pol = policy_array(design_pi_L(0.4, 0.9, T, D, B)[0])
out = {}
for name, (a, b) in {"C1": (0.56, 0.95), "C2": (19 / 30, 0.45)}.items():
    p, mu = chain(a, b)
    parts = [simulate(p, mu, None, 2_000_000) for _ in range(4)]          # 8e6 dense runs in chunks
    rd = float(np.mean([m for m, _ in parts])); sd = float(np.sqrt(sum(s_ ** 2 for _, s_ in parts)) / 4)
    rb, sb = simulate(p, mu, pol, 1_000_000)
    out[name] = dict(dense_mc=rd, dense_se=sd, dense_exact=exact["dense"][name]["float"],
                     B15_mc=rb, B15_se=sb, B15_exact=exact["closed_loop_B15"][name]["float"])
    print(name, {k: f"{v:.4e}" for k, v in out[name].items()})
json.dump(out, open("results/mc_check.json", "w"), indent=1)
