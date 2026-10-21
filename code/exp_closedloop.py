"""Closed-loop bound (Eq. (15) of the paper, Proposition S3 of the supplement). Between two
successes every observation is a NACK, so after a success at slot s with remaining budget r
the attempt slots in (s, s+D] are a deterministic NACK-retry pattern R(s, r). The crossing
count therefore decomposes as
    Lambda_pi = P(losses on R(0, B)) + sum_{s, r, w} P(reset at s, budget r, S_{s+1} = w)
                                                   * P(losses on R(s, r) | S_{s+1} = w),
and R(pi) <= Lambda_pi. The second Bonferroni inequality gives Lambda_pi (1 - nu_pi) <= R(pi),
with nu_pi the largest expected number of further crossings given the state at a crossing;
for budgeted policies nu_pi is of order one (about 1.7 for pi_L at D = 12, where this lower bound is vacuous;
0.95 and 0.63 for C2 at D = 16 and 20, where it is positive but weak; Section S3.6, Table S1).
Checked for pi_L (B = 15) over the M = 2 class."""
import json
import numpy as np
from core import words, example_chain, A_RANGE, B_RANGE
from policy import design_pi_L, evaluate_policy

T, D, B = 60, 12, 15
tab, _ = design_pi_L(0.4, 0.9, T, D, B)


def decide(t, d, r, o, k):
    if r == 0: return 0
    if d >= D: return 1                      # continuation after failure: attempt until reset
    return tab.get((t, d, r, o, k), 0)


def retry_pattern(s, r):
    """Attempt slots in (s, s+D] after a success at s (s = 0: mission start), all NACKs."""
    t, d, o, k, out = s + 1, 0, (1 if s > 0 else None), (1 if s > 0 else 0), []
    while t <= min(s + D, T):
        if decide(t, d, r, o, k):
            out.append(t); r -= 1; o, k = 0, 1
        else:
            k = k + 1 if o is not None else 0
        d += 1; t += 1
    return out


def loss_prob(ch, start_law, s, slots):
    """P(gamma_j = 0 for j in slots | law of S_{s+1} = start_law), slots in (s, s+D]."""
    dist, need = dict(start_law), set(slots)
    for t in range(s + 1, (max(slots) if slots else s) + 1):
        nd = {}
        for w, pr in dist.items():
            q = ch.p[w]
            for g, pg in ((1, q), (0, 1 - q)):
                if t in need and g == 1: continue
                k2 = w[1:] + (g,); nd[k2] = nd.get(k2, 0) + pr * pg
        dist = nd
    return sum(dist.values())


def nu_after_crossing(ch, crossing_states):
    """Largest expected number of further crossings, over the states reached at a crossing."""
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def V(t, d, r, o, k, w):                 # expected crossings in slots t..T from the state before slot t
        if t > T: return 0.0
        u = decide(t, d, r, o, k); q = ch.p[w]; tot = 0.0
        for g, pg in ((1, q), (0, 1 - q)):
            w2 = w[1:] + (g,)
            if u and g == 1: nxt = (0, r - 1, 1, 1)
            elif u: nxt = (min(d + 1, D + 1), r - 1, 0, 1)
            else: nxt = (min(d + 1, D + 1), r, o, (k + 1) if o is not None else 0)
            tot += pg * ((nxt[0] == D) + V(t + 1, *nxt, w2))
        return tot
    return max((V(t + 1, *st) for t, st in crossing_states), default=0.0)


def analyse(ch):
    W = words(ch.M)
    crossing_states = set()
    # forward recursion of the continued process: reset masses and the direct crossing count
    dist = {(0, B, None, 0, w): ch.mu[w] for w in W}
    reset = {}                                # (s, r, w') -> mass of a success at s
    EN_direct = 0.0
    for t in range(1, T + 1):
        nd = {}
        for (d, r, o, k, w), pr in dist.items():
            u = decide(t, d, r, o, k); q = ch.p[w]
            for g, pg in ((1, q), (0, 1 - q)):
                m = pr * pg; w2 = w[1:] + (g,)
                if u and g == 1:
                    key = (0, r - 1, 1, 1, w2); reset[(t, r - 1, w2)] = reset.get((t, r - 1, w2), 0) + m
                elif u:
                    key = (min(d + 1, D + 1), r - 1, 0, 1, w2)
                else:
                    key = (min(d + 1, D + 1), r, o, (k + 1) if o is not None else 0, w2)
                nd[key] = nd.get(key, 0) + m
        dist = nd
        EN_direct += sum(pr for (d, *_), pr in dist.items() if d == D)
        crossing_states |= {(t, key) for key, pr in dist.items() if key[0] == D and pr > 1e-15}
    lam = loss_prob(ch, ch.mu, 0, retry_pattern(0, B))
    # kappa (largest retry-pattern loss probability) and J = (T - 2D)^+ are kept as diagnostics of the open-loop
    # bound Lambda (1 - J kappa). Here kappa = 1: some successes leave an empty retry pattern (no budget left, or too
    # little to avoid a later failure, so pi_L idles); the lower bound therefore uses nu_pi (Section S3.6).
    kappa = 0.0
    for (s, r, w), m in reset.items():
        if s + D > T: continue
        p = loss_prob(ch, {w: 1.0}, s, retry_pattern(s, r))
        lam += m * p; kappa = max(kappa, p)
    J = max(T - 2 * D, 0)
    nu = nu_after_crossing(ch, crossing_states)
    R = float(evaluate_policy(ch, tab, T, D, B)[0])
    return dict(R=R, Lambda=lam, Lambda_direct=EN_direct, kappa=kappa, J=J, nu=nu, lower=lam * (1 - nu))


out = {"T": T, "D": D, "B": B, "patterns": {s: retry_pattern(s, r) for s, r in [(0, B), (10, 12), (20, 9)]}}
print("NACK-retry pattern after a success at s=10 with r=12:", out["patterns"][10])
rows = []
for name, (a, b) in {"nominal": (0.4, 0.9), "C1": (0.56, 0.95), "C2": (19 / 30, 0.45)}.items():
    res = analyse(example_chain(a, b)); res["name"] = name; rows.append(res)
    print(f"{name:8s} R={res['R']:.4e}  Lambda={res['Lambda']:.4e} (direct {res['Lambda_direct']:.4e})  "
          f"Lambda/R={res['Lambda'] / res['R']:.3f}  nu={res['nu']:.3f}  lower/R={res['lower'] / res['R']:.3f}")
grid = []
for a in np.linspace(float(A_RANGE[0]), float(A_RANGE[1]), 9):
    for b in np.linspace(float(B_RANGE[0]), float(B_RANGE[1]), 5):
        res = analyse(example_chain(a, b)); grid.append(dict(rho=1 - a, b=b, **res))
gap = max(r["Lambda"] / r["R"] - 1 for r in grid)
cons = max(abs(r["Lambda"] - r["Lambda_direct"]) / r["Lambda"] for r in grid + rows)
low = min(r["lower"] / r["R"] for r in grid)
print(f"class grid: max (Lambda - R)/R = {gap:.3f}, min lower/R = {low:.3f}; decomposition vs direct count max rel. diff {cons:.1e}")
for D2 in (16, 20):   # rarer failures: the bound tightens
    D = D2; tab, _ = design_pi_L(0.4, 0.9, T, D, B)
    res = analyse(example_chain(19 / 30, 0.45))
    print(f"C2, D={D}: R={res['R']:.3e} Lambda/R={res['Lambda'] / res['R']:.3f} lower/R={res['lower'] / res['R']:.3f}")
    rows.append(dict(name=f"C2_D{D}", **res))
json.dump(dict(out, rows=rows, grid=grid, max_gap=gap, min_lower_ratio=low, decomposition_err=cons), open("results/closedloop.json", "w"), indent=1)
