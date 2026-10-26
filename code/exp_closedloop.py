"""Closed-loop decomposition (Eq. (15) of the paper, Proposition S3 of the supplement). Between two
successes every observation is a NACK, so after a success at slot s with remaining budget r
the attempt slots in (s, s+D] are a deterministic NACK-retry pattern R(s, r). After a violation the
policy idles, which leaves R(pi) unchanged; every failed mission then has a unique last success before
its first violation, so
    R(pi) = Lambda_pi = P(losses on R(0, B)) + sum_{s, r, w} P(success at s before any violation, budget r, S_{s+1} = w)
                                                * P(losses on R(s, r) | S_{s+1} = w).
Checked for pi_L (B = 15) over the M = 2 class against the direct crossing count and the exact risk."""
import json
import numpy as np
from core import words, example_chain, A_RANGE, B_RANGE
from policy import design_pi_L, evaluate_policy

T, D, B = 60, 12, 15
tab, _ = design_pi_L(0.4, 0.9, T, D, B)


def decide(t, d, r, o, k):
    if r == 0 or d >= D: return 0            # after a violation the policy idles
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


def analyse(ch):
    W = words(ch.M)
    # forward recursion: masses of successes before any violation and the direct crossing count
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
    lam = loss_prob(ch, ch.mu, 0, retry_pattern(0, B))
    for (s, r, w), m in reset.items():
        if s + D > T: continue
        lam += m * loss_prob(ch, {w: 1.0}, s, retry_pattern(s, r))
    R = float(evaluate_policy(ch, tab, T, D, B)[0])
    return dict(R=R, Lambda=lam, Lambda_direct=EN_direct)


out = {"T": T, "D": D, "B": B, "patterns": {0: retry_pattern(0, B)}}
print("NACK-retry pattern at the mission start:", out["patterns"][0])
rows = []
for name, (a, b) in {"nominal": (0.4, 0.9), "C1": (0.56, 0.95), "C2": (19 / 30, 0.45)}.items():
    res = analyse(example_chain(a, b)); res["name"] = name; rows.append(res)
    print(f"{name:8s} R={res['R']:.4e}  Lambda={res['Lambda']:.4e} (direct {res['Lambda_direct']:.4e})  "
          f"Lambda/R - 1 = {res['Lambda'] / res['R'] - 1:.1e}")
grid = []
for a in np.linspace(float(A_RANGE[0]), float(A_RANGE[1]), 9):
    for b in np.linspace(float(B_RANGE[0]), float(B_RANGE[1]), 5):
        res = analyse(example_chain(a, b)); grid.append(dict(rho=1 - a, b=b, **res))
diff = max(abs(r["Lambda"] / r["R"] - 1) for r in grid + rows)
cons = max(abs(r["Lambda"] - r["Lambda_direct"]) / r["Lambda"] for r in grid + rows)
print(f"class grid: max |Lambda/R - 1| = {diff:.1e}; decomposition vs direct count max rel. diff {cons:.1e}")
json.dump(dict(out, rows=rows, grid=grid, max_rel_diff=diff, decomposition_err=cons), open("results/closedloop.json", "w"), indent=1)
