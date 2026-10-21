"""Certification baselines on the running example:
 nominal risk, grid worst case over the box (exact risks on a 47 x 21 grid in (a, b)), closed-form box certificate (dense),
 context-wise interval relaxation with a history-dependent adversary (rectangular robust MDP),
 and the worst time-invariant corner of the context-wise intervals (16 stationary chains, no coupling through c)."""
import json, itertools, numpy as np
from functools import lru_cache
from core import *
from policy import design_pi_L, evaluate_policy
q0, q1 = 0.4, 0.9
# context-wise projection of the box (success probabilities)
I = {(0,0): (0.05, 19/30), (0,1): (0.45, 0.95), (1,0): (1-1.5*19/30, 1-1.5*0.05), (1,1): (1-0.95/9, 1-0.45/9)}
mu0 = {w: float(v) for w, v in C_EXAMPLE.items()}

def robust_interval(T, D, B, table):
    """max over history-dependent choices p_t(w) in I[w] of the failure probability."""
    import sys; sys.setrecursionlimit(100000)
    @lru_cache(maxsize=None)
    def V(t, d, r, o, k, w):
        if t > T: return 0.0
        u = table.get((t, d, r, o, k), 0) if r > 0 else 0
        best = 0.0
        for p in I[w]:
            val = 0.0
            for g, pg in ((1, p), (0, 1 - p)):
                w2 = w[1:] + (g,)
                if u == 1:
                    if g == 1: val += pg * V(t+1, 0, r-1, 1, 1, w2)
                    else: val += pg * (1.0 if d+1 >= D else V(t+1, d+1, r-1, 0, 1, w2))
                else:
                    val += pg * (1.0 if d+1 >= D else V(t+1, d+1, r, o, (k+1) if o is not None else 0, w2))
            best = max(best, val)
        return best
    return max(V(1, 0, B, None, 0, w) for w in words(2))   # adversary also picks the initial context

def corner_chains():
    for ps in itertools.product(*[I[w] for w in words(2)]):
        p = dict(zip(words(2), ps))
        yield chain_from_p(2, p)

out = []
A = np.linspace(0.05, 19/30, 47); Bv = np.linspace(0.45, 0.95, 21)
for (T, D, B) in [(60, 12, 60), (60, 20, 60), (60, 12, 15)]:
    tab, mod = design_pi_L(q0, q1, T, D, B)
    Rg = max(evaluate_policy(example_chain(a, b), tab, T, D, B)[0] for a in A for b in Bv)
    if B >= T:
        # refine in rho (b irrelevant to first order): fine 1-D search at b in {0.45,0.95}
        Rg = max(Rg, max(risk_dense(example_chain(a, b), T, D) for a in np.linspace(0.05, 19/30, 2001) for b in (0.45, 0.95)))
        k, n = D - 2, T - D
        rs = rho_star(T, D, 2); rw = 0.95 if rs is None else min(max(float(rs), 11/30), 0.95)
        cert = 0.12 * rw**k * (1 + n * (1 - rw))
    else:
        cert = None
    rob = robust_interval(T, D, B, tab)
    corner = max(evaluate_policy(ch, tab, T, D, B)[0] for ch in corner_chains())
    row = dict(T=T, D=D, B=B, R_nom=mod, R_plus=Rg, box_certificate=cert, interval_robust=rob, worst_corner_stationary=corner)
    out.append(row); print(row)
json.dump(out, open("results/baselines.json", "w"), indent=1)
