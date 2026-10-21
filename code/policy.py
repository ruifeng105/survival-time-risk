"""
pi_L: the risk-minimizing budgeted scheduler under ACK/NACK feedback, designed on the
first-order (L = 1) channel model, and its exact evaluation on a reference
M-th order channel.

Under the L = 1 model with q_g = P(gamma_t = 1 | gamma_{t-1} = g), the
controller's prediction P(gamma_t = 1 | info) depends only on the last
observed outcome o and the number k of slots since that observation (delta in the paper):
    pred(o, k) = f^{k-1}(q_o),  f(x) = q_0 + (q_1 - q_0) x,
and pred = mu_1 (stationary) before the first observation (o = None).
Decision state: (t, d, r, o, k) with d the age before slot t and r the
remaining attempts. Ties are broken toward idling (attempt only if it
strictly lowers the modeled risk).
"""
from functools import lru_cache
from fractions import Fraction as Fr


def design_pi_L(q0, q1, T, D, B, tol=None):
    exact = isinstance(q0, Fr)
    mu1 = q0 / (1 - q1 + q0)

    def pred(o, k):
        if o is None:
            return mu1
        x = q1 if o == 1 else q0
        for _ in range(k - 1):
            x = q0 + (q1 - q0) * x
        return x

    return design_from_pred(pred, T, D, B, exact, tol)


def design_from_pred(pred_fn, T, D, B, exact=False, tol=None):
    """Same dynamic program with an arbitrary recovery curve pred_fn(o, k) =
    P(gamma_t = 1 | last observed outcome o, observed k slots ago); o = None before
    the first observation. design_pi_L uses the curve of the first-order model."""
    if tol is None:
        tol = 0 if exact else 1e-13
    pred = lru_cache(maxsize=None)(pred_fn)
    zero = Fr(0) if exact else 0.0
    one = Fr(1) if exact else 1.0
    table = {}

    @lru_cache(maxsize=None)
    def V(t, d, r, o, k):
        if t > T:
            return zero
        # idle
        dd = d + 1
        nk = k + 1 if o is not None else 0
        v_idle = one if dd >= D else V(t + 1, dd, r, o, nk)
        if r == 0:
            table[(t, d, r, o, k)] = 0
            return v_idle
        s = pred(o, k)
        v_fail = one if dd >= D else V(t + 1, dd, r - 1, 0, 1)
        v_tx = s * V(t + 1, 0, r - 1, 1, 1) + (1 - s) * v_fail
        if v_tx < v_idle - tol:
            table[(t, d, r, o, k)] = 1
            return v_tx
        table[(t, d, r, o, k)] = 0
        return v_idle

    import sys
    sys.setrecursionlimit(max(10000, 50 * T))
    modelled = V(1, 0, B, None, 0)
    return table, modelled


def evaluate_policy(ch, table, T, D, B):
    """Exact risk and expected number of attempts, (risk, attempts), of the fixed table policy on reference chain ch."""
    from core import words
    M = ch.M
    dist = {(0, B, None, 0, w): ch.mu[w] for w in words(M)}
    fail = 0 * next(iter(ch.mu.values()))
    attempts = 0 * fail
    for t in range(1, T + 1):
        nd = {}
        for (d, r, o, k, w), pr in dist.items():
            u = table.get((t, d, r, o, k), 0) if r > 0 else 0
            q = ch.p[w]
            for g, pg in ((1, q), (0, 1 - q)):
                m = pr * pg
                w2 = w[1:] + (g,)
                if u == 1:
                    attempts += m
                    if g == 1:
                        key = (0, r - 1, 1, 1, w2)
                    else:
                        if d + 1 >= D:
                            fail += m
                            continue
                        key = (d + 1, r - 1, 0, 1, w2)
                else:
                    if d + 1 >= D:
                        fail += m
                        continue
                    key = (d + 1, r, o, (k + 1) if o is not None else 0, w2)
                nd[key] = nd.get(key, 0) + m
        dist = nd
    return fail, attempts


def spacing_profile(table, T, D, B, q_model_chain):
    """Distribution of the gap between a NACK and the next attempt, under the
    modeled channel (diagnostic of how pi_L spreads retries)."""
    from core import words
    ch = q_model_chain
    gaps = {}
    dist = {(0, B, None, 0, w, None): ch.mu[w] for w in words(ch.M)}
    for t in range(1, T + 1):
        nd = {}
        for (d, r, o, k, w, last_nack), pr in dist.items():
            u = table.get((t, d, r, o, k), 0) if r > 0 else 0
            q = ch.p[w]
            for g, pg in ((1, q), (0, 1 - q)):
                m = pr * pg
                w2 = w[1:] + (g,)
                ln = last_nack
                if u == 1:
                    if ln is not None:
                        gaps[t - ln] = gaps.get(t - ln, 0) + m
                    if g == 1:
                        key = (0, r - 1, 1, 1, w2, None)
                    else:
                        if d + 1 >= D:
                            continue
                        key = (d + 1, r - 1, 0, 1, w2, t)
                else:
                    if d + 1 >= D:
                        continue
                    key = (d + 1, r, o, (k + 1) if o is not None else 0, w2, ln)
                nd[key] = nd.get(key, 0) + m
        dist = nd
    return gaps
