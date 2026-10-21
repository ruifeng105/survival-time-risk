"""
Exact survival-time risk evaluation for budgeted scheduling under ACK/NACK feedback
over stationary M-th order binary packet-loss channels.

Conventions (as in the paper):
  gamma_t in {0,1}, 1 = a transmission in slot t would succeed.
  Context S_t = (gamma_{t-M}, ..., gamma_{t-1}), oldest symbol first.
  A channel is stored through its stationary (M+1)-block measure nu on {0,1}^{M+1};
  p(w) = P(gamma_t = 1 | S_t = w) = nu(w1) / mu(w), mu(w) = nu(w0) + nu(w1).
  S_1 is drawn from the stationary context law mu, the age starts at d_0 = 0.
  Age: d_t = 0 if u_t = 1 and gamma_t = 1, otherwise d_t = d_{t-1} + 1.
  Failure tau = first t with d_t >= D; risk R = P(tau <= T).
The chain and risk routines work with Python floats or fractions.Fraction (exact arithmetic);
entropy_rate, loss_matrix and sigma use floats.
"""
from fractions import Fraction as Fr
import itertools


def words(n):
    return list(itertools.product((0, 1), repeat=n))


class Chain:
    def __init__(self, M, nu):
        """nu: dict {(x_1..x_{M+1}): prob}. Must be shift consistent and positive."""
        self.M = M
        self.nu = dict(nu)
        self.mu = {w: self.nu[w + (0,)] + self.nu[w + (1,)] for w in words(M)}
        self.p = {w: self.nu[w + (1,)] / self.mu[w] for w in words(M)}

    # ---- basic statistics -------------------------------------------------
    def check(self, tol=1e-12):
        M = self.M
        for w in words(M):
            left = self.nu[(0,) + w] + self.nu[(1,) + w]
            right = self.nu[w + (0,)] + self.nu[w + (1,)]
            assert abs(left - right) <= tol, ("shift consistency", w, left, right)
        assert abs(sum(self.nu.values()) - 1) <= tol

    def block(self, pattern):
        """Stationary probability of a contiguous pattern (tuple of 0/1)."""
        M = self.M
        pat = tuple(pattern)
        if len(pat) <= M + 1:
            n = len(pat)
            return sum(v for w, v in self.nu.items() if w[:n] == pat)
        prob = self.block(pat[:M + 1])
        for i in range(M + 1, len(pat)):
            ctx = pat[i - M:i]
            prob *= self.p[ctx] if pat[i] == 1 else (1 - self.p[ctx])
        return prob

    def marginal(self, n):
        return {w: self.block(w) for w in words(n)}

    def rho(self):
        """Burst persistence beyond the memory: P(loss | 0^M)."""
        return 1 - self.p[(0,) * self.M]

    def entropy_rate(self):
        import math
        h = 0.0
        for w in words(self.M):
            q = float(self.p[w])
            h -= float(self.mu[w]) * (q * math.log2(q) + (1 - q) * math.log2(1 - q))
        return h

    def step(self, w, g):
        return w[1:] + (g,)


# ---- channels from transition vectors ---------------------------------------

def chain_from_p(M, p, exact=False):
    """Build a Chain from transition probs p[w]; stationary law solved linearly."""
    import numpy as np
    W = words(M)
    idx = {w: i for i, w in enumerate(W)}
    if exact:
        n = len(W)
        A = [[Fr(0)] * n for _ in range(n)]
        for w in W:
            for g in (0, 1):
                q = p[w] if g == 1 else 1 - p[w]
                A[idx[w[1:] + (g,)]][idx[w]] += q
        for i in range(n):
            A[i][i] -= 1
        A[-1] = [Fr(1)] * n
        b = [Fr(0)] * (n - 1) + [Fr(1)]
        mu = solve_exact(A, b)
    else:
        n = len(W)
        A = np.zeros((n, n))
        for w in W:
            for g in (0, 1):
                q = p[w] if g == 1 else 1 - p[w]
                A[idx[w[1:] + (g,)], idx[w]] += q
        A -= np.eye(n)
        A[-1, :] = 1
        b = np.zeros(n)
        b[-1] = 1
        mu = list(np.linalg.solve(A, b))
    nu = {}
    for w in W:
        m = mu[idx[w]]
        nu[w + (1,)] = m * p[w]
        nu[w + (0,)] = m * (1 - p[w])
    return Chain(M, nu)


def solve_exact(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        M[c] = [v / pv for v in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [vr - f * vc for vr, vc in zip(M[r], M[c])]
    return [M[i][n] for i in range(n)]


# ---- the running example: M = 2, L = 1 compatible family ---------------------

C_EXAMPLE = {(0, 0): Fr(3, 25), (0, 1): Fr(2, 25), (1, 0): Fr(2, 25), (1, 1): Fr(18, 25)}
ETA_EXAMPLE = Fr(1, 20)
A_RANGE = (Fr(1, 20), Fr(19, 30))
B_RANGE = (Fr(9, 20), Fr(19, 20))
NOMINAL_AB = (Fr(2, 5), Fr(9, 10))


def example_chain(a, b):
    """Second-order chain compatible with C_EXAMPLE.
    p(00)=a, p(01)=b, p(10)=1-3a/2, p(11)=1-b/9 (success probabilities)."""
    one = Fr(1) if isinstance(a, Fr) else 1.0
    p = {(0, 0): a, (0, 1): b, (1, 0): one - Fr(3, 2) * a if isinstance(a, Fr) else 1 - 1.5 * a,
         (1, 1): one - b / 9}
    nu = {}
    for w, m in C_EXAMPLE.items():
        mm = m if isinstance(a, Fr) else float(m)
        nu[w + (1,)] = mm * p[w]
        nu[w + (0,)] = mm * (one - p[w])
    return Chain(2, nu)


# ---- risk evaluators ----------------------------------------------------------

def risk_dense(ch, T, D):
    """Always-attempt policy (optimal whenever B >= T)."""
    M = ch.M
    dist = {(w, 0): ch.mu[w] for w in words(M)}
    fail = 0 * next(iter(ch.mu.values()))
    for t in range(1, T + 1):
        nd = {}
        for (w, d), pr in dist.items():
            q = ch.p[w]
            k = (w[1:] + (1,), 0)
            nd[k] = nd.get(k, 0) + pr * q
            dd = d + 1
            if dd >= D:
                fail += pr * (1 - q)
            else:
                k = (w[1:] + (0,), dd)
                nd[k] = nd.get(k, 0) + pr * (1 - q)
        dist = nd
    return fail


def risk_schedule(ch, T, D, attempt_slots):
    """Open-loop schedule: transmit exactly in the given slots (1-indexed)."""
    S = set(attempt_slots)
    M = ch.M
    dist = {(w, 0): ch.mu[w] for w in words(M)}
    fail = 0 * next(iter(ch.mu.values()))
    for t in range(1, T + 1):
        tx = t in S
        nd = {}
        for (w, d), pr in dist.items():
            q = ch.p[w]
            for g, pg in ((1, q), (0, 1 - q)):
                dd = 0 if (tx and g == 1) else d + 1
                if dd >= D:
                    fail += pr * pg
                else:
                    k = (w[1:] + (g,), dd)
                    nd[k] = nd.get(k, 0) + pr * pg
        dist = nd
    return fail


def periodic_slots(T, g):
    return [t for t in range(1, T + 1) if t % g == 0]


# ---- dense-regime statistics (Theorem 1) --------------------------------------

def dense_stats(ch, T, D):
    """Dense attempts: EN = Lambda of Theorem 1 (closed form Eq. (12) for M = L + 1), lam = Pr(1 0^D),
    muD = Pr(0^D), kappa, and the two-sided bound lower = Lambda (1 - J kappa) <= R <= upper = Lambda
    with J = (T - 2D)^+."""
    M = ch.M
    assert D >= M
    muD = ch.block((0,) * D)
    lam = ch.block((1,) + (0,) * D)
    EN = muD + (T - D) * lam
    # kappa = max_u P(gamma_1 = 1, gamma_{2..D+1} = 0^D | S_1 = u)
    kap = None
    for u in words(M):
        pr = ch.p[u]
        w = u[1:] + (1,)
        for _ in range(D):
            pr = pr * (1 - ch.p[w])
            w = w[1:] + (0,)
        kap = pr if kap is None or pr > kap else kap
    pairs = max(T - 2 * D, 0)
    lower = EN * (1 - pairs * kap)
    return dict(EN=EN, lam=lam, muD=muD, kappa=kap, lower=lower, upper=EN)


def EN_closed_form(y, rho, T, D, M):
    k, n = D - M, T - D
    return y * rho ** k * (1 + n * (1 - rho))


def rho_star(T, D, M):
    k, n = D - M, T - D
    if n <= k:
        return None  # monotone increasing in rho
    return Fr(k * (n + 1), (k + 1) * n)


# ---- M = L + 1: the compatible class is a box of 2^L Frechet intervals ---------

def box_ranges(c, L, eta):
    """c: stationary (L+1)-block law (dict over words(L+1)). For every middle word
    m in {0,1}^L the 2x2 table nu(x m z) has row sums c(x m), column sums c(m z) and one
    free entry t_m = nu(0 m 0) (x_m in Lemma 1). Returns {m: (lo, hi)} = [l_m, u_m] including eta-positivity."""
    R = {}
    for m in words(L):
        r0, r1 = c[(0,) + m], c[(1,) + m]          # rows: contexts (0m), (1m)
        k0, k1 = c[m + (0,)], c[m + (1,)]          # columns
        # entries: nu(0m0)=t, nu(0m1)=r0-t, nu(1m0)=k0-t, nu(1m1)=r1-k0+t
        lo = max(eta * r0, k0 - (1 - eta) * r1, k0 - r1 + eta * r1)
        hi = min((1 - eta) * r0, k0 - eta * r1, k0 - r1 + (1 - eta) * r1)
        lo = max(lo, 0 * lo)
        R[m] = (lo, hi)
    return R


def box_chain(c, L, t):
    nu = {}
    for m in words(L):
        r0, r1 = c[(0,) + m], c[(1,) + m]
        k0 = c[m + (0,)]
        nu[(0,) + m + (0,)] = t[m]
        nu[(0,) + m + (1,)] = r0 - t[m]
        nu[(1,) + m + (0,)] = k0 - t[m]
        nu[(1,) + m + (1,)] = r1 - k0 + t[m]
    return Chain(L + 1, nu)


def box_nominal(c, L):
    """Max-entropy (L-th order Markov) point: independent coupling in every table."""
    t = {}
    for m in words(L):
        cm = c[(0,) + m] + c[(1,) + m]
        t[m] = c[(0,) + m] * c[m + (0,)] / cm
    return t


def rho_interval(c, L, eta):
    A0 = c[(0,) * (L + 1)]
    A1 = c[(1,) + (0,) * L]
    lo = max(eta, 1 - (1 - eta) * A1 / A0)
    hi = min(1 - eta, 1 - eta * A1 / A0)
    return lo, hi, A0 / (A0 + A1)


# ---- general open-loop schedules: Theorem 1 in gapped-pattern form -------------

def _pattern_prob(ch, start_ctx_law, constraints, length):
    """P(gamma_j = v_j for (j, v_j) in constraints), positions 1..length, starting from
    a context law at position 1."""
    dist = dict(start_ctx_law)
    cons = dict(constraints)
    for j in range(1, length + 1):
        nd = {}
        for w, pr in dist.items():
            q = ch.p[w]
            for g, pg in ((1, q), (0, 1 - q)):
                if j in cons and cons[j] != g:
                    continue
                k = w[1:] + (g,)
                nd[k] = nd.get(k, 0) + pr * pg
        dist = nd
    return sum(dist.values())


def schedule_stats(ch, T, D, S):
    """E[N_S] = Lambda_S of Theorem 1 (expected number of slots with age exactly D, age not reset at failure),
    kappa_S, pairs = J_S and the Bonferroni lower bound of Theorem 1 for an open-loop schedule S."""
    S = sorted(set(S))
    Sset = set(S)
    EN = 0 * next(iter(ch.mu.values()))
    # s = 0: no successful attempt in slots 1..D
    if D <= T:
        cons = {r: 0 for r in S if 1 <= r <= D}
        EN += _pattern_prob(ch, ch.mu, cons, D)
    for s in S:
        if s + D > T:
            continue
        cons = {1: 1}
        cons.update({r - s + 1: 0 for r in S if s < r <= s + D})
        EN += _pattern_prob(ch, ch.mu, cons, D + 1)
    # kappa: worst context, worst attempt slot
    kap = None
    for s in S:
        if s + D > T:
            continue
        cons = {1: 1}
        cons.update({r - s + 1: 0 for r in S if s < r <= s + D})
        for u in words(ch.M):
            v = _pattern_prob(ch, {u: 1}, cons, D + 1)
            kap = v if kap is None or v > kap else kap
    pairs = 0
    for s in [0] + S:
        cnt = sum(1 for s2 in S if s2 > s + D and s2 + D <= T)
        pairs = max(pairs, cnt)
    kap = kap if kap is not None else 0
    return dict(EN=EN, kappa=kap, pairs=pairs, lower=EN * (1 - pairs * kap), upper=EN)


# ---- spaced attempts: g-step loss-to-loss matrix and its spectral radius ---------

def loss_matrix(ch, g):
    """Q_g(w, w') = P(gamma_{t+g} = 0 and S_{t+g+1} = w' | S_{t+1} = w) for contexts
    w, w' ending in a loss; intermediate g-1 symbols are unobserved (summed out)."""
    import numpy as np
    W0 = [w for w in words(ch.M) if w[-1] == 0]
    idx = {w: i for i, w in enumerate(W0)}
    Q = np.zeros((len(W0), len(W0)))
    for w in W0:
        dist = {w: 1.0}
        for j in range(g):
            nd = {}
            for u, pr in dist.items():
                q = float(ch.p[u])
                outs = ((0, 1 - q),) if j == g - 1 else ((1, q), (0, 1 - q))
                for x, px in outs:
                    k = u[1:] + (x,)
                    nd[k] = nd.get(k, 0) + pr * px
            dist = nd
        for u, pr in dist.items():
            Q[idx[w], idx[u]] += pr
    return Q, W0


def sigma(ch, g):
    import numpy as np
    Q, _ = loss_matrix(ch, g)
    return float(max(abs(np.linalg.eigvals(Q))))
