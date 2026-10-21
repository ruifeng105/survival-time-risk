"""Hidden two-state Gilbert-Elliott channels and exact risk evaluation for them.
State s in {G,B}; in slot t, gamma_t = 0 with prob e_s; then s_{t+1} ~ P(s, .)."""
import numpy as np
from core import words, Chain

class GE:
    def __init__(self, eG, eB, pGB, pBG):
        self.e = {"G": eG, "B": eB}
        self.P = {"G": {"G": 1 - pGB, "B": pGB}, "B": {"G": pBG, "B": 1 - pBG}}
        piB = pGB / (pGB + pBG)
        self.pi = {"G": 1 - piB, "B": piB}
    def trans(self, s):
        out = []
        for g in (0, 1):
            pg = self.e[s] if g == 0 else 1 - self.e[s]
            for s2, ps in self.P[s].items():
                out.append((g, pg * ps, s2))
        return out
    def block(self, pat):
        dist = dict(self.pi)
        for g in pat:
            nd = {}
            for s, pr in dist.items():
                pg = self.e[s] if g == 0 else 1 - self.e[s]
                for s2, ps in self.P[s].items():
                    nd[s2] = nd.get(s2, 0) + pr * pg * ps
            dist = nd
        return sum(dist.values())
    def projection(self, M):
        """M-th order chain with the same (M+1)-block law."""
        nu = {w: self.block(w) for w in words(M + 1)}
        return Chain(M, nu)
    def deep_persistence(self):
        """lim P(loss | many losses) = spectral radius of diag(e) P."""
        E = np.diag([self.e["G"], self.e["B"]])
        P = np.array([[self.P["G"]["G"], self.P["G"]["B"]], [self.P["B"]["G"], self.P["B"]["B"]]])
        return float(max(abs(np.linalg.eigvals(E @ P))))

def ge_matching(eG, eB, loss=0.2, p00=0.12):
    """GE channel with given loss rate and P(loss, loss) (i.e., the same Gilbert fit)."""
    piB = (loss - eG) / (eB - eG)
    lam = (p00 - loss**2) / ((1 - piB) * piB * (eB - eG) ** 2)
    assert 0 < piB < 1 and lam < 1, (piB, lam)
    s = 1 - lam
    return GE(eG, eB, piB * s, (1 - piB) * s), lam

def risk_dense_ge(ch, T, D):
    dist = {(s, 0): p for s, p in ch.pi.items()}
    fail = 0.0
    for t in range(1, T + 1):
        nd = {}
        for (s, d), pr in dist.items():
            for g, pp, s2 in ch.trans(s):
                dd = 0 if g == 1 else d + 1
                if dd >= D: fail += pr * pp; continue
                nd[(s2, dd)] = nd.get((s2, dd), 0) + pr * pp
        dist = nd
    return fail

def evaluate_policy_ge(ch, table, T, D, B):
    dist = {(0, B, None, 0, s): p for s, p in ch.pi.items()}
    fail = 0.0
    for t in range(1, T + 1):
        nd = {}
        for (d, r, o, k, s), pr in dist.items():
            u = table.get((t, d, r, o, k), 0) if r > 0 else 0
            for g, pp, s2 in ch.trans(s):
                m = pr * pp
                if u == 1:
                    if g == 1: key = (0, r - 1, 1, 1, s2)
                    else:
                        if d + 1 >= D: fail += m; continue
                        key = (d + 1, r - 1, 0, 1, s2)
                else:
                    if d + 1 >= D: fail += m; continue
                    key = (d + 1, r, o, (k + 1) if o is not None else 0, s2)
                nd[key] = nd.get(key, 0) + m
        dist = nd
    return fail
