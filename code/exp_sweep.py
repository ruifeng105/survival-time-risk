"""How large is the ambiguity across links? Gilbert fits with loss rate eps and mean burst
length ell; for each, D is the smallest survival time at which the fitted model certifies
R_nom <= 1e-4 with dense attempts (T = 60). Exact risks over the M = 2 compatible class
(eta = 1e-4) for dense attempts and for pi_L with B = 15, evaluated vectorized over channels."""
import json, time
import numpy as np
from core import words, box_ranges, box_chain, box_nominal
from policy import design_pi_L

T, B, TARGET, ETA = 60, 15, 1e-4, 1e-4
EPS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
ELL = [1.25, 1.5, 2.0, 2.5, 3.0, 5.0]
W = words(2)


def gilbert_c(eps, ell):
    c01 = eps / ell
    return {(0, 0): eps - c01, (0, 1): c01, (1, 0): c01, (1, 1): 1 - eps - c01}


def class_grid(c, n0, n1):
    R = box_ranges(c, 1, ETA)
    x0 = np.linspace(*R[(0,)], n0)
    x1 = np.linspace(*R[(1,)], n1)
    chans = [box_chain(c, 1, {(0,): a, (1,): b}) for a in x0 for b in x1]
    return chans, x0, x1


def stack(chans):
    P = np.array([[ch.p[w] for w in W] for ch in chans])          # success prob per context
    MU = np.array([[ch.mu[w] for w in W] for ch in chans])
    return P, MU


def nxt(i, g):   # context index after appending g (contexts ordered 00,01,10,11)
    return 2 * (i & 1) + g


def risk_dense_vec(P, MU, D):
    K = P.shape[0]
    dist = np.zeros((K, 4, D)); dist[:, :, 0] = MU
    fail = np.zeros(K)
    for _ in range(T):
        nd = np.zeros_like(dist)
        for w in range(4):
            q = P[:, w][:, None]
            nd[:, nxt(w, 1), 0] += (dist[:, w, :] * q).sum(1)
            nd[:, nxt(w, 0), 1:] += dist[:, w, :-1] * (1 - q)
            fail += dist[:, w, -1] * (1 - P[:, w])
        dist = nd
    return fail


def risk_policy_vec(P, MU, table, D):
    K = P.shape[0]
    dist = {(0, B, None, 0, w): MU[:, w].copy() for w in range(4)}
    fail = np.zeros(K)
    for t in range(1, T + 1):
        nd = {}
        for (d, r, o, k, w), pr in dist.items():
            u = table.get((t, d, r, o, k), 0) if r > 0 else 0
            q = P[:, w]
            for g, pg in ((1, q), (0, 1 - q)):
                m = pr * pg
                w2 = nxt(w, g)
                if u == 1 and g == 1:
                    key = (0, r - 1, 1, 1, w2)
                elif d + 1 >= D:
                    fail += m; continue
                elif u == 1:
                    key = (d + 1, r - 1, 0, 1, w2)
                else:
                    key = (d + 1, r, o, (k + 1) if o is not None else 0, w2)
                if key in nd: nd[key] += m
                else: nd[key] = m.copy()
        dist = nd
    return fail


rows = []
t0 = time.time()
for eps in EPS:
    for ell in ELL:
        c = gilbert_c(eps, ell)
        nom = box_chain(c, 1, box_nominal(c, 1))
        Pn, MUn = stack([nom])
        D = next(D for D in range(3, T + 1) if risk_dense_vec(Pn, MUn, D)[0] <= TARGET)
        chans, x0, x1 = class_grid(c, 121, 3)
        P, MU = stack(chans)
        Rd = risk_dense_vec(P, MU, D)
        Rd_nom = risk_dense_vec(Pn, MUn, D)[0]
        q0, q1 = c[(0, 1)] / (c[(0, 0)] + c[(0, 1)]), c[(1, 1)] / (c[(1, 0)] + c[(1, 1)])
        tab, modB = design_pi_L(q0, q1, T, D, B)
        chB, x0B, x1B = class_grid(c, 41, 5)
        PB, MUB = stack(chB)
        RB = risk_policy_vec(PB, MUB, tab, D)
        RB_nom = risk_policy_vec(Pn, MUn, tab, D)[0]
        rho = lambda ch: 1 - ch.p[(0, 0)]
        i, j = int(Rd.argmax()), int(RB.argmax())
        row = dict(eps=eps, ell=ell, D=D, rho_nom=1 - 1 / ell,
                   dense=dict(R_nom=Rd_nom, R_plus=float(Rd.max()), R_minus=float(Rd.min()), rho_worst=rho(chans[i])),
                   B15=dict(R_nom=RB_nom, modelled=float(modB), R_plus=float(RB.max()), R_minus=float(RB.min()),
                            rho_worst=rho(chB[j]), b_worst=chB[j].p[(0, 1)]))
        rows.append(row)
        print(f"eps={eps:<6} ell={ell:<4} D={D:2d} | dense R_nom={Rd_nom:.2e} R+/R_nom={Rd.max()/Rd_nom:7.1f} (rho*={rho(chans[i]):.3f})"
              f" | B=15 R_nom={RB_nom:.2e} R+/R_nom={RB.max()/RB_nom:6.1f}  [{time.time()-t0:.0f}s]", flush=True)
json.dump(dict(T=T, B=B, target=TARGET, eta=ETA, rows=rows), open("results/sweep.json", "w"), indent=1)
