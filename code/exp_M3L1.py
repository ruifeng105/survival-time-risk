"""M = L + 2 (M=3, L=1): the compatible class is a general polytope. Dense worst case via the
2-D projection onto (y, x) = (mu(000), mu(0000)); exact risk at the maximizer."""
import json, numpy as np
from scipy.optimize import linprog
from core import *
c = {w: float(v) for w, v in C_EXAMPLE.items()}; eta = 0.05; M = 3
W4 = words(4); ix = {w: i for i, w in enumerate(W4)}
Aeq, beq, Aub, bub = [], [], [], []
for w in words(3):                      # shift consistency
    r = np.zeros(16); r[ix[(0,)+w]] += 1; r[ix[(1,)+w]] += 1; r[ix[w+(0,)]] -= 1; r[ix[w+(1,)]] -= 1
    Aeq.append(r); beq.append(0)
for u in words(2):                      # 2-block marginal (first two coordinates) = c
    r = np.zeros(16)
    for w in W4:
        if w[:2] == u: r[ix[w]] = 1
    Aeq.append(r); beq.append(c[u])
for w in words(3):                      # eta-positivity: nu(wx) >= eta * mu(w)
    for x in (0, 1):
        r = np.zeros(16); r[ix[w+(x,)]] -= 1; r[ix[w+(0,)]] += eta; r[ix[w+(1,)]] += eta
        Aub.append(r); bub.append(0)
Aeq, beq, Aub, bub = map(np.array, (Aeq, beq, Aub, bub))
yv = np.zeros(16); yv[ix[(0,0,0,0)]] = 1; yv[ix[(0,0,0,1)]] = 1      # y = mu(000)
xv = np.zeros(16); xv[ix[(0,0,0,0)]] = 1                              # x = mu(0000)
def lp(cost, extra_eq=None):
    Ae, be = Aeq, beq
    if extra_eq is not None:
        Ae = np.vstack([Aeq] + [e[0] for e in extra_eq]); be = np.concatenate([beq, [e[1] for e in extra_eq]])
    return linprog(cost, A_ub=Aub, b_ub=bub, A_eq=Ae, b_eq=be, bounds=[(0, 1)] * 16, method="highs")
# projection polygon by support functions
pts = []
for th in np.linspace(0, 2*np.pi, 721):
    res = lp(-(np.cos(th) * yv + np.sin(th) * xv))
    pts.append((yv @ res.x, xv @ res.x))
pts = np.array(pts)
from scipy.spatial import ConvexHull
hull = ConvexHull(pts); poly = pts[hull.vertices]
ymin, ymax = poly[:, 0].min(), poly[:, 0].max()
print("y = mu(000) range: [%.4f, %.4f]" % (ymin, ymax))
T, D = 60, 12; k, n = D - M, T - D
def Lam(y, x): return y * (x / y) ** k * (1 + n * (1 - x / y))
# dense sampling of the polygon
from matplotlib.path import Path
P = Path(poly)
Y, X = np.meshgrid(np.linspace(ymin, ymax, 400), np.linspace(pts[:, 1].min(), pts[:, 1].max(), 400))
cand = np.c_[Y.ravel(), X.ravel()]; cand = cand[P.contains_points(cand)]
vals = np.array([Lam(y, x) for y, x in cand]); j = vals.argmax(); ys, xs = cand[j]
print("max Lambda %.4f at y=%.4f, rho=x/y=%.4f" % (vals[j], ys, xs / ys))
res = lp(np.zeros(16), extra_eq=[(yv, ys), (xv, xs)])
nu = {w: max(res.x[ix[w]], 0.0) for w in W4}; s_ = sum(nu.values()); nu = {w: v / s_ for w, v in nu.items()}
ch = Chain(3, nu)
R = risk_dense(ch, T, D)
# nominal y = Pr(000) under the fitted L = 1 model: c(00) * Pr(0 | 0) = 0.12 * 0.6 (Remark S1)
y_nom = 0.12 * 0.6
print("exact risk at maximizer: %.4f ; nominal y = %.4f ; M=2 worst 0.242 ; nominal 0.0146" % (R, y_nom))
json.dump(dict(y_range=[ymin, ymax], Lambda_max=float(vals[j]), y_star=float(ys), rho_star=float(xs/ys), R_at_max=float(R), y_nom=y_nom),
          open("results/M3L1.json", "w"), indent=1)
