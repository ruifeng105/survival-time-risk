"""M=3, L=2: the compatible class is a 4-dimensional box. Check (i) the two-sided bound
for random channels in the box, (ii) that the worst case follows rho*(n,k) with M=3,
(iii) that the 3 coordinates other than t_{00} barely move the risk."""
import json, random, numpy as np
from fractions import Fraction as Fr
from core import *
random.seed(1)
p2 = {(0,0):0.35,(0,1):0.85,(1,0):0.55,(1,1):0.92}
src = chain_from_p(2, p2)
c = src.marginal(3)
L, eta = 2, 0.05
R = box_ranges(c, L, eta)
lo, hi, rnom = rho_interval(c, L, eta)
A0 = c[(0,0,0)]
print("c(000)=%.4f c(100)=%.4f  rho in [%.4f, %.4f], nominal %.4f" % (A0, c[(1,0,0)], lo, hi, rnom))
print("box:", {m:(round(a,4),round(b,4)) for m,(a,b) in R.items()})
out = {"c":{''.join(map(str,k)):v for k,v in c.items()}, "rho_lo":lo, "rho_hi":hi, "rho_nom":rnom, "cases":[]}
viol = 0.0
for (T,D) in [(8,8),(60,8),(200,8),(100,12),(400,12),(150,30)]:
    rhos = np.linspace(lo, hi, 301)
    best = []; spread=[]
    for rho in rhos:
        vals=[]
        for rep in range(4):
            t = {m: random.uniform(*R[m]) for m in words(L)}
            t[(0,0)] = rho*A0
            ch = box_chain(c, L, t)
            r = risk_dense(ch,T,D); s = dense_stats(ch,T,D)
            viol = max(viol, r-s["EN"], s["lower"]-r)
            vals.append(r)
        best.append(max(vals)); spread.append((max(vals)-min(vals))/max(vals))
    best=np.array(best)
    i=int(best.argmax()); rs=rho_star(T,D,3)
    pred = hi if rs is None else min(max(float(rs),lo),hi)
    nomch = box_chain(c,L,box_nominal(c,L))
    case=dict(T=T,D=D,rho_argmax=float(rhos[i]),rho_pred=pred,Rmax=float(best[i]),Rmin=float(best.min()),
              R_nom=float(risk_dense(nomch,T,D)), max_rel_spread_other_coords=float(max(spread)))
    out["cases"].append(case)
    print(case)
out["max_bound_violation"]=viol
print("max bound violation:", viol)
json.dump(out, open("results/dense_M3.json","w"), indent=1)
