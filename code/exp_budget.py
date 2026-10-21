"""Budget-limited regime: Theorem 1 for open-loop schedules, b-sensitivity (dense, periodic g = 2 and 4,
pi_L with B = 15) and the closed-loop sweep of pi_L over the class."""
import json, random, numpy as np
from fractions import Fraction as Fr
from core import *
from policy import design_pi_L, evaluate_policy
T, D = 60, 12
q0,q1 = 0.4, 0.9
out = {"T":T, "D":D}
# 1) Theorem 1 for open-loop schedules (dense, periodic g=2,3,4, irregular), random channels
random.seed(7); viol=0.0; rows=[]
scheds = {"dense":list(range(1,T+1)), "g2":periodic_slots(T,2), "g3":periodic_slots(T,3), "g4":periodic_slots(T,4),
          "irregular":sorted(random.sample(range(1,T+1), 18))}
for name,S in scheds.items():
    for rep in range(25):
        a = random.uniform(0.05,19/30); b = random.uniform(0.45,0.95)
        ch = example_chain(a,b)
        r = risk_schedule(ch,T,D,S); st = schedule_stats(ch,T,D,S)
        viol = max(viol, r - st["upper"], st["lower"] - r)
        rows.append((name, a, b, r, st["EN"], st["lower"]))
out["thm1_schedule_max_violation"] = viol
print("Theorem 1 (schedules) max violation:", viol)
for name in scheds:
    rr=[x for x in rows if x[0]==name]; rel=[(x[4]-x[3])/x[3] for x in rr if x[3]>0]
    print(f"  {name}: median rel. gap (E[N]-R)/R = {np.median(rel):.3f}, max {max(rel):.3f}")
out["thm1_schedule_max_rel_gap"] = max((x[4]-x[3])/x[3] for x in rows if x[3]>0)
# 2) b-sensitivity at fixed rho
tabB,modB = design_pi_L(q0,q1,T,D,15)
sens=[]
for rho in [11/30, 0.44, 0.6, 0.8, 0.95]:
    a=1-rho
    lo,hi = example_chain(a,0.45), example_chain(a,0.95)
    d = (risk_dense(lo,T,D), risk_dense(hi,T,D))
    p = (risk_schedule(lo,T,D,periodic_slots(T,4)), risk_schedule(hi,T,D,periodic_slots(T,4)))
    p2 = (risk_schedule(lo,T,D,periodic_slots(T,2)), risk_schedule(hi,T,D,periodic_slots(T,2)))
    c = (evaluate_policy(lo,tabB,T,D,15)[0], evaluate_policy(hi,tabB,T,D,15)[0])
    sens.append(dict(rho=rho, dense=d, periodic_g4=p, periodic_g2=p2, closed_B15=c,
                     ratio_dense=d[0]/d[1], ratio_periodic=p[0]/p[1], ratio_periodic_g2=p2[0]/p2[1], ratio_closed=c[0]/c[1]))
    print(f"rho={rho:.3f}: R(b=.45)/R(b=.95) dense {d[0]/d[1]:.3f} | periodic g=2 {p2[0]/p2[1]:.1f} | periodic g=4 {p[0]/p[1]:.2f} | closed B=15 {c[0]/c[1]:.2f}")
out["b_sensitivity"]=sens
# 3) closed-loop sweep over the class
A = np.linspace(0.05, 19/30, 47); Bv = np.linspace(0.45, 0.95, 21)
sweep=[]; curves={}
for B in [60,30,20,15,12]:
    tab, mod = design_pi_L(q0,q1,T,D,B)
    Rg = np.array([[evaluate_policy(example_chain(a,b),tab,T,D,B)[0] for b in Bv] for a in A])
    i,j = np.unravel_index(Rg.argmax(),Rg.shape); k,l=np.unravel_index(Rg.argmin(),Rg.shape)
    sweep.append(dict(B=B, modelled=mod, Rmax=Rg.max(), rho_max=1-A[i], b_max=Bv[j], Rmin=Rg.min(), rho_min=1-A[k], b_min=Bv[l]))
    curves[B] = {"rho":list(1-A), "b045":list(Rg[:,0]), "b095":list(Rg[:,-1]), "upper_env":list(Rg.max(axis=1)), "lower_env":list(Rg.min(axis=1))}
    print(f"B={B}: nominal {mod:.3e}  worst {Rg.max():.3e} at rho={1-A[i]:.3f}, b={Bv[j]:.3f}  best {Rg.min():.3e} at rho={1-A[k]:.3f}, b={Bv[l]:.3f}")
out["closed_loop_sweep"]=sweep; out["curves"]=curves
json.dump(out, open("results/budget.json","w"), indent=1, default=float)
