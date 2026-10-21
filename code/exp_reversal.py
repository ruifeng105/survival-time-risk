"""Exact rational certificate of the risk-order reversal between two compatible channels."""
import json, time
from fractions import Fraction as Fr
from core import *
from policy import design_pi_L, evaluate_policy
T, D, Bt, g = 60, 12, 15, 4
C1ab = (Fr(14,25), Fr(19,20)); C2ab = (Fr(19,30), Fr(9,20))
C1, C2, NOM = example_chain(*C1ab), example_chain(*C2ab), example_chain(*NOMINAL_AB)
for ch in (C1,C2): ch.check(tol=0); assert ch.marginal(2)==C_EXAMPLE
out = {"T":T,"D":D,"B_tight":Bt,"g":g,"C1":[str(x) for x in C1ab],"C2":[str(x) for x in C2ab],
       "rho1":str(C1.rho()),"rho2":str(C2.rho())}
t0=time.time()
rd = {k: risk_dense(ch,T,D) for k,ch in (("C1",C1),("C2",C2),("nom",NOM))}
rp = {k: risk_schedule(ch,T,D,periodic_slots(T,g)) for k,ch in (("C1",C1),("C2",C2),("nom",NOM))}
rp2 = {k: risk_schedule(ch,T,D,periodic_slots(T,2)) for k,ch in (("C1",C1),("C2",C2),("nom",NOM))}
print("open-loop done", time.time()-t0)
q0,q1 = Fr(2,5), Fr(9,10)
tabD, modD = design_pi_L(q0,q1,T,D,T)
tabB, modB = design_pi_L(q0,q1,T,D,Bt)
print("design done", time.time()-t0)
rcD = {k: evaluate_policy(ch,tabD,T,D,T)[0] for k,ch in (("C1",C1),("C2",C2))}
rcB = {k: evaluate_policy(ch,tabB,T,D,Bt)[0] for k,ch in (("C1",C1),("C2",C2))}
print("closed-loop done", time.time()-t0)
assert rcD["C1"]==rd["C1"] and rcD["C2"]==rd["C2"]   # with B = T, pi_L has the risk of dense attempts
for name, d in (("dense",rd),("periodic_g2",rp2),("periodic_g4",rp),("closed_loop_B15",rcB)):
    out[name] = {k:{"exact":str(v),"float":float(v)} for k,v in d.items()}
out["modelled_dense"]=float(modD); out["modelled_B15"]=float(modB)
out["dense_order_C1_gt_C2"] = rd["C1"] > rd["C2"]
out["periodic_order_C1_lt_C2"] = rp["C1"] < rp["C2"]
out["closed_loop_order_C1_lt_C2"] = rcB["C1"] < rcB["C2"]
out["dense_ratio_C1_over_C2"] = float(rd["C1"]/rd["C2"])
out["closed_ratio_C2_over_C1"] = float(rcB["C2"]/rcB["C1"])
out["periodic_ratio_C2_over_C1"] = float(rp["C2"]/rp["C1"])
out["periodic_g2_ratio_C2_over_C1"] = float(rp2["C2"]/rp2["C1"])
out["periodic_g2_order_C1_lt_C2"] = rp2["C1"] < rp2["C2"]
from policy import spacing_profile
gaps = spacing_profile(tabB,T,D,Bt,NOM); tot=sum(gaps.values())
out["pi_L_B15_retry_gap_distribution_nominal"] = {int(k):float(v/tot) for k,v in sorted(gaps.items())}
out["pair_loss_lag2"] = {"C1":str(C1.block((0,0,0))+C1.block((0,1,0))), "C2":str(C2.block((0,0,0))+C2.block((0,1,0)))}
out["box_t"] = {"C1":[str(C1.nu[(0,0,0)]),str(C1.nu[(0,1,0)])], "C2":[str(C2.nu[(0,0,0)]),str(C2.nu[(0,1,0)])]}
print(json.dumps(out, indent=1))
json.dump(out, open("results/reversal_exact.json","w"), indent=1)
