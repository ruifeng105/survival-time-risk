"""Consistency checks of the running-example family and of pi_L (run: python3 test_core.py)."""
from fractions import Fraction as Fr
from core import *
from policy import design_pi_L, evaluate_policy

for (a,b) in [(Fr(1,10),Fr(1,2)), NOMINAL_AB, (A_RANGE[0],B_RANGE[0]), (A_RANGE[1],B_RANGE[1])]:
    ch = example_chain(a,b); ch.check(tol=0)
    assert ch.marginal(2) == C_EXAMPLE, ch.marginal(2)
print("family consistent with c (exact)")
print("R(a=.1,b=.5)", float(risk_dense(example_chain(Fr(1,10),Fr(1,2)),6,6)), "expect 0.078732")
print("R(nominal)  ", float(risk_dense(example_chain(*NOMINAL_AB),6,6)), "expect 0.015552")
# with B = T, pi_L has the risk of dense attempts
q0,q1 = Fr(2,5), Fr(9,10)
T,D = 20,6
tab, mod = design_pi_L(q0,q1,T,D,T)
nom = example_chain(*NOMINAL_AB)
r_eval,_ = evaluate_policy(nom, tab, T, D, T)
print("full budget: modelled", float(mod), "eval on nominal", float(r_eval), "dense", float(risk_dense(nom,T,D)))
# tight budget consistency: modeled risk equals evaluation on the nominal chain
for B in [10,7,5]:
    tab, mod = design_pi_L(q0,q1,T,D,B)
    r_eval,att = evaluate_policy(nom, tab, T, D, B)
    print(f"B={B}: modelled {float(mod):.10f} eval-on-nominal {float(r_eval):.10f} E[attempts] {float(att):.3f}")
    assert mod == r_eval
print("ok")
