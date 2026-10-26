"""Gilbert-Elliott channels that share the Gilbert fit of the running example: exact dense and
budgeted risks and their second-order projections (Section V of the paper, Table S7)."""
import json
from core import *
from ge import *
from policy import design_pi_L, evaluate_policy
T, D = 60, 12
tabD, _ = design_pi_L(0.4, 0.9, T, D, T); tabB, modB = design_pi_L(0.4, 0.9, T, D, 15)
rows = []
for eG, eB in [(0.0, 1.0), (0.01, 0.9), (0.0, 0.8), (0.0, 0.7), (0.0, 0.65)]:
    ge, lam = ge_matching(eG, eB)
    c2 = {w: ge.block(w) for w in words(2)}
    proj = ge.projection(2)
    a, b = float(proj.p[(0,0)]), float(proj.p[(0,1)])
    rd, rb = risk_dense_ge(ge, T, D), evaluate_policy_ge(ge, tabB, T, D, 15)
    rdp, rbp = risk_dense(proj, T, D), evaluate_policy(proj, tabB, T, D, 15)[0]
    row = dict(eG=eG, eB=eB, memory=lam, c2=[c2[w] for w in words(2)], rho2=1-a, b2=b, in_box=(0.05 <= a <= 19/30 and 0.45 <= b <= 0.95),
               rho_deep=ge.deep_persistence(), sigma2=sigma(proj, 2), R_dense=rd, R_B15=rb, R_dense_proj2=rdp, R_B15_proj2=rbp)
    rows.append(row)
    print(f"eG={eG} eB={eB} lam={lam:.3f} c2={[round(x,4) for x in row['c2']]} | proj2 rho={1-a:.3f} b={b:.3f} inbox={row['in_box']} | rho_deep={row['rho_deep']:.3f} | "
          f"dense {rd:.3e} (proj {rdp:.3e}) | B15 {rb:.3e} (proj {rbp:.3e})")
print("nominal (fit) dense", risk_dense(example_chain(0.4,0.9),T,D), "B15", modB)
json.dump(rows, open("results/ge.json", "w"), indent=1)
