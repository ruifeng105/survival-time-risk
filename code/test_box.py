"""Check that the box parameterization of Lemma 1 reproduces the running-example family (run: python3 test_box.py)."""
from fractions import Fraction as Fr
from core import *
# M=2 family reproduced by the box parametrization
R = box_ranges(C_EXAMPLE, 1, ETA_EXAMPLE)
print("box ranges M=2:", {m:(str(a),str(b)) for m,(a,b) in R.items()})
print("rho interval:", [str(x) for x in rho_interval(C_EXAMPLE,1,ETA_EXAMPLE)])
# compare with (a,b) family: t_0 = nu(000) = 0.12(1-a); t_1 = nu(010) = 0.08(1-b)
for (a,b) in [(Fr(1,10),Fr(1,2)), NOMINAL_AB]:
    ch1 = example_chain(a,b)
    ch2 = box_chain(C_EXAMPLE,1,{(0,):Fr(3,25)*(1-a),(1,):Fr(2,25)*(1-b)})
    assert ch1.nu == ch2.nu
nom = box_nominal(C_EXAMPLE,1)
print("nominal t:", {m:str(v) for m,v in nom.items()}, "-> a=", str(1-nom[(0,)]/Fr(3,25)), " b=", str(1-nom[(1,)]/Fr(2,25)))
