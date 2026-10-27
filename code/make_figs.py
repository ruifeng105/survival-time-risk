"""Generate Figs. 2-4 of the paper and the phase diagram (Fig. S2 of the supplement) in figs/."""
import json, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core import *
os.makedirs("figs", exist_ok=True)
plt.rcParams.update({"font.family":"serif","font.size":8,"axes.labelsize":8,"legend.fontsize":7,
    "xtick.labelsize":7,"ytick.labelsize":7,"axes.linewidth":0.6,"lines.linewidth":1.4,
    "axes.edgecolor":"#52514e","xtick.color":"#52514e","ytick.color":"#52514e","axes.labelcolor":"#0b0b0b",
    "pdf.fonttype":42,"mathtext.fontset":"stix","font.serif":["Times New Roman","Nimbus Roman","Times"]})
C = ["#2a78d6","#eb6834","#1baf7a","#eda100"]; LS=["-","--","-.",":"]; MK=["o","s","^","D"]
GRID = dict(color="#d9d8d4", linewidth=0.4)
rho_lo, rho_hi, rho_nom = 11/30, 0.95, 0.6

# ---- Fig. 2: dense risk vs persistence ------------------------------------------
D = 10; rhos = np.linspace(rho_lo, rho_hi, 300)
fig, ax = plt.subplots(figsize=(3.5, 1.56))
for i, T in enumerate([10, 30, 60, 200]):
    R = np.array([risk_dense(example_chain(1-r, 0.7), T, D) for r in rhos])
    ax.semilogy(rhos, R, color=C[i], ls=LS[i], label=f"$T={T}$")
    j = R.argmax(); ax.plot(rhos[j], R[j], MK[i], color=C[i], ms=4.5, mec="white", mew=0.6)
    rs = rho_star(T, D, 2)
    if rs is not None and float(rs) < rho_hi:
        ax.axvline(float(rs), color=C[i], lw=0.6, ls=":", alpha=0.9)
ax.axvline(rho_nom, color="#52514e", lw=0.8, ls="--")
ax.text(rho_nom-0.008, 3.2e-5, "nominal\n($L=1$ model)", color="#52514e", va="bottom", ha="right")
ax.set_xlabel(r"persistence $\rho$")
ax.set_ylabel(r"risk $R$ (dense attempts)")
ax.set_xlim(rho_lo, 0.957); ax.grid(True, which="major", **GRID)
ax.legend(loc="lower right", bbox_to_anchor=(0.88, 0.0), frameon=False, ncol=2, title=f"$D={D}$", title_fontsize=7)
fig.tight_layout(pad=0.3); fig.savefig("figs/fig_dense_rho.pdf"); plt.close(fig)

# ---- Fig. S2 (supplement): phase diagram + span ----------------------------------
ph = json.load(open("results/phase.json"))
Ds = list(range(3, 33))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.45, 1.95))
labels = {"12":"$n=12$","48":"$n=48$","200":"$n=200$","1000000000":r"$n\to\infty$"}
for i, n in enumerate(["12","48","200","1000000000"]):
    a1.plot(Ds, ph["closed_form"][n], color=C[i], ls=LS[i], lw=1.1, label=labels[n])
    ex = [e for e in ph["exact"] if str(e["n"]) == n]
    if ex: a1.plot([e["D"] for e in ex], [e["rho_argmax"] for e in ex], MK[i], color=C[i], ms=3, mec="white", mew=0.4)
a1.axvline(21, color="#52514e", lw=0.6, ls="--"); a1.text(21.8, 0.968, "$D^\\star$", fontsize=7, color="#52514e")
a1.set_xlabel("survival-time threshold $D$"); a1.set_ylabel(r"worst-case $\rho$"); a1.set_ylim(0.45, 1.0)
a1.grid(True, **GRID); a1.legend(frameon=False, fontsize=6, loc="lower right", handlelength=1.6)
sp = ph["span"]; Dd = [s["D"] for s in sp]
a2.semilogy(Dd, [s["Rmax"]/s["Rnom"] for s in sp], "o", color=C[0], ms=2.6, label=r"$R^+/R_{\rm nom}$")
a2.semilogy(Dd, [s["pred_nom"] for s in sp], "-", color=C[0], lw=0.9)
a2.semilogy(Dd, [s["Rmax"]/s["Rmin"] for s in sp], "s", color=C[1], ms=2.6, label=r"$R^+/R^-$")
a2.semilogy(Dd, [s["pred_span"] for s in sp], "--", color=C[1], lw=0.9)
a2.set_xlabel("survival-time threshold $D$"); a2.set_ylabel("risk ratio ($n=48$)")
a2.grid(True, which="major", **GRID); a2.legend(frameon=False, fontsize=6, loc="upper left")
fig.tight_layout(pad=0.3, w_pad=0.6); fig.savefig("figs/fig_phase.pdf"); plt.close(fig)

# ---- Fig. 3: closed loop, budget changes the statistic ----------------------------
bj = json.load(open("results/budget.json")); rv = json.load(open("results/reversal_exact.json"))
fig, ax = plt.subplots(figsize=(3.5, 1.58))
k = 0
cv = bj["curves"]["60"]
ax.semilogy(cv["rho"], cv["b045"], color="#8a8986", ls="-", lw=1.0, label="$B=60$ (both $b$)")
cv = bj["curves"]["15"]
for bkey, col, lab, ls in (("b095", C[0], "0.95", (0, (5, 2))), ("b045", C[1], "0.45", (0, (2, 1.2)))):
    ax.semilogy(cv["rho"], cv[bkey], color=col, ls=ls, lw=1.2, label=f"$B=15$, $b={lab}$")
r1, r2 = 11/25, 11/30
for key, mk in (("dense", "o"), ("closed_loop_B15", "s")):
    v1, v2 = rv[key]["C1"]["float"], rv[key]["C2"]["float"]
    ax.plot([r1], [v1], mk, color="#0b0b0b", ms=4, mfc="white", mew=0.9)
    ax.plot([r2], [v2], mk, color="#0b0b0b", ms=4, mew=0.9)
ax.annotate("$C_1$", (r1, rv["dense"]["C1"]["float"]), xytext=(4, -9), textcoords="offset points", fontsize=7)
ax.annotate("$C_2$", (r2, rv["dense"]["C2"]["float"]), xytext=(6, -2), textcoords="offset points", fontsize=7)
ax.annotate("$C_1$", (r1, rv["closed_loop_B15"]["C1"]["float"]), xytext=(4, -9), textcoords="offset points", fontsize=7)
ax.annotate("$C_2$", (r2, rv["closed_loop_B15"]["C2"]["float"]), xytext=(3, 3), textcoords="offset points", fontsize=7)
ax.set_xlabel(r"persistence $\rho$"); ax.set_ylabel(r"risk $R$ of $\pi_L$"); ax.set_ylim(1e-4, None)
ax.set_xlim(0.352, rho_hi); ax.grid(True, which="major", **GRID)
ax.legend(frameon=False, loc="lower right", ncol=1, handlelength=2.2)
fig.tight_layout(pad=0.3); fig.savefig("figs/fig_budget.pdf"); plt.close(fig)

# ---- Fig. 4: ambiguity across links (Gilbert fits certifying R_nom <= 1e-4) --------
sw = json.load(open("results/sweep.json"))
eps = sorted({r["eps"] for r in sw["rows"]}); ell = sorted({r["ell"] for r in sw["rows"]})
fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.95), sharey=True)
for ax, key, title in ((axs[0], "dense", "dense attempts"), (axs[1], "B15", r"$\pi_L$, $B=15$")):
    Z = np.full((len(eps), len(ell)), np.nan)
    for r in sw["rows"]:
        Z[eps.index(r["eps"]), ell.index(r["ell"])] = r[key]["R_plus"] / r[key]["R_nom"]
    ax.imshow(np.log10(Z), cmap="Blues", vmin=0, vmax=4.2, origin="lower", aspect="auto")
    for i in range(len(eps)):
        for j in range(len(ell)):
            z = Z[i, j]; s = f"{z:.0f}" if z < 1e3 else f"{z/1e3:.1f}k"
            ax.text(j, i, s, ha="center", va="center", fontsize=6.5, color="white" if np.log10(z) > 2.4 else "#0b0b0b")
    ax.set_xticks(range(len(ell))); ax.set_xticklabels([f"{v:g}" for v in ell])
    ax.set_xlabel("mean burst length"); ax.set_title(title, fontsize=7, pad=2)
    ax.tick_params(length=0)
    for sp_ in ax.spines.values(): sp_.set_visible(False)
axs[0].set_yticks(range(len(eps))); axs[0].set_yticklabels([f"{100*v:g}%" for v in eps])
axs[0].set_ylabel("loss rate")
fig.tight_layout(pad=0.3, w_pad=0.5); fig.savefig("figs/fig_sweep.pdf"); plt.close(fig)
print("figures written")
