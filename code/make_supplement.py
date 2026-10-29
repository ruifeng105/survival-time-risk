"""Generate the tables (paper/supp/*.tex) and figures (paper/figs/supp_*.pdf) of the
supplementary material from results/*.json, so that every number in the supplement is
traceable to a script. Run make_figs.py first: Fig. S2 is copied from figs/fig_phase.pdf."""
import json, os, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SUPP = os.path.join(HERE, "..", "paper", "supp"); FIGS = os.path.join(HERE, "..", "paper", "figs")
os.makedirs(SUPP, exist_ok=True); os.makedirs(FIGS, exist_ok=True)
J = lambda f: json.load(open(os.path.join(HERE, "results", f)))
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.3,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "pdf.fonttype": 42, "mathtext.fontset": "cm"})
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a5cc7"]; GRID = dict(color="#d9d8d4", linewidth=0.4)


def sci(x, d=2):
    if x is None: return "--"
    if x == 0: return "$0$"
    m, e = f"{x:.{d - 1}e}".split("e")
    e = int(e)
    return f"${m}$" if e == 0 else f"${m}\\!\\times\\!10^{{{e}}}$"


def f3(x, d): return sci(x, 2).strip("$") if x < 10 ** -d else f"{x:.4f}" if x > 1 - 10 ** -d else f"{x:.{d}f}"


def g2(x): return f"{x:.2g}" if x < 10 else f"{x:.0f}" if x < 1000 else f"{x / 1e3:.1f}k"


def table(name, colspec, header, rows, caption, label, size="small", sep=4):
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    tex = (f"\\begin{{table}}[!htbp]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\{size}\\setlength{{\\tabcolsep}}{{{sep}pt}}\n"
           f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")
    open(os.path.join(SUPP, name), "w", encoding="utf-8", newline="\n").write(tex)


# ---- S-Table: closed-loop budget sweep -----------------------------------------------------
bj = J("budget.json")
table("tab_budget.tex", "@{}rccccc@{}",
      r"$B$ & $R_{\mathrm{nom}}$ & $R^+$ & at $(\rho,b)$ & $R^-$ & $R^+/R_{\mathrm{nom}}$",
      [[str(r["B"]), sci(r["modelled"]), f"${r['Rmax']:.3f}$", f"$({r['rho_max']:.3f},\\,{r['b_max']:.2f})$",
        sci(r["Rmin"]), f"${r['Rmax'] / r['modelled']:.1f}$"] for r in bj["closed_loop_sweep"]],
      r"Closed-loop risk of $\pi_L$ over the compatible class",
      "tab:s-budget")

# ---- S-Table: certificates and baselines -----------------------------------------------------
bl = J("baselines.json")
table("tab_baselines.tex", "@{}lccccc@{}",
      r"Task & $R_{\mathrm{nom}}$ & $R^+$ (grid) & box $\Lambda_{\max}$ & interval relax. & time-inv.\ corner",
      [[("dense" if r["B"] == r["T"] else f"$B={r['B']}$") + f", $D={r['D']}$", sci(r["R_nom"]), f"${r['R_plus']:.3f}$",
        ("--" if r["box_certificate"] is None else f"${r['box_certificate']:.3f}$"), f"${r['interval_robust']:.3f}$",
        f"${r['worst_corner_stationary']:.3f}$"] for r in bl],
      r"Risk certificates for the running example",
      "tab:s-baselines")

# ---- S-Table: eta sensitivity ------------------------------------------------------------------
et = J("eta.json")
table("tab_eta.tex", "@{}cccccc@{}",
      r"$\eta$ & $[\rho_{\mathrm{lo}},\rho_{\mathrm{hi}}]$ & worst $\rho$ & $R^+$ & $R^-$ & $R^+/R_{\mathrm{nom}}$",
      [[f"${r['eta']:g}$", f"$[{r['rho_lo']:.3f},{r['rho_hi']:.3f}]$", f"${r['rho_worst']:.3f}$", f"${r['R_plus']:.3f}$",
        sci(r["R_minus"]), f"${r['ratio_nom']:.1f}$"] for r in et["rows"]],
      r"Sensitivity of the dense worst case to $\eta$",
      "tab:s-eta")

# ---- S-Table: dense cases (Theorem 2 vs exact) ------------------------------------------------
dm = J("dense_M2.json")
rows = [c for c in dm["cases"] if abs(c["b"] - 0.7) < 1e-9]
table("tab_dense.tex", "@{}rrccccccc@{}",
      r"$T$ & $D$ & $\rho_w$ (Thm.) & argmax $\Lambda$ & argmax $R$ & $R^+$ & $R^-$ & $R_{\mathrm{nom}}$ & $R^+/R_{\mathrm{nom}}$",
      [[str(c["T"]), str(c["D"]), f"${c['rho_pred']:.3f}$", f"${c['rho_argmax_EN']:.3f}$", f"${c['rho_argmax_R']:.3f}$",
        sci(c["Rmax"]), sci(c["Rmin"]), sci(c["R_nominal"]), f"${g2(c['nominal_ratio'])}$"] for c in rows],
      r"Dense attempts over $\rho$ at $b=0.7$ in the running example",
      "tab:s-dense")

# ---- S-Table: third-order class (L = 2) --------------------------------------------------------
d3 = J("dense_M3.json")
table("tab_m3.tex", "@{}rrcccccc@{}",
      r"$T$ & $D$ & $\rho_w$ (Thm.) & argmax $R$ & $R^+$ & $R^-$ & $R_{\mathrm{nom}}$ & spread",
      [[str(c["T"]), str(c["D"]), f"${c['rho_pred']:.3f}$", f"${c['rho_argmax']:.3f}$", sci(c["Rmax"]), sci(c["Rmin"]),
        sci(c["R_nom"]), f"${100 * c['max_rel_spread_other_coords']:.1f}\\%$"] for c in d3["cases"]],
      r"Dense attempts over the third-order class ($L=2$, $M=3$)",
      "tab:s-m3")

# ---- S-Table: Gilbert-Elliott channels, with redesign ------------------------------------------
ge = J("ge.json"); rd = J("redesign.json")
red = {(r["eG"], r["eB"]): r for r in rd["ge"]}
table("tab_ge.tex", "@{}cccccccccc@{}",
      r"$(e_G,e_B)$ & memory & $\rho$ & $b$ & $\rho_\infty$ & dense & proj. & $\pi_L$, $B{=}15$ & proj. & $\pi_{\mathrm{rec}}$ / genie",
      [[f"$({r['eG']:g},{r['eB']:g})$", f"${r['memory']:.2f}$", f"${r['rho2']:.3f}$", f"${r['b2']:.3f}$", f"${r['rho_deep']:.3f}$",
        sci(r["R_dense"], 3), sci(r["R_dense_proj2"], 3), sci(r["R_B15"], 3), sci(r["R_B15_proj2"], 3),
        f"{sci(red[(r['eG'], r['eB'])]['R_rec'])} / {sci(red[(r['eG'], r['eB'])]['R_genie'])}"]
       for r in sorted(ge, key=lambda r: (r["eG"], -r["eB"]))],
      r"Gilbert--Elliott channels sharing the Gilbert fit of the running example",
      "tab:s-ge", size="footnotesize", sep=2.5)

# ---- S-Table: redesign over the class ----------------------------------------------------------
grid = rd["grid"]
rat = np.array([g["R_piL"] / g["R_rec"] for g in grid]); best = np.array([g["R_piL"] / min(g["R_piL"], g["R_rec"]) for g in grid])
gap = np.array([min(g["R_piL"], g["R_rec"]) / g["R_genie"] for g in grid])
rows = [[n["name"] if n["name"] == "nominal" else f"${n['name'][0]}_{n['name'][1]}$", f"${n['rho']:.3f}$", f"${n['b']:.2f}$",
         sci(n["R_piL"], 3), sci(n["R_rec"], 3), sci(n["R_genie"], 3)] for n in rd["chains"]]
table("tab_redesign.tex", "@{}lccccc@{}",
      r"channel & $\rho$ & $b$ & $R(\pi_L)$ & $R(\pi_{\mathrm{rec}})$ & genie",
      rows,
      r"Designing with the true recovery curve",
      "tab:s-redesign")

# ---- S-Table: sweep (Fig. 4 of the main paper) -------------------------------------------------
sw = J("sweep.json")
table("tab_sweep.tex", "@{}rrrcccccc@{}",
      r"loss & $\ell$ & $D$ & $R_{\mathrm{nom}}$ & $R^+/R_{\mathrm{nom}}$ & worst $\rho$ & $R_{\mathrm{nom}}$, $B{=}15$ & $R^+/R_{\mathrm{nom}}$, $B{=}15$ & worst $(\rho,b)$, $B{=}15$",
      [[f"${100 * r['eps']:g}\\%$", f"${r['ell']:g}$", str(r["D"]), sci(r["dense"]["R_nom"]), f"${g2(r['dense']['R_plus'] / r['dense']['R_nom'])}$",
        f"${f3(r['dense']['rho_worst'], 3)}$", sci(r["B15"]["R_nom"]), f"${g2(r['B15']['R_plus'] / r['B15']['R_nom'])}$",
        f"$({f3(r['B15']['rho_worst'], 3)},\\,{f3(r['B15']['b_worst'], 2)})$"] for r in sw["rows"]],
      r"Ambiguity across Gilbert fits",
      "tab:s-sweep", size="footnotesize", sep=2.5)

# ---- S-Tables: fading channels, all configurations --------------------------------------------
fr = J("fading.json")["rows"]; fs = J("fading_shadow.json")["rows"]
def fade_rows(rows, label):
    out = []
    for r in rows:
        d, p, b = r["dense"], r["periodic2"], r["piL_B15"]
        out.append([f"{label}, ${100 * r['eps']:g}\\%$, $f_D={r['fd']:g}$", str(r["D"]), sci(d["truth"]),
                    *[f"${d[f'M{M}'] / d['truth']:.2g}$" for M in (1, 2, 3, 4, 6)],
                    f"[{sci(d['class_min'])}, {sci(d['class_max'])}]",
                    sci(p["truth"]), f"${p['M1'] / p['truth']:.2g}$", sci(b["truth"]),
                    f"${b['M1'] / b['truth']:.2g}\\,/\\,{b['M6'] / b['truth']:.2g}$"])
    return out
table("tab_fading.tex", "@{}lrcccccclcccc@{}",
      r"channel & $D$ & $R$ dense & $L{=}1$ & $2$ & $3$ & $4$ & $6$ & $M{=}2$ class & $R$, $g{=}2$ & $L{=}1$ & $R$, $\pi_L$ & $L{=}1$/$L{=}6$",
      fade_rows(fr, "Rayleigh") + fade_rows(fs, "shadowed"),
      r"All fading configurations",
      "tab:s-fading", size="footnotesize", sep=2.5)

# ---- S-Table: FlockLab representative links ----------------------------------------------------
fl = J("flocklab.json")
rep = [r for r in fl["rows"] if "profile" in r]
table("tab_flocklab.tex", "@{}lrccrcccccccc@{}",
      r"link & tests & loss & $\ell$ & $D$ & $R$ dense & $L{=}1$ & $2$ & $3$ & $4$ & $6$ & $R$, $\pi_L$ & $L{=}1$ / rec.",
      [[f"${r['link'][0]}\\to{r['link'][1]}$", str(r["tests"]), f"${100 * r['loss']:.1f}\\%$", f"${r['mean_burst']:.2f}$", str(r["D"]),
        f"{sci(r['dense']['truth'])}", *[f"${r['dense'][f'M{M}'] / r['dense']['truth']:.2g}$" for M in (1, 2, 3, 4, 6)],
        sci(r["B15"]["truth"]), f"${r['B15']['M1'] / r['B15']['truth']:.2g}$ / {sci(r['B15']['rec'])}"] for r in rep],
      r"Representative measured links",
      "tab:s-flocklab", size="footnotesize", sep=2.5)

# ---- S-Table: Monte Carlo cross-check ----------------------------------------------------------
mc = J("mc_check.json")
table("tab_mc.tex", "@{}lcccc@{}",
      r"channel, task & exact & Monte Carlo & std.\ error & deviation (std.\ errors)",
      [[f"${n[0]}_{n[1]}$, {t}", sci(v[f'{k}_exact'], 4), sci(v[f'{k}_mc'], 4), sci(v[f'{k}_se'], 2),
        f"${abs(v[f'{k}_mc'] - v[f'{k}_exact']) / v[f'{k}_se']:.1f}$"]
       for n, v in mc.items() for k, t in (("dense", "dense"), ("B15", r"$\pi_L$, $B{=}15$"))],
      r"Monte Carlo check of the exact evaluators",
      "tab:s-mc")

# ---- numbers quoted next to the tables in the supplement ---------------------------------------
print(f"tab:s-dense:    largest bound violation {dm['max_bound_violation']:.1e}")
print(f"tab:s-m3:       rho in [{d3['rho_lo']:.3f}, {d3['rho_hi']:.3f}], rho_nom = {d3['rho_nom']:.3f}")
print(f"tab:s-eta:      R_nom = {et['R_nom']:.3g}")
print(f"tab:s-redesign: R(pi_L)/R(pi_rec) in [{rat.min():.2f}, {rat.max():.2f}], best gain {best.max():.2f}x, "
      f"{gap.min():.1f}--{gap.max():.1f} times above the genie")

# ---- exact fractions of Proposition 2 (risk-order reversal) ------------------------------------
rv = J("reversal_exact.json")
lines = []
for key, title in (("dense", "dense attempts"), ("periodic_g2", "periodic attempts every two slots"), ("closed_loop_B15", r"$\pi_L$ with $B=15$")):
    for ch in ("C1", "C2"):
        num, den = rv[key][ch]["exact"].split("/")
        lines.append(f"\\item {title}, $R({ch[0]}_{ch[1]})\\approx$ {sci(rv[key][ch]['float'], 7)}:\\\\ "
                     f"\\texttt{{\\seqsplit{{{num}}}}}\\,/\\,\\texttt{{\\seqsplit{{{den}}}}}")
open(os.path.join(SUPP, "fractions.tex"), "w", encoding="utf-8", newline="\n").write(
    "\\begin{itemize}\\footnotesize\n" + "\n".join(lines) + "\n\\end{itemize}\n")

# ---- figures -----------------------------------------------------------------------------------
shutil.copy(os.path.join(HERE, "figs", "fig_phase.pdf"), os.path.join(FIGS, "supp_phase.pdf"))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.4))
series = [("Rayleigh, 5%, $f_D{=}0.01$", next(r for r in fr if r["fd"] == 0.01 and r["eps"] == 0.05)["profile"]),
          ("Rayleigh, 5%, $f_D{=}0.03$", next(r for r in fr if r["fd"] == 0.03)["profile"]),
          ("shadowed, 5%", next(r for r in fs if r["eps"] == 0.05)["profile"]),
          *[(f"FlockLab ${a}\\to{b}$ ({100 * r['loss']:.0f}%)", r["profile"])
            for a, b in (("15", "33"), ("3", "32")) for r in rep if r["link"] == [a, b]]]
for i, (lab, pr) in enumerate(series):
    dn = pr["dense"]; dc = pr["dec2"]
    a1.plot(range(1, len(dn) + 1), dn, color=C[i], marker="o", ms=2.5, label=lab)
    a1.hlines(dn[0], 1, len(dn), color=C[i], lw=0.7, ls=":")
    a2.plot(range(1, len(dc) + 1), dc, color=C[i], marker="o", ms=2.5)
a1.set_xlabel(r"burst depth $j$"); a1.set_ylabel(r"persistence $\rho_j=\Pr(0\mid 0^j)$")
a2.set_xlabel(r"depth $j$ (attempts every second slot)"); a2.set_ylabel(r"$\rho^{(2)}_j$")
from matplotlib.ticker import MaxNLocator
for a in (a1, a2): a.grid(True, **GRID); a.xaxis.set_major_locator(MaxNLocator(integer=True))
fig.legend(*a1.get_legend_handles_labels(), frameon=False, fontsize=6.5, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(pad=0.3, w_pad=1.0, rect=(0, 0.15, 1, 1)); fig.savefig(os.path.join(FIGS, "supp_profiles.pdf")); plt.close(fig)

rows = fl["rows"]
loss = np.array([r["loss"] for r in rows]); burst = np.array([r["mean_burst"] for r in rows])
rd_ = np.array([r["dense"]["truth"] / r["dense"]["M1"] for r in rows]); rb_ = np.array([r["B15"]["truth"] / r["B15"]["M1"] for r in rows])
rm2 = np.array([r["dense"]["truth"] / r["dense"]["M2"] for r in rows])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.4))
sc = a1.scatter(100 * loss, rd_, c=burst, cmap="viridis", s=9, linewidths=0)
a1.set_yscale("log"); a1.axhline(1, color="#52514e", lw=0.8, ls="--")
a1.set_xlabel("loss rate (%)"); a1.set_ylabel("measured / Gilbert risk (dense)")
cb = fig.colorbar(sc, ax=a1, pad=0.01); cb.set_label(r"mean burst length", fontsize=7); cb.ax.tick_params(labelsize=6)
for i, (v, lab) in enumerate(((rd_, "Gilbert, dense"), (rm2, "$L{=}2$ fit, dense"), (rb_, r"Gilbert, $\pi_L$, $B{=}15$"))):
    xs = np.sort(v); a2.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=C[i], label=lab)
a2.set_xscale("log"); a2.axvline(1, color="#52514e", lw=0.8, ls="--")
a2.set_xlabel("measured / predicted risk"); a2.set_ylabel("fraction of links (ECDF)")
for a in (a1, a2): a.grid(True, which="major", **GRID)
a2.legend(frameon=False, fontsize=6, loc="lower right")
fig.tight_layout(pad=0.3, w_pad=1.0); fig.savefig(os.path.join(FIGS, "supp_flocklab.pdf")); plt.close(fig)
print("supplement tables and figures written")
