# Survival-Time Risk Beyond Low-Order Channel Models

This repository contains the code, the stored numerical results and the supplementary material of the paper *Survival-Time Risk Beyond Low-Order Channel Models: Identifiability and Scheduling*.

Low-order packet-loss models, such as the Gilbert model, are fitted to the loss rate and the mean burst length. Channels that share these statistics can nevertheless differ widely in survival-time risk, the probability that an application goes $D$ consecutive slots without a successful update during a mission of $T$ slots. The paper characterizes the Markov channels one order above a low-order fit, bounds their risk under acknowledgment-based scheduling with a transmission budget, and evaluates the gap between fitted and true risk on fading channels and measured IEEE 802.15.4 links.

`paper/supplement.pdf` contains the full proofs, the exact fractions of Proposition 2 and additional numerical results. A link to the paper will be added once it is published.

## Contents

| Path | Contents |
| --- | --- |
| `paper/supplement.pdf` | Supplementary material: proofs, additional results and a map from claims to scripts |
| `code/` | Exact risk evaluation, experiments, figure generation and claim verification |
| `code/results/` | Stored results (JSON) behind the numbers in the paper and the supplement |

In the table below, Sec., Eq., Fig. and Table refer to the paper; numbers with the prefix S refer to the supplement.

| Script | Purpose |
| --- | --- |
| `core.py` | Channels as stationary (M+1)-block laws, the box parameterization of the compatible class for M = L+1 (Lemma 1), exact risk recursions for dense attempts and open-loop schedules, and the bounds of Theorem 1 |
| `policy.py` | Design of the fitted-model policy π_L by dynamic programming under a budget and ACK/NACK feedback, and its exact evaluation on a given channel; the same program with a given recovery curve gives the redesign π_rec |
| `ge.py` | Hidden-state Gilbert–Elliott channels and their exact risks |
| `exp_dense.py` | Dense attempts: Theorem 1 and the worst-case persistence of Theorem 2 against exact maxima (Sec. III, Table S2) |
| `exp_dense_M3.py` | The same for the third-order class, M = 3 and L = 2 (Table S3) |
| `exp_phase.py` | Phase diagram of the worst-case persistence and growth of the risk span with D (Fig. S2) |
| `exp_budget.py` | Theorem 1 for open-loop schedules, sensitivity to the relapse coordinate b, and the sweep of π_L over the class (Secs. III and IV, Fig. 3, Table S4) |
| `exp_reversal.py` | Exact rational certificate of the risk-order reversal (Proposition 2, Section S4.3) and the retry spacing of π_L |
| `exp_analytic.py` | Decay exponents σ_g of spaced attempts, σ_1 = ρ and the reversal criterion (Eq. (14), Proposition S2) |
| `exp_decimation.py` | Periodic attempts see the decimated channel (Proposition 1), also for Gilbert–Elliott channels; run-persistence profiles of C1 and C2 |
| `exp_baselines.py` | Nominal risk, grid worst case, closed-form certificate and rectangular robust MDP (Sec. V, Table S5) |
| `exp_ge.py` | Gilbert–Elliott channels that share the Gilbert fit of the running example (Sec. V, Table S8) |
| `exp_eta.py` | Sensitivity of the worst case to the positivity bound η (Table S6) |
| `exp_M3L1.py` | Dense worst case for M = L+2, where the class is a polytope (Remark S1) |
| `exp_sweep.py` | Worst-to-nominal risk ratio over 36 Gilbert fits (Fig. 4, Table S10) |
| `exp_fading.py` | Rayleigh fading with Jakes spectrum and, with the argument `shadow`, log-normal shadowing: trace-driven risk against order-L Markov fits (Table I, Table S11, Fig. S3) |
| `exp_closedloop.py` | Closed-loop bound and its tightness for π_L (Eq. (15), Proposition S3, Table S1) |
| `exp_redesign.py` | Redesign of π_L with the true recovery curve, and a genie lower bound (Sec. V, Tables S8 and S9) |
| `get_flocklab.py`, `exp_flocklab.py` | Measured FlockLab links: download of the traces and evaluation on 171 TelosB links (Table I, Sec. V, Table S12, Figs. S3 and S4) |
| `exp_mc_check.py` | Independent Monte Carlo cross-check of the exact evaluators (Table S7) |
| `verify_paper_claims.py` | Rechecks the key numerical claims of the paper from the stored results |
| `make_figs.py` | Figs. 2–4 of the paper and Fig. S2, written to `code/figs/` |
| `make_supplement.py` | Tables and data figures of the supplement, written to `paper/supp/` and `paper/figs/` |
| `test_box.py`, `test_core.py` | Consistency checks of the box parameterization and of π_L |

## Requirements

Python 3 with NumPy, SciPy and Matplotlib:

```bash
pip install -r requirements.txt
```

The results were produced with Python 3.13, NumPy 2.3, SciPy 1.17 and Matplotlib 3.10.

## Reproducing the results

Recheck the paper's numbers from the stored results and regenerate the figures and the supplement tables:

```bash
cd code
python verify_paper_claims.py   # prints ALL PASS
python make_figs.py             # writes code/figs/
python make_supplement.py       # run after make_figs.py; writes paper/supp/ and paper/figs/
```

Rerun the experiments; each one overwrites its file in `code/results/`:

```bash
cd code
python exp_dense.py         # about 40 s
python exp_dense_M3.py
python exp_phase.py         # about 20 s
python exp_budget.py        # about 20 s
python exp_reversal.py      # exact rational arithmetic, about 1 s
python exp_analytic.py
python exp_baselines.py     # about 10 s
python exp_ge.py
python exp_eta.py
python exp_M3L1.py
python exp_decimation.py
python exp_sweep.py
python exp_fading.py        # about 7 min
python exp_fading.py shadow # about 5 min
python exp_closedloop.py
python exp_redesign.py
python get_flocklab.py      # downloads about 50 MB into data/flocklab/
python exp_flocklab.py      # about 30 s
python exp_mc_check.py      # about 30 s
```

All risks are computed by exact forward recursions, except the trace-driven risks of the fading channels and the measured links. `exp_mc_check.py` cross-checks the exact evaluators by Monte Carlo simulation, and `exp_reversal.py` uses exact rational arithmetic.

## Data

The measured traces come from the FlockLab link-quality dataset: R. Jacob, R. Da Forno, R. Trüb, A. Biri and L. Thiele, "Dataset: Wireless link quality estimation on FlockLab – and beyond," in *Proc. 2nd Workshop Data Acquisition To Analysis (DATA)*, 2019, pp. 57–60; data at doi:[10.5281/zenodo.3731498](https://doi.org/10.5281/zenodo.3731498), licensed under CC BY 4.0. `get_flocklab.py` downloads them; they are not part of this repository. `code/results/flocklab.json` contains statistics derived from them.

## License

The code in `code/`, including the stored results in `code/results/`, is released under the MIT License; see `LICENSE`. The MIT License does not cover `paper/supplement.pdf`, whose copyright remains with the authors.
