"""Real measured traces: FlockLab link-quality data (Jacob et al., DATA'19, doi:10.5281/zenodo.3731498;
TelosB 802.15.4, channel 26, 0 dBm; each test sends 100 strobes per directed link; Jul-Sep 2019).
Download with get_flocklab.py. Every test stream is a short trace; T = 60 missions slide inside streams.

All links with loss rate in [1%, 25%] and >= 500 distinct tests: at the survival time where the Gilbert fit
certifies R_nom <= 1e-2 with dense attempts, compare the trace-driven risk of dense attempts, of
pi_L (B = 15, designed on the Gilbert fit) and of pi_rec (same DP with the empirical recovery curve).
Every link also gets the second-order (L = 2) dense prediction; representative links (one per loss band)
additionally get order-L fits for L = 2, 3, 4, 6 (keys M2-M6, also for pi_L) and the M = 2 class range."""
import csv, io, json, os, time, zipfile
import numpy as np
from core import words, chain_from_p, box_ranges, box_chain, risk_dense
from policy import design_pi_L, design_from_pred, evaluate_policy

T, B, ETA, TARGET, MIN_TESTS = 60, 15, 1e-4, 1e-2, 500
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "flocklab")
CACHE = os.path.join(DATA, "streams_sky_dedup.npz")


def load_streams():
    """One stream per (test, sender, receiver). The preprocessed CSVs repeat most such records
    several times (about 6.4 copies on average, always with identical content), so records are
    deduplicated on that key."""
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True); return z["streams"].item()
    streams, seen = {}, set()
    for mon in ("2019-07", "2019-08", "2019-09"):
        zf = zipfile.ZipFile(os.path.join(DATA, f"{mon}_preprocessed_all.zip"))
        with zf.open(f"{mon}/{mon}_FlockLab_sky.csv") as f:
            for row in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")):
                s = row.get("rcv_stream")
                if not s or len(s) != 100 or "1" not in s: continue
                key = (row["test_number"], row["snd_id"], row["rcv_id"])
                if key in seen: continue
                seen.add(key)
                streams.setdefault((row["snd_id"], row["rcv_id"]), []).append(np.frombuffer(s.encode(), np.uint8) - 48)
    streams = {k: np.array(v, dtype=np.int8) for k, v in streams.items()}
    np.savez_compressed(CACHE, streams=np.array(streams, dtype=object))
    return streams


def fit_order(X, M):
    """Markov fit of order M (the fitted order, L in the paper) from (M+1)-blocks inside streams (0.5 pseudo-count)."""
    idx = np.zeros((X.shape[0], X.shape[1] - M), dtype=np.int64)
    for i in range(M + 1):
        idx = 2 * idx + X[:, i:X.shape[1] - M + i]
    cnt = np.bincount(idx.ravel(), minlength=2 ** (M + 1)) + 0.5
    p = {}
    for w in words(M):
        base = 2 * int("".join(map(str, w)), 2)
        p[w] = cnt[base + 1] / (cnt[base] + cnt[base + 1])
    return chain_from_p(M, p)


def recovery_curve(X):
    """Empirical P(gamma_{t+k} = 1 | gamma_t = o) inside streams, k = 1..T."""
    succ = X.mean()
    cur = {o: [None] + [X[:, k:][X[:, :-k] == o].mean() for k in range(1, T + 1)] for o in (0, 1)}
    return lambda o, k: succ if o is None else float(cur[o][min(k, T)])


def windows(X):
    """All mission windows (stream, start) of length T inside the streams."""
    starts = np.arange(X.shape[1] - T + 1)
    return X[:, starts[:, None] + np.arange(T)].reshape(-1, T)        # (n_streams * n_starts, T)


def risk_dense_trace(Wn, D):
    loss = (Wn == 0).astype(np.int32)
    cs = np.concatenate([np.zeros((len(Wn), 1), np.int32), np.cumsum(loss, axis=1)], axis=1)
    run = (cs[:, D:] - cs[:, :-D]) == D                               # a D-run of losses ends in the window
    return run.any(axis=1)


def policy_array(table, D):
    A = np.zeros((T + 2, D + 1, B + 1, 3, T + 2), dtype=np.int8)
    for (t, d, r, o, k), u in table.items():
        if u: A[t, d, r, 2 if o is None else o, k] = 1
    return A


def risk_policy_trace(Wn, pol, D):
    N = len(Wn)
    d = np.zeros(N, np.int64); r = np.full(N, B); o = np.full(N, 2); k = np.zeros(N, np.int64); failed = np.zeros(N, bool)
    for t in range(1, T + 1):
        x = Wn[:, t - 1]
        u = (r > 0) & (pol[t, np.minimum(d, D), r, o, k] == 1)
        d = np.where(u & (x == 1), 0, d + 1); failed |= d >= D
        k = np.where(u, 1, np.where(o == 2, 0, k + 1)); o = np.where(u, x, o); r = np.where(u, r - 1, r)
    return failed


def summary(fails, n_streams):
    """Risk and its standard error, treating streams as independent units."""
    per = fails.reshape(n_streams, -1).mean(axis=1)
    return float(per.mean()), float(per.std(ddof=1) / np.sqrt(n_streams))


streams = load_streams()
links = {k: X for k, X in streams.items() if len(X) >= MIN_TESTS and 0.01 <= 1 - X.mean() <= 0.25}
print(f"{len(links)} links with loss in [1%, 25%] and >= {MIN_TESTS} tests", flush=True)
rows, t0 = [], time.time()
for i, (key, X) in enumerate(sorted(links.items())):
    g = fit_order(X, 1); q0, q1 = g.p[(0,)], g.p[(1,)]
    D = next(D for D in range(2, T + 1) if risk_dense(g, T, D) <= TARGET)
    Wn = windows(X)
    tab_L, mod_L = design_pi_L(q0, q1, T, D, B)
    tab_r, _ = design_from_pred(recovery_curve(X), T, D, B)
    dense, se_d = summary(risk_dense_trace(Wn, D), len(X))
    piL, se_L = summary(risk_policy_trace(Wn, policy_array(tab_L, D), D), len(X))
    rec, se_r = summary(risk_policy_trace(Wn, policy_array(tab_r, D), D), len(X))
    m2 = fit_order(X, 2)
    rows.append(dict(link=list(key), tests=len(X), loss=float(1 - X.mean()), mean_burst=float(1 / q0), D=D,
                     dense=dict(truth=dense, se=se_d, M1=float(risk_dense(g, T, D)), M2=float(risk_dense(m2, T, D))),
                     B15=dict(truth=piL, se=se_L, M1=float(mod_L), rec=rec, se_rec=se_r)))
    if i % 25 == 0:
        print(f"[{i}/{len(links)} {time.time() - t0:.0f}s] {key} loss={1 - X.mean():.3f} D={D} dense {dense:.2e} vs Gilbert "
              f"{risk_dense(g, T, D):.2e} | B15 {piL:.2e} vs {mod_L:.2e}, rec {rec:.2e}", flush=True)

ratio_d = np.array([r["dense"]["truth"] / r["dense"]["M1"] for r in rows])
ratio_b = np.array([r["B15"]["truth"] / r["B15"]["M1"] for r in rows])
gain = np.array([r["B15"]["truth"] / max(r["B15"]["rec"], 1e-12) for r in rows])
print(f"dense truth/Gilbert: median {np.median(ratio_d):.2f}, >1 on {np.mean(ratio_d > 1):.0%}, >2 on {np.mean(ratio_d > 2):.0%}, max {ratio_d.max():.1f}")
print(f"B=15 truth/Gilbert: median {np.median(ratio_b):.2f}, >1 on {np.mean(ratio_b > 1):.0%}, >2 on {np.mean(ratio_b > 2):.0%}, max {ratio_b.max():.1f}")
print(f"pi_L/pi_rec on trace: median {np.median(gain):.2f}, >1 on {np.mean(gain > 1):.0%}, max {gain.max():.1f}")

# representative links: one per loss band, the burstiest
rep = []
for lo, hi in [(0.01, 0.04), (0.04, 0.08), (0.08, 0.15), (0.15, 0.25)]:
    cand = [r for r in rows if lo <= r["loss"] < hi]
    if cand: rep.append(max(cand, key=lambda r: r["mean_burst"]))
for r in rep:
    X = links[tuple(r["link"])]; D = r["D"]
    fits = {M: fit_order(X, M) for M in (2, 3, 4, 6)}
    c = fit_order(X, 1).marginal(2)
    Rg = box_ranges(c, 1, ETA)
    cls = [box_chain(c, 1, {(0,): a, (1,): b}) for a in np.linspace(*Rg[(0,)], 41) for b in np.linspace(*Rg[(1,)], 5)]
    vals = [float(risk_dense(ch, T, D)) for ch in cls]
    r["dense"].update({f"M{M}": float(risk_dense(ch, T, D)) for M, ch in fits.items()}, class_min=min(vals), class_max=max(vals))
    tab_L, _ = design_pi_L(fit_order(X, 1).p[(0,)], fit_order(X, 1).p[(1,)], T, D, B)
    r["B15"].update({f"M{M}": float(evaluate_policy(ch, tab_L, T, D, B)[0]) for M, ch in fits.items()})
    d = r["dense"]; b = r["B15"]
    print(f"{r['link']} loss={r['loss']:.3f} burst={r['mean_burst']:.2f} D={D}: dense {d['truth']:.2e}±{d['se']:.0e} "
          f"M1 {d['M1'] / d['truth']:.2f} M2 {d['M2'] / d['truth']:.2f} M6 {d['M6'] / d['truth']:.2f} class [{d['class_min']:.1e},{d['class_max']:.1e}] | "
          f"B15 {b['truth']:.2e} M1 {b['M1'] / b['truth']:.2f} M6 {b['M6'] / b['truth']:.2f} rec {b['rec']:.2e}")
def run_profile(X, g, jmax):
    """Within-stream run persistences rho^{(g)}_j of the g-decimated streams, j = 1..jmax."""
    P = []
    for j in range(1, jmax + 2):
        num = den = 0
        for ph in range(g):
            Z = (X[:, ph::g] == 0).astype(np.int32)
            cs = np.concatenate([np.zeros((len(Z), 1), np.int32), np.cumsum(Z, axis=1)], axis=1)
            if Z.shape[1] < j: continue
            num += int(np.sum(cs[:, j:] - cs[:, :-j] == j)); den += Z.shape[0] * (Z.shape[1] - j + 1)
        P.append(num / den)
    return [P[j] / P[j - 1] if P[j - 1] > 0 else None for j in range(1, jmax + 1)]


for r in rep:
    X = links[tuple(r["link"])]
    r["profile"] = {"dense": run_profile(X, 1, 20), "dec2": run_profile(X, 2, 10)}
json.dump(dict(T=T, B=B, target=TARGET, min_tests=MIN_TESTS, rows=rows, representative=[r["link"] for r in rep]),
          open("results/flocklab.json", "w"), indent=1)
