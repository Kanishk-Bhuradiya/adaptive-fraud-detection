# ======================= EXTRA FIGURES, SET 2: drawn from your live notebook state =======================
# Run AFTER the clean notebook through Part 10 (needs the variables checked below). Takes about 1-3 minutes.
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.metrics import precision_recall_curve, average_precision_score

need = ["y_test", "wk_t", "scores", "strategy_scores", "res", "psi", "model_cache", "cost_threshold", "model_key",
        "cost_vectors", "scenarios", "amt", "week", "y", "X", "train_mask", "test_weeks", "avg"]
missing = [n for n in need if n not in globals()]
assert not missing, f"Run the clean notebook through Part 10 first. Missing: {missing}"

OUT = "/kaggle/working/report/extra"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
def save(fig, name):
    fig.tight_layout(); fig.savefig(f"{OUT}/{name}.png", dpi=200, bbox_inches="tight"); plt.show()

early_t, late_t = np.isin(wk_t, [15, 16, 17]), np.isin(wk_t, [23, 24, 25])        # masks over the future-test rows
early_r, late_r = np.isin(week, [15, 16, 17]), np.isin(week, [23, 24, 25])        # masks over all rows
s_lgb = scores["lightgbm"]

# ------------------------------------------------------------------ L1. precision-recall curves
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
for m, lab, c in [(early_t, "weeks 15-17 (early)", "#2a9d8f"), (late_t, "weeks 23-25 (late)", "#e76f51")]:
    p, r, _ = precision_recall_curve(y_test[m], s_lgb[m])
    ax[0].plot(r, p, color=c, lw=2, label=f"{lab}: AP {average_precision_score(y_test[m], s_lgb[m]):.3f}")
    ax[0].axhline(y_test[m].mean(), color=c, ls=":", lw=1)
ax[0].set_xlabel("recall (share of frauds caught)"); ax[0].set_ylabel("precision (share of alerts that are real)")
ax[0].set_title("First LightGBM: the same model gets worse with distance"); ax[0].legend(fontsize=8); ax[0].set_ylim(0, 1.02)
for name, c in [("static", "#8d99ae"), ("every 4 weeks", "#3a86ff"), ("every 2 weeks", "#2a9d8f"), ("rolling 8w, every 2", "#8338ec")]:
    s = strategy_scores[name][late_r]; p, r, _ = precision_recall_curve(y[late_r], s)
    ax[1].plot(r, p, lw=2.2 if name == "static" else 1.6, color=c, label=f"{name}: AP {average_precision_score(y[late_r], s):.3f}")
ax[1].axhline(y[late_r].mean(), color="k", ls=":", lw=1, label="random (fraud rate)")
ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision"); ax[1].set_title("Weeks 23-25: retraining lifts the whole curve"); ax[1].legend(fontsize=8); ax[1].set_ylim(0, 1.02)
save(fig, "liveL1_precision_recall_curves")

# ------------------------------------------------------------------ L2. score distributions: why score monitoring is blind (Proposition 1)
def ecdf(a):
    a = np.sort(a); return a, np.arange(1, len(a) + 1) / len(a)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for k, (title, base) in enumerate([("ALL transactions", np.ones(len(y_test), bool)), ("FRAUDS only", y_test == 1), ("LEGITIMATE only", y_test == 0)]):
    a, b = s_lgb[early_t & base], s_lgb[late_t & base]
    for v, lab, c in [(a, "weeks 15-17", "#2a9d8f"), (b, "weeks 23-25", "#e76f51")]:
        xs, ys = ecdf(v); ax[k].plot(xs, ys, color=c, lw=2, label=lab)
    ax[k].set_xscale("log"); ax[k].set_xlabel("fraud score (log scale)"); ax[k].set_ylabel("cumulative share")
    ax[k].set_title(f"{title}   (KS = {ks_2samp(a, b).statistic:.3f})", fontsize=10); ax[k].legend(fontsize=8)
fig.suptitle("Scores of all transactions barely move, scores of frauds do: score monitoring cannot see concept drift", fontsize=11, y=1.03)
save(fig, "liveL2_score_distributions")

# ------------------------------------------------------------------ L3. cumulative gain (what a review budget buys)
fig, ax = plt.subplots(figsize=(7.4, 5))
grid = np.linspace(0, .20, 201)
def gain(yy, s):
    o = np.argsort(-s); cum = np.cumsum(yy[o]) / yy.sum(); idx = (grid * len(yy)).astype(int).clip(1, len(yy)) - 1
    return cum[idx]
for name, c in [("static", "#8d99ae"), ("every 4 weeks", "#3a86ff"), ("every 2 weeks", "#2a9d8f"), ("rolling 8w, every 2", "#8338ec")]:
    ax.plot(grid * 100, gain(y[late_r], strategy_scores[name][late_r]) * 100, lw=2.4 if name == "every 2 weeks" else 1.6, color=c, label=name)
fr_late = y[late_r].mean()
ax.plot(grid * 100, np.minimum(grid / fr_late, 1) * 100, "k--", lw=1, label="perfect model")
ax.plot(grid * 100, grid * 100, color="gray", ls=":", label="random")
ax.axvline(5, color="gray", lw=.8); ax.set_xlabel("% of transactions reviewed (highest scores first)"); ax.set_ylabel("% of all frauds caught")
ax.set_title("Weeks 23-25: share of fraud caught for a given review budget"); ax.legend(fontsize=8)
save(fig, "liveL3_cumulative_gain")

# ------------------------------------------------------------------ L4. weekly drift heatmap of the top drifting features
top = res.head(15)
Xtr = X[train_mask]
wk_masks = {w: week == w for w in test_weeks}
H = np.array([[max(psi(Xtr[f].values, X.loc[wk_masks[w], f].values) - nf, 0.0) for w in test_weeks] for f, nf in zip(top.feature, top.noise_floor)])
fig, ax = plt.subplots(figsize=(10, 5.2))
im = ax.imshow(np.clip(H, 0, 1), cmap="magma_r", aspect="auto"); ax.grid(False)
ax.set_yticks(range(len(top))); ax.set_yticklabels(top.feature); ax.set_xticks(range(len(test_weeks))); ax.set_xticklabels(test_weeks)
ax.set_xlabel("future week"); plt.colorbar(im, label="PSI above the feature's own noise floor (clipped at 1)")
ax.set_title("Top-15 drifting features, week by week (drift is persistent, not a one-off)"); save(fig, "liveL4_weekly_drift_heatmap")

# ------------------------------------------------------------------ L5. cumulative cost race
sc_name, sc = "balanced", scenarios["balanced"]
def weekly_cost(name):
    out = []
    for w in test_weeks:
        idx = np.where(week == w)[0]
        flag = strategy_scores[name][idx] >= cost_threshold(model_key(name, w), sc_name)
        fp, fn = cost_vectors(y[idx], amt[idx], sc)
        out.append(fp[flag].sum() + fn[~flag].sum())
    return np.array(out)
names5 = ["static", "every 4 weeks", "every 2 weeks", "rolling 8w, every 2"]
cum = {n: np.cumsum(weekly_cost(n)) for n in names5}
nomodel = np.cumsum([cost_vectors(y[week == w], amt[week == w], sc)[1].sum() for w in test_weeks])
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
ax[0].plot(test_weeks, nomodel / 1e3, "k:", label="no model")
for n, c in zip(names5, ["#8d99ae", "#3a86ff", "#2a9d8f", "#8338ec"]): ax[0].plot(test_weeks, cum[n] / 1e3, lw=2.2, color=c, label=n)
ax[0].set_xlabel("future week"); ax[0].set_ylabel("cumulative cost ($ thousands)"); ax[0].set_title(f"Cost race ('{sc_name}' scenario)"); ax[0].legend(fontsize=8)
for n, c in zip(names5[1:], ["#3a86ff", "#2a9d8f", "#8338ec"]): ax[1].plot(test_weeks, (cum["static"] - cum[n]) / 1e3, lw=2.2, color=c, label=n)
ax[1].axhline(0, color="k", lw=.8); ax[1].set_xlabel("future week"); ax[1].set_ylabel("money saved vs static ($ thousands)")
ax[1].set_title("Savings build up as the static model goes stale"); ax[1].legend(fontsize=8)
save(fig, "liveL5_cumulative_cost")

# ------------------------------------------------------------------ L6. drift vs importance
gain_imp = pd.Series(model_cache[(15, None)].booster_.feature_importance(importance_type="gain"), index=X.columns)
D = res.set_index("feature")[["excess", "family"]].join((gain_imp / gain_imp.sum() * 100).rename("importance"))
fig, ax = plt.subplots(figsize=(8.8, 5.6))
for f_, g in D.groupby("family"): ax.scatter(g.excess, g.importance, s=28, label=f_, alpha=.8)
lab = set(D.nlargest(6, "excess").index) | set(D.nlargest(6, "importance").index)
for f_ in lab: ax.annotate(f_, (D.loc[f_, "excess"], D.loc[f_, "importance"]), fontsize=7.5, xytext=(3, 3), textcoords="offset points")
ax.set_xscale("symlog", linthresh=.01); ax.set_xlabel("drift: PSI above noise floor (symlog)"); ax.set_ylabel("importance in the static model (% of gain)")
ax.set_title("The features that drift most are not the ones the model relies on\n(why a feature-drift alarm stayed silent)"); ax.legend(fontsize=7, ncol=2)
save(fig, "liveL6_drift_vs_importance")

# ------------------------------------------------------------------ L7. scoreboard: weekly PR-AUC of every model and strategy
rows = {}
for n, s in scores.items(): rows[f"{n} (frozen, wk 1-12)"] = [average_precision_score(y_test[wk_t == w], s[wk_t == w]) for w in test_weeks]
for n, s in strategy_scores.items(): rows[n] = [avg(s, [w]) for w in test_weeks]
Sb = pd.DataFrame(rows, index=test_weeks).T
Sb = Sb.loc[Sb.mean(axis=1).sort_values().index]
fig, ax = plt.subplots(figsize=(11, 5.4))
im = ax.imshow(Sb.values, cmap="RdYlGn", aspect="auto", vmin=np.nanmin(Sb.values), vmax=np.nanmax(Sb.values)); ax.grid(False)
for i in range(Sb.shape[0]):
    for j in range(Sb.shape[1]): ax.text(j, i, f"{Sb.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
ax.set_yticks(range(Sb.shape[0])); ax.set_yticklabels(Sb.index, fontsize=8); ax.set_xticks(range(len(test_weeks))); ax.set_xticklabels(test_weeks)
ax.set_xlabel("future week"); plt.colorbar(im, label="weekly PR-AUC"); ax.set_title("Scoreboard: weekly PR-AUC (rows sorted by average)")
save(fig, "liveL7_scoreboard")
print("saved:", sorted(f for f in os.listdir(OUT) if f.startswith("live")))