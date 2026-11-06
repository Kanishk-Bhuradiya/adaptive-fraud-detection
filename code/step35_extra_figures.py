# ======================= EXTRA FIGURES, SET 1: drawn from the exact result tables (no data needed, runs in ~30 s) =======================
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

OUT = "/kaggle/working/report/extra"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
def save(fig, name):
    fig.tight_layout(); fig.savefig(f"{OUT}/{name}.png", dpi=200, bbox_inches="tight"); plt.show()

# ------------------------------------------------------------------ A. the model ladder
# (name, PR-AUC all weeks, weeks 15-17, weeks 23-25, family)
M = [("Naive Bayes", .089, .106, .087, "weak single model"), ("Single tree, depth 6", .303, .364, .307, "weak single model"),
     ("WoE binning + L2", .365, .464, .336, "linear / additive"), ("Logistic regression (no flags)", .348, .482, .261, "linear / additive"),
     ("Linear probability (ridge)", .384, .478, .363, "linear / additive"), ("Boosting depth 1 (additive)", .389, .493, .352, "linear / additive"),
     ("Logistic reg. + flags, L1", .391, .479, .378, "linear / additive"), ("Logistic reg. + flags, L2", .394, .482, .375, "linear / additive"),
     ("Boosting depth 2", .471, .559, .454, "trees with interactions"), ("Random Forest", .487, .564, .483, "trees with interactions"),
     ("Hybrid (trees + logistic reg.)", .489, .571, .463, "hybrid"), ("Boosting depth 3", .498, .580, .489, "trees with interactions"),
     ("Same trees alone (300, depth 5)", .505, .596, .479, "trees with interactions"), ("Boosting depth 6", .508, .593, .475, "trees with interactions"),
     ("LightGBM", .523, .617, .493, "trees with interactions"), ("XGBoost", .524, .607, .499, "trees with interactions")]
col = {"weak single model": "#8d99ae", "linear / additive": "#3a86ff", "hybrid": "#8338ec", "trees with interactions": "#2a9d8f"}
M = sorted(M, key=lambda r: r[1])
fig, ax = plt.subplots(figsize=(9.5, 6.2))
for i, (n, a, e, l, f) in enumerate(M):
    ax.plot([l, e], [i, i], color=col[f], lw=3, alpha=.45)
    ax.scatter(e, i, s=55, facecolor="white", edgecolor=col[f], lw=2, zorder=3)
    ax.scatter(l, i, s=55, color=col[f], zorder=3)
    ax.scatter(a, i, marker="D", s=40, color="black", zorder=4)
    ax.text(max(e, a) + .012, i, f"{a:.3f}", va="center", fontsize=8)
ax.set_yticks(range(len(M))); ax.set_yticklabels([m[0] for m in M], fontsize=8.5)
ax.set_xlabel("PR-AUC on the future weeks"); ax.set_xlim(0, .68)
ax.scatter([], [], facecolor="white", edgecolor="gray", lw=2, label="weeks 15-17 (early)"); ax.scatter([], [], color="gray", label="weeks 23-25 (late)")
ax.scatter([], [], marker="D", color="black", label="all weeks")
for f, c in col.items(): ax.plot([], [], color=c, lw=6, label=f)
ax.axvline(.035, color="red", ls=":", lw=1); ax.text(.037, .2, "random\n(0.035)", color="red", fontsize=7)
ax.legend(fontsize=7.5, loc="lower right"); ax.set_title("Model ladder: level and decay (early -> late) of every model")
save(fig, "extraA_model_ladder")

# ------------------------------------------------------------------ B. why trees win: depth curve + hybrid vs its own trees
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
d, p = [1, 2, 3, 6], [.389, .471, .498, .508]
ax[0].plot(d, p, "o-", color="#2a9d8f", lw=2.5, ms=8, label="boosted trees of depth d")
ax[0].axhline(.394, color="#3a86ff", ls="--"); ax[0].text(3.6, .399, "logistic regression + missing flags (0.394)", color="#3a86ff", fontsize=8)
ax[0].axhline(.523, color="k", ls=":"); ax[0].text(3.6, .527, "LightGBM full (0.523)", fontsize=8)
for x, y_ in zip(d, p): ax[0].text(x, y_ - .022, f"{y_:.3f}", ha="center", fontsize=8)
ax[0].annotate("depth 1 = additive model\n= logistic regression", (1, .389), (1.3, .30), arrowprops=dict(arrowstyle="->"), fontsize=8)
ax[0].set_xticks(d); ax[0].set_xlabel("tree depth = maximum interaction order"); ax[0].set_ylabel("PR-AUC (future test weeks)"); ax[0].set_ylim(.28, .56)
ax[0].set_title("Interactions close the gap (depth test)")
lab = ["100 trees\ndepth 3", "300 trees\ndepth 3", "100 trees\ndepth 5", "300 trees\ndepth 5"]
tv, hv = np.array([.426, .471, .473, .518]), np.array([.464, .485, .484, .488]); x = np.arange(4)
ax[1].bar(x - .19, tv, .38, label="trees alone", color="#2a9d8f"); ax[1].bar(x + .19, hv, .38, label="hybrid: logistic reg. on leaves", color="#8338ec")
for i in x: ax[1].text(i, max(tv[i], hv[i]) + .008, f"{hv[i]-tv[i]:+.3f}", ha="center", fontsize=9, fontweight="bold", color="#8338ec" if hv[i] > tv[i] else "red")
ax[1].set_xticks(x); ax[1].set_xticklabels(lab, fontsize=8.5); ax[1].set_ylim(.35, .56); ax[1].set_ylabel("PR-AUC (validation weeks)")
ax[1].legend(fontsize=8, loc="upper left"); ax[1].set_title("Hybrid helps weak trees, hurts strong trees")
save(fig, "extraB_interactions_and_hybrid")

# ------------------------------------------------------------------ C. cost: savings heatmap and break-even
strat = ["every 4 weeks", "every 2 weeks", "rolling 8w, every 2", "trigger >10% drop", "trigger >15% drop"]
scen = ["cheap\nreview", "balanced", "friction-\nheavy", "fraud-\nheavy"]
S = np.array([[7.4, 12.2, 10.5, 8.9], [11.2, 17.3, 13.9, 12.6], [2.9, 13.6, 11.7, 4.9], [6.0, 9.8, 9.0, 6.9], [6.0, 10.7, 8.8, 7.1]])
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
im = ax[0].imshow(S, cmap="YlGn", vmin=0, vmax=20, aspect="auto"); ax[0].grid(False)
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(scen); ax[0].set_yticks(range(5)); ax[0].set_yticklabels(strat)
for i in range(5):
    for j in range(4): ax[0].text(j, i, f"{S[i, j]:.1f}%", ha="center", va="center", fontweight="bold")
plt.colorbar(im, ax=ax[0], label="% cost saved vs static"); ax[0].set_title("Estimated cost saving vs the static model")
be = {"every 4 weeks (first 2 extra retrains)": 41248, "trigger >15%": 36251, "trigger >10%": 33236, "every 2 weeks (5 extra)": 23391,
      "rolling 8w, every 2 (5 extra)": 18365, "moving from every 4 to every 2 weeks": 11486}
ks = list(be)[::-1]
ax[1].barh(ks, [be[k] for k in ks], color=["#e76f51" if "moving" in k else "#2a9d8f" for k in ks])
for i, k in enumerate(ks): ax[1].text(be[k] + 500, i, f"${be[k]:,}", va="center", fontsize=8)
ax[1].set_xlabel("break-even cost of ONE extra retrain ($)"); ax[1].set_xlim(0, 52000)
ax[1].set_title("Retrain only if one retrain costs less than this\n(balanced scenario)", fontsize=10)
save(fig, "extraC_cost_heatmap_breakeven")

# ------------------------------------------------------------------ D. efficiency frontier and label delay
names = ["static", "every 4 weeks", "every 2 weeks", "rolling 8w", "trigger >10%", "trigger >15%", "trigger >25%"]
n_tr, pr, rc = [1, 3, 6, 6, 3, 3, 2], [.505, .545, .556, .552, .547, .543, .535], [.519, .594, .605, .600, .594, .591, .564]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1.2, 1, 1]})
ax[0].scatter(n_tr, pr, s=[(r - .45) * 3500 for r in rc], color="#2a9d8f", alpha=.7, edgecolor="k")
off = {"static": (.12, .001), "every 4 weeks": (-1.55, -.0035), "every 2 weeks": (.12, .0012), "rolling 8w": (.12, -.003), "trigger >10%": (-1.35, .0022), "trigger >15%": (.15, -.0028), "trigger >25%": (.12, .001)}
for n, a, b in zip(names, n_tr, pr): ax[0].annotate(n, (a, b), (a + off[n][0], b + off[n][1]), fontsize=8)
ax[0].set_xlabel("number of trainings (initial + retrains)"); ax[0].set_ylabel("PR-AUC, all future weeks"); ax[0].set_xlim(.5, 7.5)
ax[0].set_title("Effort vs benefit (bubble = recall@5%, late weeks)", fontsize=10)
lab = ["static", "every 4 weeks", "every 2 weeks"]
for k, (t, a0, a1) in enumerate([("All future weeks", [.505, .545, .556], [.454, .482, .488]), ("Weeks 23-25", [.466, .550, .558], [.427, .501, .511])]):
    x = np.arange(3)
    ax[k + 1].bar(x - .19, a0, .38, label="labels instant", color="#3a86ff"); ax[k + 1].bar(x + .19, a1, .38, label="labels arrive 4 weeks late", color="#e76f51")
    for i in x: ax[k + 1].text(i + .19, a1[i] + .004, f"{a1[i]-a0[i]:+.3f}", ha="center", fontsize=8)
    ax[k + 1].set_xticks(x); ax[k + 1].set_xticklabels(lab, fontsize=8.5); ax[k + 1].set_ylim(.38, .6); ax[k + 1].set_title(f"Label delay: {t}\n(numbers = PR-AUC lost because of the delay)", fontsize=9.5)
ax[1].legend(fontsize=8, loc="upper left"); ax[1].set_ylabel("PR-AUC")
save(fig, "extraD_efficiency_and_label_delay")

# ------------------------------------------------------------------ E. SHAP stability
feats = ["TransactionAmt", "C13", "card1", "card2", "V90", "card6", "P_emaildomain", "M4", "ProductCD", "C1", "D2", "addr1"]
sh = np.array([[4.8, 4.7, 4.5], [4.7, 4.7, 4.9], [2.9, 3.0, 2.9], [2.9, 3.0, 2.9], [2.6, 2.9, 2.7], [2.5, 2.5, 2.4], [2.5, 2.4, 2.3],
               [2.4, 2.3, 2.3], [2.4, 1.6, 1.7], [2.3, 2.2, 2.0], [2.2, 2.3, 2.4], [2.0, 2.0, 1.9]])
fam = pd.DataFrame({"train": [16.6, 14.5, 6.5, 25.2, 11.2, 10.8, 15.3], "early test": [16.5, 15.7, 7.0, 25.7, 11.6, 9.3, 14.3],
                    "late test": [16.5, 15.8, 6.9, 26.5, 10.9, 9.6, 13.8]}, index=["C", "D", "M", "V", "card", "identity", "other"])
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw={"width_ratios": [1.6, .9, .9]})
x = np.arange(len(feats)); cols3 = ["#8d99ae", "#3a86ff", "#e76f51"]
for k, (lab_, c) in enumerate(zip(["train (wk 1-12)", "early test (wk 15-17)", "late test (wk 23-25)"], cols3)):
    ax[0].bar(x + (k - 1) * .27, sh[:, k], .27, label=lab_, color=c)
ax[0].set_xticks(x); ax[0].set_xticklabels(feats, rotation=45, ha="right", fontsize=8); ax[0].set_ylabel("share of total mean |SHAP| (%)")
ax[0].legend(fontsize=8); ax[0].set_title("Static model: importance of the top-12 features by period")
fam.T.plot.bar(stacked=True, ax=ax[1], colormap="tab20", width=.75, legend=False); ax[1].set_ylabel("share (%)"); ax[1].set_title("By feature family", fontsize=10)
ax[1].legend(fontsize=7, bbox_to_anchor=(1, 1)); ax[1].tick_params(axis="x", rotation=20)
cmpn = ["same period,\nnew sample\n(noise)", "train vs\nearly test", "train vs\nlate test", "static vs\nretrained", "static vs\nother seed"]
jac = [1.0, .905, .905, .905, .818]
ax[2].bar(range(5), jac, color=["#adb5bd", "#3a86ff", "#e76f51", "#2a9d8f", "#ffb703"]); ax[2].set_ylim(.7, 1.03)
for i, v in enumerate(jac): ax[2].text(i, v + .004, f"{v:.3f}", ha="center", fontsize=8)
ax[2].set_xticks(range(5)); ax[2].set_xticklabels(cmpn, fontsize=7); ax[2].set_ylabel("top-20 feature overlap (Jaccard)")
ax[2].set_title("Retraining changes the explanation\nno more than a new seed does", fontsize=10)
save(fig, "extraE_shap_stability")

# ------------------------------------------------------------------ F. coefficient drift: heatmap, similarity, share changed
rows = {"C7": [.12, 1.93, 2.29, 1.86, 1.78, .73], "V62 [missing]": [1.46, 1.63, 1.53, 1.12, .99, 1.17], "C1": [1.13, 1.12, 1.17, 1.01, .66, .97],
        "M4 [missing]": [-1.49, -.96, -.73, -.73, -1.07, -.79], "C14": [-.76, -.92, -.90, -.85, -.70, -.89],
        "card2 [missing]": [.38, .68, 1.46, .03, -.41, -.80], "card6 = code 1": [-.90, -.70, -.19, -.60, -.83, -.20],
        "D8": [-.50, -.48, -.44, -.44, -.71, -.69], "id_02 [missing]": [-.38, -.11, -1.16, -.76, -.28, -.38],
        "id_12 = code 1": [-.82, -.44, -.72, -.46, -.35, -.19], "id_12 = code 0": [-.62, -.15, -.56, -.52, -.37, -.09],
        "card5 [missing]": [.01, -.50, -.87, -.26, .12, .24]}
W = ["wk 1-4", "wk 5-8", "wk 9-12", "wk 13-16", "wk 17-20", "wk 21-24"]
C = pd.DataFrame(rows, index=W).T
sp = np.array([[1, .80, .75, .74, .67, .51], [.80, 1, .73, .68, .72, .54], [.75, .73, 1, .77, .54, .47],
               [.74, .68, .77, 1, .67, .56], [.67, .72, .54, .67, 1, .78], [.51, .54, .47, .56, .78, 1]])
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1.35, 1, .8]})
v = np.abs(C.values).max(); im = ax[0].imshow(C.values, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto"); ax[0].grid(False)
for i in range(C.shape[0]):
    for j in range(C.shape[1]): ax[0].text(j, i, f"{C.values[i, j]:.2f}", ha="center", va="center", fontsize=7.5)
ax[0].set_xticks(range(6)); ax[0].set_xticklabels(W, fontsize=8); ax[0].set_yticks(range(len(C))); ax[0].set_yticklabels(C.index, fontsize=8)
plt.colorbar(im, ax=ax[0], label="coefficient (log-odds)"); ax[0].set_title("Largest logistic-regression coefficients over time")
im2 = ax[1].imshow(sp, cmap="viridis", vmin=.4, vmax=1); ax[1].grid(False)
for i in range(6):
    for j in range(6): ax[1].text(j, i, f"{sp[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if sp[i, j] < .75 else "black")
ax[1].set_xticks(range(6)); ax[1].set_xticklabels(W, fontsize=7, rotation=30); ax[1].set_yticks(range(6)); ax[1].set_yticklabels(W, fontsize=7)
ax[1].set_title("Similarity of coefficient vectors\n(Spearman)", fontsize=10); plt.colorbar(im2, ax=ax[1], fraction=.046)
ch = [31, 39, 44, 39, 47]
ax[2].bar(W[1:], ch, color="#e76f51"); ax[2].set_ylabel("% of coefficients changed > 3x noise"); ax[2].tick_params(axis="x", rotation=40, labelsize=8)
for i, c_ in enumerate(ch): ax[2].text(i, c_ + .8, f"{c_}%", ha="center", fontsize=8)
ax[2].set_title("Change vs weeks 1-4", fontsize=10); ax[2].set_ylim(0, 55)
save(fig, "extraF_coefficient_drift_overview")

# ------------------------------------------------------------------ G. availability heatmap (share of missing values per week)
M1 = [61.2,67.5,73.7,74.8,56.0,54.3,59.7,63.7,58.8,52.6,53.2,49.6,45.7,29.9,27.9,29.5,33.3,25.1,26.5,29.6,25.3,26.2,26.3,33.1,29.8,25.3]
M7 = [79.7,83.5,86.7,87.6,77.5,76.8,74.1,72.0,67.0,63.8,63.5,60.7,56.0,39.2,43.5,38.6,42.1,35.3,36.9,46.5,42.5,36.9,36.9,43.2,41.1,34.9]
V4 = [70.9,67.4,73.6,74.8,56.0,54.3,59.7,63.7,58.9,52.6,54.5,49.6,45.7,30.0,29.8,29.5,33.3,25.1,43.2,35.2,27.6,26.2,27.1,33.1,29.8,25.3]
A1 = [6.3,5.0,13.3,14.2,13.9,11.3,9.3,10.8,11.6,11.8,12.5,12.7,11.9,9.1,11.1,13.5,18.1,10.5,10.4,11.0,8.8,8.7,10.5,12.5,10.4,9.7]
H = np.array([M7, M1, V4, A1]); lab = ["M7", "M1", "V4 (= V2-V10, D11)", "addr1 (control)"]
fig, ax = plt.subplots(figsize=(13, 2.9))
im = ax.imshow(H, cmap="YlOrRd", aspect="auto", vmin=0, vmax=90); ax.grid(False)
for i in range(4):
    for j in range(26): ax.text(j, i, f"{H[i, j]:.0f}", ha="center", va="center", fontsize=6.5)
ax.axvline(3.5, color="k", ls="--"); ax.axvline(12.5, color="k", ls="--"); ax.axvline(14.5, color="k", ls=":")
ax.set_xticks(range(26)); ax.set_xticklabels(range(26), fontsize=7); ax.set_yticks(range(4)); ax.set_yticklabels(lab, fontsize=8.5)
ax.set_xlabel("week   (dashed: week 4 and week 13 steps; dotted: start of future test)"); plt.colorbar(im, ax=ax, label="% missing", pad=.01)
ax.set_title("Feature availability by week: groups become far more complete, the control does not")
save(fig, "extraG_missing_heatmap")

# ------------------------------------------------------------------ H. regression view of the week-4 level shift (exact weekly counts)
tx = np.array([27596,28463,35701,34906,24539,20919,20564,19761,21432,22534,20953,20175,22391,27429,21269,20891,21078,23575,19258,18572,18549,21443,19811,21038,19941,17752])
fr = np.array([804,717,869,724,889,803,823,862,927,996,880,676,912,884,718,866,1069,833,727,611,639,660,565,729,733,747])
wk = pd.DataFrame({"week": np.arange(26), "n": tx, "frauds": fr})[1:].reset_index(drop=True)
wk["log_n"] = np.log(wk.n); wk["legit"] = wk.n - wk.frauds; wk["step4"] = (wk.week >= 4).astype(int); wk["step13"] = (wk.week >= 13).astype(int)
glm = lambda f, off=True: smf.glm(f, data=wk, family=sm.families.Poisson(), offset=wk.log_n if off else None).fit(scale="X2")
trend, steps = glm("frauds ~ week"), glm("frauds ~ step4 + step13 + week")
f_ct, l_ct = glm("frauds ~ step4 + week", False), glm("legit ~ step4 + week", False)
print("check against earlier output -> rate step4 %.3f (1.750), step13 %.3f (0.924), trend %.3f (0.997); frauds step4 %.3f (1.226), legit step4 %.3f (0.675)" % (
    np.exp(steps.params["step4"]), np.exp(steps.params["step13"]), np.exp(steps.params["week"]), np.exp(f_ct.params["step4"]), np.exp(l_ct.params["step4"])))
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
ax[0].plot(wk.week, wk.frauds / wk.n * 100, "o", color="k", label="observed weekly fraud rate")
ax[0].plot(wk.week, trend.fittedvalues / wk.n * 100, "--", color="#e76f51", lw=2, label="one straight trend (fits badly)")
ax[0].plot(wk.week, steps.fittedvalues / wk.n * 100, "-", color="#2a9d8f", lw=2.5, label="level shifts at weeks 4 and 13")
ax[0].axvline(3.5, color="gray", ls="--", lw=1); ax[0].axvline(12.5, color="gray", ls=":", lw=1)
ax[0].set_xlabel("week"); ax[0].set_ylabel("fraud rate (%)"); ax[0].legend(fontsize=8); ax[0].set_title("A level shift, not a trend")
m = [np.exp(f_ct.params["step4"]), np.exp(l_ct.params["step4"]), np.exp(steps.params["step4"])]
ax[1].bar(["frauds\n(count)", "legitimate\n(count)", "fraud rate"], m, color=["#e76f51", "#3a86ff", "#8338ec"]); ax[1].axhline(1, color="k", lw=1)
for i, v_ in enumerate(m): ax[1].text(i, v_ + .03, f"x{v_:.2f}", ha="center", fontweight="bold")
ax[1].set_ylabel("multiplier at week 4"); ax[1].set_title("The rate jump is mostly a falling denominator", fontsize=10); ax[1].set_ylim(0, 2.1)
save(fig, "extraH_week4_level_shift")
print("saved:", sorted(os.listdir(OUT)))