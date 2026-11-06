# ================= Coefficient drift: does a regression's view of "what predicts fraud" change over time? =================
# Logistic regression with a FIXED design (same scaling for all windows), refitted on consecutive 4-week windows.
# Sampling noise is measured by splitting each window into two random halves: a coefficient change only counts
# if it is clearly larger than that noise.
rng = np.random.RandomState(0)

# ---- 1. the 40 features the static model relies on most ----
if B and "model_cache" in B:
    gain = pd.Series(B["model_cache"][(15, None)].booster_.feature_importance(importance_type="gain"), index=X.columns)
else:                                                                     # session without the saved bundle: quick small model
    quick = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=64, colsample_bytree=0.5, subsample=0.8,
                               subsample_freq=1, scale_pos_weight=5, random_state=42, verbose=-1).fit(X[train_mask], y[train_mask])
    gain = pd.Series(quick.booster_.feature_importance(importance_type="gain"), index=X.columns)
top = list(gain.sort_values(ascending=False).index[:40])
num_top = [c for c in top if c not in text_cols]
cat_top = [c for c in top if c in text_cols and df[c].nunique() <= 30]       # small text columns become yes/no dummies
print(f"{len(num_top)} numeric + {len(cat_top)} categorical features")

# ---- 2. fixed design, fitted on the training weeks only ----
tr_rows = X[train_mask]
spec = {}
for c in num_top:
    col = tr_rows[c]
    q1, q3 = col.quantile([0.25, 0.75])
    scale = (q3 - q1) or col.std() or 1.0                                  # one unit = one inter-quartile range
    spec[c] = (col.median(), scale, col.isna().mean() > 0.005)
levels = {c: [k for k, s in tr_rows[c].value_counts(normalize=True).items() if s >= 0.02] for c in cat_top}
has_missing = {c: tr_rows[c].isna().mean() > 0.005 for c in cat_top}

def design(rows):
    parts = {}
    for c in num_top:
        med, scale, flag = spec[c]
        v = rows[c]
        parts[c] = ((v.fillna(med) - med) / scale).clip(-5, 5)
        if flag:
            parts[c + " [missing]"] = v.isna().astype(float)
    for c in cat_top:
        for code in levels[c]:
            parts[f"{c} = code {int(code)}"] = (rows[c] == code).astype(float)
        if has_missing[c]:
            parts[c + " [missing]"] = rows[c].isna().astype(float)
    return pd.DataFrame(parts).astype("float32")

D = design(X)
names = list(D.columns)
print("design matrix:", D.shape)

# ---- 3. one logistic regression per 4-week window (+ two random halves for the noise estimate) ----
windows = {"wk 1-4": (1, 4), "wk 5-8": (5, 8), "wk 9-12": (9, 12), "wk 13-16": (13, 16), "wk 17-20": (17, 20), "wk 21-24": (21, 24)}
def fit(idx):
    return LogisticRegression(C=0.1, max_iter=500).fit(D.iloc[idx], y[idx]).coef_[0]

beta, noise_sq = pd.DataFrame(index=names), np.zeros(len(names))
for w, (a, b) in windows.items():
    idx = np.where((week >= a) & (week <= b))[0]
    beta[w] = fit(idx)
    perm = rng.permutation(idx)
    half = len(perm) // 2
    noise_sq += (fit(perm[:half]) - fit(perm[half:])) ** 2
sigma = pd.Series(np.sqrt(noise_sq / len(windows)) / 2, index=names)       # sampling noise (std) of one full-window coefficient
se_diff = sigma * np.sqrt(2)                                              # noise of a difference between two windows
z_first = beta.sub(beta.iloc[:, 0], axis=0).div(se_diff, axis=0)           # change vs the first window
z_prev = beta.diff(axis=1).div(se_diff, axis=0)                           # change vs the previous window

# ---- 4. results ----
big = beta.abs().mean(axis=1).sort_values(ascending=False).index[:12]
tab = beta.loc[big].round(2); tab["noise (+/-)"] = sigma.loc[big].round(2)
print("\nCoefficients of the 12 largest effects (log-odds per IQR, or for a yes/no column):")
print(tab.to_string())

print("\nShare of coefficients that changed MORE than 3x the sampling noise:")
res = pd.DataFrame({"vs first window (wk 1-4)": (z_first.abs() > 3).mean(), "vs previous window": (z_prev.abs() > 3).mean()}).drop(index="wk 1-4", errors="ignore")
print((res * 100).round(0).astype(int).astype(str).add("%").to_string())

is_miss = pd.Series([n.endswith("[missing]") for n in names], index=names)
print("\nOf the coefficients that changed (vs first window, last window wk 21-24):")
print(f"   'missing' flags : {int((z_first['wk 21-24'].abs() > 3)[is_miss].sum())} of {int(is_miss.sum())}")
print(f"   value features  : {int((z_first['wk 21-24'].abs() > 3)[~is_miss].sum())} of {int((~is_miss).sum())}")

move = z_first["wk 21-24"].abs().sort_values(ascending=False).index[:10]
print("\nLargest changes, first vs last window:")
print(pd.DataFrame({"wk 1-4": beta.loc[move, "wk 1-4"], "wk 21-24": beta.loc[move, "wk 21-24"],
                    "z (change / noise)": z_first.loc[move, "wk 21-24"]}).round(2).to_string())

strong = [n for n in names if abs(beta.loc[n, "wk 1-4"]) > 2 * sigma[n] and abs(beta.loc[n, "wk 21-24"]) > 2 * sigma[n]]
flips = [n for n in strong if np.sign(beta.loc[n, "wk 1-4"]) != np.sign(beta.loc[n, "wk 21-24"])]
print("\nSign flips among clearly non-zero coefficients:", flips if flips else "none")
print("\nSpearman correlation between the coefficient vectors of different windows:")
print(beta.corr(method="spearman").round(2).to_string())

fig, ax = plt.subplots(figsize=(10, 4.5))
for n in big[:8]:
    ax.errorbar(range(len(windows)), beta.loc[n], yerr=1.96 * sigma[n], marker="o", capsize=3, label=n)
ax.set_xticks(range(len(windows))); ax.set_xticklabels(list(windows)); ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("coefficient (log-odds)"); ax.set_title("Coefficients over time (bars = 95% sampling noise)"); ax.legend(fontsize=7, ncol=2)
plt.tight_layout(); plt.show()
