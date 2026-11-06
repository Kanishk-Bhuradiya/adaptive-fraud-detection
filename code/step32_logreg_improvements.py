import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score

# ================= 0. labels and period masks for this step =================
if "scores" not in globals() and B:                                     # reference scores of the earlier models, from the saved bundle
    scores = B["scores"]
ytr, yva, yte = y[train_mask], y[val_mask], y[test_mask]
wk_te = week[test_mask]
early_m, late_m = np.isin(wk_te, [15, 16, 17]), np.isin(wk_te, [23, 24, 25])

# ================= 1. scaled matrix + "was missing" flags (fitted on TRAINING rows only) =================
numeric  = [c for c in X.columns if c not in text_cols]
low_card = [c for c in text_cols if df[c].nunique() <= 30]
prep = ColumnTransformer([
    ("num", Pipeline([("fill", SimpleImputer(strategy="median", add_indicator=True)),
                      ("scale", QuantileTransformer(n_quantiles=200, subsample=100_000,
                                                    output_distribution="normal", random_state=42))]), numeric),
    ("cat", Pipeline([("fill", SimpleImputer(strategy="constant", fill_value=-1)),
                      ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20))]), low_card),
], sparse_threshold=0)
t0 = time.time()
Xtr_t = prep.fit_transform(X[train_mask]).astype("float32")
Xva_t = prep.transform(X[val_mask]).astype("float32")
Xte_t = prep.transform(X[test_mask]).astype("float32")
print("scaled matrix:", Xtr_t.shape, "|", round(time.time() - t0), "s")

# ================= 2. helper: fit one logistic regression and remember how it did on VALIDATION =================
rows, fits = [], {}

def run(name, family, Xa_tr, Xa_va, Xa_te, **kw):
    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = LogisticRegression(class_weight="balanced", **kw).fit(Xa_tr, ytr)
    converged = not any(issubclass(c.category, ConvergenceWarning) for c in caught)
    val_pr = average_precision_score(yva, m.decision_function(Xa_va))
    rows.append({"model": name, "family": family, "val_PR_AUC": val_pr, "converged": converged,
                 "nonzero_coefs": int((m.coef_ != 0).sum()), "seconds": round(time.time() - t0)})
    fits[name] = (m, Xa_te)
    print(f"{name:42s} val PR-AUC {val_pr:.3f} | converged: {converged} | {round(time.time() - t0)} s")

# ================= 3. penalties: L2, L1, elastic net (smaller C = stronger penalty) =================
for reg in [0.001, 0.01, 0.1]:
    run(f"L2, C={reg}", "L2 (ridge)", Xtr_t, Xva_t, Xte_t, solver="lbfgs", max_iter=300, C=reg)
for reg in [0.003, 0.01, 0.1]:                                       # liblinear: converges reliably and gives exact zeros
    run(f"L1, C={reg}", "L1 (lasso)", Xtr_t, Xva_t, Xte_t, penalty="l1", solver="liblinear", max_iter=200, tol=1e-3, C=reg)
for reg in [0.01, 0.1]:
    run(f"Elastic net (50/50), C={reg}", "Elastic net", Xtr_t, Xva_t, Xte_t, penalty="elasticnet", l1_ratio=0.5,
        solver="saga", max_iter=100, tol=1e-3, C=reg)

# ================= 4. weight-of-evidence (WoE) binning, the credit-scorecard approach =================
# Each feature is cut into up to 10 quantile bins (plus a "missing" bin). Every bin gets ln(share of frauds / share of normals);
# positive = riskier than average. Text-coded columns use their categories (rare ones pooled). Fitted on TRAINING rows only.
def woe_key(x, spec):
    if spec["is_cat"]:
        key = np.where(np.isin(x, spec["known"]), x, -2.0)           # -2 = rare / unseen category
    else:
        key = np.searchsorted(spec["edges"], x, side="right").astype("float64")
    return np.where(np.isnan(x), -1.0, key)                          # -1 = missing

def woe_fit(x, yy, is_cat, n_bins=10, min_count=100):
    spec = {"is_cat": is_cat}
    nn = x[~np.isnan(x)]
    if is_cat:
        vals, cnt = np.unique(nn, return_counts=True)
        spec["known"] = vals[cnt >= min_count]
    else:
        spec["edges"] = np.unique(np.quantile(nn, np.linspace(0, 1, n_bins + 1))[1:-1]) if len(nn) else np.array([])
    tab = pd.DataFrame({"k": woe_key(x, spec), "bad": yy}).groupby("k")["bad"].agg(bad="sum", n="count")
    good, k = tab["n"] - tab["bad"], len(tab)
    pb = (tab["bad"] + 0.5) / (tab["bad"].sum() + 0.5 * k)           # smoothed share of frauds in each bin
    pg = (good + 0.5) / (good.sum() + 0.5 * k)                       # smoothed share of normal transactions
    woe = np.log(pb / pg)
    spec["woe"], spec["iv"] = woe.to_dict(), float(((pb - pg) * woe).sum())   # iv = information value (strength of the feature)
    return spec

def woe_apply(x, spec):
    return pd.Series(woe_key(x, spec)).map(spec["woe"]).fillna(0.0).values.astype("float32")

t0 = time.time()
cols, cat_set = list(X.columns), set(text_cols)
Xtr_df, Xva_df, Xte_df = X[train_mask], X[val_mask], X[test_mask]
specs = {c: woe_fit(Xtr_df[c].values, ytr, c in cat_set) for c in cols}
Wtr = np.column_stack([woe_apply(Xtr_df[c].values, specs[c]) for c in cols])
Wva = np.column_stack([woe_apply(Xva_df[c].values, specs[c]) for c in cols])
Wte = np.column_stack([woe_apply(Xte_df[c].values, specs[c]) for c in cols])
print("WoE matrix:", Wtr.shape, "|", round(time.time() - t0), "s")

iv = pd.Series({c: specs[c]["iv"] for c in cols}).sort_values(ascending=False)
print("\nStrongest single features by information value (>0.3 strong, 0.1-0.3 medium, <0.02 useless):")
print(pd.DataFrame({"IV": iv.head(10).round(3), "family": [family(c) for c in iv.head(10).index]}).to_string())
print("features with IV > 0.1:", int((iv > 0.1).sum()), "of", len(iv), "\n")

for reg in [0.01, 0.1, 1]:
    run(f"WoE + L2, C={reg}", "WoE + L2", Wtr, Wva, Wte, solver="lbfgs", max_iter=300, C=reg)

# ================= 5. choose by VALIDATION, then look at the future test once =================
res = pd.DataFrame(rows)
print("\nAll settings tried, judged on the VALIDATION weeks only:")
print(res.sort_values("val_PR_AUC", ascending=False).round(3).to_string(index=False))

best = res.loc[res.groupby("family")["val_PR_AUC"].idxmax()]
final = []
for _, r in best.iterrows():
    m, Xa_te = fits[r.model]
    s = m.decision_function(Xa_te)
    final.append({"model": r.model, "val_PR_AUC": r.val_PR_AUC, "ROC-AUC": roc_auc_score(yte, s),
                  "PR-AUC all": average_precision_score(yte, s),
                  "PR-AUC early": average_precision_score(yte[early_m], s[early_m]),
                  "PR-AUC late": average_precision_score(yte[late_m], s[late_m]), "nonzero": r.nonzero_coefs, "converged": r.converged})
if "scores" in globals() and "lightgbm" in scores:                      # reference rows from earlier in the notebook
    for n in ["lightgbm", "logreg"]:
        s = scores[n]
        final.append({"model": f"(reference) {n}", "val_PR_AUC": np.nan, "ROC-AUC": roc_auc_score(yte, s),
                      "PR-AUC all": average_precision_score(yte, s), "PR-AUC early": average_precision_score(yte[early_m], s[early_m]),
                      "PR-AUC late": average_precision_score(yte[late_m], s[late_m]), "nonzero": np.nan, "converged": np.nan})
final = pd.DataFrame(final).set_index("model")
final["relative drop %"] = ((final["PR-AUC early"] - final["PR-AUC late"]) / final["PR-AUC early"] * 100).round(0)
print("\nBest setting of each kind (chosen on validation), scored on the future test weeks:")
print(final.round(3).to_string())
