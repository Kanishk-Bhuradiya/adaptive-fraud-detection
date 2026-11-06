# Step 33 - tree + logistic-regression hybrid tuned on the VALIDATION weeks; the test weeks are looked at once.
# Run after the clean notebook's Parts 0-5 (needs: X, y, week, train_mask, val_mask, test_mask, scores, and the Part 0 imports).
import warnings
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score

X_tr, X_va, X_te = X[train_mask], X[val_mask], X[test_mask]
ytr, yva, yte = y[train_mask], y[val_mask], y[test_mask]
wk_te = week[test_mask]
early_m, late_m = np.isin(wk_te, [15, 16, 17]), np.isin(wk_te, [23, 24, 25])

rows, fits = [], {}
for n_trees, depth in [(100, 3), (300, 3), (100, 5), (300, 5)]:
    t0 = time.time()
    gbm = lgb.LGBMClassifier(n_estimators=n_trees, learning_rate=0.1, max_depth=depth, num_leaves=2 ** depth,
                             colsample_bytree=0.5, subsample=0.8, subsample_freq=1, scale_pos_weight=5,
                             metric="average_precision", random_state=42, verbose=-1).fit(X_tr, ytr)
    trees_val = average_precision_score(yva, gbm.predict_proba(X_va)[:, 1])
    enc = OneHotEncoder(handle_unknown="ignore")                       # one yes/no column per leaf
    L_tr = enc.fit_transform(gbm.predict(X_tr, pred_leaf=True))
    L_va = enc.transform(gbm.predict(X_va, pred_leaf=True))
    L_te = enc.transform(gbm.predict(X_te, pred_leaf=True))
    print(f"{n_trees} trees, depth {depth}: {L_tr.shape[1]} leaf columns | trees fitted in {round(time.time() - t0)} s")
    for reg in [0.003, 0.01, 0.03]:
        t1 = time.time()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            lr = LogisticRegression(C=reg, max_iter=200, class_weight="balanced").fit(L_tr, ytr)
        ok = not any(issubclass(c.category, ConvergenceWarning) for c in caught)
        hyb_val = average_precision_score(yva, lr.decision_function(L_va))
        rows.append({"trees": n_trees, "depth": depth, "C": reg, "leaf_columns": L_tr.shape[1],
                     "trees_alone_val": trees_val, "hybrid_val": hyb_val, "converged": ok})
        fits[(n_trees, depth, reg)] = (lr, L_te, gbm)
        print(f"     C={reg}: hybrid val PR-AUC {hyb_val:.3f} (trees alone {trees_val:.3f}) | converged: {ok} | {round(time.time() - t1)} s")

res = pd.DataFrame(rows)
print("\nAll settings, judged on the VALIDATION weeks:")
print(res.sort_values("hybrid_val", ascending=False).round(3).to_string(index=False))

# ---- choose on validation, then look at the future test once ----
b = res.loc[res["hybrid_val"].idxmax()]
key = (int(b.trees), int(b.depth), float(b.C))
lr, L_te, gbm = fits[key]
def summarize(name, s):
    return {"model": name, "ROC-AUC": roc_auc_score(yte, s), "PR-AUC all": average_precision_score(yte, s),
            "PR-AUC early": average_precision_score(yte[early_m], s[early_m]), "PR-AUC late": average_precision_score(yte[late_m], s[late_m])}
final = [summarize(f"hybrid (chosen: {key[0]} trees, depth {key[1]}, C={key[2]})", lr.decision_function(L_te)),
         summarize(f"trees alone (same {key[0]} trees, depth {key[1]})", gbm.predict_proba(X_te)[:, 1])]
if "scores" in globals() and "lightgbm" in scores:
    final += [summarize("(reference) lightgbm, full boosting", scores["lightgbm"]), summarize("(reference) old logreg", scores["logreg"])]
final = pd.DataFrame(final).set_index("model")
final["relative drop %"] = ((final["PR-AUC early"] - final["PR-AUC late"]) / final["PR-AUC early"] * 100).round(0)
print("\nChosen on validation, scored on the future test weeks:")
print(final.round(3).to_string())
