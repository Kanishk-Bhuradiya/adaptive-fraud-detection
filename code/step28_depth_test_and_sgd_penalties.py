# Step 28 - boosted trees of increasing depth (interaction test) and SGD-based penalties (found INCONCLUSIVE, see report).
# Run after step27 (needs: X_train, y_train, X_val, y_val, X_test, report, Xtr_t, Xte_t, ytr).
from sklearn.linear_model import SGDClassifier
import lightgbm as lgb, time

rows = []

# ---------- A. boosted trees of increasing depth ----------
for depth in [1, 2, 3, 6]:
    t0 = time.time()
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.1, max_depth=depth,
                           num_leaves=2 if depth == 1 else 2 ** depth,
                           colsample_bytree=0.5, subsample=0.8, subsample_freq=1,
                           scale_pos_weight=5, metric="average_precision",
                           random_state=42, verbose=-1)
    m.fit(X_train, y_train, eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(30, verbose=False)])
    r = report(f"boosted trees, depth {depth}", m.predict_proba(X_test)[:, 1])
    r["detail"] = f"{m.best_iteration_} trees"
    rows.append(r)
    print(f"depth {depth} done: {round(time.time() - t0)} s")

# ---------- B. penalties for logistic regression (SGD solver: did NOT converge well -> inconclusive) ----------
for penalty, alpha in [("l2", 1e-5), ("l2", 1e-4), ("l1", 1e-5), ("l1", 1e-4),
                       ("elasticnet", 1e-5), ("elasticnet", 1e-4)]:
    t0 = time.time()
    clf = SGDClassifier(loss="log_loss", penalty=penalty, alpha=alpha, l1_ratio=0.5,
                        class_weight="balanced", max_iter=50, tol=1e-4, random_state=42)
    clf.fit(Xtr_t, ytr)
    r = report(f"logreg {penalty}, alpha={alpha:g}", clf.decision_function(Xte_t))
    r["detail"] = f"{int((clf.coef_ != 0).sum())} of {Xtr_t.shape[1]} coefficients non-zero"
    rows.append(r)
    print(f"{penalty} alpha={alpha:g} done: {round(time.time() - t0)} s")

tab2 = pd.DataFrame(rows).set_index("model")
tab2["relative drop %"] = ((tab2["PR-AUC early"] - tab2["PR-AUC late"]) / tab2["PR-AUC early"] * 100).round(0)
print()
print(tab2.round(3).to_string())
