# Step 29 - first tree + logistic-regression hybrid (100 trees, depth 3; fixed settings; see step33 for the tuned version).
# Run after step27 (needs: X_train, y_train, X_test, report, Xtr_t, Xte_t, ytr).
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb, time

rows = []

# 1. small boosted-tree model that creates the "leaves"
t0 = time.time()
gbm = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, num_leaves=8,
                         colsample_bytree=0.5, subsample=0.8, subsample_freq=1,
                         scale_pos_weight=5, metric="average_precision",
                         random_state=42, verbose=-1)
gbm.fit(X_train, y_train)
r = report("trees alone (depth 3, 100 trees)", gbm.predict_proba(X_test)[:, 1])
r["detail"] = "reference"
rows.append(r)

leaf_tr = gbm.predict(X_train, pred_leaf=True)         # which leaf of each tree every transaction falls into
leaf_te = gbm.predict(X_test,  pred_leaf=True)
enc = OneHotEncoder(handle_unknown="ignore")
L_tr = enc.fit_transform(leaf_tr)                      # sparse yes/no table: one column per leaf
L_te = enc.transform(leaf_te)
print("leaf features:", L_tr.shape, "|", round(time.time() - t0), "s")

# 2. A: logistic regression on leaves only
for reg in [0.01, 0.1]:
    t0 = time.time()
    lr = LogisticRegression(C=reg, max_iter=500, class_weight="balanced").fit(L_tr, ytr)
    r = report(f"hybrid A: logreg on leaves, C={reg}", lr.predict_proba(L_te)[:, 1])
    r["detail"] = f"{int(lr.n_iter_[0])} iterations"
    rows.append(r)
    print(f"A, C={reg} done: {round(time.time() - t0)} s")

# 3. B: leaves + the scaled original features
t0 = time.time()
B_tr = sparse.hstack([L_tr, sparse.csr_matrix(Xtr_t)]).tocsr()
B_te = sparse.hstack([L_te, sparse.csr_matrix(Xte_t)]).tocsr()
lr = LogisticRegression(C=0.1, max_iter=500, class_weight="balanced").fit(B_tr, ytr)
r = report("hybrid B: logreg on leaves + features, C=0.1", lr.predict_proba(B_te)[:, 1])
r["detail"] = f"{int(lr.n_iter_[0])} iterations"
rows.append(r)
print(f"B done: {round(time.time() - t0)} s")

tab3 = pd.DataFrame(rows).set_index("model")
tab3["relative drop %"] = ((tab3["PR-AUC early"] - tab3["PR-AUC late"]) / tab3["PR-AUC early"] * 100).round(0)
print()
print(tab3.round(3).to_string())
