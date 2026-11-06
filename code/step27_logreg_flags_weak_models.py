# Step 27 - fair logistic regression (with missing-value flags) and weaker models, plus ROC-AUC for every model.
# Run as a cell AFTER the clean notebook's Parts 0-5 (needs: X, y, week, df, text_cols, train_mask, test_mask, scores).
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import time

low_card = [c for c in text_cols if df[c].nunique() <= 30]
numeric  = [c for c in X.columns if c not in text_cols]

prep = ColumnTransformer([
    ("num", Pipeline([("fill", SimpleImputer(strategy="median", add_indicator=True)),   # adds "was missing" flags
                      ("scale", QuantileTransformer(n_quantiles=200, subsample=100_000,
                                                    output_distribution="normal", random_state=42))]), numeric),
    ("cat", Pipeline([("fill", SimpleImputer(strategy="constant", fill_value=-1)),
                      ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20))]), low_card),
], sparse_threshold=0)

t0 = time.time()
Xtr_t = prep.fit_transform(X[train_mask]).astype("float32")
Xte_t = prep.transform(X[test_mask]).astype("float32")
ytr, y_t, wk_t = y[train_mask], y[test_mask], week[test_mask]
print("transformed shape:", Xtr_t.shape, "|", round(time.time() - t0), "s")

early, late = np.isin(wk_t, [15, 16, 17]), np.isin(wk_t, [23, 24, 25])
def report(name, s):
    return {"model": name,
            "ROC-AUC": roc_auc_score(y_t, s),
            "PR-AUC all": average_precision_score(y_t, s),
            "PR-AUC early": average_precision_score(y_t[early], s[early]),
            "PR-AUC late": average_precision_score(y_t[late], s[late])}

rows = [report(n, scores[n]) for n in ["logreg", "random_forest", "xgboost", "lightgbm"]]
rows[0]["model"] = "logreg (old: no missing flags)"

new_scores = {}
for reg in [0.1, 0.01]:
    t0 = time.time()
    m = LogisticRegression(C=reg, max_iter=1000, class_weight="balanced").fit(Xtr_t, ytr)
    new_scores[f"logreg + flags, C={reg}"] = m.predict_proba(Xte_t)[:, 1]
    print(f"logreg C={reg} done:", round(time.time() - t0), "s | iterations used:", int(m.n_iter_[0]))

new_scores["linear probability (Ridge)"] = Ridge(alpha=10).fit(Xtr_t, ytr).predict(Xte_t)
new_scores["naive Bayes"] = GaussianNB().fit(Xtr_t, ytr).predict_proba(Xte_t)[:, 1]
new_scores["decision tree (depth 6)"] = DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, class_weight="balanced",
                                                               random_state=42).fit(Xtr_t, ytr).predict_proba(Xte_t)[:, 1]
for n, s in new_scores.items():
    rows.append(report(n, s))

tab = pd.DataFrame(rows).set_index("model")
tab["drop (early-late)"] = tab["PR-AUC early"] - tab["PR-AUC late"]
print()
print(tab.round(3).to_string())
