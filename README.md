[README.md](https://github.com/user-attachments/files/34571838/README.md)
# Adaptive and Explainable Fraud Detection Under Concept Drift

**A regression-based study using the IEEE-CIS Fraud Detection dataset**

[![Dataset](https://img.shields.io/badge/Dataset-IEEE--CIS%20Fraud%20Detection-blue)](https://www.kaggle.com/competitions/ieee-fraud-detection)
[![Course](https://img.shields.io/badge/Course-Regression%20Techniques-informational)](#project-information)

This project studies how credit-card fraud detection models behave when transaction patterns and the relationship between transaction features and fraud change over time. Instead of relying only on a conventional random train–test split, we use a chronological evaluation to simulate deployment: models are trained on earlier transactions and evaluated on later, unseen weeks.

The work combines classification, regression modelling, drift analysis, retraining experiments, cost-sensitive evaluation, and model explainability. A particular emphasis is placed on the logistic-regression family and on using regression methods to analyse changes in fraud rates and model coefficients.

## Project information

| | |
|---|---|
| **Course** | Regression Techniques |
| **Instructor** | Prof. Minerva Mukhopadhyay |
| **Group** | Group C-DH |
| **Authors** | Kanishk Bhuradiya and Adarsh Kumar |
| **Report date** | 6 November 2026 |
| **Dataset** | IEEE-CIS Fraud Detection (Vesta Corporation) |

## Research questions

The project investigates:

1. How much does fraud-detection performance deteriorate when evaluation is performed on future transactions?
2. Which input features and feature-availability patterns change over time?
3. How robust are Logistic Regression, Random Forest, XGBoost, and LightGBM under temporal drift?
4. Can periodic or drift-triggered retraining recover lost performance?
5. How do delayed fraud labels affect retraining?
6. Does adaptive retraining reduce estimated operational cost?
7. Are model explanations stable across time?
8. How do regularisation, feature binning, and tree–regression hybrids affect the performance of regression-based models?

## Main findings

The principal results reported in our chronological experiments are:

- **Random splits can be misleading:** the random-split PR-AUC was 0.816, compared with 0.523 under the future-period evaluation.
- **Performance deteriorates over time:** frozen tree models lost approximately 15–20% of PR-AUC between the early and late future-test periods.
- **Retraining helps:** retraining on more recent labelled data recovered a substantial part of the performance loss. Schedules retraining every two or four weeks reduced estimated costs by approximately 7–17% relative to the static model across the assumed cost scenarios.
- **Feature availability matters:** much of the observed input drift was associated with changes in missingness/feature availability rather than only changes in observed feature values.
- **Unlabelled monitoring has limitations:** score-distribution and feature-distribution monitoring did not reliably reveal the deterioration in predictive performance in these experiments.
- **Interactions matter:** regularisation alone did not close the performance gap between logistic regression and boosted trees. A tree-leaf/logistic-regression hybrid improved on a plain linear model, but did not outperform its underlying tree model.

These are results from this dataset and experimental design, not guarantees about all fraud-detection systems. The estimated financial savings depend on the cost assumptions described in the report.

## Dataset

The project uses the labelled files from the [IEEE-CIS Fraud Detection competition on Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data):

- `train_transaction.csv`
- `train_identity.csv`

The two files are joined using `TransactionID`. The labelled transaction table contains 590,540 transactions. The competition's unlabelled test files are not used in this study.

**The dataset is not included in this repository.** Please obtain it through Kaggle and comply with the competition's terms. The project does not redistribute the original data.

## Methodology

### 1. Chronological evaluation

Transactions are sorted by `TransactionDT` and divided into time periods:

| Period | Weeks | Purpose |
|---|---:|---|
| Warm-up | 0 | Excluded from model fitting |
| Training | 1–12 | Model fitting |
| Validation | 13–14 | Early stopping and threshold selection |
| Future test | 15–25 | Week-by-week evaluation on later transactions |

This design is intended to reduce temporal leakage and better reflect a real deployment setting. Models are evaluated on future weeks that were not used to fit them.

### 2. Preprocessing and missingness

- Feature screening is performed using the training period only.
- Highly missing and constant columns are removed according to the notebook's documented rules.
- Highly correlated variables in the `V` feature family are filtered to reduce redundancy.
- Categorical text values are encoded in order of first appearance in time.
- `TransactionID` and `TransactionDT` are not used as model predictors.
- Missing values are retained where the model can handle them; linear-model pipelines use imputation and missingness indicators.
- Class imbalance is addressed through model-level class weighting where applicable.

### 3. Models and regression methods

The experiments include:

- Logistic Regression and regularised variants (L1, L2, and elastic net)
- Random Forest
- XGBoost
- LightGBM
- Naive Bayes and a shallow decision-tree baseline
- Weight of Evidence (WoE) feature transformation with Logistic Regression
- Tree-leaf features combined with Logistic Regression
- Poisson and negative-binomial regression for weekly fraud counts/rates, including change-point and interaction analyses
- Window-wise Logistic Regression coefficient analysis to study changes in fitted relationships

### 4. Drift, adaptation, and explainability

The analysis includes Population Stability Index (PSI), Kolmogorov–Smirnov (KS) statistics, Wasserstein distance, feature-missingness comparisons, fixed and rolling retraining schedules, label-delay experiments, cost-sensitive threshold selection, and SHAP-based explanation stability analysis.

### 5. Evaluation

PR-AUC (average precision) is the principal ranking metric because fraud is a highly imbalanced class. The notebooks also report metrics such as ROC-AUC, precision, recall, F1, and recall under fixed review-capacity budgets. Financial comparisons use explicitly stated hypothetical cost scenarios; they should be interpreted as scenario analyses rather than observed business costs.

## Repository structure

```text
.
├── README.md
├── 01-dataset-audit_final.ipynb
├── fraud_drift_report_final.pdf
├── code/
│   ├── 01-dataset-audit_final.ipynb
│   ├── exploratory_original_notebook.ipynb
│   ├── fraud-drift-clean.ipynb
│   ├── bundle.joblib                 # optional saved model/results bundle
│   ├── step27_logreg_flags_weak_models.py
│   ├── step28_depth_test_and_sgd_penalties.py
│   ├── step29_hybrid_first_try.py
│   ├── step30_31_poisson_changepoints.py
│   ├── step32_logreg_improvements.py
│   ├── step33_hybrid_tuning.py
│   └── step34_coefficient_drift.py
└── report/
    ├── fraud_drift_report_final.pdf
    └── figures/                      # figures used in the report and analysis
```

The dataset-audit notebook is also present at the repository root for convenience. The report PDF is available both at the root and under `report/` in the supplied project archive.

## How to run the notebooks

The experiments were developed in **Kaggle Notebooks**, where the IEEE-CIS competition data were attached as notebook inputs. The clean main notebook is:

`code/fraud-drift-clean.ipynb`

### Option A — Run with the saved bundle

1. Create a Kaggle notebook.
2. Add the IEEE-CIS competition dataset as an input.
3. Add the saved `bundle.joblib` file as another input (available separately; see [Saved model bundle](#saved-model-bundle)).
4. Open `code/fraud-drift-clean.ipynb` in the notebook, or upload/copy its cells into the Kaggle notebook.
5. In Part 0, set `DATA_DIR` to the competition input directory and `BUNDLE` to the bundle file path.
6. Keep `TRAIN_FROM_SCRATCH = False` and run the notebook.

With saved models, the notebook's documented runtime is approximately **15–25 minutes**, depending on the Kaggle environment.

### Option B — Retrain models from scratch

Set:

```python
TRAIN_FROM_SCRATCH = True
```

This fits the models rather than loading the saved versions. The notebook estimates approximately **one hour**, although runtime depends on the hardware and execution environment.

### Optional experiments

Set:

```python
RUN_EXTRA_EXPERIMENTS = True
```

This enables the optional random-split comparison and five-seed noise experiment. These experiments add to the runtime.

### Supplementary step scripts

The `step27`–`step34` Python files contain additional experiments. Their header comments describe the variables and notebook state they expect. Run the main notebook sections first where required. In particular, `step30_31_poisson_changepoints.py` is self-contained and replaces the dataframe it uses; run it last or in a separate session to avoid overwriting the main notebook's dataframe.

> **Reproducibility note:** Paths, package versions, hardware, and Kaggle input locations may need adjustment. Runtime estimates are approximate. The saved bundle contains selected trained models and intermediate results; it does not contain the IEEE-CIS dataset.

## Saved model bundle

The optional `bundle.joblib` file is a large supporting artifact containing selected trained models and intermediate experiment results. It is not required if you choose to retrain from scratch.

[Download the saved model bundle from Google Drive](https://drive.google.com/file/d/11cN9a-m2gUzDWRSY9hMPJ5nPtj_OVLnr/view?usp=sharing)

For convenience, the project also includes a [Kaggle dataset-audit notebook](https://www.kaggle.com/code/kanishkbhuradiya/01-dataset-audit).

## Report and figures

- [Read the final project report](fraud_drift_report_final.pdf)
- [Open the main analysis notebook](code/fraud-drift-clean.ipynb)
- [Open the dataset-audit notebook](01-dataset-audit_final.ipynb)

The `report/figures/` directory contains the plots generated for the report and supplementary analysis, including model comparisons, weekly performance, drift measures, retraining results, cost scenarios, SHAP stability, and coefficient drift.

## Limitations

The findings should be interpreted in light of several limitations:

- The analysis uses one 26-week labelled history; results may differ on other periods or datasets.
- The weekly observations are temporally related and should not be treated as fully independent.
- Several cost quantities are assumptions because the dataset does not provide real investigation or fraud-loss costs.
- Many features are anonymised or masked, limiting business-level interpretation of why they drift.
- The main experiments assume immediately available labels; a four-week label delay is examined as a separate scenario.
- Some modelling choices and exploratory comparisons have limitations discussed in the report, including a short validation period and convergence issues in some logistic-regression fits.

For the full experimental design, detailed tables, discussion, and references, please consult the [final report](fraud_drift_report_final.pdf).

## References and data acknowledgement

The primary data source is the [IEEE-CIS Fraud Detection competition](https://www.kaggle.com/competitions/ieee-fraud-detection), provided by Vesta Corporation through Kaggle. Further methodological references are listed in the project report.

---

*Academic course project — Regression Techniques, Group C-DH, Indian Statistical Institute, Delhi Centre.*
