# Adaptive and Explainable Fraud Detection Under Concept Drift

## Overview

This project investigates the temporal robustness of machine-learning models for fraud detection using the IEEE-CIS Fraud Detection dataset.

The study examines whether fraud prevalence and feature distributions change over time, whether predictive performance varies across future transaction periods, and whether adaptive retraining can maintain performance. It also investigates estimated financial decision costs and the stability of SHAP-based explanations.

## Research Questions

1. How does fraud-detection performance change across future temporal windows?
2. Which features exhibit the strongest temporal drift?
3. Which models are most robust to temporal drift?
4. Can adaptive retraining maintain performance, and what retraining frequency is appropriate?
5. Does adaptive retraining reduce estimated financial decision cost?
6. How stable are SHAP explanations across time?

## Dataset

Link to dataset : https://www.kaggle.com/competitions/ieee-fraud-detection/data
link to bundle data set which i used to run my later notebook : https://drive.google.com/drive/folders/1LrkwSvs5macXOWNVVJGdb1YUt22biUN3?usp=drive_link

The project uses the IEEE-CIS Fraud Detection competition dataset.

The training transaction table contains 590,540 transactions and 394 columns, including the `isFraud` target label and the `TransactionDT` relative time variable.

The associated identity table contains 144,233 records with 41 columns.

The original competition data is not included in this repository. Follow the competition's terms and access the data through Kaggle.

## Preliminary Dataset Audit

Initial findings from the training transaction data:

- Total transactions: 590,540
- Fraudulent transactions: 20,663 (3.499%)
- Non-fraudulent transactions: 569,877 (96.501%)
- Duplicate transaction IDs: none detected
- Transaction features with more than 50% missing values: 174 of 394
- Transactions with matching identity records: 144,233 (24.42%)

Fraud prevalence varied across five preliminary chronological windows, ranging from 2.393% to 4.007%.

Transactions with matching identity records had an observed fraud rate of 7.847%, compared with 2.094% for transactions without matching identity records.

These are descriptive findings. They do not, by themselves, establish concept drift or demonstrate that a model's performance changes over time.

## Planned Methodology

1. Complete dataset auditing and feature inspection.
2. Establish chronological training, validation, and test periods.
3. Train baseline classification models.
4. Measure feature-distribution drift and evaluate predictive performance over time.
5. Compare static and adaptive retraining strategies.
6. Estimate financial decision costs under explicit assumptions.
7. Evaluate SHAP explanation stability across time.
8. Report results, limitations, and reproducibility details.

## Evaluation

Precision, recall, PR-AUC, and other suitable classification metrics will be used to evaluate the models. Chronological evaluation will be used to simulate predictions on future transactions.

Data preprocessing, feature selection, threshold selection, and model tuning will be designed to avoid future-data leakage.

## Current Status

Completed:
- Kaggle notebook setup
- Initial sample inspection
- Full transaction audit
- Overall missing-value analysis
- Preliminary chronological fraud-rate analysis
- Identity-table coverage analysis
- Fraud-rate comparison by identity availability

In progress:
- Finalizing chronological evaluation periods
- Preparing baseline models

## Repository Structure

- `notebooks/`: executable analysis and experiment notebooks
- `research_notes/`: research questions, methodology, literature notes, and experiment logs
- `results/`: generated figures and tables

## Reproducibility

The analysis is developed in Kaggle notebooks using Python and Pandas. Notebook versions, data-access requirements, package versions, experiment settings, and results should be documented as the project develops.v
