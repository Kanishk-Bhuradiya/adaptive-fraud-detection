# Dataset Audit

## 1. Initial Sample Inspection

Loaded the first 5,000 rows of the transaction and identity tables to verify file access and inspect the schema.

- Transaction sample: 5,000 rows and 394 columns.
- Identity sample: 5,000 rows and 41 columns.
- Sample fraud rate: 2.18%.
- The first 5,000 transaction rows were ordered by TransactionDT.

The sample was used only for initial inspection. Its fraud rate was not treated as the overall dataset fraud rate.

## 2. Full Transaction Audit

- Transactions: 590,540.
- Fraud cases: 20,663.
- Non-fraud cases: 569,877.
- Fraud rate: 3.499%.
- Duplicate TransactionID values: 0.
- No missing values in the five selected audit columns.
- TransactionDT range: 86,400 to 15,811,131.
- Full transaction file sorted by TransactionDT: yes.

## 3. Missing-Value Analysis

The complete transaction table was processed in chunks of 50,000 rows.

- Columns examined: 394.
- Columns with no missing values: 20.
- Columns with more than 50% missing values: 174.
- Examples of highly incomplete features include dist2, D7, D13, and D14.

No feature-removal decisions were made based solely on missingness.

## 4. Temporal Fraud-Rate Analysis

The transactions were sorted by TransactionDT and divided into five consecutive windows of equal size.

| Window | Transactions | Fraud cases | Fraud rate |
|---|---:|---:|---:|
| 1 | 118,108 | 2,826 | 2.393% |
| 2 | 118,108 | 4,732 | 4.007% |
| 3 | 118,108 | 4,430 | 3.751% |
| 4 | 118,108 | 4,611 | 3.904% |
| 5 | 118,108 | 4,064 | 3.441% |

The observed fraud rate varied across windows. This motivates further analysis but does not establish concept drift.

## 5. Identity-Table Coverage

- Transaction records: 590,540.
- Identity records: 144,233.
- Transactions with matching identity records: 144,233.
- Transactions without matching identity records: 446,307.
- Identity coverage: 24.42%.
- Duplicate IDs in the identity table: 0.

## 6. Fraud Rate by Identity Availability

| Group | Transactions | Fraud cases | Fraud rate |
|---|---:|---:|---:|
| Identity unavailable | 446,307 | 9,345 | 2.094% |
| Identity available | 144,233 | 11,318 | 7.847% |

Identity availability is associated with different observed fraud prevalence. This descriptive association does not establish causation.

## 7. Limitations and Next Steps

The audit is descriptive. It has not yet evaluated model performance or established feature drift or concept drift.

Next steps:
1. Finalize chronological training, validation, and test periods.
2. Establish leakage-safe preprocessing.
3. Train baseline models.
4. Measure temporal feature drift.
5. Compare predictive performance across future periods.

## Equal-duration temporal analysis

The transaction dataset was divided into eight equal-duration intervals using `TransactionDT`, a relative time offset. Transaction counts consequently differed across intervals.

The observed fraud rate varied over time, reaching its highest level in Window 3 (approximately 4.26%) and its lowest level in Window 1 (approximately 2.55%). Identity-data coverage also changed substantially, declining from approximately 42% in Window 1 to approximately 17–20% across several later windows, before increasing in Window 8.

These findings indicate temporal variation in fraud prevalence and identity-data availability. They do not establish concept drift or explain the cause of the changes. Further analysis is needed to investigate transaction composition, feature distributions, and future-period model performance.

**Limitation:** The intervals represent equal durations in relative `TransactionDT` units and should not be interpreted as calendar weeks.
