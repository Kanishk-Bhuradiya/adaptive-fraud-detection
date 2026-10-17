# Experiment Log

## EXP-001: Initial Sample Inspection

**Question:** Can the transaction and identity files be accessed and inspected?

**Method:** Loaded 5,000 rows from each file.

**Result:** Transaction sample contained 394 columns; identity sample contained 41 columns. Sample fraud rate was 2.18%.

**Interpretation:** Data access and basic inspection worked. The sample rate is not the overall fraud rate.

**Next step:** Audit the full transaction table.

## EXP-002: Full Transaction Audit

**Question:** What is the size, label distribution, and time coverage of the full transaction table?

**Method:** Loaded five selected columns from the complete transaction file.

**Result:** 590,540 transactions; 20,663 fraud cases; 3.499% fraud prevalence; no duplicate transaction IDs; TransactionDT was sorted in ascending order.

**Interpretation:** The dataset is class-imbalanced and suitable for chronological evaluation, subject to further validation.

**Next step:** Analyze temporal prevalence and missingness.

## EXP-003: Temporal Fraud Prevalence

**Question:** Does the observed fraud rate vary across chronological periods?

**Method:** Divided time-sorted transactions into five consecutive equal-sized windows.

**Result:** Fraud rates ranged from 2.393% to 4.007%.

**Interpretation:** Fraud prevalence varied across the selected windows. Concept drift is not established.

**Next step:** Analyze feature-distribution changes and model performance over time.

## EXP-004: Missing-Value Audit

**Question:** How widespread are missing values across transaction features?

**Method:** Processed all 394 columns in chunks of 50,000 rows.

**Result:** 174 columns had more than 50% missing values; 20 had no missing values.

**Interpretation:** Missingness must be handled carefully, and highly incomplete features should not be discarded automatically.

**Next step:** Investigate temporal missingness and develop leakage-safe preprocessing.

## EXP-005: Identity Coverage and Fraud Prevalence

**Question:** How many transactions have matching identity records, and does fraud prevalence differ by identity availability?

**Method:** Matched transaction IDs and compared fraud rates by identity availability.

**Result:** Identity coverage was 24.42%. Fraud rates were 7.847% with identity records and 2.094% without them.

**Interpretation:** Identity availability is associated with observed fraud prevalence. Causation and predictive value remain untested.

**Next step:** Compare transaction-only and transaction-plus-identity models.
