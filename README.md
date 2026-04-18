# Loan Default Prediction from Relational Banking Data

Production-style Python repository that turns messy multi-table banking data into borrower-level default features and trains a prediction pipeline for credit screening.

The project is positioned around the real engineering problem: default models are rarely built from a single clean table. They are built from linked account, client, transaction, and loan data that must be standardized and aggregated first.

## Problem

Banks and fintech lenders need to identify higher-risk borrowers before credit losses occur. In practice, that requires joining fragmented operational data, engineering stable borrower-level features, and producing a reusable training dataset for model development.

This repository focuses on that workflow.

## Why This Matters

The strongest signal in this project is not just model fitting. It is the ability to work with messy relational data and convert millions of transaction-level records into analysis-ready borrower features.

## Data Approach

- `demo` mode generates Berka-style relational banking tables:
  - clients
  - accounts
  - transactions
  - loans
  - district attributes
- the pipeline aggregates transaction and balance history to the loan level
- the final modeling table is stored for downstream model development

## Pipeline

```text
synthetic relational tables
          |
   standardize and join keys
          |
 account and transaction aggregation
          |
 borrower-level feature table
          |
 model training and evaluation
          |
 exported scored sample and metrics
```

## Repository Layout

```text
Predicting-loan-default/
├── .github/workflows/ci.yml
├── data/{raw,processed,gold}/
├── examples/
├── logs/
├── notebooks/
├── src/loan_default_prediction/
├── tests/
├── Makefile
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Modeling Approach

- borrower feature construction from relational source tables
- logistic regression and random forest comparison
- holdout evaluation with AUC, precision, recall, and default-rate summaries
- emphasis on usable features and reproducibility rather than notebook-only exploration

## Outputs

The demo pipeline writes:

- `examples/model_summary.json`
- `examples/feature_table_sample.csv`
- `examples/scored_loans_sample.csv`

## Use Cases

- consumer credit screening
- borrower risk segmentation
- feature-store style preparation for lending models
- data engineering demonstration on multi-table financial data

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -e .
pytest
python -m loan_default_prediction.cli run --mode demo
```

## Resume-Style Summary

Built a Python credit risk pipeline that aggregates relational banking tables into borrower-level features and trains default prediction models for credit screening and portfolio analytics.

