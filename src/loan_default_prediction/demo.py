from __future__ import annotations

import numpy as np
import pandas as pd


def generate_demo_tables(seed: int = 21) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    districts = pd.DataFrame(
        {
            "district_id": range(1, 31),
            "avg_salary": rng.integers(24000, 52000, 30),
            "unemployment_rate": rng.uniform(0.02, 0.11, 30),
        }
    )

    clients = pd.DataFrame(
        {
            "client_id": range(1, 2501),
            "district_id": rng.integers(1, 31, 2500),
            "age": rng.integers(21, 74, 2500),
        }
    )

    accounts = pd.DataFrame(
        {
            "account_id": range(1, 2501),
            "client_id": clients["client_id"],
            "account_age_months": rng.integers(6, 180, 2500),
            "account_type": rng.choice(["standard", "premium", "student"], 2500, p=[0.68, 0.22, 0.10]),
        }
    )

    tx_rows = 90000
    base = accounts.merge(clients, on="client_id").merge(districts, on="district_id")
    risk_index = (
        4.5 * base["unemployment_rate"]
        - 0.000025 * base["avg_salary"]
        - 0.01 * base["account_age_months"]
        + 0.015 * (50 - base["age"]).clip(lower=0)
        + 0.35 * (base["account_type"] == "student").astype(int)
    )
    account_risk = pd.Series(risk_index.values, index=base["account_id"])

    sampled_accounts = rng.choice(accounts["account_id"], tx_rows)
    sampled_risk = account_risk.loc[sampled_accounts].to_numpy()
    credit_probability = np.clip(0.52 - 0.08 * sampled_risk, 0.1, 0.8)
    txn_type = np.where(rng.random(tx_rows) < credit_probability, "credit", "debit")
    base_balance = 3200 - 1600 * sampled_risk + rng.normal(0, 500, tx_rows)
    txn_amount = rng.normal(200 - 260 * sampled_risk, 650, tx_rows)
    debit_mask = txn_type == "debit"
    txn_amount[debit_mask] *= -1

    transactions = pd.DataFrame(
        {
            "account_id": sampled_accounts,
            "amount": txn_amount,
            "balance": base_balance,
            "txn_type": txn_type,
        }
    )

    tx_summary = transactions.groupby("account_id").agg(
        mean_balance=("balance", "mean"),
        credit_share=("txn_type", lambda x: (x == "credit").mean()),
    )
    base = base.merge(tx_summary, left_on="account_id", right_index=True, how="left")

    loans = pd.DataFrame(
        {
            "loan_id": range(1, 2501),
            "account_id": accounts["account_id"],
            "issue_date": rng.choice(pd.date_range("2015-01-01", "2020-12-01", freq="MS"), 2500),
            "loan_amount": rng.integers(1500, 22000, 2500),
            "duration_months": rng.choice([12, 24, 36, 48], 2500, p=[0.2, 0.35, 0.3, 0.15]),
        }
    )
    loan_base = loans.merge(base, on="account_id", how="left")
    logit = (
        -2.0
        + 0.00008 * loan_base["loan_amount"]
        + 0.014 * loan_base["duration_months"]
        + 12.0 * loan_base["unemployment_rate"]
        - 0.00005 * loan_base["avg_salary"]
        - 0.025 * loan_base["account_age_months"]
        + 0.03 * (48 - loan_base["age"]).clip(lower=0)
        + 0.8 * (loan_base["account_type"] == "student").astype(int)
        - 0.00022 * loan_base["mean_balance"].fillna(0)
        - 1.6 * loan_base["credit_share"].fillna(0.4)
    )
    pd_prob = 1 / (1 + np.exp(-logit))
    loans["default_flag"] = rng.binomial(1, np.clip(pd_prob, 0.04, 0.7))

    return {
        "districts": districts,
        "clients": clients,
        "accounts": accounts,
        "transactions": transactions,
        "loans": loans,
    }
