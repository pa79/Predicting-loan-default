from __future__ import annotations

import pandas as pd


def build_feature_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    tx = tables["transactions"]
    tx_summary = (
        tx.groupby("account_id")
        .agg(
            txn_count=("amount", "size"),
            avg_txn_amount=("amount", "mean"),
            avg_balance=("balance", "mean"),
            balance_volatility=("balance", "std"),
            credit_share=("txn_type", lambda x: (x == "credit").mean()),
        )
        .reset_index()
    )

    frame = (
        tables["loans"]
        .merge(tables["accounts"], on="account_id")
        .merge(tables["clients"], on="client_id")
        .merge(tables["districts"], on="district_id")
        .merge(tx_summary, on="account_id", how="left")
    )
    return frame.fillna(
        {
            "txn_count": 0,
            "avg_txn_amount": 0.0,
            "avg_balance": 0.0,
            "balance_volatility": 0.0,
            "credit_share": 0.0,
        }
    )

