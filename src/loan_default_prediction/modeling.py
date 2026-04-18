from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_COLUMNS = [
    "loan_amount",
    "duration_months",
    "account_age_months",
    "age",
    "avg_salary",
    "unemployment_rate",
    "txn_count",
    "avg_txn_amount",
    "avg_balance",
    "balance_volatility",
    "credit_share",
]
CATEGORICAL_COLUMNS = ["account_type"]


def train_models(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    frame = frame.sort_values("issue_date").reset_index(drop=True)
    y = frame["default_flag"]
    x = frame[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
    cutoff = int(len(frame) * 0.8)
    x_train, x_test = x.iloc[:cutoff], x.iloc[cutoff:]
    y_train, y_test = y.iloc[:cutoff], y.iloc[cutoff:]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
                ),
                NUMERIC_COLUMNS,
            ),
            (
                "cat",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]
                ),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=500),
        "random_forest": RandomForestClassifier(n_estimators=150, random_state=42, min_samples_leaf=10),
    }

    metrics: dict[str, dict[str, float]] = {}
    scored = x_test.copy()
    scored["actual_default"] = y_test.values

    for name, model in models.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(x_train, y_train)
        pd_hat = pipeline.predict_proba(x_test)[:, 1]
        threshold = float(y_train.mean())
        pred = (pd_hat >= threshold).astype(int)
        metrics[name] = {
            "auc": round(float(roc_auc_score(y_test, pd_hat)), 4),
            "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        }
        scored[f"{name}_pd"] = pd_hat

    return scored.reset_index(drop=True), metrics
