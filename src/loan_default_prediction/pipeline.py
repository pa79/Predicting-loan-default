from __future__ import annotations

import json
import logging

from loan_default_prediction.demo import generate_demo_tables
from loan_default_prediction.features import build_feature_table
from loan_default_prediction.modeling import train_models
from loan_default_prediction.paths import EXAMPLES_DIR, PROCESSED_DIR, RAW_DIR, ensure_directories


def run_demo_pipeline(logger: logging.Logger) -> dict[str, dict[str, float]]:
    ensure_directories()
    tables = generate_demo_tables()
    for name, frame in tables.items():
        frame.to_parquet(RAW_DIR / f"{name}.parquet", index=False)

    feature_table = build_feature_table(tables)
    feature_table.to_parquet(PROCESSED_DIR / "loan_features.parquet", index=False)

    scored, metrics = train_models(feature_table)
    scored.head(250).to_csv(EXAMPLES_DIR / "scored_loans_sample.csv", index=False)
    feature_table.head(250).to_csv(EXAMPLES_DIR / "feature_table_sample.csv", index=False)
    (EXAMPLES_DIR / "model_summary.json").write_text(json.dumps(metrics, indent=2))
    logger.info("Loan default demo pipeline completed.")
    return metrics

