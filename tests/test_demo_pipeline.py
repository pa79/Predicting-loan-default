import json

from loan_default_prediction.logging_utils import configure_logging
from loan_default_prediction.paths import EXAMPLES_DIR, LOG_DIR
from loan_default_prediction.pipeline import run_demo_pipeline


def test_demo_pipeline_runs() -> None:
    logger = configure_logging(LOG_DIR / "test.log")
    metrics = run_demo_pipeline(logger)
    assert metrics["logistic_regression"]["auc"] > 0.6
    assert "random_forest" in json.loads((EXAMPLES_DIR / "model_summary.json").read_text())
