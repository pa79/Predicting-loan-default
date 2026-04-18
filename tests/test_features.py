from loan_default_prediction.demo import generate_demo_tables
from loan_default_prediction.features import build_feature_table


def test_feature_table_contains_transaction_aggregates() -> None:
    tables = generate_demo_tables(seed=2)
    feature_table = build_feature_table(tables)
    assert "txn_count" in feature_table.columns
    assert "avg_balance" in feature_table.columns
    assert len(feature_table) == len(tables["loans"])

