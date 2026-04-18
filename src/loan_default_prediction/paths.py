from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
GOLD_DIR = DATA_DIR / "gold"
LOG_DIR = ROOT_DIR / "logs"
EXAMPLES_DIR = ROOT_DIR / "examples"


def ensure_directories() -> None:
    for directory in (RAW_DIR, PROCESSED_DIR, GOLD_DIR, LOG_DIR, EXAMPLES_DIR):
        directory.mkdir(parents=True, exist_ok=True)

