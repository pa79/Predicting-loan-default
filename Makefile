PYTHON := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: venv install test demo

venv:
	python3 -m venv .venv

install: venv
	$(PIP) install -r requirements.txt -e .

test:
	$(PYTHON) -m pytest

demo:
	$(PYTHON) -m loan_default_prediction.cli run --mode demo

