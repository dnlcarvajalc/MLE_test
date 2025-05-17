VENV_DIR := .venv

.PHONY: all env data_ingestion

all: env data_ingestion

env:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Creating virtual environment..."; \
		python3 -m venv $(VENV_DIR); \
	fi
	@. $(VENV_DIR)/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

data_ingestion:
	@. $(VENV_DIR)/bin/activate && python src/data_ingestion.py
