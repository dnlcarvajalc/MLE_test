VENV_DIR := .venv

.PHONY: all env data_ingestion data_transformation

all: env data_ingestion data_transformation

env:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Creating virtual environment..."; \
		python3 -m venv $(VENV_DIR); \
	fi
	@. $(VENV_DIR)/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

data_ingestion:
	@. $(VENV_DIR)/bin/activate && python src/data_ingestion.py

data_transformation:
	@. $(VENV_DIR)/bin/activate && python src/data_transformation.py
