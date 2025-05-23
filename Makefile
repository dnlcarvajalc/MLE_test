VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

.PHONY: all env lint data_ingestion data_transformation classify train_model run_api

all: env lint data_ingestion data_transformation classify train_model run_api

env:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Creating virtual environment..."; \
		python3 -m venv $(VENV_DIR); \
	fi
	@$(PIP) install --upgrade pip && $(PIP) install -r requirements.txt

lint:
	@echo "Running linter (ruff) on src/ "
	-@$(VENV_DIR)/bin/ruff check src || echo "Lint errors found, but continuing..."

data_ingestion:
	@$(PYTHON) src/a_data_ingestion.py

data_transformation:
	@$(PYTHON) src/b_data_transformation.py

classify:
	@$(PYTHON) src/c_classify.py

train_model:
	@$(PYTHON) src/d_train_model.py

run_api:
	@echo "Checking for Docker containers using port 8000..."
	@CID=$$(docker ps -q --filter "publish=8000"); \
	if [ -n "$$CID" ]; then \
		echo "Stopping and removing container using port 8000 (CID: $$CID)..."; \
		docker stop $$CID && docker rm $$CID; \
	else \
		echo "No container using port 8000."; \
	fi
	@echo "Building Docker image..."
	@docker build -t fastapi-bert .
	@echo "Running Docker container..."
	@docker run -p 8000:8000 fastapi-bert
