# Convenience targets. On Windows use Git Bash/WSL, or copy the commands from the README.
PY ?= python
IMAGE ?= heart-api:1.0.0

.PHONY: help install data eda train summary test lint format run docker-build docker-run smoke \
        compose-up compose-down traffic k8s-deploy k8s-status notebooks diagram report mlflow-ui clean

help:            ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-14s %s\n",$$1,$$2}'

install:         ## install all dependencies (use inside a virtualenv)
	$(PY) -m pip install -r requirements-dev.txt

data:            ## download the dataset and build data/processed/heart_clean.csv
	$(PY) -m src.download_data
	$(PY) -m src.data_processing --force

eda:             ## EDA figures -> reports/figures
	$(PY) -m src.eda

train:           ## train 3 models, log to MLflow, save models/model.joblib
	$(PY) -m src.train
	$(PY) scripts/mlflow_summary.py

summary:         ## regenerate reports/experiment_summary.md from MLflow
	$(PY) scripts/mlflow_summary.py

mlflow-ui:       ## open the MLflow UI at http://localhost:5000
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

test:            ## unit tests with coverage
	$(PY) -m pytest --cov=src --cov=api --cov-report=term-missing

lint:            ## flake8 + black --check
	flake8 src api tests scripts
	black --check src api tests scripts

format:          ## auto-format with black
	black src api tests scripts

run:             ## run the API locally with auto-reload
	uvicorn api.main:app --reload --port 8000

docker-build:    ## build the serving image (train first!)
	docker build -t $(IMAGE) .

docker-run:      ## run the container on port 8000
	docker run --rm -p 8000:8000 $(IMAGE)

smoke:           ## smoke-test a running API (default http://localhost:8000)
	bash scripts/smoke_test.sh

compose-up:      ## API + Prometheus + Grafana
	docker compose up --build -d

compose-down:
	docker compose down

traffic:         ## generate traffic for the Grafana dashboard
	$(PY) scripts/generate_traffic.py --requests 300

k8s-deploy:      ## deploy to the current kubectl context (Minikube/Docker Desktop)
	kubectl apply -k k8s/
	kubectl -n heart-mlops rollout status deployment/heart-api --timeout=120s
	kubectl -n heart-mlops get pods,svc

k8s-status:
	kubectl -n heart-mlops get pods,svc,deploy

notebooks:       ## build and execute the three notebooks in place
	$(PY) scripts/build_notebooks.py
	jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

diagram:         ## redraw report/architecture.png
	$(PY) scripts/make_architecture_diagram.py

report:          ## convert report/REPORT.md to report/MLOps_Assignment1_Report.docx (needs pandoc)
	cd report && pandoc REPORT.md -o MLOps_Assignment1_Report.docx --toc --resource-path=.:..

clean:           ## remove local run artefacts (keeps data and committed model)
	rm -rf .pytest_cache .coverage coverage.xml htmlcov test-results.xml mlflow.db mlartifacts mlruns
