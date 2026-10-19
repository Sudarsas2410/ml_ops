# Submission checklist and runbook

Work through this top to bottom. Each step lists the command, the evidence to capture (save screenshots in
`screenshots/` with exactly these names so `report/REPORT.md` picks them up) and the rubric marks it covers.
Take screenshots at a readable window size and include the terminal prompt, command and output.

## 0. One-time setup (15 min)

```bash
git clone <your-repo-url> && cd <your-repo>      # or: git init, add remote
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

## 1. Data, EDA, training (Tasks 1-4: 25 marks, ~40 min)

| Step | Command | Evidence |
|---|---|---|
| Download + clean | `python -m src.download_data && python -m src.data_processing` | `01_data_download.png` |
| EDA figures | `python -m src.eda` | figures already saved in `reports/figures/` |
| Train + track | `python -m src.train` | terminal summary table |
| Experiment summary | `python scripts/mlflow_summary.py` | `reports/experiment_summary.md` (commit it) |
| MLflow UI | `mlflow ui --backend-store-uri sqlite:///mlflow.db` then open http://localhost:5000 | `03_mlflow_runs_table.png`, `04_mlflow_compare_runs.png` (select the 3 child runs, click Compare), `05_mlflow_run_artifacts.png` (open one run, Artifacts tab) |
| Notebooks with outputs | `make notebooks` (or `python scripts/build_notebooks.py` then `jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb`) | commit the executed notebooks |

**Before moving on:** open `reports/figures/*.png` and read the numbers. Then write the EDA findings and the
model-selection paragraph in your own words (Report sections 3-4). Run `python -m src.train` two or three times with
a change in between (for example `--seed 7`, or edit a grid in `src/train.py`) so MLflow shows several experiments
and you have something real to discuss.

**Commit the data and model** (the rubric asks for the cleaned dataset; the Docker image needs the model):
`git add data/raw data/processed models reports`.

## 2. Tests and CI/CD (Task 5: 8 marks, ~60 min)

```bash
pytest --cov=src --cov=api --cov-report=term-missing     # screenshot: 06_pytest_coverage.png
flake8 src api tests scripts && black --check src api tests scripts
git add -A && git commit -m "MLOps assignment" && git push origin main
```

GitHub → **Actions** tab → open the run.

* `07_github_actions_success.png`: the run overview with four green jobs.
* `08_github_actions_job_logs.png`: expand the *Train* job log.
* `09_github_actions_artifacts.png`: scroll to *Artifacts* at the bottom of the run page.
* **Show that it fails on errors (required):** on a branch, break a test, push and open a PR.

```bash
git checkout -b ci-failure-demo
sed -i 's/== "ok"/== "broken"/' tests/test_api.py      # or edit any assertion by hand
git commit -am "demo: failing test" && git push -u origin ci-failure-demo
```

  Screenshot the red run as `10_github_actions_failure.png` (and that the later jobs were skipped). Then
  `git checkout main` and delete the branch, so `main` stays green.

If the first CI run fails, read the log of the failing step; the usual causes are a missing committed file
(`data/`, `models/`) or a formatting difference (`black src api tests scripts`).

## 3. Docker (Task 6: 5 marks, ~20 min)

```bash
docker build -t heart-api:1.0.0 .                         # 11_docker_build.png
docker run -d --name heart-api -p 8000:8000 heart-api:1.0.0
docker ps                                                 # shows (healthy) after ~10 s
curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d @scripts/sample_input.json
bash scripts/smoke_test.sh http://localhost:8000          # 12_docker_run_predict.png (curl + smoke test)
docker logs heart-api | tail                              # 19_api_logs.png
docker rm -f heart-api
```

PowerShell users: `curl.exe -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "@scripts/sample_input.json"`.
`scripts/smoke_test.sh` needs Git Bash or WSL on Windows.

## 4. Kubernetes (Task 7: 7 marks, ~45 min)

```bash
minikube start --driver=docker
minikube image load heart-api:1.0.0
kubectl apply -k k8s/
kubectl -n heart-mlops rollout status deployment/heart-api
kubectl -n heart-mlops get pods,svc,deploy                # 14_kubectl_get_all.png
minikube service heart-api -n heart-mlops --url           # keep open; copy the URL
bash scripts/smoke_test.sh <url>                          # 15_k8s_predict.png
kubectl -n heart-mlops describe deployment heart-api      # optional: probes and resources
```

Also show the Service (`LoadBalancer`) and that two replicas are running. Optional extras that look good:
`kubectl apply -f k8s/optional/hpa.yaml` (after `minikube addons enable metrics-server`), or the Ingress.
Docker Desktop users: see README section 7.

## 5. Monitoring (Task 8: 3 marks, ~30 min)

```bash
docker compose up --build -d
python scripts/generate_traffic.py --requests 300 --delay 0.05
```

* `17_prometheus_targets.png`: http://localhost:9090/targets (heart-api should be UP).
* `18_grafana_dashboard.png`: http://localhost:3000 (admin/admin) → *Heart Disease API*, with the time range set to
  the last 5 minutes while traffic runs.
* `19_api_logs.png`: `docker compose logs api | tail -20`.

`docker compose down` when done. If Prometheus shows the target DOWN, wait 10 s and refresh; the API must be healthy.

## 6. Report, video, final checks (Task 9 + deliverables: ~2.5 h)

1. Fill `report/REPORT.md` (every **[FILL]**), rebuild the diagram if you changed anything (`make diagram`), then
   `make report` and open `report/MLOps_Assignment1_Report.docx`. Check it is about 10 pages, images render and tables
   are readable. Export to PDF from Word if you prefer. Commit the report (remove the `report/*.docx` line from
   `.gitignore` or use `git add -f`).
2. Record the video (see `docs/VIDEO_SCRIPT.md`, 4-5 minutes). Upload it (Drive/YouTube unlisted) and put the link in
   the README and report, or commit a small `.mp4` if the course allows.
3. **Clean-room test:** clone the repo into a fresh folder, create a new venv, `pip install -r requirements.txt`,
   run `python -m src.train`, `pytest`, `docker build`. If anything fails, a grader's machine will fail too.
4. README: replace `<your-username>/<your-repo>`, add the video link, API access instructions and final metrics.
5. Push, confirm the last CI run on `main` is green, and submit the repository link (plus report and video as the
   course portal requires).

## Rubric self-check

| Marks | Task | Covered by | Done |
|---|---|---|---|
| 5 | Data + EDA | download script, cleaning, 7 figures, notebook | ☐ |
| 8 | Features + models | pipeline, 3 models, grid search, CV + metrics, documented choices | ☐ |
| 5 | Experiment tracking | MLflow params, metrics, plots, artifacts for all runs | ☐ |
| 7 | Packaging + reproducibility | joblib pipeline, pinned requirements, preprocessing inside pipeline | ☐ |
| 8 | CI/CD + tests | pytest, GitHub Actions with lint/test/train/docker, artifacts, failure demo | ☐ |
| 5 | Containerisation | Dockerfile, `/predict`, build/run proof | ☐ |
| 7 | Deployment | manifests, LoadBalancer/Ingress, screenshots | ☐ |
| 3 | Monitoring + logging | JSON logs, Prometheus, Grafana | ☐ |
| 2 | Documentation | README, ~10-page report, diagram, repo link | ☐ |

## Make it yours (the brief penalises near-identical submissions)

Everything here runs, but you must be able to explain and defend it. Cheap, genuine ways to personalise it:

* Change something in the experiment and report what happened: another model (e.g. SVM or XGBoost), different grids,
  class weights, a different decision threshold, or feature engineering such as an age band or `chol` log-transform.
* Add a real chart of your own to the EDA (for example age by sex by outcome) and discuss it.
* Change the Grafana dashboard layout or add a panel; rename the API fields or add a `/predict/batch` endpoint.
* Write the report analysis, conclusions and architecture walkthrough yourself from your own run's numbers.
* Be ready to explain: why imputation is inside the pipeline, why selection uses CV rather than the test set, what
  the readiness vs liveness probes do, and how the CI stages depend on each other.
