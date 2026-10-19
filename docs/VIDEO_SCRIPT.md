# Video script (4-5 minutes)

Screen-record (OBS, Zoom, QuickTime or Win+G). Keep the terminal font large. Speak over it; you do not need to type
live: run commands in advance and show the output, but let the viewer see the commands.

| Time | Show | Say |
|---|---|---|
| 0:00 | README top + architecture diagram | One sentence on the problem; the three lanes: experimentation, CI/CD, serving + monitoring |
| 0:30 | Repo tree in the editor | Where code, tests, k8s, monitoring and CI live |
| 0:50 | `reports/figures` (class balance, heatmap) | Two EDA findings you found |
| 1:15 | `python -m src.train` output, then MLflow UI compare view | Three models, grid search, why the winner won, hold-out numbers |
| 2:00 | GitHub Actions: green run, expand *Train*, show artifacts; the red failing run | Pipeline stages; it fails fast on test errors |
| 2:45 | `docker build`, `docker run`, `bash scripts/smoke_test.sh` | Container proof, `/predict` response with confidence |
| 3:20 | `kubectl get pods,svc`, curl through the service URL | Two replicas, probes, LoadBalancer |
| 3:50 | Grafana dashboard while `generate_traffic.py` runs; one line of JSON logs | What the panels show; logs per request |
| 4:30 | Back to README | Wrap-up: one limitation, one next step |

Tips: keep the total under 5 minutes, do one take of each segment and cut, say the cluster/tool names out loud
(Minikube, Prometheus, Grafana), and confirm the audio is audible before uploading.
