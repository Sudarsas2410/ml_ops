# Experiment tracking summary

- Experiment: `heart-disease-classification`  |  tracking URI: `sqlite:////home/s2025ae05524/workspace/ml_ops/mlflow.db`
- Runs logged (all sessions): **6**  |  latest comparison run: `f11b032e9d22498a8bed2cb8c632eb39`
- Seed 42, 5-fold stratified CV, hold-out test size 0.2
- Selected model (highest CV ROC-AUC): **logistic_regression**

## Cross-validation metrics (mean ± std over folds, training split)

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
| gradient_boosting | 0.802 ± 0.028 | 0.790 ± 0.050 | 0.783 ± 0.085 | 0.874 ± 0.025 |
| logistic_regression ★ | 0.830 ± 0.017 | 0.823 ± 0.043 | 0.810 ± 0.068 | 0.903 ± 0.017 |
| random_forest | 0.810 ± 0.032 | 0.810 ± 0.055 | 0.774 ± 0.083 | 0.902 ± 0.033 |

## Hold-out test metrics (never used for tuning or selection)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Fit time (s) |
|---|---|---|---|---|---|---|
| gradient_boosting | 0.902 | 0.867 | 0.929 | 0.897 | 0.961 | 5.2 |
| logistic_regression ★ | 0.885 | 0.839 | 0.929 | 0.881 | 0.965 | 7.8 |
| random_forest | 0.852 | 0.806 | 0.893 | 0.847 | 0.948 | 19.2 |

## Selected hyper-parameters (GridSearchCV, refit on ROC-AUC)

- **gradient_boosting**: `learning_rate=0.05`, `n_estimators=100`, `max_depth=2`
- **logistic_regression**: `class_weight=balanced`, `C=0.1`
- **random_forest**: `n_estimators=300`, `max_depth=None`, `min_samples_leaf=5`
