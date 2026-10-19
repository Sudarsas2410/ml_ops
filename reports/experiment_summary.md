# Experiment tracking summary

- Experiment: `heart-disease-classification`  |  tracking URI: `sqlite:///C:/Users/Sudarsan_S/Downloads/heart-disease-mlops/heart-disease-mlops/mlflow.db`
- Runs logged (all sessions): **4**  |  latest comparison run: `d2d82ff061c44f94954bd81c90d71077`
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
| gradient_boosting | 0.902 | 0.867 | 0.929 | 0.897 | 0.961 | 0.6 |
| logistic_regression ★ | 0.885 | 0.839 | 0.929 | 0.881 | 0.965 | 5.9 |
| random_forest | 0.852 | 0.806 | 0.893 | 0.847 | 0.948 | 2.0 |

## Selected hyper-parameters (GridSearchCV, refit on ROC-AUC)

- **gradient_boosting**: `max_depth=2`, `n_estimators=100`, `learning_rate=0.05`
- **logistic_regression**: `class_weight=balanced`, `C=0.1`
- **random_forest**: `max_depth=None`, `n_estimators=300`, `min_samples_leaf=5`
