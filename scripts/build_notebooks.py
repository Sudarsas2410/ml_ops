"""Generate the three notebooks in notebooks/ from source (keeps them in sync with src/).

Usage:
    python scripts/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb   # fills in outputs
(or simply `make notebooks`).
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

OUT = Path(__file__).resolve().parents[1] / "notebooks"

SETUP = dedent("""\
    import sys, warnings
    from pathlib import Path

    here = Path.cwd().resolve()
    ROOT = next((p for p in [here, *here.parents] if (p / "src" / "config.py").exists()), here)
    sys.path.insert(0, str(ROOT))
    warnings.filterwarnings("ignore")

    import json
    import pandas as pd
    from IPython.display import Image, display

    from src import config
    pd.set_option("display.max_columns", 30)
    """)


def md(text: str):
    return new_markdown_cell(dedent(text).strip())


def code(text: str):
    return new_code_cell(dedent(text).strip())


def save(name: str, cells: list) -> None:
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    OUT.mkdir(exist_ok=True)
    nbformat.write(nb, OUT / name)
    print(f"wrote notebooks/{name}")


def eda_notebook() -> None:
    cells = [
        md("""
            # 01 · Data acquisition and exploratory data analysis

            **Dataset:** UCI Heart Disease (Cleveland), 303 patients, 13 clinical features plus a 0-4
            diagnosis (`num`). **Goal of this notebook:** obtain the data, document every cleaning
            decision, and understand the features before modelling.

            The cleaning and plotting logic lives in `src/data_processing.py` and `src/eda.py`, so the
            notebook, the CI pipeline and the tests all use identical code.
            """),
        code(SETUP),
        md("## 1. Acquire the data\nRun once from the repo root: `python -m src.download_data`."),
        code("""
            from src.download_data import download
            from src.data_processing import load_raw, clean, save_clean, missing_value_report

            raw_path = download()           # no-op if the file is already there
            df_raw = load_raw(raw_path)
            print("shape:", df_raw.shape)
            df_raw.head()
            """),
        code("df_raw.dtypes.to_frame('dtype').T"),
        md("""
            ## 2. Data quality

            `?` marks missing values in the source file. They are parsed as `NaN`.
            """),
        code("missing_value_report(df_raw)"),
        md("""
            ## 3. Cleaning decisions

            | Issue | Decision | Why |
            |---|---|---|
            | `?` in `ca`, `thal` | Parsed as NaN, then **mode imputation** | Categorical codes: mode keeps valid category values; only a handful of rows affected |
            | Missing numeric values (none expected) | Median imputation | Robust to the skew in `chol` / `oldpeak` |
            | `num` is 0-4 (severity) | Binarise: `target = (num > 0)` | Task asks for presence / absence |
            | Duplicates | Dropped | Prevent leakage between CV folds |
            | Implausible ranges | Masked to NaN, then imputed | Defensive; also enforced in the API |

            The scikit-learn pipeline **repeats imputation inside cross-validation**, so no statistic from a
            validation fold leaks into training. The saved CSV is imputed only for convenience and EDA.
            """),
        code("""
            df = clean(df_raw)
            save_clean(df)
            print("clean shape:", df.shape, "| missing values left:", int(df.isna().sum().sum()))
            df.target.value_counts().rename({0: "no disease", 1: "disease"})
            """),
        code("df.describe().T.round(2)"),
        md("## 4. Visual EDA"),
        code("""
            from src import eda
            eda._style()
            raw_missing = eda.plot_missing_values(df_raw)
            if raw_missing: display(Image(filename=str(raw_missing)))
            display(Image(filename=str(eda.plot_class_balance(df))))
            """),
        md(
            "**Class balance.** The classes are close to balanced, so accuracy is meaningful, but recall is still the clinically important metric (missing a sick patient is costlier than a false alarm)."
        ),
        code("display(Image(filename=str(eda.plot_numeric_histograms(df))))"),
        code("display(Image(filename=str(eda.plot_boxplots(df))))"),
        code("display(Image(filename=str(eda.plot_categorical_counts(df))))"),
        code("display(Image(filename=str(eda.plot_correlation_heatmap(df))))"),
        code("display(Image(filename=str(eda.plot_target_correlation(df))))"),
        md("""
            ## 5. Summary statistics for the report
            """),
        code("""
            summary = eda.summarise(df_raw, df)
            print(json.dumps({k: v for k, v in summary.items() if k != "numeric_describe"}, indent=2))
            """),
        md("""
            ## 6. Observations (confirm against your own plots before quoting them)

            Patterns widely reported for the Cleveland data, which the figures above should show:

            * Patients with disease tend to have **lower maximum heart rate (`thalach`)** and **higher ST depression (`oldpeak`)**.
            * **Asymptomatic chest pain (`cp = 4`)**, **exercise-induced angina (`exang = 1`)**, a **reversible thalassemia defect (`thal = 7`)** and more **fluoroscopy-coloured vessels (`ca > 0`)** are associated with disease.
            * `chol` and `fbs` are only weakly related to the target on their own.
            * Features are only moderately correlated with each other, so multicollinearity is not a concern for logistic regression.

            **Implications for modelling:** scale continuous features (needed for logistic regression),
            one-hot encode the categorical codes (they are not ordinal: `thal` 3/6/7 is a label), and
            evaluate with stratified cross-validation because the dataset is small.
            """),
    ]
    save("01_eda.ipynb", cells)


def model_notebook() -> None:
    cells = [
        md("""
            # 02 · Feature engineering and model development

            Production training is `python -m src.train` (tracked in MLflow). This notebook documents the
            **reasoning**: baseline comparison, the tuning grids, and the final results.
            """),
        code(SETUP),
        md("""
            ## 1. Feature engineering

            * **Continuous** (`age, trestbps, chol, thalach, oldpeak`): median imputation + `StandardScaler`.
            * **Categorical** (`sex, cp, fbs, restecg, exang, slope, ca, thal`): most-frequent imputation + one-hot (`handle_unknown="ignore"` so unseen codes cannot crash serving).
            * Everything lives in one `ColumnTransformer` inside the model `Pipeline`, so training and inference are identical.
            """),
        code("""
            from src.data_processing import prepare_dataset, split_xy, build_preprocessor
            from sklearn.pipeline import Pipeline

            df = prepare_dataset()
            X, y = split_xy(df)
            pre = build_preprocessor().fit(X)
            print("input features:", X.shape[1], "-> model features:", len(pre.get_feature_names_out()))
            list(pre.get_feature_names_out())
            """),
        md("""
            ## 2. Baseline comparison (default hyper-parameters, 5-fold stratified CV)

            A quick sanity check before tuning: which model families look promising on this data?
            """),
        code("""
            from sklearn.model_selection import StratifiedKFold, cross_validate
            from src.train import model_specs, SCORING

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
            rows = []
            for name, (estimator, _) in model_specs(config.RANDOM_STATE).items():
                pipe = Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
                scores = cross_validate(pipe, X, y, cv=cv, scoring=SCORING)
                rows.append({"model": name, **{m: scores[f"test_{m}"].mean() for m in SCORING}})
            pd.DataFrame(rows).set_index("model").round(3)
            """),
        md("""
            ## 3. Tuning

            `GridSearchCV` over the grids below, 5-fold stratified CV, refit on **ROC-AUC** (threshold-free
            and robust to mild imbalance). The winner is chosen on CV ROC-AUC only; the 20 % hold-out set is
            used once for the final, unbiased estimate.
            """),
        code("""
            for name, (_, grid) in model_specs(config.RANDOM_STATE).items():
                print(name)
                for param, values in grid.items():
                    print("   ", param.removeprefix("model__"), "->", values)
            """),
        md("""
            ## 4. Results from the tracked training run

            Run `python -m src.train` first. The numbers below are read from `models/metrics.json`.
            """),
        code("""
            metrics_path = config.METRICS_FILE
            if metrics_path.exists():
                summary = json.loads(metrics_path.read_text())
                print("Selected model:", summary["best_model"], "| params:", summary["best_params"])
                display(pd.DataFrame(summary["all_models"]).T)
                display(pd.DataFrame({"CV (mean)": summary["cv"], "CV (std)": summary["cv_std"]}).T.round(3))
                display(pd.Series(summary["test"], name="hold-out test").to_frame().T)
            else:
                print("No models/metrics.json yet: run `python -m src.train`")
            """),
        code("""
            train_figs = config.FIGURES_DIR / "training"
            for name in ["model_comparison.png", "roc_overlay.png"]:
                if (train_figs / name).exists():
                    display(Image(filename=str(train_figs / name)))
            """),
        code("""
            if metrics_path.exists():
                best = summary["best_model"]
                for name in [f"confusion_{best}.png", f"importance_{best}.png"]:
                    if (train_figs / name).exists():
                        display(Image(filename=str(train_figs / name)))
            """),
        md("""
            ## 5. Interpretation checklist (write your own conclusions here)

            * Which model won, and is the gap to the others larger than the CV standard deviation?
            * Does the hold-out ROC-AUC agree with CV? A large gap would signal overfitting or an unlucky split.
            * Recall on the disease class: how many sick patients are missed? Would you lower the decision threshold in a clinical setting?
            * Which features dominate the importance plot, and does that match the EDA?
            """),
    ]
    save("02_model_development.ipynb", cells)


def inference_notebook() -> None:
    cells = [
        md("""
            # 03 · Inference

            Load the packaged pipeline (`models/model.joblib`) and score patients. The same
            `src.predict.predict` function powers the API, so results here match the service exactly.
            """),
        code(SETUP),
        code("""
            from src.predict import load_model, predict

            model = load_model()
            print(type(model).__name__, "->", [name for name, _ in model.steps])
            """),
        md("## Single patient (the sample from the API docs)"),
        code("predict(model, config.SAMPLE_PATIENT)"),
        md(
            "## Several patients from the cleaned dataset\n*(Illustration only: the model has seen most of these rows during training.)*"
        ),
        code("""
            sample = pd.read_csv(config.CLEAN_FILE).sample(8, random_state=0)
            preds = pd.DataFrame(predict(model, sample[config.FEATURES].to_dict("records")), index=sample.index)
            pd.concat([sample[["age", "sex", "cp", "thalach", "target"]], preds], axis=1).rename(columns={"target": "actual"})
            """),
        md("## Calling the running API\nStart it with `uvicorn api.main:app` or Docker, then run the cell."),
        code("""
            import json, urllib.request, urllib.error

            def call_api(payload, url="http://localhost:8000/predict"):
                req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                             headers={"Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(req, timeout=5) as r:
                        return json.loads(r.read())
                except (urllib.error.URLError, OSError) as exc:
                    return f"API not reachable ({exc}). Start it with: uvicorn api.main:app"

            call_api(config.SAMPLE_PATIENT)
            """),
        md("""
            ## Input contract

            The API rejects invalid input with HTTP 422 (see `PatientFeatures` in `api/main.py`).
            """),
        code("pd.Series(config.FEATURE_DESCRIPTIONS, name='description').to_frame()"),
    ]
    save("03_inference.ipynb", cells)


if __name__ == "__main__":
    eda_notebook()
    model_notebook()
    inference_notebook()
