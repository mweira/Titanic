"""
Titanic survival prediction pipeline.

Pipeline overview
------------------
1. Load and engineer features from the raw Kaggle Titanic data.
2. Compare five tree-based classifiers (Random Forest, Extra Trees, XGBoost,
   LightGBM, CatBoost) with `cross_validate`, tracking several metrics.
3. Tune the top models with Optuna (Bayesian / TPE search).
4. Combine the tuned models into a `StackingClassifier`.
5. Fit the final model, generate a submission file, and explain the model
   with feature importances and SHAP values.
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import optuna
except ImportError:  # pragma: no cover - optional dependency
    optuna = None

try:
    import shap
except ImportError:  # pragma: no cover - optional dependency
    shap = None

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.base import clone

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - optional dependency
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover - optional dependency
    CatBoostClassifier = None

warnings.filterwarnings("ignore", category=UserWarning)
if optuna is not None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

def find_data_file(filename: str, start_dir: Path | None = None) -> Path:
    start = (start_dir or Path(__file__).resolve()).resolve()
    for candidate_dir in [start, *start.parents]:
        if (candidate_dir / filename).exists():
            return candidate_dir / filename
        if (candidate_dir / "Titanic" / filename).exists():
            return candidate_dir / "Titanic" / filename

    cwd_candidate = Path.cwd() / filename
    if cwd_candidate.exists():
        return cwd_candidate

    return start.parent.parent / filename


ROOT = find_data_file("train.csv").parent
TRAIN_PATH = find_data_file("train.csv")
TEST_PATH = find_data_file("test.csv")
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 10
N_OPTUNA_TRIALS = 40
STACK_TOP_N = 3  

SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]

TREE_MODEL_NAMES = ["Random Forest", "Extra Trees", "XGBoost", "LightGBM", "CatBoost"]


# --------------------------------------------------------------------------- #
# Data loading & feature engineering
# --------------------------------------------------------------------------- #
def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test


def extract_title(name: str) -> str:
    match = re.search(r",\s*([^\.]+)\.", name)
    if match:
        title = match.group(1).strip()
        common = {
            "Mlle": "Miss",
            "Ms": "Miss",
            "Mme": "Mrs",
            "Lady": "Royalty",
            "Countess": "Royalty",
            "Sir": "Royalty",
            "Jonkheer": "Royalty",
            "Don": "Royalty",
            "Dona": "Royalty",
            "Capt": "Officer",
            "Col": "Officer",
            "Major": "Officer",
            "Dr": "Officer",
            "Rev": "Officer",
        }
        return common.get(title, title)
    return "Unknown"


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Deck"] = df["Cabin"].fillna("Unknown").astype(str).str[0]
    df["Deck"] = df["Deck"].replace("U", "Unknown")

    df["Title"] = df["Name"].apply(extract_title)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    df["FamilyGroup"] = pd.cut(df["FamilySize"], bins=[0, 1, 4, 20], labels=["Alone", "Small", "Large"])

    ticket_counts = df["Ticket"].value_counts()
    df["TicketGroup"] = df["Ticket"].map(ticket_counts)
    df["TicketGroup"] = pd.cut(df["TicketGroup"], bins=[0, 1, 4, 20], labels=["Single", "Small", "Large"])

    df["CabinKnown"] = df["Cabin"].notna().astype(int)
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]
    df["AgeClass"] = df["Age"] * df["Pclass"]
    df["Child"] = (df["Age"] < 16).astype(int)
    df["Mother"] = ((df["Sex"] == "female") & (df["Parch"] > 0) & (df["Age"] > 18) & (df["Title"] != "Miss")).astype(int)

    df["Age"] = df["Age"].fillna(df.groupby(["Title", "Sex", "Pclass"])["Age"].transform("median"))
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna("S")

    selected = [
        "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Title",
        "FamilySize", "IsAlone", "Deck", "CabinKnown", "FarePerPerson",
        "AgeClass", "Child", "Mother", "FamilyGroup", "TicketGroup",
    ]
    return df[selected]


NUMERIC_FEATURES = [
    "Age", "Fare", "Pclass", "SibSp", "Parch", "FamilySize",
    "FarePerPerson", "AgeClass", "Child", "Mother", "CabinKnown", "IsAlone",
]
CATEGORICAL_FEATURES = ["Sex", "Embarked", "Title", "Deck", "FamilyGroup", "TicketGroup"]


def build_preprocessor() -> ColumnTransformer:
    """Single preprocessor for all models in this pipeline (all tree-based:
    no scaling is needed, only imputation + one-hot encoding)."""
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline(model) -> Pipeline:
    return Pipeline([("preprocessor", build_preprocessor()), ("model", model)])


# --------------------------------------------------------------------------- #
# Baseline models & evaluation
# --------------------------------------------------------------------------- #
def build_models() -> dict:
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=500, max_depth=8, min_samples_leaf=2,
            max_features="sqrt", class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=500, max_depth=8, min_samples_leaf=2,
            max_features="sqrt", class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }

    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1,
        )
    if LGBMClassifier is not None:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=500, learning_rate=0.03, num_leaves=31,
            random_state=RANDOM_STATE, verbose=-1,
        )
    if CatBoostClassifier is not None:
        models["CatBoost"] = CatBoostClassifier(
            iterations=500, learning_rate=0.03, depth=6,
            loss_function="Logloss", verbose=False, random_state=RANDOM_STATE,
        )

    return models


def evaluate_models(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Cross-validate every candidate model on multiple metrics with
    `cross_validate` (instead of the single-metric `cross_val_score`)."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for model_name, model in build_models().items():
        pipeline = build_pipeline(model)
        cv_results = cross_validate(
            pipeline, X_train, y_train, cv=cv, scoring=SCORING,
            n_jobs=-1, return_train_score=False,
        )
        row = {"model": model_name}
        for metric in SCORING:
            scores = cv_results[f"test_{metric}"]
            row[f"{metric}_mean"] = scores.mean()
            row[f"{metric}_std"] = scores.std()
        rows.append(row)

    results = pd.DataFrame(rows).sort_values("accuracy_mean", ascending=False).reset_index(drop=True)
    return results


# --------------------------------------------------------------------------- #
# Optuna hyperparameter optimization
# --------------------------------------------------------------------------- #
def _suggest_params(trial, model_name: str) -> dict:
    if model_name == "Random Forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        }
    if model_name == "Extra Trees":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
        }
    if model_name == "XGBoost":
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
    if model_name == "LightGBM":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=100),
            "num_leaves": trial.suggest_int("num_leaves", 8, 64),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
        }
    if model_name == "CatBoost":
        return {
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "iterations": trial.suggest_int("iterations", 200, 900, step=100),
        }
    raise ValueError(f"Unknown model: {model_name}")


def _build_model_from_params(model_name: str, params: dict):
    if model_name == "Random Forest":
        return RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, **params)
    if model_name == "Extra Trees":
        return ExtraTreesClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, **params)
    if model_name == "XGBoost":
        return XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1, **params)
    if model_name == "LightGBM":
        return LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, **params)
    if model_name == "CatBoost":
        return CatBoostClassifier(loss_function="Logloss", verbose=False, random_state=RANDOM_STATE, **params)
    raise ValueError(f"Unknown model: {model_name}")


def tune_model_optuna(model_name: str, X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = N_OPTUNA_TRIALS):
    """Bayesian hyperparameter search with Optuna, maximizing mean CV accuracy."""
    if optuna is None:
        return build_models()[model_name], None

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial) -> float:
        params = _suggest_params(trial, model_name)
        model = _build_model_from_params(model_name, params)
        pipeline = build_pipeline(model)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        study_name=f"{model_name}_optuna",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_model = _build_model_from_params(model_name, study.best_params)
    return best_model, study


# --------------------------------------------------------------------------- #
# Stacking ensemble
# --------------------------------------------------------------------------- #
def build_stacking_model(tuned_models: dict) -> StackingClassifier:
    """Stack the tuned base models with a logistic-regression meta-learner."""
    estimators = [(name.lower().replace(" ", "_"), clone(model)) for name, model in tuned_models.items()]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        cv=5,
        n_jobs=-1,
        passthrough=False,
    )


# --------------------------------------------------------------------------- #
# Interpretation: feature importance & SHAP
# --------------------------------------------------------------------------- #
def plot_feature_importance(pipeline: Pipeline, model_name: str, top_n: int = 15):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    fitted_model = pipeline.named_steps["model"]

    if hasattr(fitted_model, "feature_importances_"):
        values = fitted_model.feature_importances_
        label = "Importance"
    elif hasattr(fitted_model, "coef_"):
        values = np.abs(fitted_model.coef_).ravel()
        label = "Absolute coefficient"
    else:
        return None

    importance = (
        pd.DataFrame({"Feature": feature_names, label: values})
        .sort_values(label, ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(importance["Feature"][::-1], importance[label][::-1], color="#6f7db8")
    ax.set_title(f"Top {top_n} features — {model_name}", fontweight="bold")
    ax.set_xlabel(label)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)
    return importance


def plot_shap_summary(pipeline: Pipeline, X_sample: pd.DataFrame, max_display: int = 15):
    """SHAP summary plot for the fitted tree model inside the pipeline."""
    if shap is None:
        print("SHAP is not installed; skipping explanation plot.")
        return None

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X_transformed = preprocessor.transform(X_sample)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    feature_names = preprocessor.get_feature_names_out()

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)
        if isinstance(shap_values, list):  # binary classifiers may return [class0, class1]
            shap_values = shap_values[1]
    except Exception as exc:  # pragma: no cover - defensive, model may not be tree-based
        print(f"SHAP explanation skipped: {exc}")
        return None

    shap.summary_plot(
        shap_values, X_transformed, feature_names=feature_names,
        max_display=max_display, show=False,
    )
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "shap_summary.png", dpi=150)
    plt.close(fig)
    return shap_values


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def train_and_predict():
    train, test = load_data()

    X_train = prepare_features(train)
    y_train = train["Survived"]
    X_test = prepare_features(test)

    # 1) Baseline comparison across candidate models
    print("\n===== BASELINE CROSS-VALIDATION (cross_validate, multiple metrics) =====\n")
    results = evaluate_models(X_train, y_train)
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(results[["model", "accuracy_mean", "precision_mean", "recall_mean", "f1_mean", "roc_auc_mean"]])

    top_models = results["model"].head(STACK_TOP_N).tolist()
    print(f"\nModels selected for Optuna tuning + stacking: {top_models}")

    # 2) Optuna tuning of the top models
    tuned_models = {}
    for model_name in top_models:
        print(f"\n--- Tuning {model_name} with Optuna ({N_OPTUNA_TRIALS} trials) ---")
        best_model, study = tune_model_optuna(model_name, X_train, y_train)
        if study is None:
            print("Optuna is not installed; using default model configuration.")
        else:
            print(f"Best CV accuracy: {study.best_value:.4f} | Best params: {study.best_params}")
        tuned_models[model_name] = best_model

    # 3) Stacking ensemble of the tuned models
    print("\n===== STACKING CLASSIFIER =====\n")
    stacking_model = build_stacking_model(tuned_models)
    stacking_pipeline = build_pipeline(stacking_model)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    stack_scores = cross_validate(stacking_pipeline, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)
    print(f"Stacking accuracy: {stack_scores['test_accuracy'].mean():.4f} ± {stack_scores['test_accuracy'].std():.4f}")
    print(f"Stacking ROC AUC:  {stack_scores['test_roc_auc'].mean():.4f} ± {stack_scores['test_roc_auc'].std():.4f}")

    best_single_acc = results["accuracy_mean"].iloc[0]
    final_choice = "stacking" if stack_scores["test_accuracy"].mean() >= best_single_acc else "best_single"

    if final_choice == "stacking":
        print("\nStacking ensemble outperforms (or matches) the best single model — using it for the submission.")
        final_pipeline = stacking_pipeline
        final_name = "Stacking Ensemble"
    else:
        best_name = results["model"].iloc[0]
        print(f"\nBest single tuned model ({best_name}) outperforms the stack — using it for the submission.")
        final_pipeline = build_pipeline(tuned_models.get(best_name, build_models()[best_name]))
        final_name = best_name

    # 4) Fit final model & predict
    final_pipeline.fit(X_train, y_train)
    predictions = final_pipeline.predict(X_test)

    submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": predictions.astype(int)})
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSubmission saved to {OUTPUT_PATH}")

    # 5) Interpretation: feature importance + SHAP
    #    (SHAP needs a single tree model, so use the best individual tuned model
    #    even if the stack was chosen for the submission.)
    interp_name = results["model"].iloc[0]
    interp_model = tuned_models.get(interp_name, build_models()[interp_name])
    interp_pipeline = build_pipeline(interp_model)
    interp_pipeline.fit(X_train, y_train)

    print(f"\nGenerating feature importance and SHAP plots for {interp_name}...")
    plot_feature_importance(interp_pipeline, interp_name)
    plot_shap_summary(interp_pipeline, X_train.sample(min(200, len(X_train)), random_state=RANDOM_STATE))
    print(f"Figures saved to {FIGURES_DIR}")

    print(f"\nFinal model used for submission: {final_name}")


if __name__ == "__main__":
    train_and_predict()
