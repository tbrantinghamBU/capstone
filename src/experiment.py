import time
import os
import uuid
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    f1_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split
from utils import log_results_to_csv


def run_experiment(model, X, y, preprocessor, mend_results, df_full=None,
                   problem_type="regression", random_state=42,
                   log_file="/logs/mend_results_v3.csv",
                   cv_log_file="/logs/cv_results_v3.csv",
                   sample_frac=1.0):
    """
    Generic experiment runner for regression or classification models.
    Logs metrics, metadata, hyperparameters, and (if applicable) CV results.
    """

    start_time = time.perf_counter()
    run_id = str(uuid.uuid4())

    # Optionally shrink dataset
    if sample_frac < 1.0:
        X, _, y, _ = train_test_split(
            X, y,
            train_size=sample_frac,
            stratify=y if problem_type == "classification" else None,
            random_state=random_state
        )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state,
        stratify=y if problem_type == "classification" else None
    )

    # Build pipeline
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Fit
    pipe.fit(X_train, y_train)

    # Predictions
    y_pred_train = pipe.predict(X_train)
    y_pred_test = pipe.predict(X_test)

    # Base results
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),   # NEW
        "run_id": run_id,
        "model_type": type(model).__name__,                # NEW explicit field
        "row_count": len(df_full) if df_full is not None else len(X),
        "sampled_row_count": len(X),
        "feature_count": X.shape[1],
        "seed": random_state,
        "sample_frac": sample_frac
    }

    # Regression metrics
    if problem_type == "regression":
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)

        # Adjusted R²
        p = pipe.named_steps["preprocessor"].transform(X_train).shape[1]
        n = len(y_test)
        adjusted_r2 = 1 - (1 - test_r2) * (n - 1) / (n - p - 1)

        results.update({
            "train_r2": train_r2,
            "train_rmse": train_rmse,
            "train_mae": train_mae,
            "test_r2": test_r2,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
            "adjusted_r2": adjusted_r2
        })

    # Classification metrics
    elif problem_type == "classification":
        results.update({
            "train_accuracy": accuracy_score(y_train, y_pred_train),
            "test_accuracy": accuracy_score(y_test, y_pred_test),
            "train_balanced_accuracy": balanced_accuracy_score(y_train, y_pred_train),
            "test_balanced_accuracy": balanced_accuracy_score(y_test, y_pred_test),
            "train_precision": precision_score(y_train, y_pred_train, average="weighted", zero_division=0),
            "test_precision": precision_score(y_test, y_pred_test, average="weighted", zero_division=0),
            "train_recall": recall_score(y_train, y_pred_train, average="weighted", zero_division=0),
            "test_recall": recall_score(y_test, y_pred_test, average="weighted", zero_division=0),
            "train_f1": f1_score(y_train, y_pred_train, average="weighted"),
            "test_f1": f1_score(y_test, y_pred_test, average="weighted"),
        })

        # AUC if available
        if hasattr(pipe.named_steps["model"], "predict_proba"):
            try:
                y_proba_train = pipe.predict_proba(X_train)[:, 1]
                y_proba_test = pipe.predict_proba(X_test)[:, 1]
                results["train_auc"] = roc_auc_score(y_train, y_proba_train)
                results["test_auc"] = roc_auc_score(y_test, y_proba_test)
            except Exception:
                pass

    # Timing
    train_time = time.perf_counter() - start_time
    results["train_time_sec"] = train_time

    # Hyperparameters
    if hasattr(model, "best_estimator_"):
        params = model.best_estimator_.get_params()
        results["best_params"] = str(model.best_params_)
        results["best_score"] = model.best_score_
    else:
        params = model.get_params()

    for k, v in params.items():
        results[f"param_{k}"] = v

    # If CV results exist, log them separately
    if hasattr(model, "cv_results_"):
        cv_df = pd.DataFrame(model.cv_results_)

        # Add metadata
        cv_df["model_type"] = type(model).__name__
        cv_df["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        cv_df["run_id"] = run_id

        # Ensure directory exists
        os.makedirs(os.path.dirname(cv_log_file), exist_ok=True)

        # If file exists, align columns with existing header
        if os.path.exists(cv_log_file):
            existing = pd.read_csv(cv_log_file, nrows=0)  # just header
            existing_cols = list(existing.columns)

            # Add any new columns at the end
            new_cols = [c for c in cv_df.columns if c not in existing_cols]
            ordered_cols = existing_cols + new_cols

            # Reorder and fill missing
            cv_df = cv_df.reindex(columns=ordered_cols)
            header = False
        else:
            # First write: timestamp + model_type first, then rest alphabetically
            base_cols = ["timestamp", "model_type"]
            other_cols = sorted([c for c in cv_df.columns if c not in base_cols])
            ordered_cols = base_cols + other_cols
            cv_df = cv_df[ordered_cols]
            header = True

        # Append safely
        cv_df.to_csv(cv_log_file, mode="a", header=header, index=False)

    # Append and log
    mend_results.append(results)
    log_results_to_csv(mend_results, log_file)

    return results