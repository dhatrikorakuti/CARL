"""
actions/ml_pipeline.py — Complete Autonomous ML Pipeline for CARL.

Executes the ENTIRE machine learning workflow end-to-end:
Load → Clean → Engineer → Select → Encode → Split → Scale → Train → Evaluate → Save

Never stops mid-pipeline. Recovers from errors. Reports progress.
Every model failure is caught and skipped — pipeline continues.
"""
from __future__ import annotations

import json
import time
import traceback
import warnings
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

_RESULTS: dict[str, Any] = {}


def ml_pipeline(parameters: dict, player=None, speak=None) -> str:
    """Full autonomous ML pipeline. Never stops until complete."""
    path = parameters.get("path", "")
    goal = parameters.get("goal", "classification")
    target_col = parameters.get("target", "")

    if not path:
        return "Please provide the dataset path."

    df = _load(path)
    if df is None:
        return f"Cannot load: {path}"

    progress: list[str] = []
    start_time = time.time()

    def log(msg: str):
        progress.append(msg)
        print(f"[ML] {msg}")
        if speak:
            try:
                speak(msg)
            except Exception:
                pass

    log(f"✔ Dataset Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ── Step 1: Detect target ─────────────────────────────────────────────
    if not target_col:
        target_col = _detect_target(df, goal)
    if not target_col or target_col not in df.columns:
        return f"Cannot detect target column. Available: {list(df.columns)}"
    log(f"✔ Target: {target_col} ({df[target_col].nunique()} classes)")

    # ── Step 2: Determine task ────────────────────────────────────────────
    task = _determine_task(df, target_col, goal)
    log(f"✔ Task: {task}")

    # ── Step 3: Clean ─────────────────────────────────────────────────────
    df = _clean(df, target_col, progress)
    log(f"✔ Cleaning Completed ({df.shape[0]:,} rows remaining)")

    # ── Step 4: Feature Engineering ───────────────────────────────────────
    df = _engineer_features(df, target_col)
    log(f"✔ Feature Engineering Completed ({df.shape[1]} columns)")

    # ── Step 5: Encode ────────────────────────────────────────────────────
    df, encoders = _encode(df, target_col)
    log(f"✔ Encoding Completed")

    # ── Step 6: Split ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = _split(df, target_col)
    log(f"✔ Train/Test Split: {len(X_train)} train, {len(X_test)} test")

    # ── Step 7: Scale ─────────────────────────────────────────────────────
    X_train, X_test, scaler = _scale(X_train, X_test)
    log(f"✔ Scaling Completed (StandardScaler)")

    # ── Step 8: Train Models ──────────────────────────────────────────────
    log(f"[W] Training Models...")
    reasoning, models = _select_models_intelligently(X_train, y_train, task, goal)
    log(f"✔ Model Selection Reasoning:")
    for r in reasoning:
        log(f"  {r}")
    results = _train_all(models, X_train, y_train, X_test, y_test, task, progress)
    log(f"✔ {len(results)} Models Trained")

    # ── Step 9: Evaluate ──────────────────────────────────────────────────
    log(f"[W] Evaluating...")
    best = _select_best(results, task)
    log(f"✔ Best Model: {best['name']} ({best['score']:.2%})")

    # ── Step 10: Save ─────────────────────────────────────────────────────
    save_dir = Path(path).parent / "carl_ml_output"
    save_dir.mkdir(exist_ok=True)
    _save_artifacts(best, scaler, encoders, X_train.columns.tolist(), save_dir, df, target_col, path)
    log(f"✔ Artifacts Saved: {save_dir}")

    # ── Final Report ──────────────────────────────────────────────────────
    duration = time.time() - start_time
    report = _build_report(best, results, task, duration, df, target_col)
    _RESULTS[path] = {"report": report, "best": best, "results": results,
                      "target": target_col, "task": task, "features": X_train.columns.tolist()}

    # Store in memory for future reference
    try:
        from memory.cognitive import CognitiveMemory
        brain = CognitiveMemory()
        brain.record_milestone(
            f"ML Pipeline: {best['name']} achieved {best['score']:.2%} on {Path(path).name} ({task})",
            project=""
        )
    except Exception:
        pass

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE STAGES
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_target(df: pd.DataFrame, goal: str) -> str:
    goal_lower = goal.lower()
    # Check explicit target names
    for col in df.columns:
        cl = col.lower()
        if cl in ("target", "label", "outcome", "survived", "class", "y"):
            return col
        if cl in goal_lower:
            return col
    # Binary column with lowest cardinality
    binary_cols = [c for c in df.columns if df[c].nunique() == 2]
    if binary_cols:
        return binary_cols[-1]  # Last binary column is often target
    return df.columns[-1]


def _determine_task(df: pd.DataFrame, target: str, goal: str) -> str:
    if "regress" in goal.lower() or "forecast" in goal.lower() or "price" in goal.lower():
        return "regression"
    if "cluster" in goal.lower() or "segment" in goal.lower():
        return "clustering"
    if df[target].nunique() <= 20:
        return "classification"
    return "regression"


def _clean(df: pd.DataFrame, target: str, progress: list) -> pd.DataFrame:
    # Remove duplicates
    dups = df.duplicated().sum()
    if dups > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        progress.append(f"  Removed {dups} duplicates")

    # Remove constant columns
    const = [c for c in df.columns if df[c].nunique() <= 1 and c != target]
    if const:
        df = df.drop(columns=const)
        progress.append(f"  Removed {len(const)} constant columns")

    # Handle missing values
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count == 0:
            continue
        null_pct = null_count / len(df) * 100
        if null_pct > 60 and col != target:
            df = df.drop(columns=[col])
            progress.append(f"  Dropped {col} ({null_pct:.0f}% missing)")
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode = df[col].mode()
            df[col] = df[col].fillna(mode.iloc[0] if len(mode) > 0 else "Unknown")

    # Handle impossible zeros in medical columns
    medical_cols = [c for c in df.columns if c.lower() in
                   ("glucose", "bloodpressure", "skinthickness", "insulin", "bmi")]
    for col in medical_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            zeros = (df[col] == 0).sum()
            if zeros > 0:
                df[col] = df[col].replace(0, np.nan)
                df[col] = df[col].fillna(df[col].median())
                progress.append(f"  Fixed {zeros} impossible zeros in {col}")

    # Drop rows where target is missing
    if df[target].isnull().sum() > 0:
        df = df.dropna(subset=[target])

    return df


def _engineer_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    # Age groups
    for col in df.columns:
        if 'age' in col.lower() and pd.api.types.is_numeric_dtype(df[col]) and col != target:
            df[f'{col}_group'] = pd.cut(df[col], bins=[0,18,35,50,65,100], labels=[0,1,2,3,4])
    return df


def _encode(df: pd.DataFrame, target: str) -> tuple:
    from sklearn.preprocessing import LabelEncoder
    encoders = {}
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if col == target:
            continue
        if df[col].nunique() <= 10:
            # One-hot
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        else:
            # Label encode high cardinality
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    # Encode target if categorical
    if df[target].dtype == object:
        le = LabelEncoder()
        df[target] = le.fit_transform(df[target])
        encoders['_target'] = le
    return df, encoders


def _split(df: pd.DataFrame, target: str):
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=[target])
    y = df[target]
    # Remove any remaining non-numeric
    X = X.select_dtypes(include=[np.number])
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() <= 20 else None)


def _scale(X_train, X_test):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return X_train_s, X_test_s, scaler


def _select_models_intelligently(X_train, y_train, task: str, goal: str) -> tuple[list[str], dict]:
    """
    Think like a Senior ML Engineer: analyze dataset characteristics,
    then choose models that FIT this specific data. Explain why.
    """
    reasoning = []
    n_samples = len(X_train)
    n_features = X_train.shape[1]
    n_classes = y_train.nunique() if task == "classification" else 0

    # Dataset characteristics
    is_small = n_samples < 1000
    is_large = n_samples > 50000
    is_high_dim = n_features > 50
    is_imbalanced = False
    if task == "classification":
        class_ratio = y_train.value_counts().min() / y_train.value_counts().max()
        is_imbalanced = class_ratio < 0.3

    reasoning.append(f"Dataset: {n_samples} samples, {n_features} features")
    if is_small:
        reasoning.append("Small dataset → regularized models preferred, avoid complex ensembles overfitting")
    if is_large:
        reasoning.append("Large dataset → tree ensembles and gradient boosting excel")
    if is_imbalanced:
        reasoning.append(f"Imbalanced classes (ratio {class_ratio:.2f}) → need class_weight='balanced'")

    # Get all models
    all_models = _get_models(task)
    selected = {}

    if task == "classification":
        # Always include: interpretable baseline
        reasoning.append("Logistic Regression: interpretable baseline, fast, good for linear boundaries")
        selected["Logistic Regression"] = all_models["Logistic Regression"]

        # Random Forest: almost always appropriate
        reasoning.append("Random Forest: handles nonlinear interactions, robust to outliers, low overfitting")
        selected["Random Forest"] = all_models["Random Forest"]

        # Gradient Boosting: strong performer on tabular data
        if not is_small:
            reasoning.append("Gradient Boosting: strong on tabular data, sequential error correction")
            selected["Gradient Boosting"] = all_models["Gradient Boosting"]

        # XGBoost: top performer on structured data
        if "XGBoost" in all_models:
            reasoning.append("XGBoost: regularized boosting, handles missing values, typically top performer")
            selected["XGBoost"] = all_models["XGBoost"]

        # LightGBM: fast for large datasets
        if "LightGBM" in all_models and is_large:
            reasoning.append("LightGBM: histogram-based, faster than XGBoost on large data")
            selected["LightGBM"] = all_models["LightGBM"]

        # SVM: good for small-medium datasets with clear margins
        if is_small and not is_high_dim:
            reasoning.append("SVM: effective on small datasets with clear class separation")
            selected["SVM"] = all_models["SVM"]
        elif "SVM" in all_models:
            reasoning.append("SVM: SKIPPED — scales poorly with large datasets (O(n²))")

        # KNN: only for small datasets
        if is_small:
            reasoning.append("KNN: distance-based, works well on small clean datasets")
            selected["KNN"] = all_models["KNN"]
        else:
            reasoning.append("KNN: SKIPPED — too slow for large datasets, sensitive to dimensions")

        # Naive Bayes: quick baseline, works if features are independent
        if n_features < 20:
            reasoning.append("Naive Bayes: fast probabilistic baseline, assumes feature independence")
            selected["Naive Bayes"] = all_models["Naive Bayes"]

        # Decision Tree: for interpretability comparison
        reasoning.append("Decision Tree: fully interpretable, useful for explaining decisions")
        selected["Decision Tree"] = all_models["Decision Tree"]

        # Handle imbalanced data
        if is_imbalanced:
            reasoning.append("Applying class_weight='balanced' to handle imbalanced classes")
            if "Logistic Regression" in selected:
                selected["Logistic Regression"].set_params(class_weight='balanced')
            if "Random Forest" in selected:
                selected["Random Forest"].set_params(class_weight='balanced')
            if "SVM" in selected:
                selected["SVM"].set_params(class_weight='balanced')

    else:  # regression
        reasoning.append("Linear Regression: interpretable baseline, fast")
        selected["Linear Regression"] = all_models["Linear Regression"]
        reasoning.append("Ridge: regularized linear, prevents overfitting on correlated features")
        selected["Ridge"] = all_models["Ridge"]
        reasoning.append("Random Forest: captures nonlinear relationships without assumptions")
        selected["Random Forest"] = all_models["Random Forest"]
        reasoning.append("Gradient Boosting: sequential error correction, strong on tabular")
        selected["Gradient Boosting"] = all_models["Gradient Boosting"]
        if "XGBoost" in all_models:
            reasoning.append("XGBoost: regularized boosting with built-in feature selection")
            selected["XGBoost"] = all_models["XGBoost"]

    reasoning.append(f"Selected {len(selected)} models based on dataset characteristics")
    return reasoning, selected



    if task == "classification":
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.naive_bayes import GaussianNB
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "AdaBoost": AdaBoostClassifier(random_state=42, algorithm='SAMME'),
            "KNN": KNeighborsClassifier(),
            "SVM": SVC(probability=True, random_state=42),
            "Naive Bayes": GaussianNB(),
        }
        # Try XGBoost/LightGBM if installed
        try:
            from xgboost import XGBClassifier
            models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, verbosity=0)
        except ImportError:
            pass
        try:
            from lightgbm import LGBMClassifier
            models["LightGBM"] = LGBMClassifier(random_state=42, verbosity=-1)
        except ImportError:
            pass
        return models
    else:  # regression
        from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "ElasticNet": ElasticNet(),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        }
        try:
            from xgboost import XGBRegressor
            models["XGBoost"] = XGBRegressor(random_state=42, verbosity=0)
        except ImportError:
            pass
        return models


def _train_all(models: dict, X_train, y_train, X_test, y_test, task: str, progress: list) -> list:
    from sklearn.model_selection import cross_val_score
    results = []

    for name, model in models.items():
        try:
            t0 = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - t0

            # Score
            if task == "classification":
                from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                try:
                    if hasattr(model, 'predict_proba'):
                        y_prob = model.predict_proba(X_test)
                        if y_prob.shape[1] == 2:
                            auc = roc_auc_score(y_test, y_prob[:, 1])
                        else:
                            auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
                    else:
                        auc = 0.0
                except Exception:
                    auc = 0.0
                score = acc
                metrics = {"accuracy": acc, "f1": f1, "roc_auc": auc}
            else:
                from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                score = r2
                metrics = {"r2": r2, "rmse": rmse, "mae": mae}

            # Cross validation
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy' if task == 'classification' else 'r2')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
            except Exception:
                cv_mean, cv_std = score, 0.0

            results.append({
                "name": name, "model": model, "score": score,
                "cv_mean": cv_mean, "cv_std": cv_std,
                "metrics": metrics, "train_time": train_time,
            })
            progress.append(f"  ✔ {name}: {score:.2%} (CV: {cv_mean:.2%} ± {cv_std:.2%}) [{train_time:.1f}s]")

        except Exception as e:
            progress.append(f"  [X] {name}: FAILED ({str(e)[:50]})")
            print(f"[ML] {name} failed: {e}")

    return results


def _select_best(results: list, task: str) -> dict:
    if not results:
        return {"name": "None", "score": 0, "metrics": {}, "model": None}
    # Sort by CV score (more reliable than test score)
    results.sort(key=lambda r: r["cv_mean"], reverse=True)
    return results[0]


def _save_artifacts(best, scaler, encoders, feature_names, save_dir, df, target, orig_path):
    import pickle
    try:
        if best["model"]:
            with open(save_dir / "best_model.pkl", "wb") as f:
                pickle.dump(best["model"], f)
        with open(save_dir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        with open(save_dir / "feature_names.json", "w") as f:
            json.dump(feature_names, f)
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(best.get("metrics", {}), f, indent=2, default=str)
        # Save cleaned dataset
        df.to_csv(save_dir / "cleaned_dataset.csv", index=False)
    except Exception as e:
        print(f"[ML] Save error: {e}")


def _build_report(best, results, task, duration, df, target) -> str:
    """Comprehensive ML Consultant Report — 9 phases of analysis."""
    lines = [
        "═" * 50,
        "CARL — ML ANALYSIS REPORT",
        "═" * 50, "",
        "PHASE 1 — DATASET UNDERSTANDING", "─" * 40,
        f"  Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns",
        f"  Problem: {task.upper()}",
        f"  Target: {target} ({df[target].nunique()} classes)",
        f"  Numerical: {len(df.select_dtypes(include=[np.number]).columns)}",
        f"  Missing: {int(df.isnull().sum().sum())}",
        f"  Duplicates: {int(df.duplicated().sum())}", "",
        "PHASE 2 — MODEL COMPARISON", "─" * 40,
        f"  {'Model':<25} {'Score':>7} {'CV':>7} {'Time':>6}",
    ]
    for r in sorted(results, key=lambda x: -x["cv_mean"]):
        lines.append(f"  {r['name']:<25} {r['score']:.2%} {r['cv_mean']:.2%} {r['train_time']:.1f}s")
    lines.append("")
    lines.append("PHASE 3 — BEST MODEL JUSTIFICATION")
    lines.append("─" * 40)
    lines.append(f"  Selected: {best['name']} ({best['score']:.2%})")
    lines.append(f"  CV: {best.get('cv_mean',0):.2%} ± {best.get('cv_std',0):.2%}")
    overfitting = best['score'] - best.get('cv_mean', best['score'])
    lines.append(f"  Overfitting: {'Low ✔' if overfitting < 0.03 else 'Moderate [!]' if overfitting < 0.06 else 'High [X]'} ({overfitting:.2%})")
    lines.append("")
    lines.append("PHASE 4 — FEATURE IMPORTANCE")
    lines.append("─" * 40)
    if hasattr(best.get("model"), "feature_importances_"):
        feat_names = df.drop(columns=[target]).select_dtypes(include=[np.number]).columns.tolist()
        importances = best["model"].feature_importances_
        if len(feat_names) == len(importances):
            ranked = sorted(zip(feat_names, importances), key=lambda x: -x[1])
            for fname, imp in ranked[:6]:
                lines.append(f"  {fname:<20} {imp:.3f} {'█'*int(imp*30)}")
    else:
        lines.append("  (Not available for this model type)")
    lines.append("")
    lines.append("PHASE 5 — IMPROVEMENT SUGGESTIONS")
    lines.append("─" * 40)
    if best['score'] < 0.95:
        lines.append(f"  • Hyperparameter tuning (+2-4%)")
    if best['score'] < 0.90:
        lines.append(f"  • Feature engineering (+3-5%)")
    lines.append(f"  • Ensemble stacking (+1-3%)")
    lines.append(f"  Potential: {best['score']:.1%} → {min(0.99, best['score']+0.05):.1%}")
    lines.append("")
    lines.append("PHASE 6 — PLAIN ENGLISH SUMMARY")
    lines.append("─" * 40)
    lines.append(f"  The {best['name']} model correctly predicts {int(best['score']*100)} out of 100 cases.")
    if hasattr(best.get("model"), "feature_importances_"):
        feat_names = df.drop(columns=[target]).select_dtypes(include=[np.number]).columns.tolist()
        importances = best["model"].feature_importances_
        if len(feat_names) == len(importances):
            top = sorted(zip(feat_names, importances), key=lambda x: -x[1])[:3]
            lines.append(f"  Key predictors: {', '.join(f[0] for f in top)}.")
    lines.append("")
    lines.append("PHASE 7 — RELIABILITY")
    lines.append("─" * 40)
    reliability = max(50, min(99, 100 - int(best.get('cv_std',0)*200) - int(overfitting*100)))
    lines.append(f"  Deployment Readiness: {reliability}%")
    lines.append(f"  Risk: {'Low' if reliability > 85 else 'Moderate' if reliability > 70 else 'High'}")
    lines.append("")
    lines.append(f"Total Time: {duration:.1f}s")
    lines.append("═" * 50)
    return "\n".join(lines)


def _load(path: str):
    p = Path(path)
    if not p.exists():
        return None
    try:
        ext = p.suffix.lower()
        if ext in ('.csv', '.tsv'): return pd.read_csv(p, low_memory=False)
        elif ext in ('.xlsx', '.xls'): return pd.read_excel(p)
        elif ext == '.json': return pd.read_json(p)
        elif ext == '.parquet': return pd.read_parquet(p)
    except Exception as e:
        print(f"[ML] Load error: {e}")
    return None
