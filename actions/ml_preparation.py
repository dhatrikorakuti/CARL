"""
actions/ml_preparation.py — Enterprise ML Data Preparation Engine for CARL.

Prepares datasets for Machine Learning like a Senior Data Scientist.
Every decision is explainable. IDs are never modified. Transformations are logged.

Actions:
    analyze   — Full semantic column analysis + cleaning policy
    prepare   — Execute ML preparation pipeline
    features  — Auto-generate features
    report    — ML readiness report
    query     — Answer questions from cached profile
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── Session cache ────────────────────────────────────────────────────────────
_PROFILE_CACHE: dict[str, dict] = {}
_LOG_CACHE: dict[str, list[str]] = {}


def ml_preparation(parameters: dict, player=None, speak=None) -> str:
    action = parameters.get("action", "analyze").lower()
    path = parameters.get("path", "")
    question = parameters.get("question", "")

    if action == "analyze":
        return _analyze(path)
    elif action == "prepare":
        return _prepare(path)
    elif action == "features":
        return _engineer_features(path)
    elif action == "report":
        return _ml_report(path)
    elif action == "query":
        return _query(path, question)
    else:
        return f"Unknown action: {action}. Use: analyze, prepare, features, report, query."


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — SEMANTIC COLUMN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

_ID_PATTERNS = re.compile(r'(^id$|_id$|^pk$|^key$|^index$)', re.I)
_TICKET_PATTERNS = re.compile(r'(ticket|booking|invoice|order|receipt|ref)', re.I)
_NAME_PATTERNS = re.compile(r'(name|first.?name|last.?name|full.?name|customer.?name)', re.I)
_EMAIL_PATTERNS = re.compile(r'(email|e.?mail|mail)', re.I)
_PHONE_PATTERNS = re.compile(r'(phone|mobile|cell|tel|contact)', re.I)
_DATE_PATTERNS = re.compile(r'(date|time|created|updated|birth|dob|timestamp|_at$|_on$)', re.I)
_CURRENCY_PATTERNS = re.compile(r'(price|cost|revenue|fare|salary|amount|fee|charge|payment|total)', re.I)
_PCT_PATTERNS = re.compile(r'(percent|pct|rate|ratio)', re.I)
_TARGET_PATTERNS = re.compile(r'(target|label|survived|churn|outcome|class$|y$)', re.I)
_BOOL_PATTERNS = re.compile(r'(is_|has_|flag|active|enabled|bool)', re.I)


def _classify_column(col: str, series: pd.Series) -> dict:
    """Classify a single column semantically."""
    nunique = series.nunique()
    total = len(series)
    dtype = str(series.dtype)
    sample = series.dropna().head(20).astype(str).tolist()

    # Check patterns
    if _ID_PATTERNS.search(col):
        if nunique == total or nunique > total * 0.95:
            return {"type": "primary_key", "confidence": 0.9, "policy": "do_not_modify"}
        return {"type": "foreign_key", "confidence": 0.7, "policy": "do_not_modify"}

    if _TICKET_PATTERNS.search(col):
        return {"type": "ticket_number", "confidence": 0.85, "policy": "do_not_modify"}

    if _EMAIL_PATTERNS.search(col):
        return {"type": "email", "confidence": 0.9, "policy": "lowercase_validate"}

    if _PHONE_PATTERNS.search(col):
        return {"type": "phone", "confidence": 0.85, "policy": "normalize"}

    if _NAME_PATTERNS.search(col):
        return {"type": "name", "confidence": 0.85, "policy": "trim_capitalize"}

    if _TARGET_PATTERNS.search(col):
        return {"type": "target_variable", "confidence": 0.8, "policy": "keep_as_is"}

    if _BOOL_PATTERNS.search(col):
        return {"type": "boolean", "confidence": 0.8, "policy": "convert_bool"}

    if _DATE_PATTERNS.search(col):
        return {"type": "datetime", "confidence": 0.8, "policy": "convert_iso"}

    if _CURRENCY_PATTERNS.search(col):
        return {"type": "currency", "confidence": 0.8, "policy": "convert_numeric"}

    if _PCT_PATTERNS.search(col):
        return {"type": "percentage", "confidence": 0.75, "policy": "convert_decimal"}

    # Unique identifier detection (high cardinality non-numeric)
    if nunique == total and dtype == "object":
        return {"type": "unique_identifier", "confidence": 0.7, "policy": "do_not_modify"}

    # Numeric
    try:
        if pd.api.types.is_numeric_dtype(series):
            return {"type": "numeric", "confidence": 0.9, "policy": "handle_missing_outliers"}
    except Exception:
        pass

    # Low cardinality text = categorical
    if dtype == "object" and nunique < 30:
        return {"type": "categorical", "confidence": 0.85, "policy": "normalize_categories"}

    # High cardinality text
    if dtype == "object" and nunique > total * 0.5:
        return {"type": "free_text", "confidence": 0.6, "policy": "clean_whitespace"}

    # Date detection from content
    if dtype == "object":
        date_count = sum(1 for v in sample if _looks_like_date(v))
        if date_count > len(sample) * 0.6:
            return {"type": "datetime", "confidence": 0.7, "policy": "convert_iso"}

    return {"type": "unknown", "confidence": 0.3, "policy": "inspect_manually"}


def _looks_like_date(val: str) -> bool:
    return bool(re.search(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}', val))


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYZE
# ═══════════════════════════════════════════════════════════════════════════════

def _analyze(path: str) -> str:
    df = _load(path)
    if df is None:
        return f"Cannot load: {path}"

    print(f"[ML] Analyzing {Path(path).name} ({len(df)} rows, {len(df.columns)} cols)")

    profile = {"file": Path(path).name, "rows": len(df), "columns": len(df.columns)}
    col_analysis = {}

    for col in df.columns:
        info = _classify_column(col, df[col])
        info["missing"] = int(df[col].isnull().sum())
        info["missing_pct"] = round(df[col].isnull().sum() / len(df) * 100, 1)
        info["nunique"] = int(df[col].nunique())
        col_analysis[col] = info

    profile["columns_analysis"] = col_analysis
    profile["id_columns"] = [c for c, v in col_analysis.items() if "id" in v["type"] or v["type"] == "ticket_number"]
    profile["numeric_columns"] = [c for c, v in col_analysis.items() if v["type"] == "numeric"]
    profile["categorical_columns"] = [c for c, v in col_analysis.items() if v["type"] == "categorical"]
    profile["target"] = next((c for c, v in col_analysis.items() if v["type"] == "target_variable"), None)

    _PROFILE_CACHE[path] = profile

    # Format output
    lines = [f"Column Analysis: {profile['file']} ({profile['rows']:,} rows)"]
    lines.append("")
    for col, info in col_analysis.items():
        flag = "🔒" if "not_modify" in info["policy"] else "[T]"
        miss = f" [{info['missing_pct']}% missing]" if info["missing"] > 0 else ""
        lines.append(f"  {flag} {col}: {info['type']} (conf={info['confidence']:.0%}) → {info['policy']}{miss}")

    if profile["target"]:
        lines.append(f"\nTarget Variable: {profile['target']}")
    lines.append(f"ID Columns (protected): {profile['id_columns']}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# PREPARE — Execute ML preparation pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def _prepare(path: str) -> str:
    df = _load(path)
    if df is None:
        return f"Cannot load: {path}"

    # Run analysis first if not cached
    if path not in _PROFILE_CACHE:
        _analyze(path)
    profile = _PROFILE_CACHE[path]
    col_info = profile["columns_analysis"]

    print(f"[ML] Preparing {Path(path).name}")
    start = time.time()
    log: list[str] = []

    # Apply policies per column
    for col, info in col_info.items():
        if col not in df.columns:
            continue
        policy = info["policy"]

        if policy == "do_not_modify":
            log.append(f"SKIP {col} — {info['type']} (protected)")
            continue

        elif policy == "lowercase_validate":
            df[col] = df[col].astype(str).str.lower().str.strip()
            log.append(f"CLEAN {col} — lowercased emails")

        elif policy == "trim_capitalize":
            df[col] = df[col].astype(str).str.strip().str.title()
            log.append(f"CLEAN {col} — trimmed + capitalized names")

        elif policy == "normalize":
            df[col] = df[col].astype(str).str.replace(r'[^\d+]', '', regex=True)
            log.append(f"CLEAN {col} — normalized phone numbers")

        elif policy == "convert_iso":
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                log.append(f"CONVERT {col} → datetime")
            except Exception:
                log.append(f"WARN {col} — datetime conversion failed")

        elif policy == "convert_numeric":
            df[col] = df[col].astype(str).str.replace(r'[$€£,]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            log.append(f"CONVERT {col} → numeric (removed currency symbols)")

        elif policy == "convert_decimal":
            df[col] = df[col].astype(str).str.replace('%', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce') / 100
            log.append(f"CONVERT {col} → decimal (percentage)")

        elif policy == "convert_bool":
            df[col] = df[col].map({1: True, 0: False, '1': True, '0': False,
                                   'yes': True, 'no': False, 'true': True, 'false': False,
                                   'Yes': True, 'No': False, 'True': True, 'False': False})
            log.append(f"CONVERT {col} → boolean")

        elif policy == "normalize_categories":
            df[col] = df[col].astype(str).str.strip().str.lower()
            log.append(f"CLEAN {col} — normalized categories")

        elif policy == "handle_missing_outliers":
            null_count = int(df[col].isnull().sum())
            if null_count > 0:
                median = df[col].median()
                df[col] = df[col].fillna(median)
                log.append(f"FILL {col} — {null_count} missing → median ({median:.2f})")

        elif policy == "clean_whitespace":
            df[col] = df[col].astype(str).str.strip()
            log.append(f"CLEAN {col} — whitespace trimmed")

    # Remove fully empty rows
    empty_before = len(df)
    df = df.dropna(how='all')
    if len(df) < empty_before:
        log.append(f"DROP {empty_before - len(df)} fully empty rows")

    # Remove duplicate rows
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        log.append(f"DROP {dup_count} duplicate rows")

    # Fill remaining nulls in numeric columns
    for col in df.select_dtypes(include=[np.number]).columns:
        null_count = int(df[col].isnull().sum())
        if null_count > 0:
            df[col] = df[col].fillna(df[col].median())
            log.append(f"FILL {col} — {null_count} remaining nulls → median")

    # Save
    out_path = Path(path).parent / f"{Path(path).stem}_ml_ready{Path(path).suffix}"
    try:
        if out_path.suffix in ('.csv', '.tsv'):
            df.to_csv(out_path, index=False)
        elif out_path.suffix in ('.xlsx', '.xls'):
            df.to_excel(out_path, index=False)
        elif out_path.suffix == '.parquet':
            df.to_parquet(out_path, index=False)
        log.append(f"SAVED {out_path.name}")
    except Exception as e:
        log.append(f"SAVE ERROR: {e}")

    duration = time.time() - start
    _LOG_CACHE[path] = log

    lines = [
        f"ML Preparation Complete: {Path(path).name}",
        f"Rows: {profile['rows']:,} → {len(df):,}",
        f"Operations: {len(log)}",
        f"Time: {duration:.1f}s",
        f"Output: {out_path.name}",
        "",
        "Log:",
    ] + [f"  {op}" for op in log[:15]]

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def _engineer_features(path: str) -> str:
    df = _load(path)
    if df is None:
        return f"Cannot load: {path}"

    log: list[str] = []

    # Datetime features
    for col in df.select_dtypes(include=['datetime64']).columns:
        df[f'{col}_year'] = df[col].dt.year
        df[f'{col}_month'] = df[col].dt.month
        df[f'{col}_weekday'] = df[col].dt.weekday
        df[f'{col}_is_weekend'] = df[col].dt.weekday.isin([5, 6]).astype(int)
        log.append(f"FEATURE {col} → year, month, weekday, is_weekend")

    # Name → Title extraction
    for col in df.columns:
        if _NAME_PATTERNS.search(col) and df[col].dtype == object:
            titles = df[col].str.extract(r' ([A-Za-z]+)\.')
            if titles[0].notna().sum() > len(df) * 0.3:
                df[f'{col}_title'] = titles[0].fillna('Unknown')
                log.append(f"FEATURE {col} → title (Mr/Mrs/Miss/etc)")

    # Age → Age Group
    for col in df.columns:
        if 'age' in col.lower() and np.issubdtype(df[col].dtype, np.number):
            bins = [0, 12, 18, 35, 50, 65, 120]
            labels = ['Child', 'Teen', 'Young Adult', 'Adult', 'Middle Age', 'Senior']
            df[f'{col}_group'] = pd.cut(df[col], bins=bins, labels=labels, right=False)
            log.append(f"FEATURE {col} → age_group")

    # Family size detection
    family_cols = [c for c in df.columns if any(w in c.lower() for w in ('sib', 'spouse', 'parch', 'child', 'family'))]
    if len(family_cols) >= 2:
        df['family_size'] = df[family_cols].sum(axis=1) + 1
        df['is_alone'] = (df['family_size'] == 1).astype(int)
        log.append(f"FEATURE family_size + is_alone from {family_cols}")

    # Fare/price → bands
    for col in df.columns:
        if _CURRENCY_PATTERNS.search(col) and np.issubdtype(df[col].dtype, np.number):
            try:
                df[f'{col}_band'] = pd.qcut(df[col], q=4, labels=['Low', 'Medium', 'High', 'Premium'], duplicates='drop')
                log.append(f"FEATURE {col} → fare_band (quartiles)")
            except Exception:
                pass

    # Cabin → Deck (if exists)
    for col in df.columns:
        if 'cabin' in col.lower() and df[col].dtype == object:
            df[f'{col}_deck'] = df[col].astype(str).str[0].replace('n', np.nan)
            log.append(f"FEATURE {col} → deck (first letter)")

    # Save
    out_path = Path(path).parent / f"{Path(path).stem}_featured{Path(path).suffix}"
    try:
        df.to_csv(out_path, index=False) if out_path.suffix == '.csv' else df.to_parquet(out_path, index=False)
        log.append(f"SAVED {out_path.name}")
    except Exception as e:
        log.append(f"SAVE ERROR: {e}")

    return f"Feature Engineering ({len(log)} operations):\n" + "\n".join(f"  {op}" for op in log)


# ═══════════════════════════════════════════════════════════════════════════════
# ML READINESS REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def _ml_report(path: str) -> str:
    df = _load(path)
    if df is None:
        return f"Cannot load: {path}"

    if path not in _PROFILE_CACHE:
        _analyze(path)
    profile = _PROFILE_CACHE[path]

    missing = int(df.isnull().sum().sum())
    dups = int(df.duplicated().sum())
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    cat_cols = list(df.select_dtypes(include=['object', 'category']).columns)

    # Recommend algorithms
    target = profile.get("target")
    if target and df[target].nunique() <= 10:
        task = "Classification"
        algos = "Random Forest, XGBoost, Logistic Regression, SVM"
    elif target:
        task = "Regression"
        algos = "XGBoost, Random Forest, Linear Regression, LightGBM"
    else:
        task = "Unsupervised"
        algos = "K-Means, DBSCAN, PCA, Isolation Forest"

    # Encoding recommendations
    encode_recs = []
    for col in cat_cols:
        n = df[col].nunique()
        if n <= 5:
            encode_recs.append(f"{col}: One-Hot ({n} categories)")
        elif n <= 15:
            encode_recs.append(f"{col}: Ordinal or Target Encoding ({n})")
        else:
            encode_recs.append(f"{col}: Frequency or Drop ({n} unique)")

    lines = [
        f"ML Readiness Report: {Path(path).name}",
        f"Rows: {len(df):,} | Columns: {len(df.columns)}",
        f"Missing: {missing:,} | Duplicates: {dups:,}",
        f"Numeric: {len(numeric_cols)} | Categorical: {len(cat_cols)}",
        f"Target: {target or 'Not detected'}",
        f"Task: {task}",
        f"Recommended Algorithms: {algos}",
        "",
        "Encoding Recommendations:",
    ] + [f"  {r}" for r in encode_recs[:10]]

    if missing > 0:
        lines.append(f"\n[!] {missing} missing values remain — run 'prepare' to fix")
    else:
        lines.append("\n[OK] Dataset is ML-ready")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# QUERY
# ═══════════════════════════════════════════════════════════════════════════════

def _query(path: str, question: str) -> str:
    profile = _PROFILE_CACHE.get(path, {})
    log = _LOG_CACHE.get(path, [])
    q = question.lower()

    if not profile:
        return "No analysis cached. Run 'analyze' first."

    if "id" in q or "identifier" in q:
        return f"ID columns (protected): {profile.get('id_columns', [])}"
    if "target" in q:
        return f"Target variable: {profile.get('target', 'Not detected')}"
    if "categorical" in q:
        return f"Categorical columns: {profile.get('categorical_columns', [])}"
    if "numeric" in q:
        return f"Numeric columns: {profile.get('numeric_columns', [])}"
    if "ticket" in q or "why" in q:
        ticket_cols = [c for c, v in profile.get("columns_analysis", {}).items() if v["type"] == "ticket_number"]
        return f"Ticket columns ({ticket_cols}) are protected — multiple passengers may share a ticket."
    if "log" in q or "what" in q:
        return "Transformation log:\n" + "\n".join(f"  {op}" for op in log[:15]) if log else "No transformations yet."
    if "ready" in q or "random forest" in q.lower() or "xgboost" in q.lower():
        return _ml_report(path)

    return f"Profile: {len(profile.get('columns_analysis', {}))} columns analyzed. Ask about: IDs, target, categorical, numeric, ticket, log, readiness."


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _load(path: str) -> pd.DataFrame | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        ext = p.suffix.lower()
        if ext in ('.csv', '.tsv'):
            return pd.read_csv(p, low_memory=False)
        elif ext in ('.xlsx', '.xls'):
            return pd.read_excel(p)
        elif ext == '.json':
            return pd.read_json(p)
        elif ext == '.parquet':
            return pd.read_parquet(p)
        return None
    except Exception as e:
        print(f"[ML] Load error: {e}")
        return None
