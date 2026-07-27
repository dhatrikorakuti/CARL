"""
actions/data_cleaning.py — Enterprise-Grade Data Cleaning Engine for CARL.

Cleans datasets with the reasoning of a senior data analyst.
Every modification is logged, explainable, and reversible.

Usage:
    result = data_cleaning({"action": "audit", "path": "sales.csv"})
    result = data_cleaning({"action": "clean", "path": "sales.csv"})
    result = data_cleaning({"action": "query", "question": "how many missing values?"})
"""
from __future__ import annotations

import time
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── Session-level audit cache (avoids re-scanning) ───────────────────────────
_AUDIT_CACHE: dict[str, dict] = {}
_CLEANED_CACHE: dict[str, pd.DataFrame] = {}

# Values treated as missing
_MISSING_VALUES = {
    "", " ", "nan", "NaN", "NAN", "null", "NULL", "None", "none",
    "N/A", "n/a", "NA", "na", "-", "--", "Unknown", "unknown",
    "UNKNOWN", "inf", "-inf", "Infinity", "-Infinity",
}


def data_cleaning(parameters: dict, player=None, speak=None) -> str:
    """Main entry point for the data cleaning tool."""
    action = parameters.get("action", "audit").lower()
    path = parameters.get("path", "")
    question = parameters.get("question", "")

    if action == "audit":
        return _run_audit(path)
    elif action == "clean":
        return _run_clean(path)
    elif action == "query":
        return _answer_query(question, path)
    elif action == "report":
        return _get_report(path)
    elif action == "quality":
        return _get_quality(path)
    else:
        return f"Unknown action: {action}. Use: audit, clean, query, report, quality."


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIT
# ═══════════════════════════════════════════════════════════════════════════════

def _run_audit(path: str) -> str:
    """Complete dataset audit — generates the full audit object."""
    df = _load_dataset(path)
    if df is None:
        return f"Cannot load dataset: {path}"

    print(f"[DATA] Auditing {path} ({len(df)} rows, {len(df.columns)} cols)")
    start = time.time()

    audit = {}
    audit["file"] = Path(path).name
    audit["rows"] = len(df)
    audit["columns"] = len(df.columns)
    audit["column_names"] = list(df.columns)
    audit["memory_mb"] = round(df.memory_usage(deep=True).sum() / 1024**2, 2)

    # Column types
    audit["numeric_columns"] = list(df.select_dtypes(include=[np.number]).columns)
    audit["categorical_columns"] = list(df.select_dtypes(include=["object", "category"]).columns)
    audit["datetime_columns"] = list(df.select_dtypes(include=["datetime"]).columns)
    audit["boolean_columns"] = list(df.select_dtypes(include=["bool"]).columns)

    # Missing values
    missing = _detect_missing(df)
    audit["total_missing"] = int(missing["total"])
    audit["missing_per_column"] = missing["per_column"]
    audit["missing_pct_per_column"] = missing["pct_per_column"]

    # Duplicates
    dup_rows = int(df.duplicated().sum())
    dup_cols = _find_duplicate_columns(df)
    audit["duplicate_rows"] = dup_rows
    audit["duplicate_columns"] = dup_cols

    # Unique values
    audit["unique_per_column"] = {col: int(df[col].nunique()) for col in df.columns}

    # Constant columns (only 1 unique value)
    audit["constant_columns"] = [col for col in df.columns if df[col].nunique() <= 1]

    # Mixed types
    audit["mixed_type_columns"] = _detect_mixed_types(df)

    # Outliers (numeric only, IQR method)
    audit["outliers"] = _detect_outliers_iqr(df)

    # Quality score
    audit["quality_score"] = _calculate_quality(df, audit)

    # Primary key candidates
    audit["pk_candidates"] = [col for col in df.columns if df[col].nunique() == len(df)]

    duration = time.time() - start
    audit["audit_time_sec"] = round(duration, 2)

    # Cache
    _AUDIT_CACHE[path] = audit

    # Format response
    lines = [
        f"Dataset Audit: {audit['file']}",
        f"Rows: {audit['rows']:,} | Columns: {audit['columns']}",
        f"Memory: {audit['memory_mb']} MB",
        f"Missing Values: {audit['total_missing']:,}",
        f"Duplicate Rows: {audit['duplicate_rows']:,}",
        f"Constant Columns: {len(audit['constant_columns'])}",
        f"Outliers: {sum(v for v in audit['outliers'].values())} total",
        f"Quality Score: {audit['quality_score']}/100",
        f"Audit Time: {audit['audit_time_sec']}s",
    ]
    if audit["missing_per_column"]:
        top_missing = sorted(audit["missing_per_column"].items(), key=lambda x: -x[1])[:5]
        lines.append("Top missing: " + ", ".join(f"{k}({v})" for k, v in top_missing))

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# CLEAN
# ═══════════════════════════════════════════════════════════════════════════════

def _run_clean(path: str) -> str:
    """Intelligent cleaning with full logging."""
    df = _load_dataset(path)
    if df is None:
        return f"Cannot load dataset: {path}"

    print(f"[DATA] Cleaning {path}")
    start = time.time()
    log: list[str] = []
    rows_before = len(df)
    missing_before = int(df.isnull().sum().sum())

    # 1. Replace custom missing values with NaN
    for col in df.columns:
        mask = df[col].astype(str).str.strip().isin(_MISSING_VALUES)
        count = int(mask.sum())
        if count > 0:
            df.loc[mask, col] = np.nan
            log.append(f"Replaced {count} missing-value markers in '{col}'")

    # 2. Remove duplicate rows
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        log.append(f"Removed {dup_count} duplicate rows")

    # 3. Remove constant columns
    const_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if const_cols:
        df = df.drop(columns=const_cols)
        log.append(f"Removed {len(const_cols)} constant columns: {const_cols}")

    # 4. Fix data types
    for col in df.columns:
        # Try numeric conversion
        if df[col].dtype == object:
            # Remove currency symbols and commas
            cleaned = df[col].astype(str).str.replace(r'[$€£,]', '', regex=True).str.strip()
            try:
                numeric = pd.to_numeric(cleaned, errors='coerce')
                if numeric.notna().sum() > len(df) * 0.7:  # >70% convertible
                    df[col] = numeric
                    log.append(f"Converted '{col}' to numeric")
                    continue
            except Exception:
                pass

            # Try datetime conversion
            try:
                dates = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                if dates.notna().sum() > len(df) * 0.7:
                    df[col] = dates
                    log.append(f"Converted '{col}' to datetime")
                    continue
            except Exception:
                pass

    # 5. Fill missing values intelligently
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        if null_count == 0:
            continue

        if df[col].dtype in [np.float64, np.int64, float, int]:
            # Numeric — use median (robust to outliers)
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            log.append(f"Filled {null_count} missing in '{col}' with median ({median_val:.2f})")
        elif df[col].dtype == 'datetime64[ns]':
            # Datetime — forward fill
            df[col] = df[col].ffill()
            log.append(f"Forward-filled {null_count} missing dates in '{col}'")
        else:
            # Categorical — use mode
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val.iloc[0])
                log.append(f"Filled {null_count} missing in '{col}' with mode ('{mode_val.iloc[0]}')")

    # 6. Strip whitespace from text columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
        # Standardize case for low-cardinality columns
        if df[col].nunique() < 20:
            df[col] = df[col].str.title()

    # Summary
    duration = time.time() - start
    missing_after = int(df.isnull().sum().sum())
    rows_after = len(df)

    # Cache cleaned dataset
    _CLEANED_CACHE[path] = df

    # Save cleaned file
    out_path = Path(path).parent / f"{Path(path).stem}_cleaned{Path(path).suffix}"
    try:
        if out_path.suffix in ('.csv', '.tsv'):
            df.to_csv(out_path, index=False)
        elif out_path.suffix in ('.xlsx', '.xls'):
            df.to_excel(out_path, index=False)
        elif out_path.suffix == '.json':
            df.to_json(out_path, orient='records', indent=2)
        elif out_path.suffix == '.parquet':
            df.to_parquet(out_path, index=False)
        log.append(f"Saved cleaned dataset: {out_path.name}")
    except Exception as e:
        log.append(f"Could not save: {e}")

    # Update audit cache
    _AUDIT_CACHE[path + "_clean"] = {
        "rows_before": rows_before, "rows_after": rows_after,
        "missing_before": missing_before, "missing_after": missing_after,
        "duplicates_removed": dup_count, "operations": log,
        "duration_sec": round(duration, 2),
    }

    # Format response
    lines = [
        f"Cleaning Complete: {Path(path).name}",
        f"Rows: {rows_before:,} → {rows_after:,}",
        f"Missing: {missing_before:,} → {missing_after:,}",
        f"Duplicates removed: {dup_count:,}",
        f"Operations: {len(log)}",
        f"Time: {duration:.1f}s",
        "",
        "Actions taken:",
    ] + [f"  • {op}" for op in log[:10]]

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# QUERY
# ═══════════════════════════════════════════════════════════════════════════════

def _answer_query(question: str, path: str) -> str:
    """Answer questions about the dataset from cached audit."""
    q = question.lower()
    audit = _AUDIT_CACHE.get(path, {})
    clean_info = _AUDIT_CACHE.get(path + "_clean", {})

    if not audit and not clean_info:
        return "No audit data available. Run audit first."

    if "missing" in q:
        total = audit.get("total_missing", clean_info.get("missing_after", "unknown"))
        return f"There are {total:,} missing values." if isinstance(total, int) else f"Missing values: {total}"

    if "duplicate" in q:
        return f"There are {audit.get('duplicate_rows', 0):,} duplicate rows."

    if "outlier" in q:
        outliers = audit.get("outliers", {})
        total = sum(outliers.values())
        if total == 0:
            return "No outliers detected."
        top = sorted(outliers.items(), key=lambda x: -x[1])[:5]
        return f"{total} outliers found. Top: " + ", ".join(f"{k}({v})" for k, v in top)

    if "quality" in q:
        return f"Dataset quality score: {audit.get('quality_score', 'N/A')}/100"

    if "row" in q:
        return f"Rows: {audit.get('rows', 'unknown'):,}"

    if "column" in q:
        return f"Columns: {audit.get('columns', 'unknown')} — {', '.join(audit.get('column_names', [])[:15])}"

    if "clean" in q and "what" in q:
        ops = clean_info.get("operations", [])
        return "Cleaning operations:\n" + "\n".join(f"  • {op}" for op in ops[:10]) if ops else "No cleaning performed yet."

    return f"Audit data: {json.dumps({k: v for k, v in audit.items() if k not in ('missing_per_column', 'missing_pct_per_column', 'unique_per_column')}, default=str)[:500]}"


def _get_report(path: str) -> str:
    """Generate post-cleaning report."""
    clean_info = _AUDIT_CACHE.get(path + "_clean", {})
    if not clean_info:
        return "No cleaning has been performed yet. Run clean first."
    lines = [
        "Post-Cleaning Report",
        f"Rows: {clean_info['rows_before']:,} → {clean_info['rows_after']:,}",
        f"Missing: {clean_info['missing_before']:,} → {clean_info['missing_after']:,}",
        f"Duplicates Removed: {clean_info['duplicates_removed']:,}",
        f"Time: {clean_info['duration_sec']}s",
        f"Operations ({len(clean_info['operations'])}):",
    ] + [f"  • {op}" for op in clean_info["operations"]]
    return "\n".join(lines)


def _get_quality(path: str) -> str:
    """Return quality score breakdown."""
    audit = _AUDIT_CACHE.get(path, {})
    if not audit:
        return "No audit data. Run audit first."
    return f"Quality Score: {audit.get('quality_score', 'N/A')}/100"


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _load_dataset(path: str) -> pd.DataFrame | None:
    """Load dataset from path — supports CSV, Excel, JSON, Parquet."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        ext = p.suffix.lower()
        if ext == ".csv" or ext == ".tsv":
            return pd.read_csv(p, low_memory=False)
        elif ext in (".xlsx", ".xls"):
            return pd.read_excel(p)
        elif ext == ".json":
            return pd.read_json(p)
        elif ext == ".parquet":
            return pd.read_parquet(p)
        elif ext == ".sqlite" or ext == ".sql":
            import sqlite3
            conn = sqlite3.connect(p)
            tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
            if len(tables) > 0:
                return pd.read_sql(f"SELECT * FROM {tables.iloc[0, 0]}", conn)
        return None
    except Exception as e:
        print(f"[DATA] Load error: {e}")
        return None


def _detect_missing(df: pd.DataFrame) -> dict:
    """Comprehensive missing value detection."""
    # First: standard NaN
    missing = df.isnull().sum()

    # Also check string representations of missing
    for col in df.select_dtypes(include=['object']).columns:
        str_missing = df[col].astype(str).str.strip().isin(_MISSING_VALUES).sum()
        missing[col] = max(missing[col], str_missing)

    per_col = {col: int(v) for col, v in missing.items() if v > 0}
    pct_col = {col: round(v / len(df) * 100, 1) for col, v in per_col.items()}

    return {
        "total": int(missing.sum()),
        "per_column": per_col,
        "pct_per_column": pct_col,
    }


def _find_duplicate_columns(df: pd.DataFrame) -> list[str]:
    """Find columns with identical content."""
    dups = []
    seen = set()
    for i, col1 in enumerate(df.columns):
        if col1 in seen:
            continue
        for col2 in df.columns[i+1:]:
            if col2 in seen:
                continue
            if df[col1].equals(df[col2]):
                dups.append(col2)
                seen.add(col2)
    return dups


def _detect_mixed_types(df: pd.DataFrame) -> list[str]:
    """Find columns with mixed data types."""
    mixed = []
    for col in df.select_dtypes(include=['object']).columns:
        sample = df[col].dropna().head(100)
        types_found = set()
        for val in sample:
            try:
                float(str(val).replace(',', '').replace('$', '').replace('€', ''))
                types_found.add("numeric")
            except ValueError:
                types_found.add("text")
        if len(types_found) > 1:
            mixed.append(col)
    return mixed


def _detect_outliers_iqr(df: pd.DataFrame) -> dict[str, int]:
    """Detect outliers using IQR method on numeric columns."""
    outliers = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((df[col] < lower) | (df[col] > upper)).sum())
        if count > 0:
            outliers[col] = count
    return outliers


def _calculate_quality(df: pd.DataFrame, audit: dict) -> int:
    """Calculate overall dataset quality score (0-100)."""
    scores = []

    # Completeness (missing values)
    total_cells = audit["rows"] * audit["columns"]
    if total_cells > 0:
        completeness = 100 * (1 - audit["total_missing"] / total_cells)
        scores.append(completeness)

    # Uniqueness (no duplicates)
    if audit["rows"] > 0:
        uniqueness = 100 * (1 - audit["duplicate_rows"] / audit["rows"])
        scores.append(uniqueness)

    # Consistency (no mixed types)
    if audit["columns"] > 0:
        consistency = 100 * (1 - len(audit.get("mixed_type_columns", [])) / audit["columns"])
        scores.append(consistency)

    # Validity (no constant columns)
    if audit["columns"] > 0:
        validity = 100 * (1 - len(audit.get("constant_columns", [])) / audit["columns"])
        scores.append(validity)

    return int(np.mean(scores)) if scores else 0
