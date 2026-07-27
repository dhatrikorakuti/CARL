"""
actions/smart_data_scientist.py — CARL's Intelligent Data Science Engine.

CARL thinks like an experienced Data Scientist.
Before touching any data, CARL:
  1. Understands the user's ML goal
  2. Analyzes every column semantically (using LLM reasoning)
  3. Detects impossible values using domain knowledge
  4. Explains every decision
  5. Chooses optimal strategies based on the goal

This is NOT a blind pandas script runner.
Every operation is reasoned, explained, and logged.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Cache
_SESSION: dict[str, Any] = {}


def smart_data_scientist(parameters: dict, player=None, speak=None) -> str:
    """Main entry — CARL's data science brain."""
    action = parameters.get("action", "analyze").lower()
    path = parameters.get("path", "")
    goal = parameters.get("goal", "")
    question = parameters.get("question", "")

    if action == "analyze":
        return _think_and_analyze(path, goal)
    elif action == "clean":
        return _think_and_clean(path, goal)
    elif action == "prepare_ml":
        return _full_ml_pipeline(path, goal)
    elif action == "explain":
        return _explain_decisions(path)
    elif action == "query":
        return _query(question)
    else:
        return f"Actions: analyze, clean, prepare_ml, explain, query"


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1+2: THINK — Understand goal + columns using LLM reasoning
# ═══════════════════════════════════════════════════════════════════════════════

def _think_and_analyze(path: str, goal: str) -> str:
    """Analyze dataset with domain understanding — like a human would."""
    df = _load(path)
    if df is None:
        return f"Cannot load: {path}"

    print(f"[DATA SCIENTIST] Thinking about {Path(path).name}...")

    # Build dataset summary for reasoning
    summary = _build_summary(df)

    # Use LLM to reason about the data (falls back to heuristic if LLM unavailable)
    reasoning = _reason_about_data(summary, goal)

    # Store in session
    _SESSION["path"] = path
    _SESSION["goal"] = goal
    _SESSION["df_shape"] = df.shape
    _SESSION["summary"] = summary
    _SESSION["reasoning"] = reasoning
    _SESSION["decisions"] = []

    # Format output
    lines = [
        f"Dataset Analysis: {Path(path).name}",
        f"Goal: {goal or 'Not specified — please tell me what you want to predict/analyze'}",
        f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns",
        f"Memory: {df.memory_usage(deep=True).sum()/1024**2:.1f} MB",
        "",
        "Column Understanding:",
    ]

    for col_info in reasoning.get("columns", []):
        name = col_info.get("name", "?")
        role = col_info.get("role", "unknown")
        action = col_info.get("recommended_action", "keep")
        impossible = col_info.get("impossible_values", "")
        reason = col_info.get("reasoning", "")

        icon = {"drop": "[!]", "target": "🎯", "id": "🔒", "feature": "[T]", "keep": "📊"}.get(action, "•")
        line = f"  {icon} {name}: {role}"
        if impossible:
            line += f" [[!] impossible: {impossible}]"
        if action == "drop":
            line += f" → DROP ({reason})"
        lines.append(line)

    # Recommendations
    recs = reasoning.get("recommendations", [])
    if recs:
        lines.append("\nData Scientist Recommendations:")
        for r in recs:
            lines.append(f"  • {r}")

    return "\n".join(lines)


def _reason_about_data(summary: dict, goal: str) -> dict:
    """Use LLM or heuristics to reason about the dataset."""
    # Try LLM first
    llm_result = _llm_reason(summary, goal)
    if llm_result:
        return llm_result

    # Fallback: heuristic reasoning
    return _heuristic_reason(summary, goal)


def _llm_reason(summary: dict, goal: str) -> dict | None:
    """Ask LLM to reason about the dataset like a data scientist."""
    try:
        from core.llm_client import call_llm_text

        prompt = f"""You are a Senior Data Scientist analyzing a dataset.

GOAL: {goal or 'General ML preparation'}

DATASET:
{json.dumps(summary, indent=2, default=str)[:3000]}

For EACH column, determine:
1. "role": what it represents (target/feature/identifier/timestamp/text/drop)
2. "type": numeric/categorical/ordinal/binary/identifier/timestamp/text
3. "impossible_values": values that are domain-impossible (e.g., glucose=0, BMI=0, negative age)
4. "recommended_action": keep/drop/encode/scale/fill_missing/engineer
5. "reasoning": one sentence explaining why

Also provide:
- "recommendations": list of 3-5 key data science decisions
- "target": which column is the target variable

Output ONLY valid JSON:
{{"columns": [...], "target": "...", "recommendations": [...]}}"""

        raw = call_llm_text(prompt=prompt, system="You are a data science expert. Output only JSON.", timeout=8)

        # Parse JSON
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json\n")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                return json.loads(match.group(0))
    except Exception as e:
        print(f"[DATA SCIENTIST] LLM reasoning unavailable: {e}")

    return None


def _heuristic_reason(summary: dict, goal: str) -> dict:
    """Fallback: use heuristic domain knowledge when LLM is unavailable."""
    columns = []
    goal_lower = (goal or "").lower()

    for col_name, info in summary.get("columns", {}).items():
        dtype = info.get("dtype", "")
        nunique = info.get("nunique", 0)
        null_pct = info.get("null_pct", 0)
        sample = info.get("sample", [])
        total_rows = summary.get("rows", 100)

        col_info = {"name": col_name, "role": "feature", "type": "unknown",
                    "impossible_values": "", "recommended_action": "keep", "reasoning": ""}

        name_lower = col_name.lower()

        # Target detection
        if any(t in name_lower for t in ("survived", "target", "label", "churn", "fraud", "outcome", "class")):
            col_info["role"] = "target"
            col_info["type"] = "binary" if nunique <= 2 else "categorical"
            col_info["recommended_action"] = "target"
            col_info["reasoning"] = "Detected as target variable"

        # ID columns
        elif any(t in name_lower for t in ("id", "index", "key")) and nunique > total_rows * 0.9:
            col_info["role"] = "identifier"
            col_info["type"] = "identifier"
            col_info["recommended_action"] = "drop"
            col_info["reasoning"] = "Unique identifier — no predictive value"

        # Name columns
        elif any(t in name_lower for t in ("name", "first_name", "last_name")):
            col_info["role"] = "identifier"
            col_info["type"] = "text"
            col_info["recommended_action"] = "drop"
            col_info["reasoning"] = "Name has no predictive value (extract title if useful)"

        # High missing columns
        elif null_pct > 70:
            col_info["recommended_action"] = "drop"
            col_info["reasoning"] = f"{null_pct:.0f}% missing — too incomplete to impute"

        # Medical domain: detect impossible zeros
        elif name_lower in ("glucose", "bloodpressure", "blood_pressure", "bmi",
                           "skinthickness", "skin_thickness", "insulin"):
            col_info["type"] = "numeric"
            col_info["impossible_values"] = "0 (medically impossible)"
            col_info["recommended_action"] = "fill_missing"
            col_info["reasoning"] = "Zero is impossible for this measurement — treat as missing"

        # Pregnancy: 0 is valid
        elif "pregnan" in name_lower:
            col_info["type"] = "numeric"
            col_info["recommended_action"] = "keep"
            col_info["reasoning"] = "Zero pregnancies is valid — do not modify"

        # Ticket numbers
        elif "ticket" in name_lower:
            col_info["type"] = "identifier"
            col_info["recommended_action"] = "drop"
            col_info["reasoning"] = "Ticket numbers are mostly unique — no predictive value"

        # Numeric
        elif dtype in ("int64", "float64", "int32", "float32"):
            col_info["type"] = "numeric"
            col_info["recommended_action"] = "keep"
            col_info["reasoning"] = "Numeric feature — check for outliers"

        # Low cardinality text = categorical
        elif dtype == "object" and nunique < 15:
            col_info["type"] = "categorical"
            col_info["recommended_action"] = "encode"
            col_info["reasoning"] = f"Categorical ({nunique} unique values) — needs encoding"

        # Binary
        elif nunique == 2:
            col_info["type"] = "binary"
            col_info["recommended_action"] = "encode"
            col_info["reasoning"] = "Binary — encode as 0/1"

        columns.append(col_info)

    # Recommendations based on goal
    recs = []
    if "diabetes" in goal_lower or "medical" in goal_lower:
        recs.append("Medical dataset: treat zero values in glucose/BP/BMI/insulin as missing (impossible)")
        recs.append("Use median imputation (robust to outliers in medical data)")
    if "churn" in goal_lower or "classification" in goal_lower:
        recs.append("Classification task: check class imbalance in target variable")
        recs.append("Consider SMOTE or class weights for imbalanced data")
    if "forecast" in goal_lower or "time" in goal_lower:
        recs.append("Time series: preserve temporal order, do NOT shuffle")
        recs.append("Engineer lag features and rolling statistics")

    recs.append("Remove high-cardinality identifiers before modeling")
    recs.append("Scale numeric features for distance-based models (SVM, KNN)")

    target = next((c["name"] for c in columns if c.get("role") == "target"), None)

    return {"columns": columns, "target": target, "recommendations": recs}


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3-9: CLEAN — Execute with domain knowledge
# ═══════════════════════════════════════════════════════════════════════════════

def _think_and_clean(path: str, goal: str) -> str:
    """Clean the dataset using reasoned decisions."""
    df = _load(path)
    if df is None:
        return f"Cannot load: {path}"

    # Ensure we have reasoning
    if "reasoning" not in _SESSION or _SESSION.get("path") != path:
        _think_and_analyze(path, goal)

    reasoning = _SESSION.get("reasoning", {})
    columns_info = {c["name"]: c for c in reasoning.get("columns", [])}

    print(f"[DATA SCIENTIST] Cleaning {Path(path).name} with domain knowledge...")
    start = time.time()
    decisions: list[str] = []
    rows_before = len(df)

    for col in list(df.columns):
        info = columns_info.get(col, {})
        action = info.get("recommended_action", "keep")
        impossible = info.get("impossible_values", "")
        col_type = info.get("type", "")

        # DROP columns
        if action == "drop":
            reason = info.get("reasoning", "recommended to drop")
            df = df.drop(columns=[col])
            decisions.append(f"DROPPED '{col}' — {reason}")
            continue

        # Handle impossible values (domain knowledge)
        if impossible and "0" in impossible and col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                zero_count = int((df[col] == 0).sum())
                if zero_count > 0:
                    df[col] = df[col].replace(0, np.nan)
                    median = df[col].median()
                    df[col] = df[col].fillna(median)
                    decisions.append(
                        f"FIXED '{col}' — {zero_count} zeros → NaN → filled with median ({median:.1f}). "
                        f"Reason: {impossible}"
                    )

        # Fill missing numeric
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            null_count = int(df[col].isnull().sum())
            if null_count > 0:
                median = df[col].median()
                df[col] = df[col].fillna(median)
                decisions.append(f"FILLED '{col}' — {null_count} missing → median ({median:.1f}). Reason: median is robust to outliers")

        # Fill missing categorical
        if col in df.columns and df[col].dtype == object:
            null_count = int(df[col].isnull().sum())
            if null_count > 0:
                mode = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "Unknown"
                df[col] = df[col].fillna(mode)
                decisions.append(f"FILLED '{col}' — {null_count} missing → mode ('{mode}'). Reason: most frequent category preserves distribution")

        # Encode binary
        if col in df.columns and col_type == "binary" and df[col].dtype == object:
            mapping = {v: i for i, v in enumerate(df[col].unique()[:2])}
            df[col] = df[col].map(mapping)
            decisions.append(f"ENCODED '{col}' — binary mapping {mapping}")

    # Remove duplicates
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        decisions.append(f"REMOVED {dup_count} duplicate rows")

    duration = time.time() - start
    _SESSION["decisions"] = decisions
    _SESSION["cleaned_df"] = df

    # Save
    out_path = Path(path).parent / f"{Path(path).stem}_scientist_cleaned{Path(path).suffix}"
    try:
        df.to_csv(out_path, index=False) if out_path.suffix == '.csv' else df.to_parquet(out_path, index=False)
        decisions.append(f"SAVED → {out_path.name}")
    except Exception as e:
        decisions.append(f"SAVE ERROR: {e}")

    # Report
    lines = [
        f"Intelligent Cleaning: {Path(path).name}",
        f"Goal: {goal or _SESSION.get('goal', 'General')}",
        f"Rows: {rows_before:,} → {len(df):,}",
        f"Columns: {_SESSION.get('df_shape', (0,0))[1]} → {len(df.columns)}",
        f"Decisions made: {len(decisions)}",
        f"Time: {duration:.1f}s",
        "",
        "Explanations:",
    ] + [f"  • {d}" for d in decisions]

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# FULL ML PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def _full_ml_pipeline(path: str, goal: str) -> str:
    """Full pipeline: analyze → clean → prepare for ML."""
    results = []
    results.append(_think_and_analyze(path, goal))
    results.append("")
    results.append("─" * 40)
    results.append("")
    results.append(_think_and_clean(path, goal))

    # ML readiness
    df = _SESSION.get("cleaned_df")
    if df is not None:
        reasoning = _SESSION.get("reasoning", {})
        target = reasoning.get("target")

        results.append("")
        results.append("─" * 40)
        results.append("")
        results.append("ML Readiness:")

        if target and target in df.columns:
            results.append(f"  Target: {target} ({df[target].nunique()} classes)")
            results.append(f"  Features: {len(df.columns) - 1}")
            results.append(f"  Samples: {len(df):,}")

            # Recommend models
            if df[target].nunique() <= 10:
                results.append(f"  Task: Classification")
                results.append(f"  Recommended: XGBoost, Random Forest, LightGBM")
                if df[target].value_counts().min() / len(df) < 0.2:
                    results.append(f"  [!] Imbalanced classes — use class_weight='balanced' or SMOTE")
            else:
                results.append(f"  Task: Regression")
                results.append(f"  Recommended: XGBoost, Random Forest, ElasticNet")
        else:
            results.append(f"  Target not found in cleaned data — specify with goal parameter")

    return "\n".join(results)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPLAIN + QUERY
# ═══════════════════════════════════════════════════════════════════════════════

def _explain_decisions(path: str) -> str:
    decisions = _SESSION.get("decisions", [])
    if not decisions:
        return "No decisions made yet. Run 'analyze' or 'clean' first."
    return "All decisions explained:\n" + "\n".join(f"  {i+1}. {d}" for i, d in enumerate(decisions))


def _query(question: str) -> str:
    q = question.lower()
    reasoning = _SESSION.get("reasoning", {})
    decisions = _SESSION.get("decisions", [])

    if "why" in q:
        # Find relevant decision
        for d in decisions:
            keywords = [w for w in q.split() if len(w) > 3]
            if any(kw in d.lower() for kw in keywords):
                return d
        return "I don't have a recorded decision matching that. Ask about a specific column."

    if "target" in q:
        return f"Target: {reasoning.get('target', 'Not detected')}"

    if "recommend" in q or "model" in q:
        recs = reasoning.get("recommendations", [])
        return "\n".join(recs) if recs else "No recommendations yet."

    if "decision" in q or "what did" in q:
        return _explain_decisions("")

    return "Ask: why was X cleaned? what's the target? what models do you recommend? what decisions were made?"


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _build_summary(df: pd.DataFrame) -> dict:
    """Build a concise dataset summary for LLM reasoning."""
    summary = {"rows": len(df), "columns": {}}
    for col in df.columns:
        info = {
            "dtype": str(df[col].dtype),
            "nunique": int(df[col].nunique()),
            "null_count": int(df[col].isnull().sum()),
            "null_pct": round(df[col].isnull().sum() / len(df) * 100, 1),
            "sample": df[col].dropna().head(5).tolist(),
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            info["min"] = float(df[col].min()) if not df[col].isnull().all() else None
            info["max"] = float(df[col].max()) if not df[col].isnull().all() else None
            info["mean"] = float(df[col].mean()) if not df[col].isnull().all() else None
            info["zeros"] = int((df[col] == 0).sum())
        summary["columns"][col] = info
    return summary


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
        print(f"[DATA SCIENTIST] Load error: {e}")
        return None
