"""
memory/dataset_memory.py — Dataset Memory Layer for CARL.

Tracks uploaded datasets, cleaning operations, generated insights,
and visualizations. Enables CARL to remember previous analyses.

Example:
  User uploads "sales.xlsx" again →
  CARL says: "I've seen this dataset before. Last time I found 92 duplicates
  and identified a declining furniture trend. Would you like me to run a fresh analysis?"
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict

_STORE_PATH = Path(__file__).resolve().parent.parent / "memory" / "dataset_store.json"
_lock = threading.Lock()
_MAX_DATASETS = 50


@dataclass
class DatasetRecord:
    id: str
    filename: str
    file_path: str
    file_type: str  # csv, xlsx, json, sql, parquet
    file_size: int = 0
    uploaded_at: float = field(default_factory=time.time)
    last_analyzed: float = 0.0
    analysis_count: int = 0

    # Data profile
    rows: int = 0
    columns: int = 0
    column_names: list[str] = field(default_factory=list)
    dtypes: dict = field(default_factory=dict)

    # Cleaning operations performed
    cleaning_ops: list[str] = field(default_factory=list)
    # e.g. ["Removed 92 duplicates", "Imputed 15 missing values", "Fixed currency format"]

    # Insights discovered
    insights: list[str] = field(default_factory=list)
    # e.g. ["Sales dropped 18% Q3", "North region most profitable"]

    # Visualizations generated
    visualizations: list[str] = field(default_factory=list)
    # e.g. ["bar_chart_sales_by_region", "trend_line_revenue"]

    # ML models applied
    models_used: list[str] = field(default_factory=list)

    # Tags for semantic search
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "filename": self.filename, "file_path": self.file_path,
            "file_type": self.file_type, "file_size": self.file_size,
            "uploaded_at": self.uploaded_at, "last_analyzed": self.last_analyzed,
            "analysis_count": self.analysis_count, "rows": self.rows, "columns": self.columns,
            "column_names": self.column_names[:30], "dtypes": self.dtypes,
            "cleaning_ops": self.cleaning_ops[-10:], "insights": self.insights[-20:],
            "visualizations": self.visualizations[-10:], "models_used": self.models_used,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class DatasetMemory:
    """Remembers datasets, analyses, and insights across sessions."""

    def __init__(self):
        self._records: Dict[str, DatasetRecord] = {}
        self._loaded = False

    def _ensure_loaded(self):
        if not self._loaded:
            self._load()
            self._loaded = True

    def record_upload(self, filename: str, file_path: str, file_type: str = "",
                      file_size: int = 0) -> DatasetRecord:
        """Record a new dataset upload (or retrieve existing by filename)."""
        self._ensure_loaded()
        key = filename.lower().strip()
        if key in self._records:
            rec = self._records[key]
            rec.analysis_count += 1
            rec.last_analyzed = time.time()
            self._save()
            return rec
        rec = DatasetRecord(
            id=f"ds_{int(time.time())}",
            filename=filename, file_path=file_path,
            file_type=file_type or Path(filename).suffix.lstrip("."),
            file_size=file_size,
        )
        self._records[key] = rec
        self._save()
        return rec

    def record_profile(self, filename: str, rows: int, columns: int,
                       column_names: list[str], dtypes: dict = None):
        """Record dataset profile after initial scan."""
        self._ensure_loaded()
        key = filename.lower().strip()
        if key in self._records:
            rec = self._records[key]
            rec.rows = rows
            rec.columns = columns
            rec.column_names = column_names[:30]
            if dtypes:
                rec.dtypes = dtypes
            self._save()

    def record_cleaning(self, filename: str, operation: str):
        """Record a cleaning operation performed on a dataset."""
        self._ensure_loaded()
        key = filename.lower().strip()
        if key in self._records:
            self._records[key].cleaning_ops.append(operation)
            self._save()

    def record_insight(self, filename: str, insight: str):
        """Record a business insight discovered from the dataset."""
        self._ensure_loaded()
        key = filename.lower().strip()
        if key in self._records:
            self._records[key].insights.append(insight)
            self._save()

    def get_dataset(self, filename: str) -> DatasetRecord | None:
        """Retrieve a dataset record by filename."""
        self._ensure_loaded()
        return self._records.get(filename.lower().strip())

    def has_seen(self, filename: str) -> bool:
        """Check if CARL has seen this dataset before."""
        self._ensure_loaded()
        return filename.lower().strip() in self._records

    def get_previous_context(self, filename: str) -> str:
        """Get a natural language summary of what CARL remembers about this dataset."""
        rec = self.get_dataset(filename)
        if not rec:
            return ""
        parts = [f"Previously analyzed: {rec.filename} ({rec.rows} rows, {rec.columns} columns)"]
        if rec.cleaning_ops:
            parts.append(f"Cleaning: {'; '.join(rec.cleaning_ops[-3:])}")
        if rec.insights:
            parts.append(f"Key insights: {'; '.join(rec.insights[-3:])}")
        return " | ".join(parts)

    def list_all(self) -> list[DatasetRecord]:
        """List all remembered datasets."""
        self._ensure_loaded()
        return sorted(self._records.values(), key=lambda r: -r.last_analyzed)

    def _load(self):
        if not _STORE_PATH.exists():
            return
        try:
            with _lock:
                data = json.loads(_STORE_PATH.read_text(encoding="utf-8"))
            self._records = {d["filename"].lower().strip(): DatasetRecord.from_dict(d) for d in data}
        except Exception as e:
            print(f"[Memory] Dataset load error: {e}")

    def _save(self):
        if len(self._records) > _MAX_DATASETS:
            # Keep most recently analyzed
            sorted_recs = sorted(self._records.values(), key=lambda r: r.last_analyzed, reverse=True)
            self._records = {r.filename.lower().strip(): r for r in sorted_recs[:_MAX_DATASETS]}
        with _lock:
            _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
            _STORE_PATH.write_text(
                json.dumps([r.to_dict() for r in self._records.values()], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
