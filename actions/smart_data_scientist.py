import duckdb
import polars as pl
import plotly.express as px
import plotly.io as pio
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score

class CarlDataEngine:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.con = duckdb.connect(database=':memory:')
        self._load_data()

    def _load_data(self):
        """Lazy loads CSV or Parquet into DuckDB for zero-memory overhead."""
        if self.file_path.endswith('.csv'):
            self.con.execute(f"CREATE VIEW dataset AS SELECT * FROM read_csv_auto('{self.file_path}')")
        elif self.file_path.endswith('.parquet'):
            self.con.execute(f"CREATE VIEW dataset AS SELECT * FROM read_parquet('{self.file_path}')")

    # ==================================================
    # DYNAMIC DATA DETAILS & STATISTICAL AUDIT
    # ==================================================
    def get_detailed_report(self) -> dict:
        """Generates a complete statistical and structural breakdown of ANY dataset."""
        df_raw = self.con.execute("SELECT * FROM dataset").df()
        
        # Identify and ignore index/ID columns
        id_cols = [c for c in df_raw.columns if 'unnamed' in c.lower() or c.lower() in ['id', 'index', 'case id']]
        df = df_raw.drop(columns=id_cols, errors='ignore')

        total_rows, total_cols = df.shape
        missing_counts = df.isnull().sum()
        missing_pcts = (missing_counts / total_rows * 100).round(2)

        column_details = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            unique_cnt = df[col].nunique()
            missing_cnt = int(missing_counts[col])
            missing_pct = missing_pcts[col]

            col_info = {
                "column_name": col,
                "data_type": dtype,
                "unique_values": unique_cnt,
                "missing_values": f"{missing_cnt} ({missing_pct}%)"
            }

            # Add numeric or categorical specifics
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info["min"] = float(df[col].min()) if not df[col].isnull().all() else None
                col_info["max"] = float(df[col].max()) if not df[col].isnull().all() else None
                col_info["mean"] = round(float(df[col].mean()), 2) if not df[col].isnull().all() else None
            else:
                top_cats = df[col].value_counts().head(3).to_dict()
                col_info["top_categories"] = top_cats

            column_details.append(col_info)

        # Auto-detect default target (last valid column)
        auto_target = df.columns[-1]

        return {
            "file_name": self.file_path,
            "total_rows": total_rows,
            "total_columns": total_cols,
            "ignored_id_columns": id_cols,
            "auto_detected_target": auto_target,
            "column_breakdown": column_details,
            "sample_data": df.head(3).to_dict(orient="records")
        }

    # ==================================================
    # DYNAMIC CHART GENERATION (PLOTLY)
    # ==================================================
    def generate_chart(self, chart_type: str, x_col: str, y_col: str = None) -> str:
        """Generates interactive Plotly JSON for frontend rendering."""
        if y_col:
            query = f'SELECT "{x_col}", "{y_col}" FROM dataset LIMIT 5000'
            df = self.con.execute(query).df()
            if chart_type == "bar":
                fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
            elif chart_type == "line":
                fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
            elif chart_type == "scatter":
                fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
            else:
                fig = px.box(df, x=x_col, y=y_col, title=f"Boxplot of {y_col} by {x_col}")
        else:
            query = f'SELECT "{x_col}" FROM dataset LIMIT 5000'
            df = self.con.execute(query).df()
            fig = px.histogram(df, x=x_col, title=f"Distribution of {x_col}")

        return pio.to_json(fig)

    # ==================================================
    # AUTOMATED DATA CLEANING & POLISHING
    # ==================================================
    def clean_and_polish_data(self) -> pd.DataFrame:
        """Strips dirty currency symbols, drops index/ID columns, and imputes missing values."""
        df = self.con.execute("SELECT * FROM dataset").df()

        # Drop ID/Index columns automatically
        id_cols = [c for c in df.columns if 'unnamed' in c.lower() or c.lower() in ['id', 'index', 'case id']]
        df = df.drop(columns=id_cols, errors='ignore')

        for col in df.columns:
            if df[col].dtype == object:
                # Strip text symbols if it's formatted numerical data
                cleaned_series = df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True)
                numeric_converted = pd.to_numeric(cleaned_series, errors='coerce')
                
                # If >60% converted to valid numbers, convert column
                if numeric_converted.notna().sum() > 0.6 * len(df) and df[col].nunique() > 2:
                    df[col] = numeric_converted
                else:
                    # Impute missing string categories with 'Unknown'
                    df[col] = df[col].fillna('Unknown').astype(str).str.strip()

        # Fill missing numericals with column median
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            df[col] = df[col].fillna(df[col].median())

        return df

    # ==================================================
    # DYNAMIC MACHINE LEARNING PREDICTION PIPELINE
    # ==================================================
    def predict_target(self, target_column: str = None) -> dict:
        """Predicts any target column dynamically and outputs accuracy + key drivers."""
        df = self.clean_and_polish_data()

        # Auto-select target column if none specified
        if not target_column or target_column not in df.columns:
            target_column = df.columns[-1]

        df = df.dropna(subset=[target_column])
        
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # One-hot encode categorical features
        X = pd.get_dummies(X, drop_first=True)

        if len(X) < 5:
            return {"error": "Dataset is too small to train a machine learning model."}

        # Train/Test Split
        if len(X) < 20:
            X_train, X_test, y_train, y_test = X, X, y, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Auto-detect Regression vs Classification
        if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            score = r2_score(y_test, predictions)
            task_type = "Regression (R² Score)"
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            score = accuracy_score(y_test, predictions)
            task_type = "Classification (Accuracy)"

        importances = dict(zip(X.columns, model.feature_importances_))
        sorted_importance = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "target": target_column,
            "task_type": task_type,
            "accuracy_score": f"{round(score * 100, 2)}%",
            "top_key_factors": [(factor, round(float(weight), 4)) for factor, weight in sorted_importance]
        }