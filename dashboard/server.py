import sys
import os

# Add project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Now import CarlDataEngine
from actions.smart_data_scientist import CarlDataEngine
import os
import shutil
import json
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from actions.smart_data_scientist import CarlDataEngine

app = FastAPI(title="Project Carl - Interactive Dashboard API")

# Enable CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve directory paths accurately
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR.parent / "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount static files folder
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global Carl Engine instance
current_engine: CarlDataEngine = None

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    """Serves app.html dynamically at the root URL."""
    html_file = STATIC_DIR / "app.html"
    if html_file.exists():
        return FileResponse(str(html_file))
    return f"<h2>File 'app.html' not found inside '{STATIC_DIR}'. Please check file location.</h2>"

@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """API Endpoint: Upload dataset (.csv / .parquet) and initialize Carl Engine."""
    global current_engine
    try:
        saved_file_path = UPLOAD_DIR / file.filename
        with open(saved_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        current_engine = CarlDataEngine(str(saved_file_path))
        report = current_engine.get_detailed_report()

        return {
            "status": "success",
            "message": f"Dataset '{file.filename}' loaded successfully into Carl Data Engine.",
            "report": report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chart")
def generate_chart(chart_type: str, x_col: str, y_col: str = None):
    """API Endpoint: Generate interactive Plotly JSON graph."""
    global current_engine
    if not current_engine:
        raise HTTPException(status_code=400, detail="Please upload a dataset first.")

    try:
        chart_json = current_engine.generate_chart(chart_type, x_col, y_col)
        return JSONResponse(content={"status": "success", "chart_json": json.loads(chart_json)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/predict")
def run_prediction(target_col: str = None):
    """API Endpoint: Run Machine Learning predictions on specified target column."""
    global current_engine
    if not current_engine:
        raise HTTPException(status_code=400, detail="Please upload a dataset first.")

    try:
        result = current_engine.predict_target(target_col)
        return {"status": "success", "prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading

    def open_chrome():
        webbrowser.open("http://127.0.0.1:8000")

    # Opens Chrome automatically after 1.5 seconds
    threading.Timer(1.5, open_chrome).start()
    uvicorn.run("dashboard.server:app", host="127.0.0.1", port=8000, reload=True)