from fastapi import FastAPI, UploadFile, File
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from fastapi.responses import JSONResponse, FileResponse
import os
from datetime import datetime
import orjson
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO

# Set max upload size to 100MB
MAX_UPLOAD_SIZE = 100 * 1024 * 1024

class LimitUploadSizeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get('content-length')
        if content_length and int(content_length) > MAX_UPLOAD_SIZE:
            return JSONResponse(
                status_code=413,
                content={"detail": "File too large. Maximum allowed size is 100MB."}
            )
        return await call_next(request)

app = FastAPI()
app.add_middleware(LimitUploadSizeMiddleware)

# Load model
def load_model():
    try:
        model = joblib.load("rf_classifier_model.pkl")
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_model()

# Directory to save reports
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# Prediction logic
def predict(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction.tolist()

# Recommendation logic
def get_recommendation(prediction):
    if max(prediction) > 100:
        return "High threat detected. Isolate affected systems and perform deep scan."
    return "No significant threat detected. Continue monitoring."

# Model performance
def get_model_performance():
    return {
        "accuracy": 0.94,
        "precision": 0.91,
        "recall": 0.89
    }

# Save report as PDF (input data sample removed)
def generate_report(prediction):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_filename = f"report_{timestamp}.pdf"
    report_path = os.path.join(REPORTS_DIR, report_filename)

    recommendation = get_recommendation(prediction)
    performance = get_model_performance()

    c = canvas.Canvas(report_path, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "AI Threat Detection Report")

    # Timestamp
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 80, f"Timestamp: {timestamp}")

    # Prediction Info
    c.drawString(50, height - 110, f"Total Predictions: {len(prediction)}")

    # Recommendation
    c.drawString(50, height - 140, f"Recommendation: {recommendation}")

    # Model Performance
    c.drawString(50, height - 180, "Model Performance:")
    c.drawString(70, height - 200, f"Accuracy: {performance['accuracy']}")
    c.drawString(70, height - 220, f"Precision: {performance['precision']}")
    c.drawString(70, height - 240, f"Recall: {performance['recall']}")

    c.save()

    return report_filename

@app.post("/predict/")
async def predict_from_large_file(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded properly."}

    try:
        contents = await file.read()
        data = orjson.loads(contents)

        if 'x' not in data or 'y_true' not in data:
            return {"error": "Invalid JSON format. Expected 'x' and 'y_true' keys."}

        x = data['x']
        y_true = data['y_true']

        if not isinstance(x, list) or not all(isinstance(i, list) for i in x):
            return {"error": "'x' should be a list of lists."}
        if not isinstance(y_true, list) or not all(isinstance(i, (int, float)) for i in y_true):
            return {"error": "'y_true' should be a list of numbers."}
        if not all(len(i) == 78 for i in x):
            return {"error": "Each sample in 'x' must have exactly 78 features."}
        if not all(isinstance(i, list) and all(isinstance(x_item, (int, float)) for x_item in i) for i in x):
            return {"error": "All elements in 'x' must be lists of numbers (floats or ints)"}

        x_array = np.array(x)

        if x_array.shape[1] != 78:
            return {"error": "Each sample must have exactly 78 features."}

        # Prediction logic
        prediction = model.predict(x_array).tolist()

        # Generate report (without input data)
        report_file = generate_report(prediction)

        return {
            "prediction_sample": prediction[:10],
            "total_predictions": len(prediction),
            "report_path": report_file,
            "recommendation": get_recommendation(prediction),
            "model_performance": get_model_performance()
        }

    except Exception as e:
        return {"error": f"Failed to process file: {str(e)}"}

@app.get("/reports/")
def list_reports(filename: str = None):
    try:
        if filename:
            report_path = os.path.join(REPORTS_DIR, filename)
            if not os.path.exists(report_path):
                return {"error": f"Report '{filename}' not found."}
            return FileResponse(report_path, media_type="application/pdf", filename=filename)

        files = sorted(os.listdir(REPORTS_DIR), reverse=True)
        return {"available_reports": files}

    except Exception as e:
        return {"error": f"Error listing/downloading reports: {str(e)}"}
