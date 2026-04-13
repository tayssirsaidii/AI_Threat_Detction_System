import os
import json
import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, String, Integer, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from starlette.exceptions import HTTPException as StarletteHTTPException
from sklearn.metrics import classification_report
from datetime import datetime

# FastAPI app setup
app = FastAPI()

# Database Setup
SQLALCHEMY_DATABASE_URL = "postgresql://csadmin:cssecurepassword@postgres/cshybrid_db"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

# Middleware to limit upload size
class LimitUploadSizeMiddleware:
    def __init__(self, app, max_size: int):
        self.app = app
        self.max_size = max_size

    async def __call__(self, request, call_next):
        content_length = request.headers.get("Content-Length")
        if content_length:
            try:
                if int(content_length) > self.max_size:
                    raise StarletteHTTPException(status_code=413, detail="Payload Too Large")
            except ValueError:
                pass
        response = await call_next(request)
        return response

app.add_middleware(LimitUploadSizeMiddleware, max_size=10 * 1024 * 1024)  # 10MB limit

# Load the model during app startup
model = None

@app.on_event("startup")
def load_model():
    global model
    model_path = os.path.join("rf_classifier_model.pkl")
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            print("Model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    else:
        raise RuntimeError(f"Model file not found at {model_path}!")

# Function to generate recommended actions based on threat type
def generate_recommendations(threat_class: str):
    recommendations = {
        "malware": "Activate MFA, Review network activity, Ensure proper antivirus is active.",
        "unauthorized_access": "Enable RBAC, Activate MFA, Review user permissions.",
        "suspicious_activity": "Monitor activity logs, Increase system alerting thresholds.",
    }
    return recommendations.get(threat_class, "Review system security settings.")

# Create a report model for the database
class Report(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(String, index=True)
    model_version = Column(String)
    detected_threats = Column(JSON)
    recommendations = Column(JSON)
    report_data = Column(JSON)

# Create the database tables
Base.metadata.create_all(bind=engine)

# Store the report in PostgreSQL database
async def store_report_in_db(report_data):
    db = SessionLocal()  # Sync session used here
    db_report = Report(
        timestamp=report_data["timestamp"],
        model_version=report_data["model_version"],
        detected_threats=report_data["detected_threats"],
        recommendations=report_data["recommendations"],
        report_data=report_data["report"],
    )
    db.add(db_report)
    db.commit()
    db.refresh(db_report)
    db.close()

# Endpoint to generate a classification report
@app.post("/generate-report")
async def generate_report_endpoint(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    # Read uploaded JSON file
    try:
        content = await file.read()
        data = json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format.")

    # Extract X (features) and y_true (labels)
    if "X" not in data or "y_true" not in data:
        raise HTTPException(status_code=400, detail="JSON must contain 'X' and 'y_true' keys.")

    X = pd.DataFrame(data["X"])  # Convert to DataFrame
    y_true = data["y_true"]       # Extract true labels

    if X.shape[1] != model.n_features_in_:
        raise HTTPException(status_code=400, detail=f"Expected {model.n_features_in_} features, but got {X.shape[1]}.")

    # Predict
    try:
        y_pred = model.predict(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Generate classification report
    report = classification_report(y_true, y_pred, output_dict=True)

    # Detect threat and recommend actions
    detected_threats = set(y_pred)  # Unique predicted classes (threats)
    recommendations = {str(threat): generate_recommendations(str(threat)) for threat in detected_threats}

    # Metadata for the report
    report_metadata = {
        "timestamp": datetime.now().isoformat(),
        "model_version": "1.0",  # You can update this dynamically if needed
        "detected_threats": list(detected_threats),
        "recommendations": recommendations,
    }

    # Prepare the report data to be stored in the database
    report_data = {**report_metadata, "report": report}

    # Store the report in the PostgreSQL database
    await store_report_in_db(report_data)

    return {"report": report, "metadata": report_metadata}

# Endpoint to get feature names
@app.get("/get-feature-names")
def get_feature_names():
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    return {"features": model.feature_names_in_.tolist()}
