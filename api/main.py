import io
import logging

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from api.schemas import ThresholdEstimate, model_type
from src.inference import InferenceError, estimate_thresholds_from_data

logger = logging.getLogger(__name__)

app = FastAPI()


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=ThresholdEstimate)
async def predict(file: UploadFile = File(...), model: model_type = Form("rf")):
    filename = (file.filename or "").lower()
    contents = await file.read()

    if filename.endswith((".xlsx", ".xls")):
        try:
            data = pd.read_excel(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read Excel file: {e}")
    elif filename.endswith(".csv"):
        try:
            data = pd.read_csv(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read CSV file: {e}")
    else:
        raise HTTPException(status_code=400, detail="File must be .csv, .xls, or .xlsx")

    try:
        vt1_power, vt1_hr, vt2_power, vt2_hr = estimate_thresholds_from_data(data, model_name=model)
    except InferenceError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return ThresholdEstimate(
        vt1_power_W=vt1_power,
        vt1_hr_bpm=vt1_hr,
        vt2_power_W=vt2_power,
        vt2_hr_bpm=vt2_hr,
    )
