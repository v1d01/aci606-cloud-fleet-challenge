from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.db import engine, Base, get_db
from app import models

from fastapi.responses import HTMLResponse, RedirectResponse

# Crear tablas en PostgreSQL si no existen
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="ACI606 Cloud Fleet Challenge",
    version="0.1.0-starter",
    description="API for the ACI606 Cloud Fleet Challenge with ML Inference & PostgreSQL.",
)

Status = Literal["NORMAL", "WARNING", "CRITICAL"]


class Telemetry(BaseModel):
    robot_id: str = Field(min_length=1, max_length=20)
    timestamp: float
    x: float
    y: float
    speed: float = Field(ge=0)
    battery: float = Field(ge=0, le=100)
    motor_temp: float
    vibration: float = Field(ge=0)
    motor_current: float = Field(ge=0)
    obstacle_distance: float = Field(ge=0)


class PredictionRequest(Telemetry):
    pass


# Cargar modelo de ML al iniciar
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
model = None

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Modelo cargado exitosamente desde {MODEL_PATH}")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
else:
    print(f"Advertencia: No se encontró el modelo en {MODEL_PATH}")


# Métricas operacionales en memoria
REQUEST_COUNT = 0
PREDICTION_COUNT = 0
TOTAL_LATENCY_MS = 0.0
ERROR_COUNT = 0


@app.get("/health")
def health(db: Session = Depends(get_db)) -> dict:
    unique_robots_count = db.query(models.TelemetryRecord.robot_id).distinct().count()
    return {
        "status": "ok",
        "service": "aci606-cloud-fleet",
        "version": app.version,
        "robots_in_database": unique_robots_count,
        "model_loaded": model is not None,
        "utc": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/telemetry", status_code=status.HTTP_201_CREATED)
def ingest_telemetry(sample: Telemetry, db: Session = Depends(get_db)) -> dict:
    global REQUEST_COUNT
    REQUEST_COUNT += 1

    record = models.TelemetryRecord(**sample.model_dump())
    db.add(record)
    db.commit()
    db.refresh(record)

    return {
        "accepted": True,
        "robot_id": sample.robot_id,
        "server_time": datetime.now(timezone.utc).isoformat(),
        "note": "Persistent mode: stored in PostgreSQL database.",
    }


@app.get("/robots")
def list_robots(db: Session = Depends(get_db)) -> list[dict]:
    subquery = db.query(
        models.TelemetryRecord.robot_id,
        func.max(models.TelemetryRecord.id).label("max_id")
    ).group_by(models.TelemetryRecord.robot_id).subquery()

    latest_records = db.query(models.TelemetryRecord).join(
        subquery, models.TelemetryRecord.id == subquery.c.max_id
    ).all()

    return [
        {
            "robot_id": rec.robot_id,
            "timestamp": rec.timestamp,
            "x": rec.x,
            "y": rec.y,
            "speed": rec.speed,
            "battery": rec.battery,
            "motor_temp": rec.motor_temp,
            "vibration": rec.vibration,
            "motor_current": rec.motor_current,
            "obstacle_distance": rec.obstacle_distance,
        }
        for rec in latest_records
    ]


@app.get("/robots/{robot_id}/history")
def robot_history(robot_id: str, db: Session = Depends(get_db)) -> list[dict]:
    records = db.query(models.TelemetryRecord)\
                .filter(models.TelemetryRecord.robot_id == robot_id)\
                .order_by(models.TelemetryRecord.timestamp.asc())\
                .all()

    if not records:
        raise HTTPException(status_code=404, detail="Robot not found")

    return [
        {
            "robot_id": rec.robot_id,
            "timestamp": rec.timestamp,
            "x": rec.x,
            "y": rec.y,
            "speed": rec.speed,
            "battery": rec.battery,
            "motor_temp": rec.motor_temp,
            "vibration": rec.vibration,
            "motor_current": rec.motor_current,
            "obstacle_distance": rec.obstacle_distance,
        }
        for rec in records
    ]


@app.post("/predict")
def predict(req: PredictionRequest, db: Session = Depends(get_db)) -> dict:
    global PREDICTION_COUNT, TOTAL_LATENCY_MS, ERROR_COUNT, REQUEST_COUNT

    if model is None:
        ERROR_COUNT += 1
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded on the server.",
        )

    start_time = time.perf_counter()

    try:
        # 1. Preparar dataframe con las características requeridas por el modelo
        input_data = pd.DataFrame([{
            "speed": req.speed,
            "battery": req.battery,
            "motor_temp": req.motor_temp,
            "vibration": req.vibration,
            "motor_current": req.motor_current,
            "obstacle_distance": req.obstacle_distance,
        }])

        if input_data.isnull().values.any():
            ERROR_COUNT += 1
            raise HTTPException(status_code=400, detail="Invalid data: Contains NaN values")

        # 2. Inferencia de Machine Learning
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        confidence = float(np.max(probabilities))

        end_time = time.perf_counter()
        inference_ms = round((end_time - start_time) * 1000, 2)

        # 3. Almacenar registro en PostgreSQL para reflejar datos en el Dashboard
        record = models.TelemetryRecord(**req.model_dump())
        db.add(record)
        db.commit()

        # Actualizar contadores operacionales
        REQUEST_COUNT += 1
        PREDICTION_COUNT += 1
        TOTAL_LATENCY_MS += inference_ms

        return {
            "robot_id": req.robot_id,
            "predicted_status": prediction,
            "confidence": round(confidence, 4),
            "model_version": "v1",
            "inference_ms": inference_ms,
        }

    except HTTPException:
        raise
    except Exception as e:
        ERROR_COUNT += 1
        raise HTTPException(status_code=400, detail=f"Bad request or invalid values: {str(e)}")


@app.get("/metrics")
def metrics(db: Session = Depends(get_db)) -> dict:
    total_records = db.query(models.TelemetryRecord).count()
    avg_latency = round(TOTAL_LATENCY_MS / PREDICTION_COUNT, 2) if PREDICTION_COUNT > 0 else 0.0

    return {
        "requests_received": REQUEST_COUNT,
        "prediction_requests": PREDICTION_COUNT,
        "errors": ERROR_COUNT,
        "average_prediction_latency_ms": avg_latency,
        "total_records_stored": total_records,
        "persistence": "PostgreSQL",
        "prediction_service": "OPERATIONAL" if model is not None else "UNAVAILABLE",
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    path = Path(__file__).resolve().parents[1] / "dashboard" / "index.html"
    return path.read_text(encoding="utf-8")

@app.get("/")
def root():
    """Redirige automáticamente la raíz de la aplicación hacia el Dashboard interactivo."""
    return RedirectResponse(url="/dashboard")