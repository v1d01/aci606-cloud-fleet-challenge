from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.db import engine, Base, get_db
from app import models

# Crear las tablas en la base de datos automáticamente al arrancar
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="ACI606 Cloud Fleet Challenge",
    version="0.1.0-starter",
    description="API for the ACI606 Cloud Fleet Challenge with PostgreSQL persistence.",
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


REQUEST_COUNT = 0


@app.get("/health")
def health(db: Session = Depends(get_db)) -> dict:
    unique_robots_count = db.query(models.TelemetryRecord.robot_id).distinct().count()
    return {
        "status": "ok",
        "service": "aci606-cloud-fleet",
        "version": app.version,
        "robots_in_database": unique_robots_count,
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
def predict(_: PredictionRequest) -> dict:
    raise HTTPException(
        status_code=501,
        detail="/predict is part of the challenge. Train and integrate your model.",
    )


@app.get("/metrics")
def metrics(db: Session = Depends(get_db)) -> dict:
    total_records = db.query(models.TelemetryRecord).count()
    return {
        "requests_received": REQUEST_COUNT,
        "total_records_stored": total_records,
        "persistence": "PostgreSQL",
        "prediction_service": "NOT_IMPLEMENTED",
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    path = Path(__file__).resolve().parents[1] / "dashboard" / "index.html"
    return path.read_text(encoding="utf-8")