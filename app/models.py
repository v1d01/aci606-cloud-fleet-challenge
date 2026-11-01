from sqlalchemy import Column, Integer, String, Float
from app.db import Base

class TelemetryRecord(Base):
    __tablename__ = "telemetry"

    id = Column(Integer, primary_key=True, index=True)
    robot_id = Column(String, index=True, nullable=False)
    timestamp = Column(Float, nullable=False)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    speed = Column(Float, nullable=False)
    battery = Column(Float, nullable=False)
    motor_temp = Column(Float, nullable=False)
    vibration = Column(Float, nullable=False)
    motor_current = Column(Float, nullable=False)
    obstacle_distance = Column(Float, nullable=False)