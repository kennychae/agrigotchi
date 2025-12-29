#data_routes.py 웹에서 데이터 조회
from fastapi import APIRouter
from sqlalchemy.orm import Session
from app.database import SessionLocal, SensorData

router = APIRouter()

@router.get("/api/data/latest")
def get_latest_data():
    db: Session = SessionLocal()
    data = db.query(SensorData).order_by(SensorData.timestamp.desc()).first()
    db.close()

    if not data:
        return {
            "temperature": None,
            "humidity": None,
            "co2": None,
            "timestamp": None
        }

    return {
        "temperature": data.temperature,
        "humidity": data.humidity,
        "co2": data.co2,
        "timestamp": data.timestamp.strftime("%Y-%m-%d %H:%M:%S")
    }

@router.get("/api/data/all")
def get_all_data():
    db = SessionLocal()
    rows = db.query(SensorData).order_by(SensorData.timestamp.asc()).all()
    db.close()

    result = []
    for r in rows:
        result.append({
            "temperature": r.temperature,
            "humidity": r.humidity,
            "co2": r.co2,
            "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        })

    return result

@router.get("/api/ai/latest")
def get_latest_ai():
    db = SessionLocal()
    row = db.query(SensorData).order_by(SensorData.timestamp.desc()).first()
    db.close()

    if not row:
        return {"timestamp": None, "ai_result": None}

    return {
        "timestamp": row.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "ai_result": row.ai_result
    }

@router.get("/api/ai/all")
def get_all_ai():
    db = SessionLocal()
    rows = db.query(SensorData).order_by(SensorData.timestamp.asc()).all()
    db.close()

    result = []
    for r in rows:
        result.append({
            "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "ai_result": r.ai_result
        })
    return result