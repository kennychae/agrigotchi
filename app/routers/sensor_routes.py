#sensor_routes.py 센서 -> 서버 API
from fastapi import APIRouter, UploadFile, File, Form
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import shutil
import json

from app.database import SessionLocal, SensorData as SensorDataModel

# ai모델용 준비
from app.model import leafModel
from ultralytics import YOLO

APP_DIR = Path(__file__).resolve().parents[1]               # .../app
MODEL_PATH = APP_DIR / "model" / "epoch30.pt"               # .../app/model/epoch30.pt
IMAGE_PATH = APP_DIR / "static" / "images" / "cam1.jpg"     # .../app/static/images/cam1.jpg
RESULT_PATH = APP_DIR / "static" / "images" / "result.jpg"

leaf_model = leafModel.LeafSegmentationModel(str(MODEL_PATH))

router = APIRouter()

@router.post("/api/sensor/ingest")
def ingest_sensor_data(
    temperature: float = Form(...),
    humidity: float = Form(...),
    co2: float = Form(...),
    status: str = Form(""),
    timestamp: str = Form(""),
    image: UploadFile = File(...),
):
    with save_path.open("wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # 예측 수행
    results = leaf_model.predict(payload.image or IMAGE_PATH, conf=0.25)

    # 마스크 추출
    masks = leaf_model.get_masks(results)
    ai_result_dict = {
        "leaf_count": int(len(masks) if masks is not None else 0)
    }
    ai_result_str = json.dumps(ai_result_dict, ensure_ascii=False)

    # 결과 시각화 및 저장
    model.visualize(results, save_path=RESULT_PATH)

    db = SessionLocal()
    try:
        row = SensorDataModel(
            temperature=temperature,
            humidity=humidity,
            co2=co2,
            ai_result=ai_result_str,
            timestamp=datetime.now(ZoneInfo("Asia/Seoul")),
        )
        db.add(data)
        db.commit()
    finally:
        db.close()

    return {"status": "ok", "ai_result": ai_result_dict}