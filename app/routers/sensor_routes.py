#sensor_routes.py 센서 -> 서버 API
from fastapi import APIRouter, UploadFile, File, Form, Request
from typing import Optional
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import shutil
import json

from app.database import SessionLocal, SensorData as SensorDataModel
from app.sensor_manager import save_sensor_data

# ai모델용 준비
from app.model import leafModel
from app.model import fruitModel
from ultralytics import YOLO
from app.model.fruitModel import predict_with_sahi

APP_DIR = Path(__file__).resolve().parents[1]                           # .../app
LEAFMODEL_PATH = APP_DIR / "model" / "epoch30.pt"                       # .../app/model/epoch30.pt
FRUITMODEL_PATH = APP_DIR / "model" / "best.pt"                         # .../app/model/best.pt
IMAGE_PATH = APP_DIR / "static" / "images" / "cam1.jpg"                 # .../app/static/images/cam1.jpg
LEAFRESULT_PATH = APP_DIR / "static" / "images" / "leaf_result.jpg"
FRUITRESULT_PATH = APP_DIR / "static" / "images" / "fruit_result.jpg"

model = YOLO(str(LEAFMODEL_PATH))
leaf_model = leafModel.LeafSegmentationModel(str(LEAFMODEL_PATH))
fruit_model = YOLO(str(FRUITMODEL_PATH))

router = APIRouter()

from fastapi.responses import JSONResponse
import logging

@router.post("/api/sensor/ingest")
async def ingest_sensor_data(
    request: Request,
    temperature: Optional[float] = Form(...),
    humidity: Optional[float] = Form(...),
    co2: Optional[float] = Form(...),
    status: str = Form(""),
    timestamp: str = Form(""),
    image: Optional[UploadFile] = File(...),
):
    with IMAGE_PATH.open("wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # 예측 수행
    results = leaf_model.predict(str(IMAGE_PATH), conf=0.25)
    # 결과 시각화 및 저장
    leaf_model.visualize(results, save_path=str(LEAFRESULT_PATH))

    fruit_result = predict_with_sahi(image_path=str(IMAGE_PATH),
                                     model_path=str(FRUITMODEL_PATH),
                                     overlap_width_ratio=0.5,
                                     overlap_height_ratio=0.5,
                                     conf_threshold=0.5,
                                     save_path=str(FRUITRESULT_PATH))

    leaf_result = model(str(IMAGE_PATH))

    r = leaf_result[0]  # 이미지 1장 기준

    leaf_class_counts = {}

    if r.boxes is not None:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            class_name = r.names[cls_id]

            leaf_class_counts[class_name] = leaf_class_counts.get(class_name, 0) + 1

    fruit_class_counts = {}

    for i, pred in enumerate(fruit_result):
        class_id = pred.category.id
        class_name = pred.category.name
        if class_name not in fruit_class_counts:
            fruit_class_counts[class_name] = 0
        fruit_class_counts[class_name] += 1

    leaf_ai_result_dict = {
        "type": "disease_check",
        "class_counts": leaf_class_counts,
        "total": sum(leaf_class_counts.values())
    }

    fruit_ai_result_dict = {
        "type": "disease_check",
        "class_counts": fruit_class_counts,
        "total": sum(fruit_class_counts.values())
    }

    leaf_ai_result_str = json.dumps(leaf_ai_result_dict, ensure_ascii=False)
    fruit_ai_result_str = json.dumps(fruit_ai_result_dict, ensure_ascii=False)

    save_sensor_data(temperature,
                     humidity,
                     co2,
                     leaf_ai_result_str,
                     fruit_ai_result_str,
                     datetime.now(ZoneInfo("Asia/Seoul")))

    return {"status": "ok", "leaf_result": leaf_ai_result_dict, "fruit_result": fruit_ai_result_dict}

# @router.post("/api/sensor/ingest")
# async def ingest_sensor_data(request: Request):
#     ct = request.headers.get("content-type")
#     logging.error(f"INGEST content-type: {ct}")
#
#     try:
#         form = await request.form()
#         keys = list(form.keys())
#         file_keys = [k for k, v in form.items() if hasattr(v, "filename")]
#         logging.error(f"INGEST form keys: {keys}")
#         logging.error(f"INGEST file keys: {file_keys}")
#     except Exception as e:
#         logging.error(f"INGEST form parse error: {repr(e)}")
#         return JSONResponse(status_code=400, content={"status": "bad_request", "error": repr(e)})
#
#     missing = []
#     for k in ["temperature", "humidity", "co2", "image"]:
#         if k not in form:
#             missing.append(k)
#
#     if missing:
#         return JSONResponse(
#             status_code=422,
#             content={
#                 "status": "validation_failed",
#                 "content_type": ct,
#                 "received_keys": keys,
#                 "missing": missing,
#             },
#         )
#
#     # 여기까지 오면 폼/파일은 정상적으로 들어온 것
#     return {"status": "ok", "content_type": ct, "received_keys": keys}