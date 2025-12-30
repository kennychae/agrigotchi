#main.py 서버 시작
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from app.routers import auth_routes, setup_routes, sensor_routes, data_routes, dashboard_routes
from app.config_manager import load_config
from app.database import init_db

app = FastAPI()
init_db()

# Static & Templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Load config on startup
config = load_config()
print("Loaded config:", config)

# Routers
app.include_router(auth_routes.router)
app.include_router(setup_routes.router)
app.include_router(sensor_routes.router)
app.include_router(data_routes.router)
app.include_router(dashboard_routes.router)

# ai모델용 준비
from pathlib import Path
from app.model import leafModel
from app.model import fruitModel
from ultralytics import YOLO
from app.model.fruitModel import predict_with_sahi
import shutil
import json
from app.sensor_manager import save_sensor_data
from datetime import datetime
from zoneinfo import ZoneInfo

APP_DIR = Path(__file__).resolve().parents[0]                           # .../app
LEAFMODEL_PATH = APP_DIR / "model" / "epoch30.pt"                       # .../app/model/epoch30.pt
FRUITMODEL_PATH = APP_DIR / "model" / "best.pt"                         # .../app/model/best.pt
IMAGE_PATH = APP_DIR / "static" / "images" / "cam1.jpg"                 # .../app/static/images/cam1.jpg
LEAFRESULT_PATH = APP_DIR / "static" / "images" / "leaf_result.jpg"
FRUITRESULT_PATH = APP_DIR / "static" / "images" / "fruit_result.jpg"

model = YOLO(str(LEAFMODEL_PATH))
leaf_model = leafModel.LeafSegmentationModel(str(LEAFMODEL_PATH))
fruit_model = YOLO(str(FRUITMODEL_PATH))

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import logging

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logging.error(exc.errors())
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )

@app.get("/")
def root():
    # model = YOLO(str(MODEL_PATH))
    # results = model(str(IMAGE_PATH))
    #
    # print(results)
    # quick_predict(image_path=str(IMAGE_PATH),
    #               model_path=str(MODEL_PATH),
    #               conf_threshold=0.1,
    #               visualize=False,
    #               save_result=True)
    print(str(APP_DIR))

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

    save_sensor_data(23.9,
                     50.1,
                     439.5,
                     leaf_ai_result_str,
                     fruit_ai_result_str,
                     datetime.now(ZoneInfo("Asia/Seoul")))

    return {"status": "ok", "leaf_result": leaf_ai_result_dict, "fruit_result": fruit_ai_result_dict}

    # return RedirectResponse("/login")