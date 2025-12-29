#main.py 서버 시작
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.routers import auth_routes, setup_routes, sensor_routes, data_routes, dashboard_routes
from app.config_manager import load_config
from app.sensor_manager import save_sensor_data
from app.database import init_db

from app.model import leafModel
from ultralytics import YOLO

import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "epoch30.pt")
IMAGE_PATH = os.path.join(BASE_DIR, "static", "images", "cam1.jpg")

app = FastAPI()
init_db()

# Static & Templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Load config on startup
config = load_config()
print("Loaded config:", config)

# Routers
app.include_router(auth_routes.router)
app.include_router(setup_routes.router)
app.include_router(sensor_routes.router)
app.include_router(data_routes.router)
app.include_router(dashboard_routes.router)

@app.get("/")
def root():
    model = YOLO(MODEL_PATH)
    results = model(IMAGE_PATH)

    r = results[0]  # 이미지 1장 기준

    class_counts = {}

    if r.boxes is not None:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            class_name = r.names[cls_id]

            class_counts[class_name] = class_counts.get(class_name, 0) + 1

    print(class_counts)

    ai_result_dict = {
        "type":"disease_check",
        "class_counts": class_counts,
        "total":sum(class_counts.values())
    }

    ai_result_str = json.dumps(ai_result_dict, ensure_ascii=False)

    save_sensor_data(36, 10, 15, ai_result_str)