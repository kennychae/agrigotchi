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

from pathlib import Path
from ultralytics import YOLO
from app.model.fruitModel import predict_with_sahi

APP_DIR = Path(__file__).resolve().parents[0]               # .../app
# APP_DIR = Path(__file__).resolve().parents[1]               # .../app
MODEL_PATH = APP_DIR / "model" / "best.pt"               # .../app/model/epoch30.pt
IMAGE_PATH = APP_DIR / "static" / "images" / "cam3.jpg"     # .../app/static/images/cam1.jpg
RESULT_PATH = APP_DIR / "static" / "images" / "result.jpg"

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

    return RedirectResponse("/login")