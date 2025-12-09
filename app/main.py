#main.py 서버 시작
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.routers import auth_routes, setup_routes, sensor_routes, data_routes, dashboard_routes
from app.config_manager import load_config

app = FastAPI()

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
    return {"message": "Server is running"}
