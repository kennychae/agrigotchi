#setup_routes.py 초기 설정
from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from app.config_manager import load_config, save_config
from app.auth import is_logged_in

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/setup", response_class=HTMLResponse)
def setup_page(request: Request):
    if not is_logged_in(request):
        return RedirectResponse("/login", status_code=302)

    return templates.TemplateResponse("setup.html", {"request": request})

@router.post("/setup_complete")
def setup_complete():
    cfg = load_config()
    cfg["setup_completed"] = True
    save_config(cfg)
    return {"status": "ok"}

@router.post("/change_password")
def change_password(new_password: str = Form(...)):
    cfg = load_config()
    cfg["admin_password"] = new_password
    save_config(cfg)
    return {"status": "ok", "message": "비밀번호가 변경되었습니다!"}

@router.post("/register_sensor")
def register_sensor(
    device_name: str = Form(...),
    device_ip: str = Form(...),
    device_enabled: str = Form("0"),
):
    cfg = load_config()

    # sensors 키가 없으면 초기화
    if "sensors" not in cfg or not isinstance(cfg["sensors"], list):
        cfg["sensors"] = []

    sensor_obj = {
        "id": device_name,
        "ip": device_ip,
        "enabled": (device_enabled == "1"),
    }

    cfg["sensors"].append(sensor_obj)
    save_config(cfg)

    return {"status": "ok", "message": "기기 설정이 저장되었습니다!"}