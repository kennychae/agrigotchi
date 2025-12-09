#data_routes.py 웹에서 데이터 조회
from fastapi import APIRouter

router = APIRouter()

@router.get("/api/data/latest")
def get_latest_data():
    # TODO: DB에서 최근 데이터 가져오기
    return {
        "temperature": 24.5,
        "humidity": 61,
        "ai_result": "healthy"
    }