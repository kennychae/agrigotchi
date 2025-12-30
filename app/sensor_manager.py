#sensor_manager.py 센서 검색 및 통신

from app.database import SessionLocal, SensorData

def save_sensor_data(temp, hum, co2, ai_result, ai_result2, timestamp):
    db = SessionLocal()
    data = SensorData(
        temperature=temp,
        humidity=hum,
        co2=co2,
        ai_result=ai_result,
        ai_result2=ai_result2,
        timestamp=timestamp
    )
    db.add(data)
    db.commit()
    db.close()