#ai_model.py AI 모델 들어갈 곳
def leaf_predict(image_path):
    # TODO: leaf seg 모델 구현
    # TODO(1) : 잎이미지 전처리 구현하기 
    # TODO(2) : 센서 추론 진행
    # TODO(3) : 후처리 진행하기
    # TODO(4) : 결과 시각화하고 저장하기 
    # TODO(5) : 모델 평가하고 반환하기
    return {
        "status": "healthy",
        "image_path": "path/to/image.jpg",
        "detections": [
            {
                "class": "ripe_berry",
                "confidence": 0.95,
                "bbox": [x1, y1, x2, y2],
                "mask": [[x1,y1], [x2,y2], [x3,y3], ...],  # polygon 좌표
                # 또는
                # "mask_rle": "...",  # RLE 인코딩
                # "mask_binary": np.array(...),  # HxW binary mask
            },
            {
                "class": "leaf",
                "confidence": 0.89,
                "bbox": [x1, y1, x2, y2],
                "mask": [[x1,y1], [x2,y2], ...],
            }
        ],
        "summary": {
            "total_area": 1234,  # pixel count
            "berry_area": 567,
            "leaf_area": 667,
        },
        "image_shape": [height, width, channels],
    }

#ai_model.py AI 모델 들어갈 곳
def berry_predict(image_path):
    # TODO: leaf seg 모델 구현
    # TODO(1) : 딸기이미지 전처리 구현하기 
    # TODO(2) : 센서 추론 진행
    # TODO(3) : 후처리 진행하기
    # TODO(4) : 결과 시각화하고 저장하기 
    # TODO(5) : 모델 평가하고 반환하기
    
    return {
        "status": "healthy",
        "image_path": "path/to/image.jpg", 
        "detections": [
            {
                "class": "ripe_berry",  # 또는 class_id: 0
                "confidence": 0.95,
                "bbox": [x1, y1, x2, y2],  # xyxy 형식
                "center": [cx, cy],  # optional
            },
            {
                "class": "unripe_berry",
                "confidence": 0.87,
                "bbox": [x1, y1, x2, y2],
                "center": [cx, cy],
            }
        ],
        "summary": {
            "total_count": 15,
            "ripe_count": 10,
            "unripe_count": 5,
        },
        "image_shape": [height, width, channels],
    }

#ai_model.py AI 모델 들어갈 곳, 시계열 Pred로 시계열 추출하기
def series_predict(temperature, humidity):
    # TODO: 모델 구현
    # TODO(1) : 센서값 전처리 구현하기 
    # TODO(2) : Onnx 모델 호출 해서 추론하기
    # TODO(3) : 후처리해서 값 반환하기 (Ture or False)
    return {"status": "healthy", "score": 0.91}