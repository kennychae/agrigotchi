# ai_model.py - LSTM Autoencoder Inference Template

import numpy as np
import pandas as pd
import onnxruntime as ort
from typing import Dict, List, Tuple, Union

class LSTMAutoencoderPredictor:
    """
    LSTM Autoencoder 기반 시계열 이상 탐지 클래스
    - 테이블 데이터(센서값 등)의 시계열 패턴 학습
    - Reconstruction Error 기반 이상 탐지
    """
    
    def __init__(self, 
                 model_path: str,
                 time_steps: int = 10,
                 n_features: int = 5,
                 scaler_params: Dict = None):
        """
        Args:
            model_path: ONNX 모델 경로
            time_steps: 시계열 윈도우 크기 (lookback period)
            n_features: 입력 feature 개수
            scaler_params: 정규화 파라미터 {'mean': [...], 'std': [...]}
        """
        self.time_steps = time_steps
        self.n_features = n_features
        self.scaler_params = scaler_params
        
        # TODO: ONNX Runtime Session 초기화
        # sess_options = ort.SessionOptions()
        # sess_options.intra_op_num_threads = 8
        # sess_options.inter_op_num_threads = 4
        # self.session = ort.InferenceSession(model_path, sess_options)
        # self.input_name = self.session.get_inputs()[0].name
        # self.output_name = self.session.get_outputs()[0].name
        
    def preprocess(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        테이블 데이터를 시계열 윈도우로 변환
        
        Args:
            data: 센서 데이터 (N_samples, n_features)
                 DataFrame 또는 numpy array
            
        Returns:
            windows: (N_windows, time_steps, n_features)
        """
        # TODO(1): DataFrame → numpy 변환
        # if isinstance(data, pd.DataFrame):
        #     data = data.values
        
        # TODO(2): Normalization (학습 시 사용한 scaler 파라미터)
        # if self.scaler_params:
        #     mean = np.array(self.scaler_params['mean'])
        #     std = np.array(self.scaler_params['std'])
        #     data = (data - mean) / std
        
        # TODO(3): Sliding Window 생성
        # windows = []
        # for i in range(len(data) - self.time_steps + 1):
        #     window = data[i:i + self.time_steps]
        #     windows.append(window)
        # windows = np.array(windows)  # (N_windows, time_steps, n_features)
        
        # return windows.astype(np.float32)
        
        pass
    
    def inference(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        LSTM Autoencoder 추론 (Reconstruction)
        
        Args:
            input_tensor: (N_windows, time_steps, n_features)
            
        Returns:
            reconstructed: (N_windows, time_steps, n_features)
                          원본 데이터를 재구성한 결과
        """
        # TODO: ONNX Session Run
        # input_feed = {self.input_name: input_tensor}
        # outputs = self.session.run([self.output_name], input_feed)
        # return outputs[0]
        
        pass
    
    def postprocess(self, 
                    original: np.ndarray,
                    reconstructed: np.ndarray,
                    threshold: float = None) -> Dict:
        """
        Reconstruction Error 계산 및 이상 탐지
        
        Args:
            original: 원본 입력 (N_windows, time_steps, n_features)
            reconstructed: 재구성 출력 (N_windows, time_steps, n_features)
            threshold: 이상 판단 임계값 (None이면 자동 계산)
            
        Returns:
            result: {
                "reconstruction_errors": [float, ...],  # 각 윈도우별 error
                "anomaly_scores": [float, ...],  # 정규화된 이상 점수
                "is_anomaly": [bool, ...],  # 이상 여부
                "threshold": float,
                "anomaly_indices": [int, ...],  # 이상 윈도우 인덱스
            }
        """
        # TODO(1): Reconstruction Error 계산 (MSE, MAE 등)
        # errors = np.mean((original - reconstructed) ** 2, axis=(1, 2))
        
        # TODO(2): Threshold 설정 (자동: mean + 3*std)
        # if threshold is None:
        #     threshold = np.mean(errors) + 3 * np.std(errors)
        
        # TODO(3): 이상 판단
        # is_anomaly = errors > threshold
        # anomaly_indices = np.where(is_anomaly)[0].tolist()
        
        # TODO(4): Anomaly Score 정규화 (0~1)
        # max_error = np.max(errors)
        # anomaly_scores = errors / max_error if max_error > 0 else errors
        
        # return {
        #     "reconstruction_errors": errors.tolist(),
        #     "anomaly_scores": anomaly_scores.tolist(),
        #     "is_anomaly": is_anomaly.tolist(),
        #     "threshold": threshold,
        #     "anomaly_indices": anomaly_indices,
        # }
        
        pass
    
    def predict(self, 
                data_path: str,
                threshold: float = None) -> Dict:
        """
        전체 추론 파이프라인
        
        Args:
            data_path: CSV 파일 경로 또는 DataFrame
            threshold: 이상 탐지 임계값
            
        Returns:
            result: {
                "data_path": str,
                "total_windows": int,
                "anomaly_count": int,
                "anomaly_ratio": float,
                "reconstruction_errors": [...],
                "anomaly_indices": [...],
                "feature_importance": {  # feature별 평균 error
                    "feature_0": float,
                    "feature_1": float,
                    ...
                },
                "summary": {
                    "status": "normal" or "anomaly_detected",
                    "max_error": float,
                    "avg_error": float,
                }
            }
        """
        # TODO(1): 데이터 로드
        # if isinstance(data_path, str):
        #     data = pd.read_csv(data_path)
        # else:
        #     data = data_path  # DataFrame 직접 전달
        
        # TODO(2): 전처리 (windowing)
        # windows = self.preprocess(data)
        
        # TODO(3): 추론 (reconstruction)
        # reconstructed = self.inference(windows)
        
        # TODO(4): 후처리 (anomaly detection)
        # result = self.postprocess(windows, reconstructed, threshold)
        
        # TODO(5): Feature별 중요도 계산 (어느 센서가 이상?)
        # feature_errors = np.mean((windows - reconstructed) ** 2, axis=(0, 1))
        # feature_importance = {
        #     f"feature_{i}": float(err) 
        #     for i, err in enumerate(feature_errors)
        # }
        
        # TODO(6): Summary 생성
        # summary = {
        #     "status": "anomaly_detected" if result["anomaly_indices"] else "normal",
        #     "max_error": float(np.max(result["reconstruction_errors"])),
        #     "avg_error": float(np.mean(result["reconstruction_errors"])),
        # }
        
        # return {
        #     "data_path": data_path if isinstance(data_path, str) else "dataframe",
        #     "total_windows": len(windows),
        #     "anomaly_count": len(result["anomaly_indices"]),
        #     "anomaly_ratio": len(result["anomaly_indices"]) / len(windows),
        #     **result,
        #     "feature_importance": feature_importance,
        #     "summary": summary,
        # }
        
        pass
    
    def visualize(self, 
                  original_data: pd.DataFrame,
                  result: Dict,
                  save_path: str = None) -> None:
        """
        시계열 이상 탐지 결과 시각화
        
        Args:
            original_data: 원본 데이터 DataFrame
            result: predict() 결과
            save_path: 그래프 저장 경로
        """
        # TODO: matplotlib으로 시각화
        # 1. 시계열 그래프 + 이상 구간 하이라이트
        # 2. Reconstruction Error 그래프
        # 3. Feature별 중요도 bar chart
        
        pass
    
    def predict_realtime(self, 
                        data_buffer: List[np.ndarray],
                        threshold: float) -> Dict:
        """
        실시간 추론 (버퍼 기반)
        
        Args:
            data_buffer: 최근 time_steps개 데이터 [(n_features,), ...]
            threshold: 이상 판단 임계값
            
        Returns:
            realtime_result: {
                "is_anomaly": bool,
                "reconstruction_error": float,
                "anomaly_score": float,
                "timestamp": str,
            }
        """
        # TODO(1): 버퍼 검증
        # assert len(data_buffer) == self.time_steps
        
        # TODO(2): 단일 윈도우 생성 및 정규화
        # window = np.array(data_buffer).reshape(1, self.time_steps, self.n_features)
        
        # TODO(3): 추론
        # reconstructed = self.inference(window)
        
        # TODO(4): Error 계산 및 판단
        # error = np.mean((window - reconstructed) ** 2)
        # is_anomaly = error > threshold
        
        # return {
        #     "is_anomaly": bool(is_anomaly),
        #     "reconstruction_error": float(error),
        #     "anomaly_score": float(error / threshold),
        #     "timestamp": pd.Timestamp.now().isoformat(),
        # }
        
        pass


# ===== 사용 예시 =====
if __name__ == "__main__":
    # Scaler 파라미터 (학습 시 저장한 값)
    scaler_params = {
        'mean': [0.5, 1.2, 3.4, 2.1, 0.8],
        'std': [0.1, 0.3, 0.5, 0.2, 0.15]
    }
    
    # 모델 초기화
    predictor = LSTMAutoencoderPredictor(
        model_path="path/to/lstm_autoencoder.onnx",
        time_steps=10,
        n_features=5,
        scaler_params=scaler_params
    )
    
    # 배치 추론 (CSV 파일)
    result = predictor.predict("path/to/sensor_data.csv")
    print(f"이상 탐지: {result['anomaly_count']}개 발견")
    
    # 실시간 추론 (버퍼 기반)
    # buffer = [...]  # 최근 10개 데이터
    # realtime_result = predictor.predict_realtime(buffer, threshold=0.05)