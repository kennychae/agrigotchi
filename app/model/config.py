import os
from pathlib import Path
import platform

# 환경 설정
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

CLASS_NAMES = ["OK", "NG"]

# 예측 설정
PREDICTION_CONFIG = {
    'conf_threshold': 0.25,
    'iou_threshold': 0.45,
}

# Matplotlib 한글 폰트 설정
def setup_matplotlib_font():
    """플랫폼에 따라 한글 폰트 설정"""
    import matplotlib.pyplot as plt

    if platform.system() == 'Darwin':  # macOS
        plt.rcParams['font.family'] = 'AppleGothic'
    elif platform.system() == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    else:  # Linux (Colab)
        plt.rcParams['font.family'] = 'NanumGothic'

    plt.rcParams['axes.unicode_minus'] = False
