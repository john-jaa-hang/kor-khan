# 필요 라이브러리 설치
!pip install transformers==4.31.0 kss==3.4.2 pandas tqdm torch

# Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 경로 설정 (Google Drive 기준)
INPUT_CSV_PATH = "/content/drive/MyDrive/데이터/complete_test_stratified_utf8.csv"
OUTPUT_CSV_BASE = "/content/drive/MyDrive/결과/문장분해결과" 
