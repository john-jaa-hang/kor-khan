# KOR-KHAN: 정확한 정치적 입장 예측을 위한 지식 인식 계층적 주의 네트워크

이 저장소는 KHAN(Knowledge-Aware Hierarchical Attention Networks) 모델을 기반으로 한국어 정치 기사의 정치적 성향을 분류하는 프로젝트입니다. KHAN 모델은 계층적 주의 네트워크(HAN)와 지식 인코딩(KE)의 두 가지 주요 구성 요소로 이루어져 있습니다.

## 프로젝트 개요

이 프로젝트는 온라인 뉴스 기사의 정치적 입장을 자동으로 분류하는 것을 목표로 합니다. 논문에서 제안된 KHAN 모델은 다음 두 가지 구성 요소를 통해 높은 정확도를 달성합니다:

1. **계층적 주의 네트워크(HAN)**: 단어, 문장, 제목 수준의 3단계 계층을 통해 텍스트의 로컬 및 글로벌 컨텍스트를 효과적으로 포착합니다.
2. **지식 인코딩(KE)**: 실제 엔티티에 대한 일반 및 정치적 지식을 통합하여 더 정확한 분류를 제공합니다.

**현재 상태**: 현재는 계층적 주의 네트워크(HAN) 부분만 구현되어 있으며, 지식 인코딩(KE) 부분은 아직 구현되지 않았습니다.

## 설치 방법

### 필수 요구사항

- Python 3.7 이상
- PyTorch 1.7 이상
- Windows, macOS 또는 Linux

### 패키지 설치

```bash
pip install numpy pandas torch scikit-learn matplotlib seaborn tqdm
```

### MeCab 설치 (형태소 분석기)

이 프로젝트는 한국어 텍스트 처리를 위해 MeCab 형태소 분석기를 사용합니다. Windows에서의 설치 방법은 다음과 같습니다:

#### Windows 설치 방법

1. **MeCab 설치**:
   - [mecab-ko-msvc](https://github.com/Pusnow/mecab-ko-msvc/releases)에서 `mecab-ko-msvc-x64.zip`을 다운로드하고 압축을 풉니다.
   - 압축 해제한 파일을 `C:\mecab`에 복사합니다.

2. **사전 설치**:
   - [mecab-ko-dic-msvc](https://github.com/Pusnow/mecab-ko-dic-msvc/releases)에서 `mecab-ko-dic-msvc-x64.zip`을 다운로드하고 압축을 풉니다.
   - 압축 해제한 파일을 `C:\mecab\mecab-ko-dic`에 복사합니다.

3. **Python 바인딩 설치**:
   ```bash
   pip install mecab-python3
   ```

4. **환경 변수 설정**:
   - 시스템 환경 변수 `PATH`에 `C:\mecab`을 추가합니다.
   - 필요한 경우 `MECABRC` 환경 변수를 `C:\mecab\mecab-ko-dic\mecabrc`로 설정합니다.

**참고**: MeCab 설치에 실패할 경우, 코드는 KoNLPy의 Okt 또는 기본 공백 기반 토크나이저를 대체로 사용합니다.


### 주요 모듈 설명

1. **텍스트 전처리**:
   - MeCab을 사용한 형태소 분석
   - 문장 분리 및 정제
   - 어휘 사전 구축

2. **계층적 주의 네트워크 (HAN)**:
   - 단어 수준 주의 모듈: 문장 내 단어들 간의 관계 학습
   - 문장 수준 주의 모듈: 문서 내 문장들 간의 관계 학습
   - 제목 수준 주의 모듈: 제목과 문장 간의 관계 학습

3. **학습 및 평가**:
   - 교차 엔트로피 손실 함수
   - Adam 최적화 알고리즘
   - 정확도, F1 점수, 혼동 행렬 평가

4. **결과 시각화**:
   - 학습 곡선
   - 혼동 행렬
   - 주의 가중치 시각화

## 사용 방법

### 데이터 경로 설정


```python
config = {
    'data_path': r'경로/complete_test_stratified_utf8.csv',
    'results_dir': 'results',
    # 기타 설정...
}
```

## 미구현된 부분: 지식 인코딩(KE)

지식 인코딩(KE) 모듈은 아직 구현되지 않았습니다. 이 모듈은 다음과 같은 기능을 제공할 예정입니다:

1. **지식 그래프 구축**:
   - 일반 지식 그래프(예: YAGO)
   - 정치적 지식 그래프(KG-lib, KG-con)

2. **지식 임베딩**:
   - 지식 그래프에서 엔티티 관계 학습

3. **지식 주입**:
   - 일반 및 정치적 지식을 단어 임베딩에 통합
   - 다양한 정치적 관점의 지식 융합

지식 인코딩 모듈을 구현하려면 지식 그래프와 관련 임베딩 방법(예: RotatE, ModE, HAKE)에 대한 추가 작업이 필요합니다.

## 모델 아키텍처

### 계층적 주의 네트워크(HAN)

![HAN 아키텍처](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSQtvIj3Mno-yQiEYFthjwFVwVOOt9QWV1xZw&usqp=CAU)

HAN은 다음과 같은 계층으로 구성됩니다:

1. **단어 수준 주의**: 문장 내 중요한 단어 강조
2. **문장 수준 주의**: 문서 내 중요한 문장 강조
3. **제목 수준 주의**: 제목과 관련된 문장 강조

각 계층은 양방향 GRU와 주의 메커니즘을 사용하여 문맥을 포착합니다.

## 알려진 문제점 및 해결 방법

1. **MeCab 설치 오류**:
   - Windows에서 MeCab 설치가 실패할 경우, `mecab-python3`를 재설치하거나 환경 변수를 확인하세요.
   - 코드는 MeCab 실패 시 Okt 또는 기본 토크나이저로 대체합니다.

2. **메모리 문제**:
   - 대용량 데이터셋 처리 시 메모리 오류가 발생할 수 있습니다.
   - 배치 크기를 줄이거나 문서당 최대 문장 수를 제한하여 해결할 수 있습니다.


## 참고 문헌

1. Ko, Y., Ryu, S., Han, S., Jeon, Y., Kim, J., Park, S., Han, K., Tong, H., & Kim, S. W. (2023). KHAN: Knowledge-Aware Hierarchical Attention Networks for Accurate Political Stance Prediction. In Proceedings of the ACM Web Conference 2023.

2. Kim, B., Lee, E., & Na, D. (2023). A New Korean Text Classification Benchmark for Recognizing the Political Intents in Online Newspapers. arXiv preprint arXiv:2311.01712.
