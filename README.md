# kor-khan

# KHAN 모델 구현을 위한 한국어 형태소 분석기(MeCab) 설정 가이드

## 개요
이 문서는 KHAN(Knowledge-Aware Hierarchical Attention Networks) 모델 학습을 위한 MeCab 한국어 형태소 분석기 설치 및 구성 방법을 설명합니다.

## 설치 과정

### 1. MeCab 설치
1. [mecab-ko-msvc](https://github.com/Pusnow/mecab-ko-msvc/releases)에서 최신 버전의 `mecab-ko-windows-x64.zip` 다운로드
2. 압축 해제 후 내용물을 `C:\mecab` 폴더에 복사 (폴더가 없으면 생성)
   - 설치 후 `C:\mecab` 내에 `bin`, `etc`, `include`, `lib`, `share` 폴더가 생성됩니다.

### 2. 한국어 사전(mecab-ko-dic) 설치
1. [mecab-ko-dic-msvc](https://github.com/Pusnow/mecab-ko-dic-msvc/releases)에서 최신 버전의 `mecab-ko-dic-msvc.zip` 다운로드
2. 압축 해제 후 내용물을 `C:\mecab\mecab-ko-dic` 폴더에 복사 (폴더가 없으면 생성)
   - 사전 파일들(`char.def`, `dicrc`, `feature.def`, `matrix.bin` 등)이 해당 경로에 위치해야 합니다.

### 3. mecabrc 설정 파일 생성
1. `C:\mecab` 폴더에 `mecabrc` 파일 생성 (확장자 없음)
2. 파일에 아래 내용 입력:
```
dicdir = C:/mecab/mecab-ko-dic
userdic = 
output-format-type = wakati
charset = UTF-8
```

### 4. 환경 변수 설정
1. Windows 검색에서 "환경 변수"를 검색하고 "시스템 환경 변수 편집" 선택
2. "환경 변수" 클릭
3. "시스템 변수" 섹션에서 Path 변수 찾아 편집
4. "새로 만들기" 클릭 후 `C:\mecab\bin` 추가
5. "확인" 버튼 클릭하여 저장

### 5. Python 패키지 설치
```bash
pip install mecab-python3
pip install konlpy
```

## 코드 내 MeCab 초기화 구현

```python
def initialize_mecab():
    """
    MeCab 형태소 분석기 초기화 함수
    """
    try:
        from konlpy.tag import Mecab
        mecab = Mecab(dicpath='C:/mecab/mecab-ko-dic')
        print("MeCab 초기화 성공")
        return mecab
    except Exception as e:
        print(f"MeCab 초기화 실패: {e}")
        # 대체 토크나이저 사용
        try:
            from konlpy.tag import Okt
            print("MeCab 대신 Okt를 사용합니다.")
            return Okt()
        except Exception:
            print("기본 공백 기반 토크나이저를 사용합니다.")
            class SimpleTokenizer:
                def morphs(self, text):
                    return text.split() if text else ["<unk>"]
            return SimpleTokenizer()
```

## 문제 해결

### 일반적인 오류와 해결책

1. **"사전이 없습니다" 오류**
   - 오류 메시지: `The MeCab dictionary does not exist at "C:/mecab/mecab-ko-dic"`
   - 해결책: mecab-ko-dic을 정확한 경로에 설치했는지 확인

2. **출력 형식 오류**
   - 오류 메시지: `unknown format type [chasen]`
   - 해결책: mecabrc 파일의 `output-format-type`을 `wakati`로 변경

3. **디렉토리 구조 확인**
   ```
   dir C:\mecab /s
   dir C:\mecab\mecab-ko-dic
   ```

4. **dicrc 경로 문제**
   - 오류 메시지: `[ifs] no such file or directory: C:/mecab/mecabrc`
   - 해결책: mecabrc 파일이 C:\mecab 폴더에 있는지 확인, 파일 확장자가 없어야 함

## 참고 자료
- [MeCab 공식 문서](https://taku910.github.io/mecab/)
- [mecab-python3 GitHub](https://github.com/SamuraiT/mecab-python3)
- [mecab-ko-dic GitHub](https://github.com/Pusnow/mecab-ko-dic-msvc)
