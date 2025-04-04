import pandas as pd
import torch
import re
import kss
import os
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

# 하드코딩된 경로 및 설정값
INPUT_CSV_PATH = r"C:\Users\갱보\Desktop\갱보\2025\1-2025\종프\모델 학습\dataset\기사 데이터\complete_test_stratified_utf8.csv"
OUTPUT_CSV_BASE = r"C:\Users\갱보\Desktop\갱보\2025\1-2025\종프\모델 학습\dataset\문장분해결과"  # 확장자 제외 (번호 붙임)
SUMMARY_RATIO = 0.3  # 요약 비율: 원본의 30%
SAVE_INTERVAL = 500  # 중간 결과 저장 간격 (기사 수)

class KoBARTSummarizer:
    def __init__(self):
        """KoBART 요약 모델 초기화"""
        print("🔄 KoBART 요약 모델 로드 중...")
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-summarization')
        self.model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-summarization')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"KoBART 모델 장치: {self.device}")
        self.model.eval()

    def summarize(self, text, ratio=SUMMARY_RATIO):
        """
        기사 요약 - 원본 길이의 일정 비율로 요약
        ratio: 원본 대비 요약 비율 (0.3 = 30%)
        """
        if not text or len(text) < 50:  # 너무 짧은 텍스트는 요약하지 않음
            return text
            
        try:
            # 불필요한 정보 제거 (이메일, 기자 정보 등)
            text = clean_unnecessary_info(text)
            
            # 원본 문장 수 계산
            original_sentences = split_sentences(text)
            original_sentence_count = len(original_sentences)
            
            # 타겟 문장 수 계산 (최소 3문장, 최대 원본의 ratio 비율)
            target_sentences = max(3, int(original_sentence_count * ratio))
            
            # 길이가 긴 경우 적절히 자르기 (KoBART 입력 제한)
            if len(text) > 1024:
                text = text[:1024]
                
            inputs = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
            
            # 요약 생성
            with torch.no_grad():
                # 문장 수에 따라 max_length와 min_length 조정
                max_token_length = min(512, int(len(inputs[0]) * ratio * 1.5))
                min_token_length = max(50, int(len(inputs[0]) * ratio * 0.8))
                
                summary_ids = self.model.generate(
                    inputs,
                    max_length=max_token_length,
                    min_length=min_token_length,
                    length_penalty=1.0,  # 요약 길이 조절
                    num_beams=4,
                    early_stopping=True
                )
                
            # 디코딩
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            # 반복 패턴 및 불필요한 정보 제거
            summary = clean_repetitive_patterns(summary)
            summary = clean_unnecessary_info(summary)
            
            # 중복 문장 제거 및 정제
            summary = self.clean_summary(summary)
            
            # 원하는 문장 수에 맞게 조정
            summary_sentences = split_sentences(summary)
            
            # 요약된 문장이 타겟보다 적으면 원본에서 중요 문장 추가
            if len(summary_sentences) < target_sentences:
                # 원본에서 첫 문장과 마지막 문장 (보통 중요한 정보 포함)
                additional_sentences = []
                if original_sentence_count > 0:
                    # 첫 문장 추가 (요약에 없을 경우)
                    if not any(sentence_similarity(original_sentences[0], s) > 0.7 for s in summary_sentences):
                        first_sent = clean_repetitive_patterns(original_sentences[0])
                        first_sent = clean_unnecessary_info(first_sent)
                        if is_valid_sentence(first_sent):
                            additional_sentences.append(first_sent)
                    
                    # 마지막 문장 추가 (요약에 없고 이메일이나 기자 정보가 아닌 경우)
                    last_sent = original_sentences[-1]
                    if not any(sentence_similarity(last_sent, s) > 0.7 for s in summary_sentences) and \
                       not contains_email(last_sent) and not contains_reporter_info(last_sent):
                        last_sent = clean_repetitive_patterns(last_sent)
                        last_sent = clean_unnecessary_info(last_sent)
                        if is_valid_sentence(last_sent):
                            additional_sentences.append(last_sent)
                
                # 남은 부족한 문장 수를 원본에서 추가
                remaining_needed = target_sentences - len(summary_sentences) - len(additional_sentences)
                if remaining_needed > 0:
                    # 중간 문장들에서 선택 (길이가 긴 문장 우선)
                    middle_sentences = original_sentences[1:-1] if len(original_sentences) > 2 else []
                    # 유효한 문장만 선택
                    valid_middle_sentences = []
                    for s in middle_sentences:
                        if not any(sentence_similarity(s, existing) > 0.7 for existing in summary_sentences + additional_sentences) and \
                           not contains_email(s) and not contains_reporter_info(s):
                            s_clean = clean_repetitive_patterns(s)
                            s_clean = clean_unnecessary_info(s_clean)
                            if is_valid_sentence(s_clean):
                                valid_middle_sentences.append((s_clean, len(s_clean)))
                    
                    # 길이 기준으로 정렬하고 필요한 만큼 추가
                    valid_middle_sentences.sort(key=lambda x: x[1], reverse=True)
                    additional_sentences.extend([s[0] for s in valid_middle_sentences[:remaining_needed]])
                
                # 요약 + 추가 문장을 합쳐서 최종 요약 생성
                final_sentences = summary_sentences + additional_sentences
                
                # 원본 순서대로 재정렬 (원본 문장의 순서를 보존)
                def get_original_position(sentence):
                    for i, orig in enumerate(original_sentences):
                        if sentence_similarity(sentence, orig) > 0.7:
                            return i
                    return 999  # 일치하는 문장이 없으면 맨 뒤로
                
                final_sentences.sort(key=get_original_position)
                
                # 최종 문장들 정제
                final_sentences = [clean_unnecessary_info(clean_repetitive_patterns(s)) for s in final_sentences]
                summary = ' '.join(final_sentences)
            
            return summary
        except Exception as e:
            print(f"요약 오류: {e}")
            # 오류 발생 시 원본 텍스트에서 불필요한 정보 제거 후 반환
            return clean_unnecessary_info(text)

    def clean_summary(self, summary):
        """요약문에서 중복된 내용 제거 및 정제"""
        # 중복된 문장 패턴 제거
        summary = re.sub(r'(.{20,}?)\1+', r'\1', summary)
        
        # 문장 분리
        try:
            sentences = kss.split_sentences(summary)
        except:
            sentences = re.split(r'(?<=[.!?])\s+', summary)
        
        # 중복 문장 제거
        unique_sentences = []
        seen = set()
        
        for s in sentences:
            s_clean = s.strip()
            # 핵심 내용만 비교 (조사, 어미 제거)
            core_content = re.sub(r'[은는이가을를에서의로]', '', s_clean)
            if len(s_clean) > 10 and core_content not in seen and is_valid_sentence(s_clean):
                seen.add(core_content)
                unique_sentences.append(s_clean)
        
        return ' '.join(unique_sentences)

def sentence_similarity(s1, s2):
    """두 문장 간의 유사도 계산 (간단한 구현)"""
    # 조사, 어미 제거한 핵심 내용으로 비교
    s1_core = re.sub(r'[은는이가을를에서의로]', '', s1)
    s2_core = re.sub(r'[은는이가을를에서의로]', '', s2)
    
    # 문자열을 집합으로 변환하여 자카드 유사도 계산
    if not s1_core or not s2_core:
        return 0
    
    s1_chars = set(s1_core)
    s2_chars = set(s2_core)
    
    intersection = len(s1_chars.intersection(s2_chars))
    union = len(s1_chars.union(s2_chars))
    
    return intersection / union if union > 0 else 0

def clean_repetitive_patterns(text):
    """반복되는 패턴 제거"""
    # 1. 동일 문자 3회 이상 반복 패턴 정제 (예: "킹킹킹킹" -> "킹")
    text = re.sub(r'([가-힣])\1{2,}', r'\1', text)
    
    # 2. 동일 단어 연속 반복 패턴 정제 (예: "좋아 좋아 좋아" -> "좋아")
    word_repeat_pattern = re.compile(r'(([가-힣]+) \2(?: \2)+)')
    for match in word_repeat_pattern.finditer(text):
        repeated_phrase = match.group(1)
        single_word = match.group(2)
        text = text.replace(repeated_phrase, single_word)
    
    # 3. 2음절 이상 단어 반복 패턴 제거 (예: "안녕하세요안녕하세요" -> "안녕하세요")
    for word_len in range(6, 1, -1):  # 6음절부터 2음절까지 검사
        pattern = re.compile(f'([가-힣]{{{word_len}}})\\1+')
        for match in pattern.finditer(text):
            repeated_word = match.group(0)
            single_word = match.group(1)
            text = text.replace(repeated_word, single_word)
    
    return text

def contains_email(text):
    """이메일 주소 포함 여부 확인"""
    email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    return bool(email_pattern.search(text))

def contains_reporter_info(text):
    """기자 정보 포함 여부 확인"""
    reporter_patterns = [
        r'\s기자\s*$',
        r'\[.*?기자\]',
        r'\(.*?기자\)',
        r'[가-힣]+\s*기자\s*=',
        r'[가-힣]+\s*기자\s*[a-zA-Z0-9._%+-]+@',
        r'[■◆●▶▷▶️]'  # 기사 끝에 많이 사용되는 기호
    ]
    
    for pattern in reporter_patterns:
        if re.search(pattern, text):
            return True
    
    return False

def clean_unnecessary_info(text):
    """불필요한 정보 제거 (이메일, 기자 정보 등)"""
    # 1. 이메일 주소 제거
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text)
    
    # 2. 기자 정보가 포함된 부분 제거
    reporter_patterns = [
        r'\s*\[[가-힣\s]+기자\]\s*',
        r'\s*\([가-힣\s]+기자[=]?[가-힣\s]*\)\s*',
        r'\s*[가-힣]+\s*기자\s*=\s*[가-힣]+\s*',
        r'\s*[가-힣]+\s*기자$'
    ]
    
    for pattern in reporter_patterns:
        text = re.sub(pattern, ' ', text)
    
    # 3. 언론사 표시 제거 (예: [중앙일보], [한겨레] 등)
    text = re.sub(r'\s*\[[가-힣A-Za-z\s]+\]\s*', ' ', text)
    
    # 4. 기사 날짜 정보 제거
    text = re.sub(r'\s*\d{4}년\s*\d{1,2}월\s*\d{1,2}일\s*', ' ', text)
    
    # 5. 특수 기호로 시작하는 부분 제거 (많은 기사에서 끝 부분 표시로 사용)
    text = re.sub(r'[■◆●▶▷▶️].*$', '', text)
    
    # 6. 연속된 공백 제거 및 정리
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def is_valid_sentence(text):
    """유효한 문장인지 확인"""
    # 10자 미만 짧은 문장 필터링
    if len(text) < 10:
        return False
    
    # 이메일이나 기자 정보가 있는 문장 제거
    if contains_email(text) or contains_reporter_info(text):
        return False
    
    # 특수 기호로만 구성된 문장 제거
    if re.match(r'^[^\w가-힣]+$', text):
        return False
    
    return True

def split_sentences(text):
    """문장 분리 함수 (향상된 버전)"""
    # 중복된 문장 패턴 제거
    text = clean_repetitive_patterns(text)
    
    # 이메일 및 기자 정보 제거
    text = clean_unnecessary_info(text)
    
    try:
        # KSS로 문장 분리 시도
        sentences = kss.split_sentences(text)
    except Exception as e:
        # print(f"🔍 KSS 문장 분리 오류, 기본 분리 사용: {e}")
        # 기본 구분자 사용
        sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # 추가적인 문장 분리 (긴 문장이나 복합 문장 처리)
    expanded_sentences = []
    for sentence in sentences:
        # 너무 긴 문장인 경우 추가 분리
        if len(sentence) > 100:
            # 쉼표나 세미콜론 등으로 추가 분리
            sub_sentences = re.split(r'(?<=[,;])\s+', sentence)
            # 충분히 긴 부분만 별도 문장으로 처리
            for sub in sub_sentences:
                if len(sub) >= 20:  # 20자 이상만 별도 문장으로
                    expanded_sentences.append(sub)
                else:
                    # 짧은 부분은 이전 문장에 붙이거나 새 문장으로 시작
                    if expanded_sentences:
                        expanded_sentences[-1] = expanded_sentences[-1] + ' ' + sub
                    else:
                        expanded_sentences.append(sub)
        else:
            expanded_sentences.append(sentence)
    
    # 중복 및 짧은 문장 제거, 유효성 검사
    unique_sentences = []
    seen = set()
    
    for s in expanded_sentences:
        s_clean = s.strip()
        
        # 유효하지 않은 문장 무시
        if not is_valid_sentence(s_clean):
            continue
            
        # 핵심 내용만 비교 (조사, 어미 제거)
        core_content = re.sub(r'[은는이가을를에서의로]', '', s_clean)
        
        if len(s_clean) > 10 and core_content not in seen:
            seen.add(core_content)
            unique_sentences.append(s_clean)
    
    return unique_sentences

def further_split_sentence(sentence):
    """복합 문장을 더 작은 문장으로 추가 분리"""
    result = []
    
    # 이미 충분히 짧은 문장은 분리하지 않음
    if len(sentence) < 50:
        return [sentence]
        
    # 접속사나 특정 패턴으로 분리
    connectors = [
        '그리고', '또한', '하지만', '그러나', '그런데', '하며', '따라서', 
        '이에', '이때', '또', '그래서', '이어', '반면', '한편'
    ]
    
    # 접속사 위치 찾기
    positions = []
    for connector in connectors:
        pattern = f'\\s{connector}\\s'
        for match in re.finditer(pattern, sentence):
            positions.append((match.start(), match.end(), connector))
    
    # 위치 정렬
    positions.sort(key=lambda x: x[0])
    
    # 분할점이 없으면 원래 문장 반환
    if not positions:
        return [sentence]
    
    # 문장 분할
    prev_end = 0
    for start, end, _ in positions:
        # 분할점 앞부분이 충분히 긴 경우만 별도 문장으로
        if start - prev_end >= 15:  # 최소 15자 이상
            sub_sentence = sentence[prev_end:start].strip()
            if is_valid_sentence(sub_sentence):
                result.append(sub_sentence)
        prev_end = end
    
    # 마지막 부분 추가
    if len(sentence) - prev_end >= 15:
        sub_sentence = sentence[prev_end:].strip()
        if is_valid_sentence(sub_sentence):
            result.append(sub_sentence)
    
    # 분리된 것이 없으면 원본 반환
    if not result:
        return [sentence]
        
    return result

def save_results(sentences, file_number):
    """결과 저장 함수 - 순차적 파일 번호 사용"""
    # 파일 경로 생성 (문장분해결과1.csv, 문장분해결과2.csv, ...)
    output_path = f"{OUTPUT_CSV_BASE}{file_number}.csv"
    
    # 데이터프레임 생성 및 저장
    df = pd.DataFrame(sentences)
    df.to_csv(output_path, index=False)
    print(f"💾 결과 저장 완료: {output_path} (문장 수: {len(sentences)})")
    
    return output_path

def main():
    """메인 함수 - 하드코딩된 경로 및 설정 사용"""
    # 요약기 초기화
    summarizer = KoBARTSummarizer()
    
    print(f"📑 기사 CSV 불러오는 중... ({INPUT_CSV_PATH})")
    df = pd.read_csv(INPUT_CSV_PATH)
    print(f"총 {len(df)}개 기사 로드됨")
    
    # label1 열 존재 확인
    has_label1 = 'label1' in df.columns
    if has_label1:
        print("✅ label1 열 발견: 결과에 포함됩니다.")
    else:
        print("⚠️ label1 열을 찾을 수 없습니다. 결과에 포함되지 않습니다.")
    
    all_sentences = []
    processed_articles = 0
    file_number = 1  # 결과 파일 순차 번호
    
    # 각 기사 처리
    for i, row in tqdm(df.iterrows(), total=len(df), desc="기사 처리 중"):
        article_id = i + 1
        content = str(row.get("content", ""))
        
        # label1 값 가져오기 (있는 경우)
        label1_value = row.get("label1", None) if has_label1 else None
        
        if pd.isna(content) or content.strip() == "":
            continue
        
        # 원본 문장 수 확인
        original_sentences = split_sentences(content)
        original_count = len(original_sentences)
        
        # 기사 요약 (원본의 ratio 비율로)
        print(f"\n📰 기사 #{article_id} 요약 중... (원본: {original_count}문장)")
        summarized_text = summarizer.summarize(content, ratio=SUMMARY_RATIO)
        
        # 요약된 문장 추출 및 추가 분리
        sentences = []
        for sentence in split_sentences(summarized_text):
            # 각 문장에 대해 추가 분리 수행 (복합 문장 분리)
            sub_sentences = further_split_sentence(sentence)
            sentences.extend(sub_sentences)
        
        print(f"📝 요약 결과: {len(sentences)}문장 ({len(sentences)/max(1, original_count)*100:.1f}%)")
        
        # 각 문장을 개별 행으로 저장 (label1 포함)
        for sentence in sentences:
            # 최종 유효성 검사
            if not is_valid_sentence(sentence):
                continue
                
            sentence_data = {
                "article_id": article_id,
                "sentence": sentence
            }
            
            # label1이 있으면 추가
            if has_label1 and label1_value is not None:
                sentence_data["label1"] = label1_value
                
            all_sentences.append(sentence_data)
        
        processed_articles += 1
        
        # 지정된 간격(SAVE_INTERVAL)마다 중간 결과 저장하고 새 파일 시작
        if processed_articles % SAVE_INTERVAL == 0:
            save_results(all_sentences, file_number)
            file_number += 1  # 다음 파일 번호로 증가
            all_sentences = []  # 새 파일을 위해 초기화
    
    # 마지막 배치 결과 저장 (남은 데이터가 있는 경우)
    if all_sentences:
        save_results(all_sentences, file_number)
    
    print(f"✅ 모든 처리 완료! 총 {processed_articles}개 기사를 처리했습니다.")

if __name__ == "__main__":
    main()
