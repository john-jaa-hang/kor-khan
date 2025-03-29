# -*- coding: utf-8 -*-
"""
한국어 정치 기사의 정치적 성향 분류를 위한 계층적 주의 네트워크(HAN) 구현
- MeCab 형태소 분석기를 사용한 텍스트 전처리
- 단어-문장-제목 수준의 계층적 주의 메커니즘 구현
- 정치적 성향만 분류 (친정부 성향 라벨링 제외)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
import json
from tqdm import tqdm
import random
import logging

# ======================================
# 1. 설정 및 유틸리티 함수
# ======================================

def set_seed(seed):
    """
    재현성을 위한 랜덤 시드 설정
    모든 라이브러리에 동일한 시드 적용
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 다중 GPU 경우
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_logging(log_file='han_training.log'):
    """
    로깅 설정
    훈련 과정 및 결과를 저장하기 위한 로거 설정
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

# ======================================
# 2. MeCab 설정 및 텍스트 전처리 함수
# ======================================

def initialize_mecab():
    """
    MeCab 형태소 분석기 초기화 함수
    
    다양한 환경에서의 설치 문제를 방지하기 위해 여러 방식 시도
    설치 실패 시 KoNLPy의 Okt 형태소 분석기를 대체로 사용
    
    Returns:
        mecab: 초기화된 형태소 분석기 객체
    """
    try:
        from konlpy.tag import Mecab
        # 정확한 사전 경로 지정
        mecab = Mecab(dicpath='C:/mecab/mecab-ko-dic')
        print("MeCab 초기화 성공")
        return mecab
    except Exception as e1:
        print(f"KoNLPy MeCab 초기화 실패: {e1}")
        
        # 직접 mecab-python3 사용 시도 - mecabrc 파일 지정
        try:
            import MeCab
            # mecabrc 파일 직접 지정
            mecab_tagger = MeCab.Tagger('-d "C:/mecab/mecab-ko-dic"')
            
            # 간단한 테스트
            test_result = mecab_tagger.parse("테스트 문장입니다.")
            print(f"MeCab 테스트 결과: {test_result[:30]}...")
            
            # 래퍼 클래스 생성
            class MecabWrapper:
                def __init__(self, tagger):
                    self.tagger = tagger
                
                def morphs(self, text):
                    if not text or not text.strip():
                        return ["<unk>"]
                    
                    # 형태소 분석 결과를 단어로 변환
                    result = []
                    node = self.tagger.parseToNode(text)
                    while node:
                        if node.surface:  # 빈 문자열 아닌 경우만
                            result.append(node.surface)
                        node = node.next
                    
                    # 결과가 없으면 <unk> 반환
                    return result if result else ["<unk>"]
            
            wrapper = MecabWrapper(mecab_tagger)
            print("mecab-python3 직접 사용 초기화 성공")
            return wrapper
        except Exception as e2:
            print(f"mecab-python3 직접 사용 실패: {e2}")
            
            print("MeCab 대신 Okt를 사용합니다.")
            try:
                from konlpy.tag import Okt
                return Okt()
            except Exception as e3:
                print(f"Okt 초기화 실패: {e3}")
                
                # 모든 방법이 실패한 경우 기본 토크나이저 반환
                print("기본 공백 기반 토크나이저를 사용합니다.")
                
                class SimpleTokenizer:
                    def morphs(self, text):
                        return text.split() if text and text.strip() else ["<unk>"]
                
                return SimpleTokenizer()

def clean_text(text):
    """
    텍스트 정제 함수
    
    HTML 태그, 특수문자 등을 제거하고 텍스트를 정규화
    
    Args:
        text: 정제할 원본 텍스트
        
    Returns:
        정제된 텍스트
    """
    if pd.isna(text):  # None 또는 NaN 값 처리
        return ""
    
    text = str(text)  # 문자열로 변환 (숫자 등 다른 타입이 있을 경우)
    
    # HTML 태그 제거
    text = re.sub(r'<.*?>', '', text)
    
    # 이메일, URL 등 제거 또는 대체
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)
    text = re.sub(r'http\S+', '[URL]', text)
    
    # 특수문자는 유지하되 이상한 문자 제거
    # 한글, 영문, 숫자, 기본 문장부호 유지
    text = re.sub(r'[^\w\s\.\,\?\!\(\)\[\]\{\}\:\;\-\=\+\/\\\'\"\@\#\$\%\&\*\~\^\|가-힣]', '', text)
    
    # 여러 공백을 하나로 변환
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def split_into_sentences(text):
    """
    텍스트를 문장으로 분리하는 함수
    
    한국어 문장 종결 부호를 기준으로 문장 분리
    인용부호 내의 문장 종결 부호 고려
    
    Args:
        text: 분리할 텍스트
        
    Returns:
        분리된 문장 리스트
    """
    # 인용부호 내부의 문장부호 처리
    text = re.sub(r'([.!?])"', r'\1" ', text)
    
    # 한국어 문장 분리 (마침표, 물음표, 느낌표 기준)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # 빈 문장 제거하고 정제
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences

def tokenize_with_mecab(sentence, tokenizer):
    """
    MeCab을 사용한 문장 토큰화 함수
    
    Args:
        sentence: 토큰화할 문장
        tokenizer: MeCab 또는 다른 토크나이저 객체
        
    Returns:
        토큰화된 단어 리스트
    """
    if not sentence.strip():  # 빈 문장 확인
        return ["<unk>"]
    
    try:
        # MeCab 형태소 분석
        if hasattr(tokenizer, 'morphs'):  # MeCab, Okt 등
            tokens = tokenizer.morphs(sentence)
        else:  # 기본 토큰화 (공백 기준)
            tokens = sentence.split()
        
        # 빈 결과 처리
        if not tokens:
            return ["<unk>"]
            
        return tokens
    except Exception as e:
        print(f"토큰화 오류 ('{sentence}'): {e}")
        return ["<unk>"]

def preprocess_with_mecab(dataframe, tokenizer, max_sentences=100, max_words=50):
    """
    MeCab을 사용한 데이터 전처리 함수
    
    정치적 성향(political_orientation/label1)만 사용
    
    Args:
        dataframe: 처리할 데이터프레임
        tokenizer: MeCab 토크나이저
        max_sentences: 문서당 최대 문장 수
        max_words: 문장당 최대 단어 수
        
    Returns:
        전처리된 데이터 리스트
    """
    processed_data = []
    
    # 열 이름 확인 및 매핑
    if 'political_orientation' not in dataframe.columns and 'label1' in dataframe.columns:
        dataframe = dataframe.rename(columns={'label1': 'political_orientation'})
        print("label1 → political_orientation으로 매핑됨")
    
    # article_url 열 무시 - 처리 전 명시적으로 언급
    if 'article_url' in dataframe.columns:
        print("article_url 열은 학습에 사용되지 않습니다.")
    
    # label2/gov_attitude 열 무시 - 처리 전 명시적으로 언급
    if 'gov_attitude' in dataframe.columns or 'label2' in dataframe.columns:
        print("gov_attitude/label2 열은 학습에 사용되지 않습니다.")
    
    for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="MeCab 전처리 중"):
        # 제목과 본문 결합
        title = row['title'] if not pd.isna(row['title']) else ""
        content = row['content'] if not pd.isna(row['content']) else ""
        
        # 텍스트 정제
        clean_title = clean_text(title)
        clean_content = clean_text(content)
        
        # 제목을 첫 번째 문장으로 설정 (제목 주의 메커니즘 활용)
        sentences = [clean_title] if clean_title else []
        
        # 본문 문장 분리 및 추가
        content_sentences = split_into_sentences(clean_content)
        sentences.extend(content_sentences)
        
        # 빈 문장 처리
        if not sentences:
            sentences = ["<unk>"]
        
        # 문장 수 제한
        sentences = sentences[:max_sentences]
        
        # MeCab 토큰화
        tokenized_sentences = []
        for sentence in sentences:
            tokens = tokenize_with_mecab(sentence, tokenizer)
            # 토큰 수 제한
            tokens = tokens[:max_words]
            tokenized_sentences.append(tokens)
        
        # 빈 결과 처리
        if not tokenized_sentences:
            tokenized_sentences = [["<unk>"]]
        
        # 정치적 성향 라벨만 추출 (친정부 성향 무시)
        if 'political_orientation' in row:
            label = row['political_orientation'] - 1  # 0-indexing (1~5 → 0~4)
        else:
            # 기본값 설정 (중도)
            label = 2
        
        processed_data.append({
            'text': tokenized_sentences,
            'label': label
        })
    
    return processed_data

def build_vocab(processed_data, min_freq=2, special_tokens=None):
    """
    어휘 사전 구축 함수
    
    Args:
        processed_data: 전처리된 데이터
        min_freq: 최소 등장 빈도 (이 값 미만으로 등장하는 단어는 제외)
        special_tokens: 추가할 특수 토큰 리스트
        
    Returns:
        어휘 사전 (단어→인덱스 매핑)
    """
    # 기본 특수 토큰
    if special_tokens is None:
        special_tokens = ['<pad>', '<unk>']
    
    word_counts = {}
    
    # 단어 빈도 계산 - 학습, 검증, 테스트 데이터 모두 포함
    for item in processed_data:
        for sentence in item['text']:
            for word in sentence:
                word_counts[word] = word_counts.get(word, 0) + 1
    
    # 빈도에 따른 어휘 선택
    vocab = {}
    
    # 특수 토큰 추가
    for i, token in enumerate(special_tokens):
        vocab[token] = i
    
    # 최소 빈도 이상 단어 추가
    idx = len(special_tokens)
    for word, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True):
        if count >= min_freq and word not in vocab:  # 추가: 중복 방지
            vocab[word] = idx
            idx += 1
    
    print(f"어휘 크기: {len(vocab)}개 단어")
    print(f"최소 빈도 {min_freq} 이상 단어: {len(vocab) - len(special_tokens)}개")
    
    return vocab

def load_data(data_path):
    """
    데이터 로드 함수
    
    여러 인코딩 방식을 시도하여 CSV 파일 로드
    정치적 성향 라벨만 사용 (친정부 성향 무시)
    
    Args:
        data_path: 데이터 파일 경로
        
    Returns:
        로드된 데이터프레임
    """
    print("데이터 로드 시도 중...")
    
    # 다양한 인코딩으로 시도
    encodings = ['utf-8', 'utf-8-sig', 'cp1252', 'cp949', 'euc-kr', 'latin1']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(data_path, encoding=encoding)
            print(f"성공: {encoding} 인코딩으로 데이터 로드됨")
            break
        except Exception as e:
            print(f"실패: {encoding} 인코딩 - {str(e)}")
    
    if df is None:
        raise Exception("모든 인코딩으로 시도했으나 파일을 로드할 수 없습니다.")
    
    # 열 이름 확인 및 매핑
    print("원본 열 이름:", df.columns.tolist())
    
    # 필요한 열이 있는지 확인
    required_columns = ['title', 'content']
    if not all(col in df.columns for col in required_columns):
        print("경고: 필요한 열 중 일부가 없습니다.")
    
    # 라벨 열 매핑 - 정치적 성향만 사용
    if 'political_orientation' not in df.columns and 'label1' in df.columns:
        df = df.rename(columns={'label1': 'political_orientation'})
        print("label1 → political_orientation으로 매핑됨")
    
    # article_url 열 명시적으로 제외
    if 'article_url' in df.columns:
        print("article_url 열은 학습에 사용되지 않습니다.")
    
    # 친정부 성향 라벨 명시적으로 제외
    if 'gov_attitude' in df.columns or 'label2' in df.columns:
        if 'gov_attitude' in df.columns:
            print("gov_attitude 열은 학습에 사용되지 않습니다.")
        else:
            print("label2 열은 학습에 사용되지 않습니다.")
    
    # 데이터 크기 확인
    print(f"데이터 크기: {df.shape[0]}행 x {df.shape[1]}열")
    
    # 데이터 샘플 확인
    print("데이터 샘플:")
    print(df.head(2))
    
    return df

# ======================================
# 3. 데이터셋 클래스
# ======================================

class HANDataset(Dataset):
    """
    계층적 주의 네트워크를 위한 데이터셋 클래스
    
    텍스트를 계층적 구조(단어, 문장, 문서)로 처리
    배치 처리를 위한 패딩과 가변 길이 시퀀스 처리 지원
    """
    def __init__(self, processed_data, vocab, max_sentences=100, max_words=50):
        """
        Args:
            processed_data: 전처리된 데이터 리스트
            vocab: 어휘 사전 (단어→인덱스 매핑)
            max_sentences: 문서당 최대 문장 수
            max_words: 문장당 최대 단어 수
        """
        self.data = processed_data
        self.vocab = vocab
        self.max_sentences = max_sentences
        self.max_words = max_words
        self.vocab_size = len(vocab)
        self.pad_idx = vocab.get('<pad>', 0)
        self.unk_idx = vocab.get('<unk>', 1)
        print(f"어휘 사전 크기: {self.vocab_size}")
    
    def __len__(self):
        """데이터셋 크기 반환"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        데이터셋 인덱싱 처리
        
        Args:
            idx: 데이터 인덱스
            
        Returns:
            텍스트와 라벨 정보가 담긴 딕셔너리
        """
        item = self.data[idx]
        tokenized_sentences = item['text']
        label = item['label']
        
        # 문장 및 단어 수 계산
        sentences_count = len(tokenized_sentences)
        word_counts = [len(sentence) for sentence in tokenized_sentences]
        
        # 단어 인덱스로 변환
        indexed_sentences = []
        for sentence in tokenized_sentences:
            # 인덱스가 어휘 사전 크기를 초과하지 않도록 보장
            indexed_sentence = []
            for word in sentence:
                # 어휘 사전에 없는 단어는 <unk> 토큰으로 처리
                word_idx = self.vocab.get(word, self.unk_idx)
                # 추가 안전 검사: 인덱스가 어휘 크기를 초과하지 않는지 확인
                if word_idx >= self.vocab_size:
                    word_idx = self.unk_idx
                indexed_sentence.append(word_idx)
            indexed_sentences.append(indexed_sentence)
        
        return {
            'text': indexed_sentences,
            'sentences_count': sentences_count,
            'word_counts': word_counts,
            'label': label
        }
        
    def collate_fn(self, batch):
        """
        배치 내 샘플들을 패딩하여 텐서로 변환
        
        가변 길이 문서를 배치로 처리하기 위한 커스텀 collate 함수
        
        Args:
            batch: 배치 샘플 리스트
            
        Returns:
            패딩된 텐서들 (텍스트, 문장 수, 단어 수, 라벨)
        """
        batch_size = len(batch)
        
        # 빈 문서/문장 처리
        for i, item in enumerate(batch):
            if not item['text'] or all(len(sent) == 0 for sent in item['text']):
                item['text'] = [[self.unk_idx]]
                item['sentences_count'] = 1
                item['word_counts'] = [1]
        
        # 배치에서 최대 문장 수와 최대 단어 수 계산
        max_sentences_in_batch = min(max([item['sentences_count'] for item in batch]), self.max_sentences)
        
        # 안전하게 word_counts 확인
        word_counts_per_batch = []
        for item in batch:
            if item['word_counts']:
                word_counts_per_batch.append(max(item['word_counts']))
            else:
                word_counts_per_batch.append(1)
        
        max_words_in_batch = min(max(word_counts_per_batch), self.max_words)
        
        # 패딩된 텐서 초기화
        padded_texts = torch.ones((batch_size, max_sentences_in_batch, max_words_in_batch), 
                                  dtype=torch.long) * self.pad_idx
        sentences_per_doc = torch.zeros(batch_size, dtype=torch.long)
        words_per_sentence = torch.zeros((batch_size, max_sentences_in_batch), dtype=torch.long)
        labels = torch.zeros(batch_size, dtype=torch.long)
        
        # 배치 내 각 샘플을 패딩
        for i, item in enumerate(batch):
            doc = item['text']
            sentences_per_doc[i] = min(len(doc), max_sentences_in_batch)
            
            for j, sentence in enumerate(doc[:max_sentences_in_batch]):
                words_per_sentence[i, j] = min(len(sentence), max_words_in_batch)
                sentence_len = words_per_sentence[i, j].item()
                
                if sentence_len > 0:  # 빈 문장이 아닌 경우에만 처리
                    padded_texts[i, j, :sentence_len] = torch.LongTensor(
                        sentence[:sentence_len]
                    )
            
            labels[i] = item['label']
        
        return padded_texts, sentences_per_doc, words_per_sentence, labels

# ======================================
# 4. 계층적 주의 네트워크 모델
# ======================================

class WordAttention(nn.Module):
    """
    단어 수준 주의 모듈
    
    문장 내 단어들 간의 관계 학습 및 중요한 단어에 집중
    양방향 GRU로 컨텍스트 인식 처리
    주의 메커니즘으로 중요 단어 가중치 부여
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, attention_dim):
        """
        Args:
            vocab_size: 어휘 사전 크기
            embed_dim: 임베딩 차원
            hidden_dim: GRU 은닉 상태 차원
            attention_dim: 주의 메커니즘 차원
        """
        super(WordAttention, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(hidden_dim * 2, attention_dim)
        self.context_vector = nn.Linear(attention_dim, 1, bias=False)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, text, words_per_sentence):
        """
        순전파 함수
        
        Args:
            text: 문장의 단어 인덱스 텐서 [batch_size, max_sentences, max_words]
            words_per_sentence: 각 문장의 실제 단어 수 [batch_size, max_sentences]
            
        Returns:
            sentence_representation: 문장 표현 텐서 [batch_size, max_sentences, hidden_dim*2]
            word_attention_weights: 단어 주의 가중치 [batch_size, max_sentences, max_words]
        """
        batch_size, max_sentences, max_words = text.shape
        
        # 안전 검사: 인덱스가 어휘 크기를 초과하지 않도록 클리핑
        vocab_size = self.word_embeddings.num_embeddings
        text_safe = torch.clamp(text, 0, vocab_size - 1)
        
        # 문장 처리를 위해 텍스트 평탄화
        text_reshaped = text_safe.view(batch_size * max_sentences, max_words)
        words_per_sentence_flat = words_per_sentence.view(-1)
        
        # 단어 임베딩
        embedded = self.word_embeddings(text_reshaped)  # [batch*sentences, words, embed_dim]
        embedded = self.dropout(embedded)
        
        # 유효한 문장만 처리 (단어 수가 0인 문장 제외)
        valid_sentences = words_per_sentence_flat > 0
        
        # 결과 텐서 초기화
        sentence_representation = torch.zeros(
            batch_size * max_sentences, self.gru.hidden_size * 2, 
            device=text.device
        )
        
        # 주의 가중치 초기화
        word_attention_weights = torch.zeros(
            batch_size * max_sentences, max_words, 
            device=text.device
        )
        
        if valid_sentences.sum() > 0:
            # 유효한 문장만 선택
            valid_embedded = embedded[valid_sentences]
            valid_words_per_sentence = words_per_sentence_flat[valid_sentences]
            
            # 패킹된 시퀀스 생성
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                valid_embedded, 
                valid_words_per_sentence.cpu(),
                batch_first=True,
                enforce_sorted=False
            )
            
            # GRU로 처리
            packed_output, _ = self.gru(packed_embedded)
            
            # 다시 패딩된 시퀀스로 복원
            output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, 
                batch_first=True,
                total_length=max_words
            )
            
            # 어텐션 계산
            u = torch.tanh(self.attention(output))  # [valid_sentences, words, attention_dim]
            att = self.context_vector(u).squeeze(2)  # [valid_sentences, words]
            
            # 패딩 마스킹
            mask = torch.arange(max_words, device=text.device).expand(valid_sentences.sum(), max_words)
            mask = mask < valid_words_per_sentence.unsqueeze(1)
            att = att.masked_fill(~mask, -1e9)
            
            # 소프트맥스로 정규화
            att_weights = F.softmax(att, dim=1)  # [valid_sentences, words]
            
            # 가중치를 사용하여 단어 표현 결합
            sent_repr = torch.bmm(
                att_weights.unsqueeze(1),  # [valid_sentences, 1, words]
                output  # [valid_sentences, words, hidden_dim*2]
            ).squeeze(1)  # [valid_sentences, hidden_dim*2]
            
            # 결과 저장
            sentence_representation[valid_sentences] = sent_repr
            word_attention_weights[valid_sentences] = att_weights
        
        # 원래 형태로 복원
        sentence_representation = sentence_representation.view(
            batch_size, max_sentences, -1
        )  # [batch, sentences, hidden_dim*2]
        
        word_attention_weights = word_attention_weights.view(
            batch_size, max_sentences, max_words
        )  # [batch, sentences, words]
        
        return sentence_representation, word_attention_weights

class SentenceAttention(nn.Module):
    """
    문장 수준 주의 모듈
    
    문서 내 문장들 간의 관계 학습 및 중요한 문장에 집중
    양방향 GRU로 문서 수준 컨텍스트 처리
    주의 메커니즘으로 중요 문장 가중치 부여
    """
    def __init__(self, input_dim, hidden_dim, attention_dim):
        """
        Args:
            input_dim: 입력 차원 (단어 주의 모듈 출력 차원)
            hidden_dim: GRU 은닉 상태 차원
            attention_dim: 주의 메커니즘 차원
        """
        super(SentenceAttention, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(hidden_dim * 2, attention_dim)
        self.context_vector = nn.Linear(attention_dim, 1, bias=False)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, sentence_representation, sentences_per_document):
        """
        순전파 함수
        
        Args:
            sentence_representation: 문장 표현 텐서 [batch_size, max_sentences, input_dim]
            sentences_per_document: 각 문서의 실제 문장 수 [batch_size]
            
        Returns:
            document_representation: 문서 표현 텐서 [batch_size, hidden_dim*2]
            sentence_attention_weights: 문장 주의 가중치 [batch_size, max_sentences]
        """
        batch_size, max_sentences, _ = sentence_representation.shape
        
        # 드롭아웃 적용
        sentence_representation = self.dropout(sentence_representation)
        
        # 유효한 문서만 처리 (문장 수가 0인 문서 제외)
        valid_docs = sentences_per_document > 0
        
        # 결과 텐서 초기화
        document_representation = torch.zeros(
            batch_size, self.gru.hidden_size * 2, 
            device=sentence_representation.device
        )
        
        # 주의 가중치 초기화
        sentence_attention_weights = torch.zeros(
            batch_size, max_sentences, 
            device=sentence_representation.device
        )
        
        if valid_docs.sum() > 0:
            # 유효한 문서만 선택
            valid_sentence_repr = sentence_representation[valid_docs]
            valid_sentences_per_doc = sentences_per_document[valid_docs]
            
            # 패킹된 시퀀스 생성
            packed_sentences = nn.utils.rnn.pack_padded_sequence(
                valid_sentence_repr, 
                valid_sentences_per_doc.cpu(),
                batch_first=True,
                enforce_sorted=False
            )
            
            # GRU로 처리
            packed_output, _ = self.gru(packed_sentences)
            
            # 다시 패딩된 시퀀스로 복원
            output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, 
                batch_first=True,
                total_length=max_sentences
            )
            
            # 어텐션 계산
            u = torch.tanh(self.attention(output))  # [valid_docs, sentences, attention_dim]
            att = self.context_vector(u).squeeze(2)  # [valid_docs, sentences]
            
            # 패딩 마스킹
            mask = torch.arange(max_sentences, device=sentence_representation.device)
            mask = mask.expand(valid_docs.sum(), max_sentences)
            mask = mask < valid_sentences_per_doc.unsqueeze(1)
            att = att.masked_fill(~mask, -1e9)
            
            # 소프트맥스로 정규화
            att_weights = F.softmax(att, dim=1)  # [valid_docs, sentences]
            
            # 가중치를 사용하여 문장 표현 결합
            doc_repr = torch.bmm(
                att_weights.unsqueeze(1),  # [valid_docs, 1, sentences]
                output  # [valid_docs, sentences, hidden_dim*2]
            ).squeeze(1)  # [valid_docs, hidden_dim*2]
            
            # 결과 저장
            document_representation[valid_docs] = doc_repr
            sentence_attention_weights[valid_docs] = att_weights
        
        return document_representation, sentence_attention_weights

class TitleAttention(nn.Module):
    """
    제목 수준 주의 모듈
    
    제목 정보를 활용하여 문장에 대한 추가 주의 적용
    제목과 문장 간의 관계 학습
    제목 관련 중요 문장 가중치 부여
    """
    def __init__(self, hidden_dim, attention_dim):
        """
        Args:
            hidden_dim: 입력 차원 (문장 표현 차원)
            attention_dim: 주의 메커니즘 차원
        """
        super(TitleAttention, self).__init__()
        self.title_projection = nn.Linear(hidden_dim * 2, attention_dim)
        self.sentence_projection = nn.Linear(hidden_dim * 2, attention_dim)
        self.context_vector = nn.Linear(attention_dim, 1, bias=False)
    
    def forward(self, title_representation, sentence_representations, sentences_per_document):
        """
        순전파 함수
        
        Args:
            title_representation: 제목 표현 텐서 [batch_size, hidden_dim*2]
            sentence_representations: 문장 표현 텐서 [batch_size, max_sentences, hidden_dim*2]
            sentences_per_document: 각 문서의 실제 문장 수 [batch_size]
            
        Returns:
            title_context: 제목 컨텍스트 텐서 [batch_size, hidden_dim*2]
            title_attention_weights: 제목 주의 가중치 [batch_size, max_sentences]
        """
        batch_size, max_sentences, _ = sentence_representations.shape
        
        # 제목과 문장의 표현을 같은 공간으로 투영
        title_proj = self.title_projection(title_representation).unsqueeze(1)  # [batch, 1, attention_dim]
        sent_proj = self.sentence_projection(sentence_representations)  # [batch, sentences, attention_dim]
        
        # 제목-문장 유사도 계산
        similarity = torch.tanh(title_proj + sent_proj)  # [batch, sentences, attention_dim]
        similarity = self.context_vector(similarity).squeeze(2)  # [batch, sentences]
        
        # 패딩 마스킹
        mask = torch.arange(max_sentences, device=sentence_representations.device)
        mask = mask.expand(batch_size, max_sentences)
        mask = mask < sentences_per_document.unsqueeze(1)
        similarity = similarity.masked_fill(~mask, -1e9)
        
        # 소프트맥스로 정규화
        title_attention_weights = F.softmax(similarity, dim=1)  # [batch, sentences]
        
        # 제목 어텐션으로 문장 표현 결합
        title_context = torch.bmm(
            title_attention_weights.unsqueeze(1),  # [batch, 1, sentences]
            sentence_representations  # [batch, sentences, hidden_dim*2]
        ).squeeze(1)  # [batch, hidden_dim*2]
        
        return title_context, title_attention_weights

class HierarchicalAttentionNetwork(nn.Module):
    """
    계층적 주의 네트워크 모델
    
    단어, 문장, 제목 수준의 계층적 주의 메커니즘 결합
    복잡한 문서 구조에서 중요 정보 추출
    """
    def __init__(self, vocab_size, embed_dim, word_hidden_dim, sent_hidden_dim, attention_dim, num_classes, dropout=0.5):
        """
        Args:
            vocab_size: 어휘 사전 크기
            embed_dim: 임베딩 차원
            word_hidden_dim: 단어 수준 GRU 은닉 상태 차원
            sent_hidden_dim: 문장 수준 GRU 은닉 상태 차원
            attention_dim: 주의 메커니즘 차원
            num_classes: 출력 클래스 수
            dropout: 드롭아웃 비율
        """
        super(HierarchicalAttentionNetwork, self).__init__()
        
        # 단어 수준 주의 모듈
        self.word_attention = WordAttention(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=word_hidden_dim,
            attention_dim=attention_dim
        )
        
        # 문장 수준 주의 모듈
        self.sentence_attention = SentenceAttention(
            input_dim=word_hidden_dim * 2,
            hidden_dim=sent_hidden_dim,
            attention_dim=attention_dim
        )
        
        # 제목 수준 주의 모듈 - 문서 표현에서 제목 관련 정보 강화
        self.title_attention = TitleAttention(
            hidden_dim=sent_hidden_dim,
            attention_dim=attention_dim
        )
        
        # 분류기
        self.fc = nn.Linear(sent_hidden_dim * 4, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text, sentences_per_document, words_per_sentence):
        """
        순전파 함수
        
        Args:
            text: 문서의 단어 인덱스 텐서 [batch_size, max_sentences, max_words]
            sentences_per_document: 각 문서의 실제 문장 수 [batch_size]
            words_per_sentence: 각 문장의 실제 단어 수 [batch_size, max_sentences]
            
        Returns:
            logits: 클래스 로짓 [batch_size, num_classes]
            word_attention: 단어 주의 가중치 [batch_size, max_sentences, max_words]
            sentence_attention: 문장 주의 가중치 [batch_size, max_sentences]
            title_attention: 제목 주의 가중치 [batch_size, max_sentences]
        """
        # 1. 단어 수준 주의 적용
        sentence_representation, word_attention = self.word_attention(text, words_per_sentence)
        
        # 2. 제목 표현 추출 (첫 번째 문장을 제목으로 간주)
        title_representation = sentence_representation[:, 0, :]
        
        # 3. 문장 수준 주의 적용
        document_representation, sentence_attention = self.sentence_attention(
            sentence_representation, sentences_per_document
        )
        
        # 4. 제목 주의 적용
        title_context, title_attention = self.title_attention(
            title_representation, sentence_representation, sentences_per_document
        )
        
        # 5. 문서 표현과 제목 컨텍스트 결합
        final_representation = torch.cat([document_representation, title_context], dim=1)
        final_representation = self.dropout(final_representation)
        
        # 6. 분류
        logits = self.fc(final_representation)
        
        return logits, word_attention, sentence_attention, title_attention

# ======================================
# 5. 학습 및 평가 함수
# ======================================

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, logger=None):
    """
    모델 학습 함수
    
    Args:
        model: 학습할 모델
        train_loader: 학습 데이터 로더
        val_loader: 검증 데이터 로더
        criterion: 손실 함수
        optimizer: 최적화 알고리즘
        num_epochs: 학습 에폭 수
        device: 학습에 사용할 디바이스 (CPU/GPU)
        logger: 로깅 객체
        
    Returns:
        history: 학습 히스토리
    """
    model = model.to(device)
    best_val_f1 = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # 학습 데이터 처리
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (texts, sentences_per_doc, words_per_sentence, labels) in enumerate(progress_bar):
            # 데이터를 디바이스로 이동
            texts = texts.to(device)
            sentences_per_doc = sentences_per_doc.to(device)
            words_per_sentence = words_per_sentence.to(device)
            labels = labels.to(device)
            
            # 순전파
            optimizer.zero_grad()
            logits, _, _, _ = model(texts, sentences_per_doc, words_per_sentence)
            loss = criterion(logits, labels)
            
            # 역전파
            loss.backward()
            optimizer.step()
            
            # 통계 업데이트
            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # 진행률 표시 업데이트
            progress_bar.set_postfix({
                'loss': f'{train_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # 에폭별 학습 통계
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = train_correct / train_total
        
        # 검증 수행
        val_loss, val_acc, val_f1, _, _, _ = evaluate_model(model, val_loader, criterion, device)
        
        # 결과 기록
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        log_msg = (f'Epoch {epoch+1}/{num_epochs} - '
                  f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
        print(log_msg)
        if logger:
            logger.info(log_msg)
        
        # 최적 모델 저장
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_han_model.pth')
            save_msg = f'모델 저장됨! (Val F1: {val_f1:.4f})'
            print(save_msg)
            if logger:
                logger.info(save_msg)
    
    return history

def evaluate_model(model, data_loader, criterion, device):
    """
    모델 평가 함수
    
    Args:
        model: 평가할 모델
        data_loader: 평가 데이터 로더
        criterion: 손실 함수
        device: 평가에 사용할 디바이스 (CPU/GPU)
        
    Returns:
        val_loss: 평가 손실
        val_acc: 정확도
        val_f1: F1 점수
        confusion_mat: 혼동 행렬
        val_preds: 예측 라벨
        val_labels: 실제 라벨
    """
    model = model.to(device)
    model.eval()
    
    val_loss = 0.0
    val_preds = []
    val_labels = []
    
    # 주의 가중치 저장용
    word_attentions = []
    sentence_attentions = []
    title_attentions = []
    
    with torch.no_grad():
        for texts, sentences_per_doc, words_per_sentence, labels in tqdm(data_loader, desc='평가 중'):
            # 데이터를 디바이스로 이동
            texts = texts.to(device)
            sentences_per_doc = sentences_per_doc.to(device)
            words_per_sentence = words_per_sentence.to(device)
            labels = labels.to(device)
            
            # 순전파
            logits, word_attn, sent_attn, title_attn = model(texts, sentences_per_doc, words_per_sentence)
            loss = criterion(logits, labels)
            
            # 통계 업데이트
            val_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            
            # CPU로 이동하여 저장
            val_preds.extend(predicted.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
            
            # 주의 가중치 저장 (선택적)
            word_attentions.append(word_attn.cpu().numpy())
            sentence_attentions.append(sent_attn.cpu().numpy())
            title_attentions.append(title_attn.cpu().numpy())
    
    # 결과 계산
    val_loss /= len(data_loader)
    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average='macro')
    confusion_mat = confusion_matrix(val_labels, val_preds)
    
    return val_loss, val_acc, val_f1, confusion_mat, val_preds, val_labels

# ======================================
# 6. 결과 시각화 및 저장 함수
# ======================================

def save_and_visualize_results(history, test_results, config, results_dir='results'):
    """
    결과 시각화 및 저장 함수
    
    Args:
        history: 학습 히스토리
        test_results: 테스트 결과 튜플 (loss, acc, f1, confusion_matrix, preds, labels)
        config: 모델 설정
        results_dir: 결과 저장 디렉토리
    """
    os.makedirs(results_dir, exist_ok=True)
    
    test_loss, test_acc, test_f1, cm, test_preds, test_labels = test_results
    
    # 1. 학습 곡선 시각화
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history['val_f1'], label='Val F1')
    plt.title('F1 Score vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'han_learning_curves.png'))
    
    # 2. 혼동 행렬 시각화
    plt.figure(figsize=(10, 8))
    class_names = ['매우 진보', '약간 진보', '중도', '약간 보수', '매우 보수']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'han_confusion_matrix.png'))
    
    # 3. 결과 저장
    results = {
        'test_loss': float(test_loss),
        'test_acc': float(test_acc),
        'test_f1': float(test_f1),
        'confusion_matrix': cm.tolist(),
        'history': {k: [float(val) for val in v] for k, v in history.items()},
        'vocab_size': config.get('vocab_size', 0),
        'hyperparameters': {
            'embed_dim': config.get('embed_dim', 0),
            'word_hidden_dim': config.get('word_hidden_dim', 0),
            'sent_hidden_dim': config.get('sent_hidden_dim', 0),
            'attention_dim': config.get('attention_dim', 0),
            'num_classes': config.get('num_classes', 0),
            'dropout': config.get('dropout', 0),
            'batch_size': config.get('batch_size', 0),
            'learning_rate': config.get('learning_rate', 0),
        }
    }
    
    with open(os.path.join(results_dir, 'han_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n결과가 {results_dir} 디렉토리에 저장되었습니다.")

def save_model(model, vocab, config, model_path='han_model_checkpoint.pth'):
    """
    모델 저장 함수
    
    Args:
        model: 저장할 모델
        vocab: 어휘 사전
        config: 모델 설정
        model_path: 모델 저장 경로
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'config': {
            'vocab_size': len(vocab),
            'embed_dim': config.get('embed_dim', 200),
            'word_hidden_dim': config.get('word_hidden_dim', 100),
            'sent_hidden_dim': config.get('sent_hidden_dim', 100),
            'attention_dim': config.get('attention_dim', 100),
            'num_classes': config.get('num_classes', 5),
            'max_sentences': config.get('max_sentences', 50),
            'max_words': config.get('max_words', 30)
        }
    }
    
    torch.save(checkpoint, model_path)
    print(f"모델이 {model_path}에 저장되었습니다.")

def visualize_attention(model, dataset, idx, device, tokenizer=None, vocab=None, results_dir='results'):
    """
    주의 가중치 시각화 함수
    
    Args:
        model: 학습된 모델
        dataset: 데이터셋
        idx: 시각화할 샘플 인덱스
        device: 사용할 디바이스
        tokenizer: 토크나이저 (원본 텍스트 확인용)
        vocab: 어휘 사전 (인덱스→단어 매핑용)
        results_dir: 결과 저장 디렉토리
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # 샘플 데이터 준비
    sample = dataset[idx]
    
    # 인덱스→단어 매핑 딕셔너리 생성
    if vocab:
        idx_to_word = {idx: word for word, idx in vocab.items()}
    
    # 모델에 입력할 데이터 준비
    texts = torch.LongTensor([sample['text']]).to(device)  # [1, sentences, words]
    sentences_per_doc = torch.LongTensor([sample['sentences_count']]).to(device)  # [1]
    words_per_sentence = torch.LongTensor([sample['word_counts']]).to(device)  # [1, sentences]
    
    # 모델 평가 모드로 전환
    model.eval()
    
    # 예측 및 주의 가중치 추출
    with torch.no_grad():
        logits, word_attention, sentence_attention, title_attention = model(
            texts, sentences_per_doc, words_per_sentence
        )
        predicted = torch.argmax(logits, dim=1).item()
    
    # 원본 라벨
    true_label = sample['label']
    
    # 정치적 성향 클래스 이름
    class_names = ['매우 진보', '약간 진보', '중도', '약간 보수', '매우 보수']
    
    # 결과 출력
    print(f"샘플 {idx} 분석:")
    print(f"실제 라벨: {class_names[true_label]} (클래스 {true_label+1})")
    print(f"예측 라벨: {class_names[predicted]} (클래스 {predicted+1})")
    
    # 주의 가중치 시각화 (문장 수준)
    plt.figure(figsize=(10, 5))
    
    # 문장 수준 주의 가중치
    sent_attn = sentence_attention[0, :sentences_per_doc[0]].cpu().numpy()
    plt.barh(range(len(sent_attn)), sent_attn, color='skyblue')
    plt.xlabel('Attention Weight')
    plt.ylabel('Sentence Index')
    plt.title('Sentence-level Attention Weights')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'sentence_attention_{idx}.png'))
    
    # 가장 중요한 문장 출력
    most_important_sent_idx = np.argmax(sent_attn)
    print(f"\n가장 중요한 문장 (주의 점수: {sent_attn[most_important_sent_idx]:.4f}):")
    
    # 토큰화된 문장 원본 확인 가능한 경우
    if vocab:
        important_sent_tokens = texts[0, most_important_sent_idx, :words_per_sentence[0, most_important_sent_idx]].cpu().numpy()
        important_sent_words = [idx_to_word.get(idx, '<unk>') for idx in important_sent_tokens]
        print(' '.join(important_sent_words))
    
    # 단어 수준 주의 가중치
    plt.figure(figsize=(12, 6))
    
    # 중요 문장의 단어 주의 가중치
    word_attn = word_attention[0, most_important_sent_idx, :words_per_sentence[0, most_important_sent_idx]].cpu().numpy()
    
    if vocab:
        words = [idx_to_word.get(texts[0, most_important_sent_idx, i].item(), '<unk>') for i in range(len(word_attn))]
        plt.barh(words, word_attn, color='lightcoral')
    else:
        plt.barh(range(len(word_attn)), word_attn, color='lightcoral')
    
    plt.xlabel('Attention Weight')
    plt.ylabel('Word')
    plt.title(f'Word-level Attention Weights for Sentence {most_important_sent_idx}')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'word_attention_{idx}.png'))
    
    # 제목 주의 가중치 
    plt.figure(figsize=(10, 5))
    
    # 제목 수준 주의 가중치
    title_attn = title_attention[0, :sentences_per_doc[0]].cpu().numpy()
    plt.barh(range(len(title_attn)), title_attn, color='lightgreen')
    plt.xlabel('Attention Weight')
    plt.ylabel('Sentence Index')
    plt.title('Title-level Attention Weights')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'title_attention_{idx}.png'))
    
    # 가장 중요한 단어 출력
    most_important_word_idx = np.argmax(word_attn)
    if vocab:
        most_important_word = idx_to_word.get(texts[0, most_important_sent_idx, most_important_word_idx].item(), '<unk>')
        print(f"\n가장 중요한 단어 (주의 점수: {word_attn[most_important_word_idx]:.4f}):")
        print(most_important_word)

# ======================================
# 7. 메인 함수
# ======================================

def main():
    """메인 함수 - 전체 프로세스 실행"""
    # 설정
    config = {
        # 파일 경로에 r 접두사 사용 (Windows 경로 문제 방지)
        'data_path': r'기사 데이터 파일 경로',
        'results_dir': 'results',
        
        'max_sentences': 50,         # 문서당 최대 문장 수
        'max_words': 30,             # 문장당 최대 단어 수
        'embed_dim': 200,            # 단어 임베딩 차원
        'word_hidden_dim': 100,      # 단어 GRU 은닉층 크기
        'sent_hidden_dim': 100,      # 문장 GRU 은닉층 크기
        'attention_dim': 100,        # 주의 메커니즘 차원
        'num_classes': 5,            # 정치적 성향 클래스 수 (1-5)
        'dropout': 0.5,              # 드롭아웃 비율
        'batch_size': 16,            # 배치 크기
        'num_epochs': 20,            # 학습 에폭 수
        'learning_rate': 0.001,      # 학습률
        'min_word_freq': 2,          # 최소 단어 빈도
        'seed': 42,                  # 랜덤 시드
    }
    
    # 결과 디렉토리 생성
    os.makedirs(config['results_dir'], exist_ok=True)
    
    # 로깅 설정
    logger = setup_logging(os.path.join(config['results_dir'], 'han_training.log'))
    logger.info("정치적 성향 분류를 위한 계층적 주의 네트워크(HAN) 학습 시작")
    logger.info(f"설정: {config}")
    
    # 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"사용 장치: {device}")
    
    # 랜덤 시드 설정
    set_seed(config['seed'])
    
    # 1. 데이터 로드 - 정치적 성향 라벨만 사용
    logger.info("1. 데이터 로드 중...")
    df = load_data(config['data_path'])
    
    # 2. MeCab 초기화
    logger.info("2. MeCab 초기화 중...")
    mecab = initialize_mecab()
    
    # 3. 데이터 전처리 - 정치적 성향 라벨만 사용
    logger.info("3. 데이터 전처리 중...")
    processed_data = preprocess_with_mecab(df, mecab, config['max_sentences'], config['max_words'])
    
    # 4. 어휘 사전 구축
    logger.info("4. 어휘 사전 구축 중...")
    vocab = build_vocab(processed_data, min_freq=config['min_word_freq'])
    config['vocab_size'] = len(vocab)
    
    # 5. 데이터 분할
    logger.info("5. 데이터 분할 중...")
    train_data, temp_data = train_test_split(processed_data, test_size=0.3, random_state=config['seed'])
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=config['seed'])
    
    logger.info(f"학습 데이터: {len(train_data)}개 샘플")
    logger.info(f"검증 데이터: {len(val_data)}개 샘플")
    logger.info(f"테스트 데이터: {len(test_data)}개 샘플")
    
    # 6. 데이터셋 및 데이터로더 생성
    logger.info("6. 데이터셋 및 데이터로더 생성 중...")
    train_dataset = HANDataset(train_data, vocab, config['max_sentences'], config['max_words'])
    val_dataset = HANDataset(val_data, vocab, config['max_sentences'], config['max_words'])
    test_dataset = HANDataset(test_data, vocab, config['max_sentences'], config['max_words'])
    
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], 
        shuffle=True, collate_fn=train_dataset.collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], 
        shuffle=False, collate_fn=val_dataset.collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'], 
        shuffle=False, collate_fn=test_dataset.collate_fn
    )
    
    # 7. 모델 초기화
    logger.info("7. 모델 초기화 중...")
    model = HierarchicalAttentionNetwork(
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim'],
        word_hidden_dim=config['word_hidden_dim'],
        sent_hidden_dim=config['sent_hidden_dim'],
        attention_dim=config['attention_dim'],
        num_classes=config['num_classes'],
        dropout=config['dropout']
    ).to(device)
    
    # 모델 정보 출력
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"모델 파라미터 수: {total_params:,}")
    
    # 8. 손실 함수 및 최적화 알고리즘 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # 9. 모델 학습
    logger.info("8. 모델 학습 시작...")
    history = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        config['num_epochs'], device, logger
    )
    
    # 10. 최적 모델 로드 및 평가 부분 수정
    logger.info("9. 최적 모델 평가 중...")
    model_path = os.path.join(config['results_dir'], 'best_han_model.pth')

    # 최적 모델이 저장되어 있는지 확인
    if os.path.exists('best_han_model.pth'):
        model.load_state_dict(torch.load('best_han_model.pth'))

    # 예외 처리 추가
    try:
        test_results = evaluate_model(model, test_loader, criterion, device)
        test_loss, test_acc, test_f1, _, _, _ = test_results
        
        logger.info(f"테스트 손실: {test_loss:.4f}")
        logger.info(f"테스트 정확도: {test_acc:.4f}")
        logger.info(f"테스트 F1 점수: {test_f1:.4f}")
        
        # 11. 결과 시각화 및 저장
        logger.info("10. 결과 시각화 및 저장 중...")
        save_and_visualize_results(history, test_results, config, config['results_dir'])
    except Exception as e:
        logger.error(f"모델 평가 중 오류 발생: {e}")
        logger.error("평가를 건너뛰고 모델만 저장합니다.")
        # 실패해도 학습된 모델은 저장
        save_model(model, vocab, config, os.path.join(config['results_dir'], 'han_model_checkpoint.pth'))
    
    # 12. 모델 저장
    save_model(model, vocab, config, os.path.join(config['results_dir'], 'han_model_checkpoint.pth'))
    
    # 13. 샘플 예측 및 주의 가중치 시각화
    logger.info("11. 샘플 예측 및 주의 가중치 시각화 중...")
    # 테스트 데이터에서 무작위 샘플 선택
    sample_idx = np.random.randint(0, len(test_data))
    
    # 인덱스→단어 매핑 딕셔너리 생성
    idx_to_word = {idx: word for word, idx in vocab.items()}
    
    # 주의 가중치 시각화
    visualize_attention(
        model, test_dataset, sample_idx, device, 
        tokenizer=mecab, vocab=vocab, 
        results_dir=config['results_dir']
    )
    
    logger.info("모든 과정이 완료되었습니다!")

if __name__ == "__main__":
    main()
