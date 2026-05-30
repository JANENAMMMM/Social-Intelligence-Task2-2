# Task 2.2 — Discover Insights & Value (40점)

## 개요

Task 2.1에서 구축한 Retriever를 포함한 다양한 분석 방법을 활용해,
스마트폰 리뷰 데이터에서 **비즈니스 인사이트를 발굴하고 가치로 연결**하는 과제입니다.

- 데이터: 삼성/애플 다중 모델 스마트폰 리뷰 (텍스트 + 메타데이터, 정답 레이블 없음)
- 평가 방식: 정량 점수 없음 — 교수자가 **정성 평가**

---

## 사용 가능한 분석 방법

아래 방법들을 자유롭게 조합해서 사용합니다.

| # | 방법 | 설명 |
|---|------|------|
| ① | **Topic Modeling** | LDA / BERTopic으로 리뷰의 잠재 주제 추출 |
| ② | **Network Analysis** | 키워드 공기어 그래프, 연관 분석 |
| ③ | **Clustering** | 임베딩 기반 K-means / HDBSCAN으로 리뷰·제품 그룹화 |
| ④ | **Information Retrieval** | Task 2.1 Retriever로 특정 테마/불만 리뷰 탐색 |
| ⑤ | **Generative AI** | LLM을 통한 요약, 분류, 페르소나 생성 |
| ⑥ | **기타** | 통계 검정, 시계열 분석, 회귀 등 |

---

## 평가 기준 (40점 배점)

### Idea/Novelty — 40%
- 질문의 독창성
- 시도한 방법과 기법의 다양성
- 분석 관점의 깊이
- 데이터 간 연결고리 도출

### Performance/Content — 40%
- 발견한 인사이트의 설득력
- 근거의 충분성
- 비즈니스적 의미와 임팩트

### Code/Presentation — 20%
- 재현 가능성, 코드 완성도
- 보고서 서술 및 내러티브
- 시각화 품질
- 슬라이드/문서 명확성

---

## 강한 인사이트 예시 (PDF 제시)

- 부정 리뷰에서 사용자가 직접 해결책을 공유하는 경우가 많음 → 온보딩 가이드 개선 기회
- 긍정 리뷰는 실제 사용 시나리오 중심 → 스펙보다 시나리오 중심 마케팅으로 전환
- 낮은 평점 리뷰에서 배송 문제 언급 빈도 높음 → 배송 신뢰성·커뮤니케이션 개선

---

## 권장 작업 흐름

```
데이터 탐색
    ↓
분석 질문 설정 (가설)
    ↓
방법 적용 (Topic Modeling / Clustering / IR / LLM 등)
    ↓
인사이트 발굴
    ↓
비즈니스 가치로 연결
    ↓
시각화 + 보고서 작성
```

---

## 코드 작성 시 주의사항

- **랜덤 시드 고정** — `random`, `numpy`, `torch` 모두
- **LLM 사용 시** — `temperature=0` 또는 고정 시드, 모델 버전 명시 (예: `gpt-4o-2024-08-06`)
- **경로 하드코딩 금지** — `uv` 환경 기반으로 어디서든 실행 가능하게
- **재현 가능성** — 동일 입력에 동일 출력이 나와야 함

---

## 관련 파일

- [elastic_search.py](elastic_search.py) — Task 2.1 Retriever (IR 기반 탐색에 활용)
- [si_dataset/review_for_analysis.json](si_dataset/review_for_analysis.json) — 분석용 리뷰 데이터
- [si_dataset/train_review_data.json](si_dataset/train_review_data.json) — 학습용 리뷰 데이터
