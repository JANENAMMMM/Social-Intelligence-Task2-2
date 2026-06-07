"""
Step 4: 클러스터별 BERTopic으로 세부 토픽 추출
입력: task2.2/clustered_complaints.csv
출력: task2.2/topic_summary.txt
      task2.2/topic_results.csv
"""

from pathlib import Path

import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

HERE       = Path(__file__).resolve().parent
IN_PATH    = HERE / "clustered_complaints.csv"
OUT_TXT    = HERE / "topic_summary.txt"
OUT_CSV    = HERE / "topic_results.csv"

EMBED_MODEL = "jhgan/ko-sroberta-multitask"

# 한국어 불용어
KO_STOPWORDS = [
    "이", "그", "저", "것", "수", "를", "을", "이", "가", "은", "는",
    "에", "의", "과", "와", "도", "로", "으로", "에서", "하고", "하는",
    "있는", "없는", "같은", "때", "더", "또", "안", "못", "좀", "진짜",
    "정말", "너무", "그냥", "제", "제가", "저는", "저도", "이거", "이게",
    "이건", "그거", "아니", "근데", "그런데", "그리고", "하지만", "근데",
    "왜", "어떻게", "언제", "어디", "무슨", "뭐", "어", "아", "ㅠ", "ㅜ",
    "ㅋ", "ㅎ", "나", "내", "제", "우리", "분", "거", "게", "걸", "걸로",
    "만", "다", "다가", "다고", "다는", "면", "서", "고", "지", "며",
    "해서", "해도", "했는데", "하면", "하니", "하고", "한", "할", "하면"
]

CLUSTER_LABELS = {
    0: "혼재(감정표현)",
    1: "쿠팡_배송포장CS",
    2: "쿠팡_교환절차",
    3: "제품_성능불량",
    4: "가격불만",
    5: "제품_발열마이그레이션",
}

def run_bertopic(sentences: list[str], cluster_id: int, min_topic_size: int = 10):
    if len(sentences) < 20:
        return None, None

    embed_model = SentenceTransformer(EMBED_MODEL)
    vectorizer  = CountVectorizer(
        min_df=2,
        token_pattern=r"[가-힣a-zA-Z]{2,}",  # 한글/영문 2자 이상
        stop_words=KO_STOPWORDS,
    )
    representation = KeyBERTInspired()

    topic_model = BERTopic(
        embedding_model=embed_model,
        vectorizer_model=vectorizer,
        representation_model=representation,
        min_topic_size=min_topic_size,
        nr_topics="auto",
        verbose=False,
        calculate_probabilities=False,
    )

    topics, _ = topic_model.fit_transform(sentences)
    return topic_model, topics


def summarize_cluster(topic_model, topics, sentences, cluster_id, df_cluster):
    lines = []
    label = CLUSTER_LABELS.get(cluster_id, f"C{cluster_id}")
    lines.append(f"\n{'='*60}")
    lines.append(f"클러스터 {cluster_id} — {label}  ({len(sentences)}개 문장)")
    lines.append(f"평균 별점: {df_cluster['rating'].mean():.2f}")

    topic_info = topic_model.get_topic_info()
    n_topics   = len(topic_info[topic_info["Topic"] != -1])
    n_outliers = topic_info[topic_info["Topic"] == -1]["Count"].values[0] if -1 in topic_info["Topic"].values else 0
    lines.append(f"토픽 수: {n_topics}  /  아웃라이어: {n_outliers}개")

    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            continue
        keywords = [w for w, _ in topic_model.get_topic(tid)[:8]]
        count    = row["Count"]
        lines.append(f"\n  [토픽 {tid}] ({count}개)  키워드: {', '.join(keywords)}")

        # 대표 문장 2개 (해당 토픽에서)
        idxs = [i for i, t in enumerate(topics) if t == tid][:2]
        for i in idxs:
            lines.append(f"    → {sentences[i][:100]}")

    return "\n".join(lines)


def main():
    df = pd.read_csv(IN_PATH, encoding="utf-8-sig")
    print(f"전체 부정 문장: {len(df):,}개")

    all_lines   = []
    result_rows = []

    for cid in sorted(df["cluster"].unique()):
        sub       = df[df["cluster"] == cid].reset_index(drop=True)
        sentences = sub["sentence"].tolist()
        print(f"\n[C{cid}] {CLUSTER_LABELS.get(cid, '')} — {len(sentences)}개 문장 BERTopic 실행 중...")

        min_size = max(5, len(sentences) // 20)
        topic_model, topics = run_bertopic(sentences, cid, min_topic_size=min_size)

        if topic_model is None:
            all_lines.append(f"\n[C{cid}] 문장 수 부족 — 스킵")
            continue

        summary = summarize_cluster(topic_model, topics, sentences, cid, sub)
        all_lines.append(summary)
        print(summary)

        # 결과 저장용
        for i, (topic, sent) in enumerate(zip(topics, sentences)):
            result_rows.append({
                "cluster":    cid,
                "topic":      topic,
                "sentence":   sent,
                "review_id":  sub.loc[i, "review_id"] if "review_id" in sub.columns else None,
                "rating":     sub.loc[i, "rating"] if "rating" in sub.columns else None,
            })

    full_text = "\n".join(all_lines)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"\n요약 저장: {OUT_TXT}")

    pd.DataFrame(result_rows).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"상세 결과 저장: {OUT_CSV}")


if __name__ == "__main__":
    main()
