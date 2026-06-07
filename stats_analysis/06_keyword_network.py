"""
06_keyword_network.py
키워드 공기어 네트워크 분석 (NetworkX)
- 불만 문장 내 한국어 명사 추출 (형태소 없으면 규칙 기반)
- 동일 문장 내 키워드 공기어 엣지 구성
- 커뮤니티 탐지 (Louvain 없으면 greedy modularity)
- 중심성 분석: degree / betweenness / eigenvector
- 클러스터별 핵심 키워드 서브그래프
"""
import pandas as pd
import numpy as np
import re
import networkx as nx
from collections import Counter
from pathlib import Path

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "output"

master = pd.read_csv(DATA / "master_dataset.csv", encoding="utf-8-sig")
main = master[master["cluster_display"] != "기타"].copy()

# ---- 형태소 추출 (규칙 기반) ----
STOPWORDS = {
    "이게", "이건", "이거", "그거", "그것", "그게", "것", "것도", "것은", "것이", "것을",
    "있는", "있어", "없는", "없어", "있고", "없고", "않은", "않아", "않고",
    "너무", "정말", "진짜", "조금", "좀", "좀더", "더", "매우", "아주", "완전", "굉장히",
    "하지만", "그런데", "근데", "그래서", "그리고", "또한", "또", "계속", "항상", "자꾸",
    "쿠팡", "아이폰", "갤럭시", "삼성", "애플", "폰", "기기", "제품", "스마트폰", "휴대폰",
    "사용", "이용", "구매", "구입", "작동", "설치", "연결", "해결", "개선", "문제",
    "생각", "느낌", "경우", "부분", "상태", "시간", "이상", "이하", "이후", "이전",
    "정도", "수준", "기능", "성능", "품질", "가격",
    "리뷰", "후기", "평가", "별점", "추천", "비추",
    "ㅠ", "ㅠㅠ", "ㅋ", "ㅋㅋ", "ㅎ", "...", ".", ",",
}
_PARTICLES = sorted(
    ["이에서", "에서는", "에서의", "에서도", "에서가", "에서를", "에서만",
     "이라고", "이라는", "이라면", "이라도", "이라서", "이라야",
     "이지만", "이지만은", "이지는", "이지도", "이지요",
     "에서", "에게", "에게서", "에게도", "에게만", "에게서는",
     "이라", "이며", "이고", "이다", "이야", "이요",
     "에도", "에만", "에는", "에의", "에가", "에까지",
     "으로", "으로서", "으로써", "으로도", "으로만",
     "로서", "로써", "로도", "로만", "로는",
     "이랑", "이나", "이나마", "이든", "이든지",
     "라고", "라는", "라면", "라도", "라서", "라야",
     "으로부터", "로부터",
     "에서", "부터", "까지", "마저", "조차", "이나", "이든",
     "에서는", "에서도", "에서만",
     "이다", "이었다", "였다", "입니다", "이었습니다",
     "에서는", "을", "를", "이", "가", "은", "는", "도", "만", "과", "와",
     "에", "의", "로", "에게", "서", "요", "야", "고", "지"],
    key=len, reverse=True,
)

def extract_nouns(text):
    words = str(text).split()
    nouns = []
    for word in words:
        w = word.strip(".,!?~ㅠㅋㅎ")
        for p in _PARTICLES:
            if w.endswith(p) and len(w) > len(p) + 1:
                w = w[:-len(p)]
                break
        tokens = re.findall(r'[가-힣]{2,}', w)
        for tok in tokens:
            if tok not in STOPWORDS and len(tok) >= 2:
                nouns.append(tok)
    return nouns

# ---- 전체 공기어 네트워크 구성 ----
print("공기어 엣지 계산 중...")
cooc = Counter()
word_freq = Counter()

for _, row in main.iterrows():
    nouns = extract_nouns(row["sentence"])
    nouns = list(set(nouns))  # 문장 내 중복 제거
    word_freq.update(nouns)
    for i in range(len(nouns)):
        for j in range(i + 1, len(nouns)):
            edge = tuple(sorted([nouns[i], nouns[j]]))
            cooc[edge] += 1

print(f"총 엣지: {len(cooc)}, 총 단어: {len(word_freq)}")

# 최소 빈도 필터 (노드: >=20회, 엣지: >=5회)
MIN_NODE_FREQ = 15
MIN_EDGE_FREQ = 5
top_words = {w for w, c in word_freq.items() if c >= MIN_NODE_FREQ}
print(f"필터 후 노드: {len(top_words)}")

G = nx.Graph()
for (w1, w2), cnt in cooc.items():
    if w1 in top_words and w2 in top_words and cnt >= MIN_EDGE_FREQ:
        G.add_edge(w1, w2, weight=cnt)

# 노드 빈도 추가
for node in G.nodes():
    G.nodes[node]["freq"] = word_freq[node]

print(f"그래프: {G.number_of_nodes()} 노드, {G.number_of_edges()} 엣지")

# ---- 중심성 분석 ----
degree_cent = nx.degree_centrality(G)
betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
try:
    eigenvector = nx.eigenvector_centrality(G, weight="weight", max_iter=500)
except Exception:
    eigenvector = {n: 0 for n in G.nodes()}

centrality_df = pd.DataFrame({
    "word": list(G.nodes()),
    "freq": [word_freq[n] for n in G.nodes()],
    "degree_centrality": [degree_cent[n] for n in G.nodes()],
    "betweenness": [betweenness[n] for n in G.nodes()],
    "eigenvector": [eigenvector[n] for n in G.nodes()],
    "degree": [G.degree(n) for n in G.nodes()],
})
centrality_df = centrality_df.sort_values("betweenness", ascending=False)
print("\nTop20 키워드 (betweenness 기준):")
print(centrality_df.head(20)[["word", "freq", "degree", "betweenness", "eigenvector"]].to_string())
centrality_df.to_csv(OUT / "06_keyword_centrality.csv", index=False, encoding="utf-8-sig")

# ---- 커뮤니티 탐지 ----
try:
    from networkx.algorithms import community as nx_comm
    communities = list(nx_comm.greedy_modularity_communities(G, weight="weight"))
    print(f"\n커뮤니티 탐지: {len(communities)}개")
    for i, comm in enumerate(sorted(communities, key=len, reverse=True)[:8]):
        top_words_comm = sorted(comm, key=lambda w: betweenness.get(w, 0), reverse=True)[:10]
        print(f"  커뮤니티 {i+1} (n={len(comm)}): {', '.join(top_words_comm)}")

    community_map = {}
    for i, comm in enumerate(communities):
        for word in comm:
            community_map[word] = i
    community_df = pd.DataFrame(list(community_map.items()), columns=["word", "community"])
    community_df = community_df.merge(centrality_df[["word", "freq", "betweenness"]], on="word")
    community_df.to_csv(OUT / "06_keyword_communities.csv", index=False, encoding="utf-8-sig")
except Exception as e:
    print(f"커뮤니티 탐지 오류: {e}")

# ---- 클러스터별 특징 키워드 (TF-IDF 방식) ----
print("\n=== 클러스터별 특징 키워드 ===")
cluster_word_counts = {}
for cluster in main["cluster_display"].unique():
    sub = main[main["cluster_display"] == cluster]
    words = []
    for sent in sub["sentence"].fillna(""):
        words.extend(extract_nouns(sent))
    cluster_word_counts[cluster] = Counter(words)

# TF-IDF (term = word, doc = cluster)
all_clusters = list(cluster_word_counts.keys())
vocab = set()
for cnt in cluster_word_counts.values():
    vocab.update(cnt.keys())

import math
N = len(all_clusters)
idf = {}
for word in vocab:
    df = sum(1 for cnt in cluster_word_counts.values() if word in cnt)
    idf[word] = math.log(N / (1 + df))

cluster_tfidf = {}
for cluster, cnt in cluster_word_counts.items():
    total = sum(cnt.values())
    tfidf = {w: (c / total) * idf.get(w, 0) for w, c in cnt.items() if c >= 3}
    cluster_tfidf[cluster] = sorted(tfidf.items(), key=lambda x: -x[1])[:15]

cluster_keywords_rows = []
for cluster, kws in cluster_tfidf.items():
    for rank, (word, score) in enumerate(kws, 1):
        cluster_keywords_rows.append({
            "cluster": cluster, "rank": rank, "word": word, "tfidf": round(score, 5)
        })

df_kw = pd.DataFrame(cluster_keywords_rows)
df_kw.to_csv(OUT / "06_cluster_tfidf_keywords.csv", index=False, encoding="utf-8-sig")

print("클러스터별 Top5 TF-IDF 키워드:")
for cluster in all_clusters:
    kws = [row["word"] for row in cluster_keywords_rows if row["cluster"] == cluster][:5]
    print(f"  [{cluster}]: {', '.join(kws)}")

print("\n저장 완료: 06_*.csv")
