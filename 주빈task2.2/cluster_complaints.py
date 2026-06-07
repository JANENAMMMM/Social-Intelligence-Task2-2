"""
Step 2: 불만 문장 임베딩 + KMeans 클러스터링
입력: task2.2/sentence_sentiments.csv
출력: task2.2/clustered_complaints.csv
      task2.2/cluster_summary.txt
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer

# ── 경로 ──────────────────────────────────────────────────
HERE      = Path(__file__).resolve().parent
IN_PATH   = HERE / "sentence_sentiments.csv"   # 문장 단위 입력
OUT_CSV   = HERE / "clustered_complaints.csv"
OUT_TXT   = HERE / "cluster_summary.txt"
PLOT_PATH = HERE / "cluster_umap.png"

EMBED_MODEL = "jhgan/ko-sroberta-multitask"
BATCH_SIZE  = 64
K_CLUSTERS  = 6
K_SCAN_RANGE = range(2, 12)   # 엘보우/실루엣 스캔 범위

# ── 폰트 (한글) ────────────────────────────────────────────
def set_korean_font():
    candidates = ["Malgun Gothic", "NanumGothic", "AppleGothic", "DejaVu Sans"]
    available  = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in available:
            plt.rcParams["font.family"] = c
            break
    plt.rcParams["axes.unicode_minus"] = False

# ── 1. 데이터 로드 + 정제 ──────────────────────────────────
def is_garbage(text: str) -> bool:
    """반복 특수문자, 이모지만 있는 쓰레기 문장 필터"""
    import re
    clean = re.sub(r'[\s\W_]+', '', text)          # 공백·특수문자 제거
    if len(clean) < 4:
        return True
    # 같은 문자가 80% 이상이면 쓰레기 (ㅡㅡㅡ, ----)
    most_common = max(set(clean), key=clean.count)
    if clean.count(most_common) / len(clean) >= 0.8:
        return True
    return False

def load_negatives() -> pd.DataFrame:
    df = pd.read_csv(IN_PATH, encoding="utf-8-sig")
    neg = df[df["is_negative"] == True].copy()
    before = len(neg)
    neg = neg[~neg["sentence"].apply(is_garbage)].reset_index(drop=True)
    print(f"부정 문장: {len(neg):,}개  (정제 제거: {before - len(neg)}개, 출처 리뷰: {neg['review_id'].nunique():,}개)")
    return neg

# ── 2. 임베딩 ──────────────────────────────────────────────
def embed(texts: list[str]) -> np.ndarray:
    print(f"임베딩 모델 로드: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    print("임베딩 생성 중...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    print(f"임베딩 shape: {embeddings.shape}")
    return embeddings

# ── 3. UMAP 차원 축소 ──────────────────────────────────────
def reduce_umap(embeddings: np.ndarray, n_components: int = 30) -> np.ndarray:
    print(f"UMAP 차원 축소: {embeddings.shape[1]}d → {n_components}d")
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=30,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(embeddings)

# ── 4. 엘보우 + 실루엣 스캔 ────────────────────────────────
def scan_k(reduced: np.ndarray) -> None:
    print(f"k={K_SCAN_RANGE.start}~{K_SCAN_RANGE.stop-1} 스캔 중...")
    inertias, sil_scores = [], []
    for k in K_SCAN_RANGE:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(reduced)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(reduced, labels, sample_size=2000, random_state=42))
        print(f"  k={k}: inertia={km.inertia_:.0f}, silhouette={sil_scores[-1]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ks = list(K_SCAN_RANGE)
    axes[0].plot(ks, inertias, "bo-")
    axes[0].set_title("Elbow (Inertia)")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")
    axes[1].plot(ks, sil_scores, "ro-")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Score")
    plt.tight_layout()
    scan_path = HERE / "cluster_k_scan.png"
    plt.savefig(scan_path, dpi=150)
    print(f"스캔 그래프 저장: {scan_path}")

    best_k = ks[sil_scores.index(max(sil_scores))]
    print(f"\n실루엣 최고점 k={best_k} (score={max(sil_scores):.4f})")
    print(f"현재 설정 k={K_CLUSTERS}")

# ── 5. KMeans 클러스터링 ───────────────────────────────────
def cluster(reduced: np.ndarray) -> np.ndarray:
    print(f"KMeans 클러스터링 (k={K_CLUSTERS})...")
    km = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=20)
    labels = km.fit_predict(reduced)
    sil = silhouette_score(reduced, labels, sample_size=2000, random_state=42)
    sizes = pd.Series(labels).value_counts().sort_index()
    print(f"실루엣 점수 (k={K_CLUSTERS}): {sil:.4f}")
    for cid, n in sizes.items():
        print(f"  C{cid}: {n}개")
    return labels

# ── 5. UMAP 2D 시각화용 ────────────────────────────────────
def reduce_2d(embeddings: np.ndarray) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(embeddings)

# ── 6. 클러스터 요약 ───────────────────────────────────────
def summarize(df: pd.DataFrame) -> str:
    lines = []
    cluster_ids = sorted(df["cluster"].unique())
    for cid in cluster_ids:
        sub = df[df["cluster"] == cid]
        label = f"클러스터 {cid}"
        lines.append(f"\n{'='*60}")
        lines.append(f"{label}  ({len(sub)}개)")
        lines.append(f"평균 별점: {sub['rating'].mean():.2f}")
        if "brand" in df.columns:
            lines.append(f"브랜드 분포: {sub['brand'].value_counts().to_dict()}")

        # 대표 문장 3개 (prob_negative 높은 순)
        if True:
            lines.append(f"출처 리뷰 수: {sub['review_id'].nunique()}개")
            lines.append("── 대표 문장 ──")
            top = sub.nlargest(3, "prob_negative")
            for _, row in top.iterrows():
                sent = str(row.get("sentence", ""))[:120]
                lines.append(f"  [{row.get('rating','?')}★] {sent}")
    return "\n".join(lines)

# ── 7. 시각화 ──────────────────────────────────────────────
def plot(df: pd.DataFrame, xy_2d: np.ndarray):
    set_korean_font()
    cluster_ids = sorted(df["cluster"].unique())
    cmap = plt.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=(10, 7))
    for i, cid in enumerate(cluster_ids):
        mask  = df["cluster"] == cid
        color = "#aaaaaa" if cid == -1 else cmap(i % 20)
        label = f"C{cid} (n={mask.sum()})"
        ax.scatter(xy_2d[mask, 0], xy_2d[mask, 1],
                   c=[color], label=label, s=20, alpha=0.7)

    ax.set_title("불만 리뷰 클러스터 (UMAP 2D)")
    ax.legend(loc="upper right", fontsize=8, markerscale=2)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"시각화 저장: {PLOT_PATH}")

# ── Main ───────────────────────────────────────────────────
def main():
    df = load_negatives()

    embeddings = embed(df["sentence"].tolist())

    # 클러스터링용 축소
    reduced_10d = reduce_umap(embeddings)

    # 엘보우 + 실루엣 스캔
    scan_k(reduced_10d)

    labels = cluster(reduced_10d)
    df["cluster"] = labels

    # 시각화용 2D
    xy_2d = reduce_2d(embeddings)

    # 결과 저장
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"클러스터 결과 저장: {OUT_CSV}")

    summary = summarize(df)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"요약 저장: {OUT_TXT}")
    # 터미널 인코딩 문제 방지
    safe_summary = summary.encode("cp949", errors="replace").decode("cp949")
    print(safe_summary)

    plot(df, xy_2d)

if __name__ == "__main__":
    main()
