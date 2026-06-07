"""
Step 6: 교차 분석
 - 브랜드(Apple/Samsung) × 클러스터 분포
 - 제품별 클러스터 분포
 - 시계열: 월별 클러스터 불만 트렌드
 - 공존 분석: 같은 리뷰에서 여러 클러스터 불만 조합
출력: task2.2/cross_analysis.txt, cross_analysis_*.png
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

HERE     = Path(__file__).resolve().parent
IN_PATH  = HERE / "clustered_complaints.csv"
RAW_PATH = HERE / "review_with_sentiment.csv"

CLUSTER_NAMES = {
    0: "혼재",
    1: "쿠팡_배송포장",
    2: "쿠팡_교환절차",
    3: "제품_카메라화면",
    4: "가격보상판매",
    5: "제품_배터리용량",
}

def set_font():
    for name in ["Malgun Gothic", "NanumGothic", "AppleGothic"]:
        if any(f.name == name for f in fm.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def load_data():
    df  = pd.read_csv(IN_PATH, encoding="utf-8-sig")
    raw = pd.read_csv(RAW_PATH, encoding="utf-8-sig")

    # 브랜드 추출
    raw["brand"] = raw["product_name"].apply(
        lambda x: "Apple" if "iphone" in str(x).lower() else "Samsung"
    )
    # reviewAt 파싱
    raw["reviewAt"] = pd.to_datetime(raw["reviewAt"], unit="ms", errors="coerce")
    raw["month"]    = raw["reviewAt"].dt.to_period("M")

    # review_id 기준으로 병합 (brand, month만 추가)
    review_meta = raw[["reviewId", "brand", "month"]].rename(
        columns={"reviewId": "review_id"}
    )
    df = df.merge(review_meta, on="review_id", how="left")
    df["cluster_name"] = df["cluster"].map(CLUSTER_NAMES)
    return df, raw


# ── 1. 브랜드 × 클러스터 ────────────────────────────────────
def brand_cluster(df: pd.DataFrame) -> str:
    lines = ["\n=== 브랜드 × 클러스터 분포 ===\n"]

    ct = pd.crosstab(df["brand"], df["cluster_name"], normalize="index") * 100
    lines.append(ct.round(1).to_string())

    set_font()
    ax = ct.plot(kind="bar", figsize=(10, 5), colormap="Set2", alpha=0.85)
    ax.set_title("브랜드별 불만 클러스터 비율 (%)")
    ax.set_ylabel("%")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(HERE / "cross_brand_cluster.png", dpi=150)
    plt.close()

    # 브랜드별 가장 두드러진 불만
    lines.append("\n[Apple 상위 불만]")
    lines.append(ct.loc["Apple"].nlargest(3).to_string())
    lines.append("\n[Samsung 상위 불만]")
    lines.append(ct.loc["Samsung"].nlargest(3).to_string())
    return "\n".join(lines)


# ── 2. 제품별 클러스터 ──────────────────────────────────────
def product_cluster(df: pd.DataFrame) -> str:
    lines = ["\n=== 제품별 불만 클러스터 분포 ===\n"]

    ct = pd.crosstab(df["product_name"], df["cluster_name"], normalize="index") * 100
    ct = ct.round(1)
    lines.append(ct.to_string())

    set_font()
    fig, ax = plt.subplots(figsize=(12, 6))
    ct.plot(kind="bar", ax=ax, colormap="tab10", alpha=0.85)
    ax.set_title("제품별 불만 클러스터 비율 (%)")
    ax.set_ylabel("%")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(HERE / "cross_product_cluster.png", dpi=150)
    plt.close()
    return "\n".join(lines)


# ── 3. 시계열: 월별 클러스터 불만 ──────────────────────────
def time_series(df: pd.DataFrame) -> str:
    lines = ["\n=== 월별 클러스터 불만 트렌드 ===\n"]

    monthly = (
        df.dropna(subset=["month"])
        .groupby(["month", "cluster_name"])
        .size()
        .unstack(fill_value=0)
    )
    lines.append(monthly.to_string())

    set_font()
    fig, ax = plt.subplots(figsize=(13, 5))
    for col in monthly.columns:
        ax.plot(monthly.index.astype(str), monthly[col], marker="o", label=col)
    ax.set_title("월별 불만 클러스터 추이")
    ax.set_xlabel("월")
    ax.set_ylabel("문장 수")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(HERE / "cross_timeseries.png", dpi=150)
    plt.close()
    return "\n".join(lines)


# ── 4. 공존 분석 ───────────────────────────────────────────
def cooccurrence(df: pd.DataFrame) -> str:
    lines = ["\n=== 공존 분석 (같은 리뷰 내 불만 조합) ===\n"]

    # 리뷰별 클러스터 집합
    review_clusters = (
        df.groupby("review_id")["cluster"]
        .apply(set)
        .reset_index()
    )
    multi = review_clusters[review_clusters["cluster"].apply(len) > 1]
    lines.append(f"2개 이상 불만 클러스터가 공존하는 리뷰: {len(multi):,}개")

    # 클러스터 쌍별 공존 횟수
    from itertools import combinations
    pair_counts: dict = {}
    for clusters in multi["cluster"]:
        for a, b in combinations(sorted(clusters), 2):
            key = (CLUSTER_NAMES[a], CLUSTER_NAMES[b])
            pair_counts[key] = pair_counts.get(key, 0) + 1

    pair_df = pd.DataFrame(
        [(k[0], k[1], v) for k, v in pair_counts.items()],
        columns=["클러스터A", "클러스터B", "공존 리뷰 수"]
    ).sort_values("공존 리뷰 수", ascending=False)

    lines.append("\n[자주 함께 나타나는 불만 조합 Top 10]")
    lines.append(pair_df.head(10).to_string(index=False))

    # 히트맵
    set_font()
    cluster_list = list(CLUSTER_NAMES.values())
    matrix = pd.DataFrame(0, index=cluster_list, columns=cluster_list)
    for _, row in pair_df.iterrows():
        matrix.loc[row["클러스터A"], row["클러스터B"]] = row["공존 리뷰 수"]
        matrix.loc[row["클러스터B"], row["클러스터A"]] = row["공존 리뷰 수"]

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(matrix.values, cmap="YlOrRd")
    ax.set_xticks(range(len(cluster_list)))
    ax.set_yticks(range(len(cluster_list)))
    ax.set_xticklabels(cluster_list, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(cluster_list, fontsize=9)
    for i in range(len(cluster_list)):
        for j in range(len(cluster_list)):
            v = matrix.values[i, j]
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, label="공존 리뷰 수")
    ax.set_title("불만 클러스터 공존 히트맵")
    plt.tight_layout()
    plt.savefig(HERE / "cross_cooccurrence.png", dpi=150)
    plt.close()
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────
def main():
    df, raw = load_data()
    print(f"분석 대상: {len(df):,}개 부정 문장, {df['review_id'].nunique():,}개 리뷰")

    results = []
    results.append(brand_cluster(df))
    print("브랜드 교차 분석 완료")
    results.append(product_cluster(df))
    print("제품별 분석 완료")
    results.append(time_series(df))
    print("시계열 분석 완료")
    results.append(cooccurrence(df))
    print("공존 분석 완료")

    out = "\n".join(results)
    with open(HERE / "cross_analysis.txt", "w", encoding="utf-8") as f:
        f.write(out)
    print("\n저장 완료: cross_analysis.txt")
    safe = out.encode("cp949", errors="replace").decode("cp949")
    print(safe)


if __name__ == "__main__":
    main()
