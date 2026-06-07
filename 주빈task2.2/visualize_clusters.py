"""
클러스터 + BERTopic 통합 시각화 대시보드
"""

import warnings; warnings.filterwarnings("ignore")
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

HERE    = Path(__file__).resolve().parent
CLUSTER = HERE / "clustered_complaints.csv"
TOPIC   = HERE / "topic_results.csv"
RAW     = HERE / "review_with_sentiment.csv"
OUT     = HERE / "dashboard.png"

# ── 한글 폰트 ───────────────────────────────────────────────
def set_font():
    for name in ["Malgun Gothic", "NanumGothic", "AppleGothic"]:
        if any(f.name == name for f in fm.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False

CNAMES = {
    0: "C0\nkss오류",
    1: "C1\n쿠팡배송/포장",
    2: "C2\n무게/외관",
    3: "C3\n완성도+배터리",
    4: "C4\n카메라/고스트",
    5: "C5\n가격/구성품",
}
COLORS = ["#9E9E9E", "#EF5350", "#FF9800", "#42A5F5", "#66BB6A", "#AB47BC"]

TOPIC_LABELS = {
    0: {0:"앱속도/멀티태스킹", 1:"여러앱동시", 2:"게임/영상편집", 3:"인터넷/카카오톡", 4:"앱전환", 5:"끊김", 6:"칩성능"},
    1: {0:"CS/교환불만", 1:"박스파손", 2:"완충재없음", 3:"쿠팡전반"},
    2: {0:"일반아쉬움", 1:"스크래치/케이스", 2:"무게/손목", 3:"디자인실망"},
    3: {0:"제품완성도", 1:"배터리/발열"},
    4: {0:"카메라야간/고스트", 1:"S펜미지원"},
    5: {0:"가격부담", 1:"결제/카드", 2:"통신사약정", 3:"구매후회", 4:"충전기미포함"},
}

def load():
    cluster = pd.read_csv(CLUSTER, encoding="utf-8-sig")
    topic   = pd.read_csv(TOPIC, encoding="utf-8-sig")
    raw     = pd.read_csv(RAW, encoding="utf-8-sig")
    raw["brand"] = raw["product_name"].apply(
        lambda x: "Apple" if "iphone" in str(x).lower() else "Samsung"
    )
    meta = raw[["reviewId","brand","product_name"]].rename(columns={"reviewId":"review_id"})
    cluster = cluster.drop(columns=["product_name"], errors="ignore").merge(meta, on="review_id", how="left")
    return cluster, topic

def main():
    set_font()
    cluster, topic = load()

    fig = plt.figure(figsize=(22, 26), facecolor="#F5F5F5")
    fig.suptitle("Task 2.2 — 불만 유형 클러스터 분석 대시보드",
                 fontsize=20, fontweight="bold", y=0.98)

    outer = gridspec.GridSpec(4, 1, figure=fig,
                              hspace=0.45,
                              top=0.95, bottom=0.03,
                              left=0.06, right=0.97)

    # ── Row 0: 클러스터 개요 바 차트 ───────────────────────
    ax_top = fig.add_subplot(outer[0])
    cids   = sorted(cluster["cluster"].unique())
    sizes  = [len(cluster[cluster["cluster"]==c]) for c in cids]
    bars   = ax_top.bar([CNAMES[c] for c in cids], sizes,
                        color=COLORS, alpha=0.85, edgecolor="white", linewidth=1.5)
    ax_top.set_title("클러스터별 부정 문장 수 & 평균 별점", fontsize=13, fontweight="bold")
    ax_top.set_ylabel("문장 수")
    ax_top.set_ylim(0, max(sizes) * 1.25)
    ax_top.grid(axis="y", alpha=0.3)
    ax_top.spines[["top","right"]].set_visible(False)

    for bar, cid, n in zip(bars, cids, sizes):
        avg = cluster[cluster["cluster"]==cid]["rating"].mean()
        ax_top.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                    f"{n}개\n★{avg:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    if 0 in cids:
        bars[0].set_hatch("//")
        bars[0].set_alpha(0.5)
        ax_top.text(bars[0].get_x() + bars[0].get_width()/2, 5,
                    "⚠ 신뢰도낮음", ha="center", va="bottom", fontsize=8, color="gray")

    # ── Row 1: 브랜드 분포 + 제품 Top3 ───────────────────
    inner1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1],
                                              wspace=0.35)

    # 브랜드 stacked bar
    ax_brand = fig.add_subplot(inner1[0])
    apple_n   = [cluster[(cluster["cluster"]==c)&(cluster["brand"]=="Apple")].shape[0] for c in cids]
    samsung_n = [cluster[(cluster["cluster"]==c)&(cluster["brand"]=="Samsung")].shape[0] for c in cids]
    x = np.arange(len(cids))
    ax_brand.bar(x, apple_n,   color="#42A5F5", alpha=0.85, label="Apple")
    ax_brand.bar(x, samsung_n, bottom=apple_n, color="#66BB6A", alpha=0.85, label="Samsung")
    ax_brand.set_xticks(x)
    ax_brand.set_xticklabels([CNAMES[c] for c in cids], fontsize=9)
    ax_brand.set_title("브랜드별 분포", fontsize=12, fontweight="bold")
    ax_brand.set_ylabel("문장 수")
    ax_brand.legend(loc="upper right", fontsize=9)
    ax_brand.grid(axis="y", alpha=0.3)
    ax_brand.spines[["top","right"]].set_visible(False)

    # 별점 분포 히트맵
    ax_rating = fig.add_subplot(inner1[1])
    rating_matrix = []
    for cid in cids:
        sub = cluster[cluster["cluster"]==cid]
        row = [(sub["rating"]==r).sum() / len(sub) * 100 for r in [1,2,3,4,5]]
        rating_matrix.append(row)
    rm = np.array(rating_matrix)
    im = ax_rating.imshow(rm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=80)
    ax_rating.set_xticks(range(5)); ax_rating.set_xticklabels(["★1","★2","★3","★4","★5"])
    ax_rating.set_yticks(range(len(cids)))
    ax_rating.set_yticklabels([CNAMES[c] for c in cids], fontsize=9)
    ax_rating.set_title("별점 분포 히트맵 (%)", fontsize=12, fontweight="bold")
    for i in range(len(cids)):
        for j in range(5):
            ax_rating.text(j, i, f"{rm[i,j]:.0f}%",
                           ha="center", va="center", fontsize=9,
                           color="white" if rm[i,j] > 40 else "black")
    plt.colorbar(im, ax=ax_rating, fraction=0.03)

    # ── Row 2: BERTopic 토픽 분포 (C1~C5) ────────────────
    inner2 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[2],
                                              wspace=0.4)
    valid_cids = [1, 2, 3, 4, 5]
    topic_cmaps = ["Reds", "Oranges", "Blues", "Greens", "Purples"]

    for ax_idx, (cid, cmap_name) in enumerate(zip(valid_cids, topic_cmaps)):
        ax = fig.add_subplot(inner2[ax_idx])
        tsub  = topic[topic["cluster"]==cid]
        valid = tsub[tsub["topic"] != -1]
        outlier_n = (tsub["topic"] == -1).sum()
        outlier_pct = outlier_n / len(tsub) * 100

        if len(valid) == 0:
            ax.text(0.5, 0.5, "토픽 없음", ha="center", va="center")
            continue

        tid_counts = valid["topic"].value_counts().sort_index()
        labels = [TOPIC_LABELS.get(cid, {}).get(tid, f"T{tid}") for tid in tid_counts.index]
        cmap   = plt.get_cmap(cmap_name)
        colors = [cmap(0.4 + 0.5 * i / max(len(tid_counts)-1, 1))
                  for i in range(len(tid_counts))]

        wedges, texts, autotexts = ax.pie(
            tid_counts.values,
            labels=None,
            colors=colors,
            autopct="%1.0f%%",
            startangle=90,
            pctdistance=0.75,
        )
        for at in autotexts:
            at.set_fontsize(8)

        ax.set_title(f"{CNAMES[cid]}\n아웃라이어:{outlier_pct:.0f}%",
                     fontsize=9, fontweight="bold")
        ax.legend(wedges, labels, loc="lower center",
                  bbox_to_anchor=(0.5, -0.35),
                  fontsize=7, ncol=1, frameon=False)

    # ── Row 3: 제품별 클러스터 비율 히트맵 ────────────────
    ax_prod = fig.add_subplot(outer[3])
    cluster2 = cluster.copy()
    ct = pd.crosstab(cluster2["product_name"], cluster2["cluster"]) \
           .div(cluster2.groupby("product_name").size(), axis=0) * 100
    ct.columns = [CNAMES[c].replace("\n", " ") for c in ct.columns]
    ct = ct.round(1)

    im2 = ax_prod.imshow(ct.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=40)
    ax_prod.set_xticks(range(len(ct.columns)))
    ax_prod.set_xticklabels(ct.columns, fontsize=10)
    ax_prod.set_yticks(range(len(ct.index)))
    ax_prod.set_yticklabels(ct.index, fontsize=10)
    ax_prod.set_title("제품별 클러스터 비율 히트맵 (%)", fontsize=13, fontweight="bold")
    for i in range(len(ct.index)):
        for j in range(len(ct.columns)):
            v = ct.values[i, j]
            ax_prod.text(j, i, f"{v:.0f}%",
                         ha="center", va="center", fontsize=9,
                         color="white" if v > 25 else "black", fontweight="bold")
    plt.colorbar(im2, ax=ax_prod, fraction=0.02, label="%")

    plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"저장 완료: {OUT}")

if __name__ == "__main__":
    main()
