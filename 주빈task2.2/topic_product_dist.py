"""
클러스터 × BERTopic 토픽 × 브랜드/제품 세부 분포 시각화
각 클러스터 안에서 토픽별로 어느 제품/브랜드에서 많이 나왔는지 보여줌
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

HERE    = Path(__file__).resolve().parent
CLUSTER = HERE / "clustered_complaints.csv"
TOPIC   = HERE / "topic_results.csv"
RAW     = HERE / "review_with_sentiment.csv"
OUT     = HERE / "topic_product_dist.png"

def set_font():
    for name in ["Malgun Gothic", "NanumGothic", "AppleGothic"]:
        if any(f.name == name for f in fm.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False

CLUSTER_INFO = {
    1: {"name": "C1 — 쿠팡 배송/포장",      "color": "#EF5350"},
    2: {"name": "C2 — 무게/외관/디자인",     "color": "#FF9800"},
    3: {"name": "C3 — 완성도+배터리",        "color": "#42A5F5"},
    4: {"name": "C4 — 카메라 야간/고스트",   "color": "#66BB6A"},
    5: {"name": "C5 — 가격/구성품",          "color": "#AB47BC"},
}

TOPIC_LABELS = {
    (1, 0): "CS/교환불만",
    (1, 1): "박스파손",
    (1, 2): "완충재없음",
    (1, 3): "쿠팡전반",
    (2, 0): "일반아쉬움",
    (2, 1): "스크래치/케이스",
    (2, 2): "무게/손목",
    (2, 3): "디자인실망",
    (3, 0): "제품완성도",
    (3, 1): "배터리/발열",
    (4, 0): "카메라야간/고스트",
    (4, 1): "S펜미지원",
    (5, 0): "가격부담",
    (5, 1): "결제/카드",
    (5, 2): "통신사약정",
    (5, 3): "구매후회",
    (5, 4): "충전기미포함",
}

PRODUCTS = [
    "iphone_17_pro_max", "iphone_17_pro", "iphone_17",
    "galaxy_s26_ultra",  "galaxy_s26",    "galaxy_z_fold7", "galaxy_z_flip7",
]
PROD_SHORT = {
    "iphone_17_pro_max": "i17 Pro Max",
    "iphone_17_pro":     "i17 Pro",
    "iphone_17":         "i17",
    "galaxy_s26_ultra":  "S26 Ultra",
    "galaxy_s26":        "S26",
    "galaxy_z_fold7":    "Z Fold7",
    "galaxy_z_flip7":    "Z Flip7",
}
PROD_COLORS = {
    "iphone_17_pro_max": "#0D47A1",
    "iphone_17_pro":     "#1565C0",
    "iphone_17":         "#1976D2",
    "galaxy_s26_ultra":  "#1B5E20",
    "galaxy_s26":        "#2E7D32",
    "galaxy_z_fold7":    "#388E3C",
    "galaxy_z_flip7":    "#43A047",
}

def load():
    cluster = pd.read_csv(CLUSTER, encoding="utf-8-sig")
    topic   = pd.read_csv(TOPIC,   encoding="utf-8-sig")
    raw     = pd.read_csv(RAW, encoding="utf-8-sig")
    raw["brand"] = raw["product_name"].apply(
        lambda x: "Apple" if "iphone" in str(x).lower() else "Samsung"
    )
    meta = raw[["reviewId","brand","product_name"]].rename(columns={"reviewId":"review_id"})
    cluster = cluster.drop(columns=["product_name"], errors="ignore").merge(meta, on="review_id", how="left")
    # topic에 product_name 붙이기 (review_id + sentence 텍스트 매칭)
    sent_meta = cluster[["review_id","product_name","brand"]].drop_duplicates("review_id")
    topic = topic.merge(sent_meta, on="review_id", how="left")
    return cluster, topic

def main():
    set_font()
    cluster, topic = load()

    n_clusters = len(CLUSTER_INFO)
    fig = plt.figure(figsize=(24, n_clusters * 7), facecolor="#F8F9FA")
    fig.suptitle("불만 클러스터 × BERTopic 토픽 × 제품 분포 세부 분석",
                 fontsize=20, fontweight="bold", y=0.995)

    outer = gridspec.GridSpec(n_clusters, 1, figure=fig,
                              hspace=0.6, top=0.98, bottom=0.02,
                              left=0.04, right=0.98)

    for row_idx, (cid, cinfo) in enumerate(CLUSTER_INFO.items()):
        csub   = cluster[cluster["cluster"] == cid]
        tsub   = topic[(topic["cluster"] == cid) & (topic["topic"] != -1)]
        tids   = sorted(tsub["topic"].unique())
        n_topics = len(tids)
        if n_topics == 0:
            continue

        inner = gridspec.GridSpecFromSubplotSpec(
            1, n_topics + 1, subplot_spec=outer[row_idx],
            wspace=0.35, width_ratios=[0.8] + [1] * n_topics
        )

        # ── 맨 왼쪽: 클러스터 요약 ──────────────────────
        ax_sum = fig.add_subplot(inner[0])
        apple_n   = (csub["brand"] == "Apple").sum()
        samsung_n = (csub["brand"] == "Samsung").sum()
        total     = len(csub)
        wedges, _, autotexts = ax_sum.pie(
            [apple_n, samsung_n],
            colors=["#1976D2", "#388E3C"],
            autopct="%1.0f%%",
            startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=2),
            textprops=dict(fontsize=11),
        )
        for at in autotexts:
            at.set_fontweight("bold")
            at.set_color("white")
            at.set_fontsize(12)
        ax_sum.set_title(
            f"{cinfo['name']}\n총 {total}개\n아웃라이어 제외 {len(tsub)}개",
            fontsize=12, fontweight="bold",
            color=cinfo["color"], pad=6
        )
        ax_sum.text(0, -1.55, f"Apple {apple_n}개\nSamsung {samsung_n}개",
                    ha="center", va="top", fontsize=10, color="#555")

        # ── 토픽별 제품 분포 스택 바 ────────────────────
        for t_idx, tid in enumerate(tids):
            ax = fig.add_subplot(inner[t_idx + 1])
            trows = tsub[tsub["topic"] == tid]
            t_total = len(trows)
            tlabel = TOPIC_LABELS.get((cid, tid), f"T{tid}")

            # 제품별 카운트
            prod_counts = {p: (trows["product_name"] == p).sum() for p in PRODUCTS}
            prod_pcts   = {p: v / t_total * 100 for p, v in prod_counts.items()}

            # 수평 스택 바 1개 (전체 합 = 100%)
            left = 0
            for p in PRODUCTS:
                pct = prod_pcts[p]
                if pct < 0.5:
                    left += pct
                    continue
                ax.barh(0, pct, left=left, color=PROD_COLORS[p],
                        edgecolor="white", linewidth=0.8, height=0.45)
                if pct >= 7:
                    ax.text(left + pct / 2, 0,
                            f"{PROD_SHORT[p]}\n{pct:.0f}%",
                            ha="center", va="center",
                            fontsize=8.5, color="white", fontweight="bold")
                left += pct

            # 제품별 개별 바 (아래)
            y_positions = np.arange(1, len(PRODUCTS) + 1) * 0.7
            for y, p in zip(y_positions, PRODUCTS):
                n   = prod_counts[p]
                pct = prod_pcts[p]
                bar_color = PROD_COLORS[p]
                ax.barh(y, pct, color=bar_color, alpha=0.8,
                        edgecolor="white", linewidth=0.5, height=0.5)
                if pct >= 3:
                    ax.text(pct + 0.8, y,
                            f"{n}개\n({pct:.0f}%)",
                            va="center", fontsize=8, color="#333",
                            fontweight="bold")
                ax.text(-1, y, PROD_SHORT[p], va="center",
                        ha="right", fontsize=8.5, color="#333")

            ax.set_xlim(-14, 55)
            ax.set_ylim(-0.5, (len(PRODUCTS) + 1) * 0.7 + 0.3)
            ax.axis("off")

            apple_t   = trows[trows["brand"] == "Apple"].shape[0]
            samsung_t = trows[trows["brand"] == "Samsung"].shape[0]
            ax.set_title(
                f"[토픽{tid}] {tlabel}\n{t_total}개 | "
                f"🍎{apple_t} / 🤖{samsung_t}",
                fontsize=11, fontweight="bold",
                color=cinfo["color"], pad=6
            )

            # 구분선
            ax.axhline(0.28, color="#CCCCCC", linewidth=0.8, linestyle="--")

    # 범례
    legend_patches = [mpatches.Patch(color=c, label=PROD_SHORT[p])
                      for p, c in PROD_COLORS.items()]
    legend_patches += [
        mpatches.Patch(color="#1976D2", label="Apple (전체)"),
        mpatches.Patch(color="#388E3C", label="Samsung (전체)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=5,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.005),
               facecolor="white", edgecolor="#CCC")

    plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"저장: {OUT}")

if __name__ == "__main__":
    main()
