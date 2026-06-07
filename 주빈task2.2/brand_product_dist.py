"""
각 불만 클러스터별 브랜드/제품 분포 시각화
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

HERE    = Path(__file__).resolve().parent
CLUSTER = HERE / "clustered_complaints.csv"
RAW     = HERE / "review_with_sentiment.csv"
OUT     = HERE / "brand_product_dist.png"

def set_font():
    for name in ["Malgun Gothic", "NanumGothic", "AppleGothic"]:
        if any(f.name == name for f in fm.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False

CLUSTERS = {
    1: {"name": "C1\n쿠팡 배송/포장",     "color": "#EF5350"},
    2: {"name": "C2\n무게/외관/디자인",    "color": "#FF9800"},
    3: {"name": "C3\n완성도+배터리",       "color": "#42A5F5"},
    4: {"name": "C4\n카메라 야간/고스트",  "color": "#66BB6A"},
    5: {"name": "C5\n가격/구성품",         "color": "#AB47BC"},
}

PRODUCTS = [
    "iphone_17_pro_max",
    "iphone_17_pro",
    "iphone_17",
    "galaxy_s26_ultra",
    "galaxy_s26",
    "galaxy_z_fold7",
    "galaxy_z_flip7",
]
PROD_LABELS = {
    "iphone_17_pro_max": "iPhone 17\nPro Max",
    "iphone_17_pro":     "iPhone 17\nPro",
    "iphone_17":         "iPhone 17",
    "galaxy_s26_ultra":  "Galaxy\nS26 Ultra",
    "galaxy_s26":        "Galaxy S26",
    "galaxy_z_fold7":    "Galaxy\nZ Fold7",
    "galaxy_z_flip7":    "Galaxy\nZ Flip7",
}
PROD_BRAND = {p: "Apple" if "iphone" in p else "Samsung" for p in PRODUCTS}
BRAND_COLOR = {"Apple": "#1976D2", "Samsung": "#388E3C"}

def load():
    cluster = pd.read_csv(CLUSTER, encoding="utf-8-sig")
    raw     = pd.read_csv(RAW, encoding="utf-8-sig")
    raw["brand"] = raw["product_name"].apply(
        lambda x: "Apple" if "iphone" in str(x).lower() else "Samsung"
    )
    meta = raw[["reviewId", "brand", "product_name"]].rename(columns={"reviewId": "review_id"})
    cluster = cluster.drop(columns=["product_name"], errors="ignore").merge(meta, on="review_id", how="left")
    return cluster[cluster["cluster"].isin(CLUSTERS)]

def main():
    set_font()
    df = load()

    fig = plt.figure(figsize=(22, 28), facecolor="#F8F9FA")
    fig.suptitle("불만 클러스터별 브랜드 & 제품 분포",
                 fontsize=20, fontweight="bold", y=0.99)

    outer = gridspec.GridSpec(len(CLUSTERS), 1, figure=fig,
                              hspace=0.55, top=0.96, bottom=0.03,
                              left=0.06, right=0.97)

    for row_idx, (cid, cinfo) in enumerate(CLUSTERS.items()):
        sub  = df[df["cluster"] == cid]
        total = len(sub)

        inner = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[row_idx], wspace=0.35, width_ratios=[1, 2.2]
        )

        # ── 왼쪽: 브랜드 파이차트 ─────────────────────────
        ax_pie = fig.add_subplot(inner[0])
        apple_n   = (sub["brand"] == "Apple").sum()
        samsung_n = (sub["brand"] == "Samsung").sum()
        wedges, texts, autotexts = ax_pie.pie(
            [apple_n, samsung_n],
            labels=["Apple", "Samsung"],
            colors=["#1976D2", "#388E3C"],
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=2),
            textprops=dict(fontsize=12),
        )
        for at in autotexts:
            at.set_fontsize(12)
            at.set_fontweight("bold")
            at.set_color("white")
        ax_pie.set_title(
            f"{cinfo['name']}\n총 {total}개",
            fontsize=13, fontweight="bold",
            color=cinfo["color"], pad=8
        )

        # ── 오른쪽: 제품별 수평 바차트 ───────────────────
        ax_bar = fig.add_subplot(inner[1])

        counts = []
        for p in PRODUCTS:
            n = (sub["product_name"] == p).sum()
            counts.append(n)

        pct    = [c / total * 100 for c in counts]
        colors = [BRAND_COLOR[PROD_BRAND[p]] for p in PRODUCTS]
        y_pos  = np.arange(len(PRODUCTS))

        bars = ax_bar.barh(y_pos, pct, color=colors, alpha=0.85,
                           edgecolor="white", linewidth=0.8, height=0.6)

        # 바 오른쪽에 개수 + % 표시
        for i, (bar, c, p) in enumerate(zip(bars, counts, pct)):
            ax_bar.text(p + 0.5, i, f"{c}개 ({p:.1f}%)",
                        va="center", fontsize=11, fontweight="bold",
                        color="#333333")

        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels([PROD_LABELS[p] for p in PRODUCTS], fontsize=11)
        ax_bar.set_xlabel("비율 (%)", fontsize=11)
        ax_bar.set_xlim(0, max(pct) * 1.45)
        ax_bar.invert_yaxis()
        ax_bar.spines[["top", "right", "left"]].set_visible(False)
        ax_bar.grid(axis="x", alpha=0.3)

        # 브랜드 구분선
        ax_bar.axhline(2.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
        ax_bar.text(max(pct) * 1.4, 1.0, "Apple",   fontsize=10,
                    color="#1976D2", va="center", ha="right", fontstyle="italic")
        ax_bar.text(max(pct) * 1.4, 5.0, "Samsung", fontsize=10,
                    color="#388E3C", va="center", ha="right", fontstyle="italic")

        # 클러스터 색상 강조선
        ax_bar.set_title("제품별 분포", fontsize=12, color="#555555")
        for spine in ax_bar.spines.values():
            spine.set_edgecolor("#DDDDDD")

    # 범례
    from matplotlib.patches import Patch
    legend_els = [
        Patch(color="#1976D2", label="Apple"),
        Patch(color="#388E3C", label="Samsung"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=2,
               fontsize=13, frameon=True, bbox_to_anchor=(0.5, 0.005))

    plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"저장: {OUT}")

if __name__ == "__main__":
    main()
