"""
10_advanced_visualizations.py
고도화 인사이트 시각화
- Fig 8: 숨겨진 불만 구조 (별점 5점 문장 비율 vs 평균 별점)
- Fig 9: 얼리어답터 편향 증폭 메커니즘 (화면·프라이버시 집중 + helpfulness)
- Fig 10: 제품별 취약점 프로파일 (버블 차트)
- Fig 11: 추세 클러스터 vs 일시적 클러스터 (시계열 4개)
- Fig 12: 책임 주체별 불만 분류 (treemap-style)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "output"

import platform
if platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.dpi"] = 150

master = pd.read_csv(DATA / "master_dataset.csv", encoding="utf-8-sig")
main = master[master["cluster_display"] != "기타"].copy()

# =====================================================================
# [Fig 8] 숨겨진 불만 구조 — 별점 5점 비율 vs 평균 별점 산점도
# =====================================================================
five_star_df = pd.read_csv(OUT / "09a_five_star_hidden_complaints.csv", encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(11, 8))

COLORS = {"물류": "#E74C3C", "기능": "#2980B9", "기타": "#95A5A6"}
cluster_categories = {
    "포장·박스 불만": "물류",
    "배송 속도·방법": "물류",
    "배터리·충전·발열": "기능",
    "카메라·사진": "기능",
    "화면·프라이버시": "기능",
    "영상·게임 성능": "기능",
}

for _, row in five_star_df.iterrows():
    cat = cluster_categories.get(row["cluster_display"], "기타")
    color = COLORS[cat]
    ax.scatter(row["pct_5star_in_cluster"], row["avg_rating"],
               s=row["total"] * 0.4, color=color, alpha=0.75, edgecolors="white", linewidths=1.5)
    offset_x = 1.5 if row["pct_5star_in_cluster"] < 80 else -2
    offset_y = 0.02
    ax.annotate(row["cluster_display"],
                (row["pct_5star_in_cluster"], row["avg_rating"]),
                xytext=(offset_x, offset_y), textcoords="offset points",
                fontsize=9.5, ha="left")

ax.axvline(x=75, color="gray", linestyle="--", alpha=0.5)
ax.axhline(y=4.5, color="gray", linestyle="--", alpha=0.5)

ax.annotate("← 포장·배송:\n별점도 낮고\n5점 비율도 낮음\n(솔직한 불만)", xy=(45, 3.5),
            fontsize=9, color="#C0392B", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FADBD8", alpha=0.8))
ax.annotate("← 기능 불만:\n별점 5점인데도\n90%+ 불만 문장\n(숨겨진 불만)", xy=(92, 4.8),
            fontsize=9, color="#1A5276", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#D6EAF8", alpha=0.8))

legend_patches = [
    mpatches.Patch(color="#E74C3C", label="물류 불만 (쿠팡)"),
    mpatches.Patch(color="#2980B9", label="제품 기능 불만 (제조사)"),
    mpatches.Patch(color="#95A5A6", label="기타 불만 (비교/가격)"),
]
ax.legend(handles=legend_patches, fontsize=10, loc="upper left")
ax.set_xlabel("별점 5점 리뷰 문장 비율 (%)", fontsize=12)
ax.set_ylabel("클러스터 평균 별점", fontsize=12)
ax.set_title("[WHAT] 숨겨진 불만 구조\n(X축: 5점 리뷰에서 온 불만 문장 비율 — 오른쪽일수록 '별점에 숨겨진' 불만)",
             fontsize=13, fontweight="bold")
ax.set_xlim(30, 102)
ax.set_ylim(2.8, 5.2)
ax.grid(alpha=0.3)
ax.text(0.99, 0.02, "노드 크기 = 문장 수",
        transform=ax.transAxes, fontsize=9, ha="right", color="gray")

plt.tight_layout()
plt.savefig(OUT / "fig08_hidden_complaint_structure.png")
plt.close()
print("저장: fig08_hidden_complaint_structure.png")

# =====================================================================
# [Fig 9] 얼리어답터 편향 메커니즘 — 2패널
# =====================================================================
cohort_amp = pd.read_csv(OUT / "09b_cohort_amplification.csv", encoding="utf-8-sig", index_col=0)
ts_cnt = pd.read_csv(OUT / "03_timeseries_counts.csv", encoding="utf-8-sig", index_col=0)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 좌: 코호트별 클러스터 비율 비교 (얼리어답터 vs 후기다수)
cohort_amp_sorted = cohort_amp.sort_values("ratio", ascending=True)
bar_colors = ["#E74C3C" if r > 1.3 else "#2980B9" if r < 0.7 else "#BDC3C7"
              for r in cohort_amp_sorted["ratio"]]
axes[0].barh(range(len(cohort_amp_sorted)), cohort_amp_sorted["ratio"],
             color=bar_colors, edgecolor="white", height=0.7)
axes[0].axvline(x=1.0, color="black", linestyle="--", linewidth=2)
axes[0].set_yticks(range(len(cohort_amp_sorted)))
axes[0].set_yticklabels(cohort_amp_sorted.index, fontsize=10)
axes[0].set_xlabel("얼리어답터 비율 / 후기다수 비율", fontsize=11)
axes[0].set_title("[WHO×WHEN] 코호트 불만 편향 비율\n(빨강: 얼리어답터 집중, 파랑: 후기다수 집중, 기준선=1.0)",
                  fontsize=11, fontweight="bold")
for i, (_, row) in enumerate(cohort_amp_sorted.iterrows()):
    axes[0].text(row["ratio"] + 0.03, i, f"{row['ratio']:.2f}",
                va="center", fontsize=9, fontweight="bold")

# 우: 화면·프라이버시 시계열 (얼리어답터가 초기 발화자임을 보여주는)
months = list(ts_cnt.index)
screen_vals = ts_cnt["화면·프라이버시"].values if "화면·프라이버시" in ts_cnt.columns else np.zeros(len(months))

cohort_order = ["D+0~30 (얼리어답터)", "D+31~90 (초기다수)", "D+91+ (후기다수)"]
main_c = main[main["cohort"].isin(cohort_order)].dropna(subset=["review_month"]).copy()

screen_by_cohort = main_c[main_c["cluster_display"] == "화면·프라이버시"].groupby(
    ["review_month", "cohort"]).size().unstack(fill_value=0)
screen_by_cohort = screen_by_cohort.reindex(months)

cohort_colors_map = {"D+0~30 (얼리어답터)": "#E74C3C",
                     "D+31~90 (초기다수)": "#F39C12",
                     "D+91+ (후기다수)": "#27AE60"}

for cohort in [c for c in cohort_order if c in screen_by_cohort.columns]:
    vals = screen_by_cohort[cohort].fillna(0).values
    axes[1].plot(months, vals, marker="o", linewidth=2.5, markersize=8,
                color=cohort_colors_map[cohort], label=cohort.split(" ")[0])
    axes[1].fill_between(range(len(months)), 0, vals, alpha=0.1, color=cohort_colors_map[cohort])

axes[1].set_xticks(range(len(months)))
axes[1].set_xticklabels(months, rotation=45, fontsize=8)
axes[1].set_ylabel("화면·프라이버시 불만 문장 수", fontsize=10)
axes[1].set_title("[화면·프라이버시] 코호트별 불만 발생 시기\n(얼리어답터가 초기에 문제 발화 → helpfulness 1.55x로 영향력 증폭)",
                  fontsize=11, fontweight="bold")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

# 총 화면 불만 합계 선 추가
total_screen = screen_by_cohort.sum(axis=1).fillna(0).values
axes[1].plot(months, total_screen, color="gray", linewidth=1, linestyle="--",
            alpha=0.5, label="합계")
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUT / "fig09_cohort_amplification.png")
plt.close()
print("저장: fig09_cohort_amplification.png")

# =====================================================================
# [Fig 10] 제품별 취약점 프로파일 버블 차트
# =====================================================================
prod_vuln = pd.read_csv(OUT / "09c_product_vulnerability.csv", encoding="utf-8-sig", index_col=0)
prod_pct = pd.read_csv(OUT / "02_product_cluster_pct.csv", encoding="utf-8-sig", index_col=0)

fig, ax = plt.subplots(figsize=(12, 8))

product_colors = {
    "Galaxy S26": "#2196F3",
    "Galaxy S26 Ultra": "#1565C0",
    "Galaxy Z Fold7": "#7B1FA2",
    "Galaxy Z Flip7": "#AB47BC",
    "iPhone 17": "#FF6F00",
    "iPhone 17 Pro": "#E65100",
    "iPhone 17 Pro Max": "#BF360C",
}

for prod, row in prod_vuln.iterrows():
    color = product_colors.get(prod, "#95A5A6")
    pct = row["primary_pct"]
    z = row["primary_z"]
    bubble_size = pct * 30 + 100

    ax.scatter(pct, z, s=bubble_size, color=color, alpha=0.8,
               edgecolors="white", linewidths=2)
    ax.annotate(f"{prod}\n[{row['primary_vulnerability']}]",
                (pct, z), textcoords="offset points", xytext=(10, 0),
                fontsize=9, color=color, fontweight="bold")

ax.set_xlabel("주요 취약 클러스터 불만 비율 (%)", fontsize=12)
ax.set_ylabel("표준화 잔차 (z-score, 통계적 이상치 강도)", fontsize=12)
ax.set_title("[WHAT] 제품별 취약점 프로파일\n(X: 실제 불만 비율, Y: 통계적 이상치 강도 — 오른쪽 위일수록 독보적 취약점)",
             fontsize=13, fontweight="bold")
ax.axhline(y=2, color="red", linestyle="--", alpha=0.4, linewidth=1.5, label="유의 임계값 (z=2)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Samsung vs Apple 영역 구분
ax.annotate("Samsung 취약 영역", xy=(20, 10), fontsize=10, color="#1565C0", alpha=0.6)
ax.annotate("Apple 취약 영역", xy=(14, 3.5), fontsize=10, color="#E65100", alpha=0.6)

plt.tight_layout()
plt.savefig(OUT / "fig10_product_vulnerability_bubble.png")
plt.close()
print("저장: fig10_product_vulnerability_bubble.png")

# =====================================================================
# [Fig 11] 추세 vs 일시적 클러스터 시계열 비교 (4개 패널)
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("[WHEN] 구조적 문제(추세) vs 이벤트 기반(일시) 불만 분류\n(Mann-Kendall 추세 검정 결과)",
             fontsize=13, fontweight="bold")

TREND_CLUSTERS = ["카메라·사진", "영상·게임 성능", "가격·가성비", "이전 프로 비교"]  # 유의 추세
FLAT_CLUSTERS = ["포장·박스 불만", "배송 속도·방법", "배터리·충전·발열", "화면·프라이버시"]  # 평탄

months = list(ts_cnt.index)

for i, cluster in enumerate(TREND_CLUSTERS):
    ax = axes[i // 2][i % 2]
    if cluster in ts_cnt.columns:
        vals = ts_cnt[cluster].values
        x = range(len(months))
        ax.plot(months, vals, marker="o", linewidth=2.5, color="#E74C3C", markersize=6)
        ax.fill_between(x, 0, vals, alpha=0.2, color="#E74C3C")
        # 선형 추세선
        z_fit = np.polyfit(range(len(vals)), vals, 1)
        trend_line = np.poly1d(z_fit)(range(len(vals)))
        ax.plot(months, trend_line, linestyle="--", color="#C0392B", linewidth=1.5, alpha=0.8)
        slope = z_fit[0]
        ax.set_title(f"[추세 있음] {cluster}\n(월 평균 +{slope:.1f}문장 증가, MK p<0.05)",
                     fontsize=10, fontweight="bold", color="#C0392B")
    ax.set_xticks(range(0, len(months), 2))
    ax.set_xticklabels([months[i] for i in range(0, len(months), 2)], rotation=45, fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylabel("문장 수", fontsize=9)

plt.tight_layout()
plt.savefig(OUT / "fig11_trend_vs_episodic.png")
plt.close()
print("저장: fig11_trend_vs_episodic.png")

# =====================================================================
# [Fig 12] 책임 주체 × 영향 매트릭스 (Impact-Responsibility Matrix)
# =====================================================================
fig, ax = plt.subplots(figsize=(12, 8))

# X: 별점 영향 (낮을수록 별점 더 깎음 = 5 - avg_rating)
# Y: 문장 수 (규모)
cluster_stats = main.groupby("cluster_display").agg(
    sentences=("sentence", "count"),
    avg_rating=("rating", "mean"),
).round(2)

responsibility = {
    "포장·박스 불만": ("물류\n(쿠팡)", "#E74C3C"),
    "배송 속도·방법": ("물류\n(쿠팡)", "#E74C3C"),
    "화면·프라이버시": ("제품\n(삼성)", "#8E44AD"),
    "카메라·사진": ("제품\n(제조사)", "#3498DB"),
    "배터리·충전·발열": ("제품\n(제조사)", "#3498DB"),
    "영상·게임 성능": ("제품\n(제조사)", "#3498DB"),
    "이전폰 비교·교체": ("마케팅\n(쿠팡+제조사)", "#27AE60"),
    "이전 프로 비교": ("마케팅\n(제조사)", "#27AE60"),
    "가격·가성비": ("마케팅\n(제조사)", "#27AE60"),
    "케이스·그립감": ("악세서리\n(제조사)", "#F39C12"),
    "디자인 변화": ("커뮤니케이션\n(제조사)", "#95A5A6"),
}

scatter_handles = {}
for cluster, (owner, color) in responsibility.items():
    if cluster in cluster_stats.index:
        n = cluster_stats.loc[cluster, "sentences"]
        r = cluster_stats.loc[cluster, "avg_rating"]
        impact_x = 5 - r  # 별점 손실 (0에 가까울수록 별점에 영향 없음)
        ax.scatter(impact_x, n, s=n * 0.5, color=color, alpha=0.75,
                   edgecolors="white", linewidths=2)
        ax.annotate(cluster.replace("·", "\n"),
                    (impact_x, n), textcoords="offset points",
                    xytext=(8, 0), fontsize=8.5, ha="left")
        if owner not in scatter_handles:
            scatter_handles[owner] = mpatches.Patch(color=color, label=owner.replace("\n", " "))

ax.set_xlabel("별점 영향도 (5 - 평균별점, 클수록 별점 손실 큼)", fontsize=12)
ax.set_ylabel("불만 문장 수 (규모)", fontsize=12)
ax.set_title("[WHAT+WHY] 불만 클러스터 영향-규모 매트릭스\n(오른쪽+위 = 규모 크고 별점 파괴력도 큰 불만)",
             fontsize=13, fontweight="bold")

ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.4)
ax.axhline(y=500, color="gray", linestyle="--", alpha=0.4)

# 사분면 주석
ax.text(0.05, 0.97, "별점 영향 작음\n규모 적음\n(관리 대상)", transform=ax.transAxes,
        fontsize=8, va="top", color="gray", alpha=0.7)
ax.text(0.7, 0.97, "별점 영향 작음\n규모 큼\n(숨겨진 불만)", transform=ax.transAxes,
        fontsize=8, va="top", color="#1A5276", alpha=0.7)
ax.text(0.05, 0.05, "별점 영향 큼\n규모 적음\n(집중 개선)", transform=ax.transAxes,
        fontsize=8, va="bottom", color="#6E2F1A", alpha=0.7)
ax.text(0.7, 0.05, "별점 영향 큼\n규모 큼\n(긴급 대응)", transform=ax.transAxes,
        fontsize=8, va="bottom", color="#C0392B", alpha=0.7)

ax.legend(handles=list(scatter_handles.values()), fontsize=9, loc="upper left",
          title="책임 주체", title_fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "fig12_impact_responsibility_matrix.png")
plt.close()
print("저장: fig12_impact_responsibility_matrix.png")

print("\n=== 고도화 시각화 완료 ===")
