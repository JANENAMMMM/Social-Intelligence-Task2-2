"""
08_final_visualizations.py
통계 검정 결과 기반 발표 품질 시각화
- 클러스터 별점 분포 (KW 결과 주석 포함)
- 스파이크 Z-score 히트맵
- 코호트 × 배송/포장 집중도
- 도메인 × 클러스터 (Gmail 패턴)
- 키워드 네트워크 그래프
- 종합 인사이트 패널
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

DATA = Path(__file__).parent / "data"
OUT_DATA = Path(__file__).parent / "output"
OUT_VIZ = Path(__file__).parent / "output"

# 한글 폰트 설정
import platform
if platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.dpi"] = 150

master = pd.read_csv(DATA / "master_dataset.csv", encoding="utf-8-sig")
main = master[master["cluster_display"] != "기타"].copy()
pairs_df = pd.read_csv(OUT_DATA / "01_cluster_pairwise_mwu.csv", encoding="utf-8-sig")
ts_z = pd.read_csv(OUT_DATA / "03_timeseries_zscore.csv", encoding="utf-8-sig", index_col=0)
ts_cnt = pd.read_csv(OUT_DATA / "03_timeseries_counts.csv", encoding="utf-8-sig", index_col=0)
poisson_df = pd.read_csv(OUT_DATA / "03_poisson_spike_test.csv", encoding="utf-8-sig")
cohort_pct = pd.read_csv(OUT_DATA / "04_cohort_cluster_pct.csv", encoding="utf-8-sig", index_col=0)

# =====================================================================
# [Fig 1] 클러스터 별점 분포 + KW 통계 주석
# =====================================================================
fig, ax = plt.subplots(figsize=(13, 7))

cluster_stats = main.groupby("cluster_display")["rating"].agg(["mean", "median", "sem"]).sort_values("mean")
clusters = cluster_stats.index.tolist()
means = cluster_stats["mean"].values
sems = cluster_stats["sem"].values

# 색상: 물류 = 빨강, 기능 = 파랑, 비교/기타 = 회색
LOGISTICS = {"포장·박스 불만", "배송 속도·방법"}
FUNCTION = {"배터리·충전·발열", "카메라·사진", "영상·게임 성능", "화면·프라이버시", "디자인 변화"}
colors = []
for c in clusters:
    if c in LOGISTICS: colors.append("#E74C3C")
    elif c in FUNCTION: colors.append("#2980B9")
    else: colors.append("#95A5A6")

bars = ax.barh(range(len(clusters)), means, xerr=sems, color=colors, edgecolor="white",
               height=0.7, capsize=4, error_kw={"elinewidth": 1.5, "ecolor": "gray"})

for i, (c, m) in enumerate(zip(clusters, means)):
    ax.text(m + 0.02, i, f"★{m:.2f}", va="center", fontsize=10, fontweight="bold")

ax.set_yticks(range(len(clusters)))
ax.set_yticklabels(clusters, fontsize=11)
ax.set_xlim(2.8, 5.3)
ax.set_xlabel("평균 별점", fontsize=12)
ax.set_title("[WHAT] 불만 클러스터별 별점 분포\n(Kruskal-Wallis: H=유의, p<0.001 — 물류 불만이 기능 불만보다 통계적으로 낮음)",
             fontsize=13, fontweight="bold")

legend_patches = [
    mpatches.Patch(color="#E74C3C", label="물류 불만 (포장·배송)"),
    mpatches.Patch(color="#2980B9", label="제품 기능 불만"),
    mpatches.Patch(color="#95A5A6", label="기타 불만"),
]
ax.legend(handles=legend_patches, loc="lower right", fontsize=10)
ax.axvline(x=means.mean(), color="gray", linestyle="--", alpha=0.5, label="전체 평균")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_VIZ / "fig01_cluster_rating_kw.png")
plt.close()
print("저장: fig01_cluster_rating_kw.png")

# =====================================================================
# [Fig 2] 시계열 Z-score 히트맵 (스파이크 시각화)
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 좌: Z-score 히트맵
cluster_order = list(ts_z.columns)
im = axes[0].imshow(ts_z[cluster_order].T, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
axes[0].set_xticks(range(len(ts_z.index)))
axes[0].set_xticklabels(ts_z.index, rotation=45, ha="right", fontsize=9)
axes[0].set_yticks(range(len(cluster_order)))
axes[0].set_yticklabels(cluster_order, fontsize=9)
plt.colorbar(im, ax=axes[0], label="Z-score")
axes[0].set_title("[WHEN] 클러스터별 월간 이상치 탐지\n(빨강=급증, 파랑=급감, |z|>2 = 통계적 이상치)", fontsize=11, fontweight="bold")

# |z|>2 표시
for i, month in enumerate(ts_z.index):
    for j, cluster in enumerate(cluster_order):
        z = ts_z.loc[month, cluster]
        if abs(z) > 2.0:
            axes[0].text(i, j, f"{z:.1f}", ha="center", va="center", fontsize=7,
                        fontweight="bold", color="white" if abs(z) > 2.5 else "black")

# 우: 2026-03 Poisson 검정 결과
sig_p = poisson_df.sort_values("ratio", ascending=False)
bar_colors = ["#E74C3C" if s else "#BDC3C7" for s in sig_p["significant"]]
axes[1].barh(range(len(sig_p)), sig_p["ratio"], color=bar_colors, edgecolor="white")
axes[1].set_yticks(range(len(sig_p)))
axes[1].set_yticklabels(sig_p["cluster"], fontsize=9)
axes[1].axvline(x=1.0, color="black", linestyle="--", linewidth=1.5, label="기준선(배율=1)")
axes[1].set_xlabel("2026-03 / 이전 평균 배율", fontsize=10)
axes[1].set_title("[WHEN] 2026-03 스파이크 Poisson 검정\n(빨강=p<0.05 통계적 유의)", fontsize=11, fontweight="bold")
for i, (_, row) in enumerate(sig_p.iterrows()):
    pstar = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
    axes[1].text(row["ratio"] + 0.05, i, f"{row['ratio']:.1f}x {pstar}", va="center", fontsize=9)

axes[1].grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_VIZ / "fig02_timeseries_spike_poisson.png")
plt.close()
print("저장: fig02_timeseries_spike_poisson.png")

# =====================================================================
# [Fig 3] 코호트 × 클러스터 비율 변화 (배송 집중도 강조)
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

cohort_order = ["D+0~30 (얼리어답터)", "D+31~90 (초기다수)", "D+91+ (후기다수)"]
available_cohorts = [c for c in cohort_order if c in cohort_pct.index]

# 좌: 배송·포장 클러스터 코호트별 비율 변화
logistics_cols = ["배송 속도·방법", "포장·박스 불만"]
available_logistics = [c for c in logistics_cols if c in cohort_pct.columns]

for col in available_logistics:
    vals = [cohort_pct.loc[c, col] if c in cohort_pct.index else 0 for c in cohort_order]
    axes[0].plot(cohort_order, vals, marker="o", linewidth=2.5, markersize=10, label=col)
    for x, y in zip(range(len(cohort_order)), vals):
        axes[0].annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)

# 기능 불만 (비교용)
func_cols = ["배터리·충전·발열", "카메라·사진"]
available_func = [c for c in func_cols if c in cohort_pct.columns]
for col in available_func:
    vals = [cohort_pct.loc[c, col] if c in cohort_pct.index else 0 for c in cohort_order]
    axes[0].plot(cohort_order, vals, marker="s", linewidth=1.5, markersize=7, linestyle="--",
                alpha=0.7, label=col)

axes[0].set_xlabel("코호트 (출시 후 경과 기간)", fontsize=11)
axes[0].set_ylabel("해당 클러스터 비율 (%)", fontsize=11)
axes[0].set_title("[WHO×WHEN] 코호트별 불만 클러스터 비율 변화\n(포장·배송은 전 코호트 평탄 — 출시 물류 문제는 '절대량' 급증이지 '비율' 아님)",
                  fontsize=11, fontweight="bold")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)
axes[0].set_xticks(range(len(cohort_order)))
axes[0].set_xticklabels(cohort_order, fontsize=9)

# 우: Helpfulness 코호트 비교 박스플롯
rev_level = master.drop_duplicates(subset=["review_id"]).dropna(subset=["helpfulTrueCount", "cohort"]).copy()
rev_level["cohort"] = rev_level["cohort"].str.strip()
rev_plot = rev_level[rev_level["cohort"].isin(cohort_order)].copy()

cohort_help_data = [rev_plot[rev_plot["cohort"] == c]["helpfulTrueCount"].values for c in cohort_order]
bp = axes[1].boxplot(cohort_help_data, labels=[c.split(" ")[0] for c in cohort_order],
                     patch_artist=True, showfliers=False, widths=0.5)

colors_bp = ["#E74C3C", "#F39C12", "#27AE60"]
for patch, color in zip(bp["boxes"], colors_bp):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

means_help = [np.mean(g) for g in cohort_help_data]
for i, m in enumerate(means_help):
    axes[1].scatter(i + 1, m, color="white", s=80, zorder=5)
    axes[1].annotate(f"평균\n{m:.1f}", (i + 1, m), textcoords="offset points",
                    xytext=(12, 0), fontsize=9, ha="left")

axes[1].set_ylabel("도움돼요 (helpfulTrueCount)", fontsize=11)
axes[1].set_title("[WHO] 코호트별 helpfulness 분포\n(KW H=92.94, p<0.001 — 얼리어답터가 구매 결정에 불균형 영향)",
                  fontsize=11, fontweight="bold")
axes[1].grid(axis="y", alpha=0.3)
axes[1].set_ylim(0, max(np.percentile(g, 95) for g in cohort_help_data) * 1.3)

plt.tight_layout()
plt.savefig(OUT_VIZ / "fig03_cohort_logistics_helpfulness.png")
plt.close()
print("저장: fig03_cohort_logistics_helpfulness.png")

# =====================================================================
# [Fig 4] 도메인 × 클러스터 Chi-square 결과 (표준화 잔차)
# =====================================================================
domain_resid = pd.read_csv(OUT_DATA / "05_domain_cluster_stdresid.csv",
                            encoding="utf-8-sig", index_col=0)
domain_pct = pd.read_csv(OUT_DATA / "05_domain_cluster_pct.csv",
                          encoding="utf-8-sig", index_col=0)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 좌: 표준화 잔차 히트맵
sns.heatmap(domain_resid.round(2), annot=True, fmt=".1f", cmap="RdBu_r",
            center=0, vmin=-3, vmax=3,
            ax=axes[0], cbar_kws={"label": "표준화 잔차"})
axes[0].set_title("[WHO] 도메인 × 클러스터 Chi-square 잔차\n(Chi2 p<0.001, 빨강=과대표현, 파랑=과소표현)",
                  fontsize=11, fontweight="bold")
axes[0].set_xlabel("")
axes[0].set_ylabel("이메일 도메인 그룹", fontsize=10)
axes[0].tick_params(axis="x", rotation=45)

# 우: 도메인별 클러스터 비율 히트맵
sns.heatmap(domain_pct.round(1), annot=True, fmt=".0f", cmap="YlOrRd",
            ax=axes[1], cbar_kws={"label": "비율 (%)"})
axes[1].set_title("[WHO] 도메인별 불만 클러스터 비율 (%)\n(Gmail: 배터리·AS 집중, 카카오: 가격 집중)",
                  fontsize=11, fontweight="bold")
axes[1].set_xlabel("")
axes[1].set_ylabel("")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(OUT_VIZ / "fig04_domain_cluster_chi2.png")
plt.close()
print("저장: fig04_domain_cluster_chi2.png")

# =====================================================================
# [Fig 5] 제품별 클러스터 집중도 + S26 Ultra 화면 이상치 강조
# =====================================================================
prod_stdresid = pd.read_csv(OUT_DATA / "02_product_cluster_stdresid.csv",
                             encoding="utf-8-sig", index_col=0)
prod_pct = pd.read_csv(OUT_DATA / "02_product_cluster_pct.csv",
                        encoding="utf-8-sig", index_col=0)

fig, ax = plt.subplots(figsize=(14, 7))
# 표준화 잔차 히트맵 (어떤 제품-클러스터 조합이 통계적으로 특이한가)
mask = abs(prod_stdresid) < 1.5  # 1.5 이하는 무시
annot_data = prod_stdresid.round(1).astype(str)
annot_data[abs(prod_stdresid) < 1.5] = ""

sns.heatmap(prod_stdresid, annot=annot_data, fmt="", cmap="RdBu_r",
            center=0, vmin=-4, vmax=4,
            ax=ax, cbar_kws={"label": "표준화 잔차 (|z|>2 주목)"})
ax.set_title("[WHAT] 제품 × 클러스터 Chi-square 표준화 잔차\n(Chi2 p<0.001 — 숫자: |z|>1.5 유의미한 이탈)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("불만 클러스터", fontsize=11)
ax.set_ylabel("제품", fontsize=11)
ax.tick_params(axis="x", rotation=45)

# S26 Ultra 화면 강조 박스
if "Galaxy S26 Ultra" in prod_stdresid.index and "화면·프라이버시" in prod_stdresid.columns:
    row_idx = list(prod_stdresid.index).index("Galaxy S26 Ultra")
    col_idx = list(prod_stdresid.columns).index("화면·프라이버시")
    ax.add_patch(plt.Rectangle((col_idx, row_idx), 1, 1, fill=False,
                                edgecolor="yellow", lw=3, label="S26 Ultra 화면 집중"))

plt.tight_layout()
plt.savefig(OUT_VIZ / "fig05_product_cluster_stdresid.png")
plt.close()
print("저장: fig05_product_cluster_stdresid.png")

# =====================================================================
# [Fig 6] 종합 인사이트 요약 패널
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("통합 분석 결과 요약 — 발표 준비", fontsize=14, fontweight="bold", y=1.01)

# 6-1: 물류 vs 기능 별점 차이
logistics_ratings = main[main["cluster_display"].isin(["포장·박스 불만", "배송 속도·방법"])]["rating"]
function_ratings = main[main["cluster_display"].isin(["배터리·충전·발열", "카메라·사진", "영상·게임 성능"])]["rating"]

axes[0,0].hist(logistics_ratings, bins=[0.5,1.5,2.5,3.5,4.5,5.5], density=True,
               alpha=0.7, color="#E74C3C", label=f"물류 불만 (n={len(logistics_ratings)})")
axes[0,0].hist(function_ratings, bins=[0.5,1.5,2.5,3.5,4.5,5.5], density=True,
               alpha=0.7, color="#2980B9", label=f"기능 불만 (n={len(function_ratings)})")
axes[0,0].axvline(logistics_ratings.mean(), color="#C0392B", linestyle="--", linewidth=2,
                   label=f"물류 평균 ★{logistics_ratings.mean():.2f}")
axes[0,0].axvline(function_ratings.mean(), color="#1A5276", linestyle="--", linewidth=2,
                   label=f"기능 평균 ★{function_ratings.mean():.2f}")
axes[0,0].set_xlabel("별점", fontsize=10)
axes[0,0].set_ylabel("밀도", fontsize=10)
axes[0,0].set_title("물류 vs 기능 불만 별점 분포\n(MWU p<0.001)", fontsize=11, fontweight="bold")
axes[0,0].legend(fontsize=8)

# 6-2: 2026-03 카메라·화면 스파이크
months = list(ts_cnt.index)
cam_vals = ts_cnt["카메라·사진"].values if "카메라·사진" in ts_cnt.columns else np.zeros(len(months))
screen_vals = ts_cnt["화면·프라이버시"].values if "화면·프라이버시" in ts_cnt.columns else np.zeros(len(months))

axes[0,1].plot(months, cam_vals, marker="o", color="#E74C3C", linewidth=2.5, label="카메라·사진")
axes[0,1].plot(months, screen_vals, marker="s", color="#8E44AD", linewidth=2.5, label="화면·프라이버시")
mar_idx = months.index("2026-03") if "2026-03" in months else -1
if mar_idx >= 0:
    axes[0,1].axvline(x=mar_idx, color="gray", linestyle="--", linewidth=2, alpha=0.7)
    axes[0,1].annotate("2026-03\n스파이크", (mar_idx, max(cam_vals.max(), screen_vals.max()) * 0.9),
                       xytext=(mar_idx - 1.5, max(cam_vals.max(), screen_vals.max()) * 0.95),
                       fontsize=9, color="red",
                       arrowprops=dict(arrowstyle="->", color="red"))
axes[0,1].set_xticks(range(len(months)))
axes[0,1].set_xticklabels(months, rotation=45, fontsize=8)
axes[0,1].set_ylabel("불만 문장 수", fontsize=10)
axes[0,1].set_title("카메라·화면 불만 시계열\n(Poisson 검정: 2026-03 p<0.001)", fontsize=11, fontweight="bold")
axes[0,1].legend(fontsize=9)
axes[0,1].grid(alpha=0.3)

# 6-3: 도메인별 AS 불만 집중도
domain_pct_reload = pd.read_csv(OUT_DATA / "05_domain_cluster_pct.csv",
                                  encoding="utf-8-sig", index_col=0)
as_clusters_for_plot = [c for c in ["배터리·충전·발열", "화면·프라이버시", "영상·게임 성능"]
                         if c in domain_pct_reload.columns]
domains_for_plot = [d for d in ["네이버", "Gmail", "카카오", "네이트"]
                    if d in domain_pct_reload.index]

if as_clusters_for_plot and domains_for_plot:
    domain_as = domain_pct_reload.loc[domains_for_plot, as_clusters_for_plot].sum(axis=1)
    bar_colors_dom = ["#3498DB" if d == "Gmail" else "#BDC3C7" for d in domains_for_plot]
    axes[1,0].bar(domains_for_plot, domain_as.values, color=bar_colors_dom, edgecolor="white")
    for i, (d, v) in enumerate(zip(domains_for_plot, domain_as.values)):
        axes[1,0].text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
    axes[1,0].set_ylabel("AS·기능 불만 클러스터 비율 합 (%)", fontsize=9)
    axes[1,0].set_title("도메인별 기능·AS 불만 집중도\n(파랑=Gmail, Chi2 p<0.001)", fontsize=11, fontweight="bold")
    axes[1,0].grid(axis="y", alpha=0.3)

# 6-4: 얼리어답터 helpfulness 우위
help_means = [g_early_m if len(g) > 0 else 0
              for g, g_early_m in zip(
                  [rev_plot[rev_plot["cohort"] == c]["helpfulTrueCount"].values if
                   len(rev_plot[rev_plot["cohort"] == c]) > 0 else [0] for c in cohort_order],
                  [rev_plot[rev_plot["cohort"] == c]["helpfulTrueCount"].mean() if
                   len(rev_plot[rev_plot["cohort"] == c]) > 0 else 0 for c in cohort_order]
              )]

cohort_short = ["D+0~30\n얼리어답터", "D+31~90\n초기다수", "D+91+\n후기다수"]
bar_c = ["#E74C3C", "#F39C12", "#27AE60"]
axes[1,1].bar(cohort_short, help_means, color=bar_c, edgecolor="white", width=0.5)
for i, (label, val) in enumerate(zip(cohort_short, help_means)):
    axes[1,1].text(i, val + 0.05, f"{val:.2f}", ha="center", fontsize=11, fontweight="bold")
if len(help_means) >= 3 and help_means[2] > 0:
    pct_diff = (help_means[0] - help_means[2]) / help_means[2] * 100
    axes[1,1].annotate(f"얼리어답터\n+{pct_diff:.0f}%\n더 공감받음",
                       xy=(0, help_means[0]), xytext=(1.5, help_means[0] * 1.1),
                       fontsize=9, color="#C0392B",
                       arrowprops=dict(arrowstyle="->", color="#C0392B"))
axes[1,1].set_ylabel("평균 도움돼요 수", fontsize=10)
axes[1,1].set_title("코호트별 평균 helpfulness\n(KW H=92.94, p<0.001)", fontsize=11, fontweight="bold")
axes[1,1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_VIZ / "fig06_comprehensive_insight_panel.png")
plt.close()
print("저장: fig06_comprehensive_insight_panel.png")

# =====================================================================
# [Fig 7] 키워드 네트워크 그래프
# =====================================================================
import networkx as nx

try:
    centrality_df = pd.read_csv(OUT_DATA / "06_keyword_centrality.csv", encoding="utf-8-sig")
    communities_df = pd.read_csv(OUT_DATA / "06_keyword_communities.csv", encoding="utf-8-sig")
    kw_df = pd.read_csv(OUT_DATA / "06_cluster_tfidf_keywords.csv", encoding="utf-8-sig")

    # 클러스터별 Top5 키워드만으로 서브그래프 구성
    top_kws = set(kw_df[kw_df["rank"] <= 5]["word"].tolist())
    top_central = set(centrality_df.head(30)["word"].tolist())
    key_words = top_kws | top_central

    master_temp = pd.read_csv(DATA / "master_dataset.csv", encoding="utf-8-sig")
    main_temp = master_temp[master_temp["cluster_display"] != "기타"].copy()

    import re
    STOPWORDS2 = {
        "이게", "이건", "이거", "그거", "그것", "그게", "것", "있는", "있어", "없는",
        "너무", "정말", "진짜", "하지만", "그런데", "그래서",
        "쿠팡", "아이폰", "갤럭시", "삼성", "애플", "폰", "제품",
        "사용", "이용", "구매", "생각", "느낌", "경우", "부분",
        "정도", "기능", "성능", "품질", "가격", "리뷰",
    }
    _PARTICLES2 = sorted(["에서", "에게", "이라", "이며", "이고", "이다",
                           "으로", "에서는", "에서도", "을", "를", "이", "가",
                           "은", "는", "도", "만", "과", "와", "에", "의", "로"], key=len, reverse=True)

    def extract_nouns2(text):
        words = str(text).split()
        nouns = []
        for word in words:
            w = word.strip(".,!?~ㅠㅋㅎ…")
            for p in _PARTICLES2:
                if w.endswith(p) and len(w) > len(p) + 1:
                    w = w[:-len(p)]
                    break
            for tok in re.findall(r'[가-힣]{2,}', w):
                if tok not in STOPWORDS2:
                    nouns.append(tok)
        return nouns

    cooc_kw = {}
    for _, row in main_temp.iterrows():
        nouns = [n for n in extract_nouns2(row["sentence"]) if n in key_words]
        nouns = list(set(nouns))
        for i in range(len(nouns)):
            for j in range(i + 1, len(nouns)):
                edge = tuple(sorted([nouns[i], nouns[j]]))
                cooc_kw[edge] = cooc_kw.get(edge, 0) + 1

    G_kw = nx.Graph()
    for (w1, w2), cnt in cooc_kw.items():
        if cnt >= 3:
            G_kw.add_edge(w1, w2, weight=cnt)
    for node in list(G_kw.nodes()):
        if node in centrality_df["word"].values:
            G_kw.nodes[node]["betweenness"] = centrality_df[centrality_df["word"] == node]["betweenness"].values[0]
        else:
            G_kw.nodes[node]["betweenness"] = 0.001

    if G_kw.number_of_nodes() > 5:
        fig, ax = plt.subplots(figsize=(14, 10))
        pos = nx.spring_layout(G_kw, seed=42, k=1.5)
        node_sizes = [G_kw.nodes[n].get("betweenness", 0.001) * 30000 + 200 for n in G_kw.nodes()]
        edge_weights = [G_kw[u][v]["weight"] for u, v in G_kw.edges()]
        max_w = max(edge_weights) if edge_weights else 1
        edge_widths = [w / max_w * 3 + 0.3 for w in edge_weights]

        # 커뮤니티별 색상
        try:
            comm_df_tmp = pd.read_csv(OUT_DATA / "06_keyword_communities.csv", encoding="utf-8-sig")
            comm_map = dict(zip(comm_df_tmp["word"], comm_df_tmp["community"]))
            comm_palette = ["#E74C3C","#3498DB","#2ECC71","#F39C12","#9B59B6","#1ABC9C","#E67E22","#95A5A6"]
            node_colors = [comm_palette[comm_map.get(n, 0) % len(comm_palette)] for n in G_kw.nodes()]
        except Exception:
            node_colors = ["#3498DB"] * G_kw.number_of_nodes()

        nx.draw_networkx_edges(G_kw, pos, ax=ax, width=edge_widths, alpha=0.3, edge_color="gray")
        nx.draw_networkx_nodes(G_kw, pos, ax=ax, node_size=node_sizes, node_color=node_colors, alpha=0.85)
        nx.draw_networkx_labels(G_kw, pos, ax=ax, font_size=8, font_family="Malgun Gothic",
                                font_weight="bold", font_color="white")
        ax.set_title("불만 키워드 공기어 네트워크\n(노드 크기 = 매개중심성, 선 굵기 = 공기 빈도)",
                     fontsize=13, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(OUT_VIZ / "fig07_keyword_network.png")
        plt.close()
        print("저장: fig07_keyword_network.png")
except Exception as e:
    print(f"네트워크 그래프 오류: {e}")

print("\n=== 모든 시각화 완료 ===")
print(f"출력 폴더: {OUT_VIZ}")
