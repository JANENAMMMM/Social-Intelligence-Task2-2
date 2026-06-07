"""
04_cohort_crossanalysis.py
코호트 x 클러스터 교차 분석
- 전체 Chi-square
- Fisher's exact (배송 클러스터 집중도: D+0~30 vs D+91+)
- Helpfulness 코호트 차이 KW + 사후 검정
- 코호트별 불만 프로파일 변화
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "output"

master = pd.read_csv(DATA / "master_dataset.csv", encoding="utf-8-sig")
main = master[master["cluster_display"] != "기타"].dropna(subset=["cohort"]).copy()

cohort_order = ["D+0~30 (얼리어답터)", "D+31~90 (초기다수)", "D+91+ (후기다수)"]
main["cohort"] = main["cohort"].str.strip()
main = main[main["cohort"].isin(cohort_order)].copy()

# ---- 1. 코호트 x 클러스터 Chi-square ----
cohort_cluster = pd.crosstab(main["cohort"], main["cluster_display"])
chi2, p, dof, expected = stats.chi2_contingency(cohort_cluster)
print(f"=== 코호트 x 클러스터 Chi-square ===")
print(f"chi2={chi2:.4f}, p={p:.2e}, dof={dof}")

cohort_pct = cohort_cluster.div(cohort_cluster.sum(axis=1), axis=0) * 100
std_resid = (cohort_cluster - expected) / np.sqrt(expected)

print("\n코호트별 클러스터 비율(%):")
print(cohort_pct.round(1).to_string())
print("\n표준화 잔차 (|z|>2 주목):")
print(std_resid.round(2).to_string())

cohort_pct.to_csv(OUT / "04_cohort_cluster_pct.csv", encoding="utf-8-sig")
std_resid.to_csv(OUT / "04_cohort_cluster_stdresid.csv", encoding="utf-8-sig")

# ---- 2. Fisher's exact: 배송 불만이 D+0~30에 집중되는가? ----
delivery_clusters = ["배송 속도·방법", "포장·박스 불만"]
for dc in delivery_clusters:
    early = main[main["cohort"] == "D+0~30 (얼리어답터)"]
    late = main[main["cohort"] == "D+91+ (후기다수)"]

    a = (early["cluster_display"] == dc).sum()
    b = len(early) - a
    c = (late["cluster_display"] == dc).sum()
    d = len(late) - c

    contingency = np.array([[a, b], [c, d]])
    odds_ratio, p_fisher = stats.fisher_exact(contingency, alternative="greater")

    early_pct = a / len(early) * 100 if len(early) > 0 else 0
    late_pct = c / len(late) * 100 if len(late) > 0 else 0

    print(f"\nFisher's exact [{dc}]:")
    print(f"  D+0~30: {a}/{len(early)} ({early_pct:.1f}%)")
    print(f"  D+91+:  {c}/{len(late)} ({late_pct:.1f}%)")
    print(f"  Odds Ratio={odds_ratio:.3f}, p={p_fisher:.4f} ({'유의' if p_fisher<0.05 else '비유의'})")

# ---- 3. Helpfulness 코호트 차이 (리뷰 단위) ----
rev_level = (
    master[master["cohort"].isin(cohort_order)]
    .drop_duplicates(subset=["review_id"])
    .dropna(subset=["helpfulTrueCount"])
    .copy()
)
rev_level["cohort"] = rev_level["cohort"].str.strip()

g_early = rev_level[rev_level["cohort"] == "D+0~30 (얼리어답터)"]["helpfulTrueCount"]
g_mid = rev_level[rev_level["cohort"] == "D+31~90 (초기다수)"]["helpfulTrueCount"]
g_late = rev_level[rev_level["cohort"] == "D+91+ (후기다수)"]["helpfulTrueCount"]

kw_stat, kw_p = stats.kruskal(g_early, g_mid, g_late)
print(f"\n=== Helpfulness KW ===")
print(f"H={kw_stat:.4f}, p={kw_p:.2e}")
for label, g in [("D+0~30", g_early), ("D+31~90", g_mid), ("D+91+", g_late)]:
    print(f"  {label}: n={len(g)}, mean={g.mean():.2f}, median={g.median():.1f}")

# 쌍별 MWU
for (l1, g1), (l2, g2) in [
    (("D+0~30", g_early), ("D+91+", g_late)),
    (("D+0~30", g_early), ("D+31~90", g_mid)),
    (("D+31~90", g_mid), ("D+91+", g_late)),
]:
    u, p = stats.mannwhitneyu(g1, g2, alternative="greater")
    r = 1 - 2 * u / (len(g1) * len(g2))
    print(f"  MWU {l1} > {l2}: p={p:.4f}, r={r:.3f}")

# ---- 4. 코호트별 불만 프로파일 Top3 ----
print("\n=== 코호트별 Top3 불만 클러스터 ===")
for cohort in cohort_order:
    sub = main[main["cohort"] == cohort]
    top3 = sub["cluster_display"].value_counts().head(3)
    pct = top3 / len(sub) * 100
    print(f"\n{cohort} (n={len(sub)}):")
    for cluster, cnt in top3.items():
        print(f"  {cluster}: {cnt}건 ({pct[cluster]:.1f}%)")

print("\n저장 완료: 04_*.csv")
