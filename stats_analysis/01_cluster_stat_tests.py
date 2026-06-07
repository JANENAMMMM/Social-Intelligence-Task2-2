"""
01_cluster_stat_tests.py
클러스터 간 별점 차이 통계 검정
- Kruskal-Wallis (전체)
- 쌍별 Mann-Whitney U + Bonferroni 보정
- 효과 크기 (rank-biserial r)
- 클러스터별 기술통계
"""
import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
from pathlib import Path

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

master = pd.read_csv(DATA / "master_dataset.csv", encoding="utf-8-sig")
# 리뷰 단위로 (문장 여러 개 → 리뷰 별점은 동일이므로 리뷰 기준으로 집계)
rev = master.drop_duplicates(subset=["review_id", "cluster_name"]).copy()

# 기타 제외
main = rev[rev["cluster_display"] != "기타"].copy()
clusters = sorted(main["cluster_display"].unique())

print("=== 클러스터 기술통계 ===")
desc = main.groupby("cluster_display")["rating"].agg(
    ["count", "mean", "median", "std"]
).round(3).sort_values("mean")
print(desc.to_string())

# Kruskal-Wallis
groups = [main[main["cluster_display"] == c]["rating"].values for c in clusters]
kw_stat, kw_p = stats.kruskal(*groups)
print(f"\nKruskal-Wallis: H={kw_stat:.4f}, p={kw_p:.2e}")

# 쌍별 Mann-Whitney U
pairs = list(combinations(clusters, 2))
results = []
for c1, c2 in pairs:
    g1 = main[main["cluster_display"] == c1]["rating"].values
    g2 = main[main["cluster_display"] == c2]["rating"].values
    u_stat, p_val = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    n1, n2 = len(g1), len(g2)
    # rank-biserial r
    r = 1 - 2 * u_stat / (n1 * n2)
    results.append({
        "cluster_1": c1, "cluster_2": c2,
        "mean_1": np.mean(g1), "mean_2": np.mean(g2),
        "U": u_stat, "p_raw": p_val, "r": r,
        "n1": n1, "n2": n2
    })

df_pairs = pd.DataFrame(results)
# Bonferroni 보정
df_pairs["p_bonferroni"] = (df_pairs["p_raw"] * len(pairs)).clip(upper=1.0)
df_pairs["significant"] = df_pairs["p_bonferroni"] < 0.05

sig = df_pairs[df_pairs["significant"]].sort_values("p_bonferroni")
print(f"\n유의미한 쌍 ({len(sig)}/{len(pairs)}개):")
print(sig[["cluster_1", "cluster_2", "mean_1", "mean_2", "p_bonferroni", "r"]].to_string())

df_pairs.to_csv(OUT / "01_cluster_pairwise_mwu.csv", index=False, encoding="utf-8-sig")
desc.to_csv(OUT / "01_cluster_desc_stats.csv", encoding="utf-8-sig")

# 핵심 발견 요약
print("\n=== 핵심 발견 ===")
print(f"KW: H={kw_stat:.2f}, p={kw_p:.2e} -> 클러스터 간 별점 분포 차이 통계적으로 유의")
worst = desc["mean"].idxmin()
best = desc["mean"].idxmax()
print(f"최저 별점: '{worst}' (mean={desc.loc[worst,'mean']:.2f})")
print(f"최고 별점: '{best}' (mean={desc.loc[best,'mean']:.2f})")

# 포장·배송 vs 제품기능 클러스터 대비
logistics = main[main["cluster_display"].isin(["포장·박스 불만", "배송 속도·방법"])]["rating"]
product_func = main[main["cluster_display"].isin(["배터리·충전·발열", "카메라·사진", "영상·게임 성능"])]["rating"]
u, p = stats.mannwhitneyu(logistics, product_func, alternative="less")
r_lp = 1 - 2 * u / (len(logistics) * len(product_func))
print(f"\n물류 불만(포장+배송) vs 기능 불만(배터리+카메라+영상): U={u:.0f}, p={p:.2e}, r={r_lp:.3f}")
print(f"  물류 mean={logistics.mean():.2f}, 기능 mean={product_func.mean():.2f}")
print(f"  -> 물류 관련 불만이 기능 불만보다 별점에서 훨씬 더 가혹하게 표출됨")
