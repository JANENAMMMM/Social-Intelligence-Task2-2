"""
02_brand_product_chi2.py
브랜드/제품별 불만 클러스터 분포 Chi-square 검정
- 브랜드(Apple vs Samsung) x 클러스터: Chi-square
- 제품 x 클러스터: Chi-square + 잔차 분석으로 특이 제품-클러스터 조합 탐색
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "output"

master = pd.read_csv(DATA / "master_dataset.csv", encoding="utf-8-sig")
main = master[master["cluster_display"] != "기타"].copy()

# ---- 1. 브랜드 x 클러스터 Chi-square ----
brand_cluster = pd.crosstab(main["brand"], main["cluster_display"])
chi2, p, dof, expected = stats.chi2_contingency(brand_cluster)
print("=== 브랜드 x 클러스터 Chi-square ===")
print(f"chi2={chi2:.4f}, p={p:.2e}, dof={dof}")

# 표준화 잔차: (observed - expected) / sqrt(expected)
std_resid_brand = (brand_cluster - expected) / np.sqrt(expected)
print("\n표준화 잔차 (|z|>2 = 유의미한 과대/과소):")
print(std_resid_brand.round(2).to_string())

brand_cluster_pct = brand_cluster.div(brand_cluster.sum(axis=1), axis=0) * 100
print("\n브랜드별 클러스터 비율(%):")
print(brand_cluster_pct.round(1).to_string())

brand_cluster_pct.to_csv(OUT / "02_brand_cluster_pct.csv", encoding="utf-8-sig")
std_resid_brand.to_csv(OUT / "02_brand_cluster_stdresid.csv", encoding="utf-8-sig")

# 핵심: Apple vs Samsung 가장 큰 차이 클러스터
diff = brand_cluster_pct.loc["Apple"] - brand_cluster_pct.loc["Samsung"]
print("\nApple - Samsung 비율 차이 (크면 Apple이 더 많이 불만):")
print(diff.sort_values(ascending=False).round(2))

# ---- 2. 제품 x 클러스터 Chi-square ----
prod_cluster = pd.crosstab(main["product_label"], main["cluster_display"])
chi2_p, p_p, dof_p, expected_p = stats.chi2_contingency(prod_cluster)
print(f"\n=== 제품 x 클러스터 Chi-square ===")
print(f"chi2={chi2_p:.4f}, p={p_p:.2e}, dof={dof_p}")

std_resid_prod = (prod_cluster - expected_p) / np.sqrt(expected_p)
prod_cluster_pct = prod_cluster.div(prod_cluster.sum(axis=1), axis=0) * 100

print("\n제품별 클러스터 비율(%):")
print(prod_cluster_pct.round(1).to_string())

# 각 클러스터별 최다 불만 제품
print("\n각 클러스터에서 가장 높은 비율의 제품:")
for col in prod_cluster_pct.columns:
    top_prod = prod_cluster_pct[col].idxmax()
    top_pct = prod_cluster_pct[col].max()
    resid = std_resid_prod.loc[top_prod, col]
    print(f"  [{col}] -> {top_prod}: {top_pct:.1f}% (표준화잔차={resid:.2f})")

std_resid_prod.to_csv(OUT / "02_product_cluster_stdresid.csv", encoding="utf-8-sig")
prod_cluster_pct.to_csv(OUT / "02_product_cluster_pct.csv", encoding="utf-8-sig")

# ---- 3. 핵심 발견: S26 Ultra 화면 불만 집중도 ----
s26u_screen = prod_cluster_pct.loc["Galaxy S26 Ultra", "화면·프라이버시"]
all_screen = prod_cluster_pct["화면·프라이버시"]
print(f"\n[인사이트] S26 Ultra 화면·프라이버시 비율: {s26u_screen:.1f}%")
print(f"  타 제품 평균: {all_screen.drop('Galaxy S26 Ultra').mean():.1f}%")
print(f"  표준화잔차: {std_resid_prod.loc['Galaxy S26 Ultra', '화면·프라이버시']:.2f}")

# ---- 4. Z Fold7 '이전폰 비교' 집중도 ----
fold_comp = prod_cluster_pct.loc["Galaxy Z Fold7", "이전폰 비교·교체"]
print(f"\n[인사이트] Z Fold7 '이전폰 비교' 비율: {fold_comp:.1f}%")
print(f"  타 제품 평균: {prod_cluster_pct['이전폰 비교·교체'].drop('Galaxy Z Fold7').mean():.1f}%")

print("\n저장 완료: 02_*.csv")
