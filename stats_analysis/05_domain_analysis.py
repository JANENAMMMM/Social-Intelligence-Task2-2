"""
05_domain_analysis.py
도메인 그룹별 정밀 분석
- 도메인 x 클러스터 Chi-square
- 도메인 x 브랜드 Chi-square (Gmail=Samsung 반전 재확인)
- Poisson GLM: helpfulness 예측 요인 재확인
- 도메인별 리뷰 품질 차이 (길이, 사진, helpfulness)
"""
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
from pathlib import Path

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "output"

master = pd.read_csv(DATA / "master_dataset.csv", encoding="utf-8-sig")
main = master[master["cluster_display"] != "기타"].copy()

# 리뷰 단위 데이터 (중복 제거)
rev = master.drop_duplicates(subset=["review_id"]).copy()
rev_main = rev[rev["domain_group"].isin(["네이버", "Gmail", "카카오", "네이트"])].copy()

# ---- 1. 도메인 x 클러스터 Chi-square ----
domain_cluster = pd.crosstab(
    main[main["domain_group"].isin(["네이버", "Gmail", "카카오", "네이트"])]["domain_group"],
    main[main["domain_group"].isin(["네이버", "Gmail", "카카오", "네이트"])]["cluster_display"]
)
chi2, p, dof, expected = stats.chi2_contingency(domain_cluster)
print(f"=== 도메인 x 클러스터 Chi-square ===")
print(f"chi2={chi2:.4f}, p={p:.2e}, dof={dof}")

std_resid = (domain_cluster - expected) / np.sqrt(expected)
domain_cluster_pct = domain_cluster.div(domain_cluster.sum(axis=1), axis=0) * 100

print("\n도메인별 클러스터 비율(%):")
print(domain_cluster_pct.round(1).to_string())
print("\n표준화 잔차 (|z|>2 = 특이 패턴):")
print(std_resid.round(2).to_string())

domain_cluster_pct.to_csv(OUT / "05_domain_cluster_pct.csv", encoding="utf-8-sig")
std_resid.to_csv(OUT / "05_domain_cluster_stdresid.csv", encoding="utf-8-sig")

# ---- 2. 도메인 x 브랜드 Chi-square ----
domain_brand = pd.crosstab(rev_main["domain_group"], rev_main["brand"])
chi2_b, p_b, dof_b, _ = stats.chi2_contingency(domain_brand)
domain_brand_pct = domain_brand.div(domain_brand.sum(axis=1), axis=0) * 100
print(f"\n=== 도메인 x 브랜드 Chi-square ===")
print(f"chi2={chi2_b:.4f}, p={p_b:.2e}")
print(domain_brand_pct.round(1).to_string())
print("-> Gmail도 Samsung이 다수인가?")
gmail_samsung_pct = domain_brand_pct.loc["Gmail", "Samsung"] if "Gmail" in domain_brand_pct.index else None
print(f"   Gmail 중 Samsung 비율: {gmail_samsung_pct:.1f}%")

domain_brand_pct.to_csv(OUT / "05_domain_brand_pct.csv", encoding="utf-8-sig")

# ---- 3. 도메인별 기술통계 (rating, content_len, helpfulness) ----
print("\n=== 도메인별 기술통계 ===")
for col in ["rating", "content_len", "helpfulTrueCount"]:
    print(f"\n[{col}]")
    g = rev_main.groupby("domain_group")[col].agg(["mean", "median", "std", "count"]).round(2)
    print(g.to_string())
    h_stat, h_p = stats.kruskal(
        *[rev_main[rev_main["domain_group"] == dg][col].dropna().values
          for dg in ["네이버", "Gmail", "카카오", "네이트"]]
    )
    print(f"  KW: H={h_stat:.2f}, p={h_p:.4f}")

# ---- 4. Poisson GLM (helpfulness 예측) ----
reg_df = rev_main.dropna(subset=["helpfulTrueCount", "content_len", "has_photo", "domain_group", "rating"]).copy()
reg_df["helpfulTrueCount"] = reg_df["helpfulTrueCount"].astype(int)
reg_df["has_photo_int"] = reg_df["has_photo"].astype(int)
reg_df["log_len"] = np.log1p(reg_df["content_len"])
reg_df["domain_naver"] = (reg_df["domain_group"] == "네이버").astype(int)
reg_df["domain_gmail"] = (reg_df["domain_group"] == "Gmail").astype(int)
reg_df["domain_kakao"] = (reg_df["domain_group"] == "카카오").astype(int)

formula = "helpfulTrueCount ~ log_len + has_photo_int + domain_naver + domain_gmail + domain_kakao + rating"
try:
    model = smf.poisson(formula, data=reg_df).fit(disp=False)
    print("\n=== Poisson GLM (helpfulness) ===")
    summary_df = pd.DataFrame({
        "coef": model.params,
        "IRR": np.exp(model.params),
        "p": model.pvalues,
        "sig": model.pvalues.apply(lambda x: "***" if x<0.001 else "**" if x<0.01 else "*" if x<0.05 else "")
    }).round(4)
    print(summary_df.to_string())
    summary_df.to_csv(OUT / "05_poisson_glm.csv", encoding="utf-8-sig")
except Exception as e:
    print(f"GLM error: {e}")

print("\n저장 완료: 05_*.csv")
