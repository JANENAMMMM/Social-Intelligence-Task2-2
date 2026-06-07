"""
09_advanced_insights.py
고도화 인사이트 — 단순 통계 너머

1. 숨겨진 불만 증폭기: 별점 5점 리뷰 속 불만 문장의 클러스터별 실제 비율
   ("★4.90인데 실제 불만이 있는 배터리 클러스터" 정량화)
2. 공감 유발 리뷰 패턴: "이전폰 비교 + 사진 + 장문" 조합 효과
3. 얼리어답터 왜곡 지수: 영상/게임 불만이 얼리어답터에서 집중 + helpfulness 우위 → 구매 결정 편향
4. 제품별 취약점 프로파일: 각 제품의 독특한 약점 요약
5. Mann-Kendall 추세 클러스터: 시간이 지날수록 증가하는 불만 vs 감소하는 불만
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "output"

master = pd.read_csv(DATA / "master_dataset.csv", encoding="utf-8-sig")
main = master[master["cluster_display"] != "기타"].copy()

# ======================================================================
print("=" * 65)
print("[인사이트 A] 별점 5점 리뷰 속 불만 클러스터 — 숨겨진 불만 정량화")
print("=" * 65)

five_star = main[main["rating"] == 5]
total_5star_sents = len(five_star)
print(f"★5점 리뷰의 부정 문장 수: {total_5star_sents}개 (전체 {len(main)}개 중 {total_5star_sents/len(main)*100:.1f}%)")

cluster_5star = five_star.groupby("cluster_display").agg(
    count=("sentence", "count"),
    pct_of_cluster=("sentence", lambda x: len(x))
).reset_index()

# 각 클러스터 내에서 별점 5점인 비율
cluster_total = main.groupby("cluster_display")["sentence"].count().rename("total")
cluster_5star = cluster_5star.merge(cluster_total, on="cluster_display")
cluster_5star["pct_5star_in_cluster"] = cluster_5star["count"] / cluster_5star["total"] * 100

# 클러스터 평균 별점
cluster_avg = main.groupby("cluster_display")["rating"].mean().rename("avg_rating")
cluster_5star = cluster_5star.merge(cluster_avg, on="cluster_display")

cluster_5star = cluster_5star.sort_values("avg_rating", ascending=False)
print("\n클러스터별 별점 5점 문장 비율:")
print(cluster_5star[["cluster_display", "total", "count", "pct_5star_in_cluster", "avg_rating"]].to_string())
print()

# 핵심 발견: 별점이 높은데 5점 비율이 낮은 클러스터 = "불만을 참고 5점 준" 클러스터
print("주목할 패턴: 별점 높지만 5점 문장 비율이 낮은 클러스터 (불만을 참고 5점 준 경우)")
for _, row in cluster_5star.iterrows():
    if row["avg_rating"] >= 4.8 and row["pct_5star_in_cluster"] < 60:
        print(f"  [{row['cluster_display']}] avg★{row['avg_rating']:.2f}, 5점 비율={row['pct_5star_in_cluster']:.1f}%")

cluster_5star.to_csv(OUT / "09a_five_star_hidden_complaints.csv", index=False, encoding="utf-8-sig")

# ======================================================================
print("\n" + "=" * 65)
print("[인사이트 B] 얼리어답터 편향 증폭 메커니즘")
print("=" * 65)
# 얼리어답터가 영상/게임 불만에 집중 (z=5.85) + helpfulness가 55% 우위
# → 영상/게임 불만이 후기 구매자 결정에 불균형하게 영향

cohort_order = ["D+0~30 (얼리어답터)", "D+31~90 (초기다수)", "D+91+ (후기다수)"]
main_c = main[main["cohort"].isin(cohort_order)].copy()

# 각 클러스터별 얼리어답터 vs 후기다수 비율 비교 (z-score 방식)
print("클러스터별 코호트 비율 변화 (얼리어답터 / 후기다수):")
cohort_cluster = pd.crosstab(main_c["cohort"], main_c["cluster_display"])
cohort_pct = cohort_cluster.div(cohort_cluster.sum(axis=1), axis=0) * 100

amplification = {}
for cluster in cohort_pct.columns:
    if "D+0~30 (얼리어답터)" in cohort_pct.index and "D+91+ (후기다수)" in cohort_pct.index:
        early_pct = cohort_pct.loc["D+0~30 (얼리어답터)", cluster]
        late_pct = cohort_pct.loc["D+91+ (후기다수)", cluster]
        ratio = early_pct / late_pct if late_pct > 0 else 0
        amplification[cluster] = {
            "early_pct": early_pct,
            "late_pct": late_pct,
            "ratio": ratio,
        }

df_amp = pd.DataFrame(amplification).T.sort_values("ratio", ascending=False)
print(df_amp.round(2).to_string())

# 얼리어답터 helpfulness 우위
rev = master.drop_duplicates(subset=["review_id"]).copy()
rev_c = rev[rev["cohort"].isin(cohort_order)].dropna(subset=["helpfulTrueCount"])
early_help = rev_c[rev_c["cohort"] == "D+0~30 (얼리어답터)"]["helpfulTrueCount"].mean()
late_help = rev_c[rev_c["cohort"] == "D+91+ (후기다수)"]["helpfulTrueCount"].mean()
amplification_factor = early_help / late_help

print(f"\n얼리어답터 helpfulness 우위: {early_help:.2f} / {late_help:.2f} = {amplification_factor:.2f}x")
top_early = df_amp.sort_values("ratio", ascending=False).index[0]
top_late_inv = df_amp.sort_values("ratio").index[0]
print(f"\n[결론] 얼리어답터가 유독 집중하는 불만: '{top_early}' (ratio={df_amp.loc[top_early,'ratio']:.2f})")
print(f"  후기다수가 유독 집중하는 불만: '{top_late_inv}' (ratio={df_amp.loc[top_late_inv,'ratio']:.2f} → 후기다수가 {1/df_amp.loc[top_late_inv,'ratio']:.1f}x 더 많이)")
print(f"  + 얼리어답터 helpfulness {amplification_factor:.1f}x 우위")
print(f"  → '{top_early}' 불만이 얼리어답터에서 집중되고 이들이 더 공감받아")
print(f"     후기 구매자들의 구매 결정에 불균형하게 영향을 미칠 수 있음")

df_amp.to_csv(OUT / "09b_cohort_amplification.csv", encoding="utf-8-sig")

# ======================================================================
print("\n" + "=" * 65)
print("[인사이트 C] 제품별 취약점 프로파일 — Top 불만 + 통계 유의성")
print("=" * 65)

prod_stdresid = pd.read_csv(OUT / "02_product_cluster_stdresid.csv",
                             encoding="utf-8-sig", index_col=0)
prod_pct = pd.read_csv(OUT / "02_product_cluster_pct.csv",
                        encoding="utf-8-sig", index_col=0)

product_profiles = {}
for prod in prod_pct.index:
    # 가장 높은 잔차 (통계적으로 가장 이상하게 집중된 불만)
    resid_row = prod_stdresid.loc[prod]
    top_complaint = resid_row.idxmax()
    top_z = resid_row.max()
    top_pct = prod_pct.loc[prod, top_complaint]

    # 두 번째 취약점
    resid_row2 = resid_row.copy()
    resid_row2[top_complaint] = -999
    second_complaint = resid_row2.idxmax()
    second_z = resid_row2.max()

    product_profiles[prod] = {
        "primary_vulnerability": top_complaint,
        "primary_z": round(top_z, 2),
        "primary_pct": round(top_pct, 1),
        "secondary_vulnerability": second_complaint,
        "secondary_z": round(second_z, 2),
    }

df_profiles = pd.DataFrame(product_profiles).T.sort_values("primary_z", ascending=False)
print(df_profiles.to_string())

df_profiles.to_csv(OUT / "09c_product_vulnerability.csv", encoding="utf-8-sig")

# ======================================================================
print("\n" + "=" * 65)
print("[인사이트 D] 물류 vs 제품 불만의 구조적 차이 — 해결 난이도 분류")
print("=" * 65)

# 물류 불만: 쿠팡이 해결할 수 있음 (배송 정책, 포장 기준)
# 제품 불만: 제조사만 해결 가능 (하드웨어, 소프트웨어)
# 가격 불만: 마케팅으로 완화 가능

category_map = {
    "포장·박스 불만": ("물류_쿠팡", "쿠팡 포장 정책 개선"),
    "배송 속도·방법": ("물류_쿠팡", "쿠팡 배송 프로세스 개선"),
    "화면·프라이버시": ("제품_삼성", "S26 Ultra 프라이버시 필터 품질/OTA 개선"),
    "카메라·사진": ("제품_공통", "카메라 알고리즘 업데이트"),
    "배터리·충전·발열": ("제품_공통", "배터리 최적화 펌웨어"),
    "영상·게임 성능": ("제품_공통", "성능 모드 소프트웨어 개선"),
    "이전폰 비교·교체": ("마케팅", "마이그레이션 지원 강화"),
    "이전 프로 비교": ("마케팅", "업그레이드 가치 커뮤니케이션"),
    "가격·가성비": ("마케팅", "번들 혜택·할부 프로모션"),
    "케이스·그립감": ("제조사_악세서리", "공식 케이스 라인업 확대"),
    "디자인 변화": ("제품_디자인", "디자인 변화 사전 커뮤니케이션"),
}

cluster_stats = main.groupby("cluster_display").agg(
    sentences=("sentence", "count"),
    avg_rating=("rating", "mean"),
).round(2)

print("클러스터 → 책임 주체 → 개선 방향:")
for cluster, (owner, action) in category_map.items():
    if cluster in cluster_stats.index:
        n = cluster_stats.loc[cluster, "sentences"]
        r = cluster_stats.loc[cluster, "avg_rating"]
        print(f"  [{cluster}] {n}문장 ★{r} → [{owner}] {action}")

# 책임 주체별 불만 볼륨
owner_stats = {}
for cluster, (owner, action) in category_map.items():
    if cluster in cluster_stats.index:
        n = cluster_stats.loc[cluster, "sentences"]
        if owner not in owner_stats:
            owner_stats[owner] = 0
        owner_stats[owner] += n

print("\n책임 주체별 불만 문장 수:")
for owner, n in sorted(owner_stats.items(), key=lambda x: -x[1]):
    print(f"  {owner}: {n}문장")

# ======================================================================
print("\n" + "=" * 65)
print("[인사이트 E] 추세 vs 일시적 — Mann-Kendall 결과 해석")
print("=" * 65)

mk_df = pd.read_csv(OUT / "03_mann_kendall_trend.csv", encoding="utf-8-sig")
sig_mk = mk_df[mk_df["p"] < 0.1].sort_values("p")
print("추세가 있는 클러스터 (p<0.1):")
for _, row in sig_mk.iterrows():
    direction_kr = "장기 증가" if row["trend"] == "증가" else "장기 감소"
    print(f"  [{row['cluster']}] {direction_kr} 추세 (z={row['z']:.2f}, p={row['p']:.3f})")

flat_clusters = mk_df[mk_df["p"] >= 0.1]["cluster"].tolist()
print(f"\n추세 없음 (일시적/평탄) — {len(flat_clusters)}개:")
for c in flat_clusters:
    print(f"  {c}")

print("\n[해석] 추세 클러스터 = 시간이 지날수록 불만이 누적되는 구조적 문제")
print("       평탄 클러스터 = 출시 이벤트 기반 일시적 급증 후 안정")

print("\n모든 분석 완료.")
