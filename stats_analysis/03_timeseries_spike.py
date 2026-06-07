"""
03_timeseries_spike.py
시계열 이상치(스파이크) 통계적 탐지
- 월별 클러스터 문장 수 Z-score 계산
- Poisson 검정: 2026-03 스파이크 유의성
- Mann-Kendall 추세 검정 (pymannkendall 없으면 간이 구현)
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "output"

master = pd.read_csv(DATA / "master_dataset.csv", encoding="utf-8-sig")
main = master[master["cluster_display"] != "기타"].dropna(subset=["review_month"]).copy()

# ---- 1. 월별 클러스터 문장 수 피벗 ----
ts = main.groupby(["review_month", "cluster_display"]).size().unstack(fill_value=0)
ts = ts.sort_index()
print("=== 월별 클러스터 문장 수 ===")
print(ts.to_string())

# ---- 2. Z-score 계산 (각 클러스터 내에서) ----
ts_zscore = (ts - ts.mean()) / ts.std()
print("\n=== Z-score (|z|>2 = 이상치) ===")
print(ts_zscore.round(2).to_string())

# 이상치 탐지 (|z| > 2)
anomalies = []
for month in ts_zscore.index:
    for cluster in ts_zscore.columns:
        z = ts_zscore.loc[month, cluster]
        if abs(z) > 2.0:
            anomalies.append({
                "month": month,
                "cluster": cluster,
                "count": ts.loc[month, cluster],
                "z_score": z,
                "direction": "급증" if z > 0 else "급감"
            })

df_anom = pd.DataFrame(anomalies).sort_values("z_score", ascending=False)
print(f"\n=== 이상치 탐지 결과 ({len(df_anom)}건) ===")
print(df_anom.to_string())
df_anom.to_csv(OUT / "03_timeseries_anomalies.csv", index=False, encoding="utf-8-sig")

# ---- 3. Poisson 검정: 2026-03 스파이크 ----
# 귀무가설: 2026-03 count가 이전 달들의 Poisson 평균과 같다
# 2025-07 ~ 2026-02를 기준으로 평균 계산, 2026-03을 검정
baseline_months = [m for m in ts.index if m < "2026-03"]
test_month = "2026-03"

print(f"\n=== Poisson 검정: {test_month} 스파이크 ===")
print(f"기준 기간: {baseline_months[0]} ~ {baseline_months[-1]} ({len(baseline_months)}개월)")

poisson_results = []
for cluster in ts.columns:
    baseline = ts.loc[baseline_months, cluster].values
    baseline_mean = baseline.mean()
    observed = ts.loc[test_month, cluster]
    # 기대 기간 1개월
    # scipy.stats.poisson.sf(observed-1, baseline_mean) = P(X >= observed)
    if baseline_mean > 0:
        p_val = stats.poisson.sf(observed - 1, baseline_mean)
        ratio = observed / baseline_mean
        poisson_results.append({
            "cluster": cluster,
            "baseline_mean": round(baseline_mean, 1),
            "observed_2026_03": observed,
            "ratio": round(ratio, 2),
            "p_value": p_val,
            "significant": p_val < 0.05
        })

df_poisson = pd.DataFrame(poisson_results).sort_values("p_value")
print(df_poisson.to_string())
df_poisson.to_csv(OUT / "03_poisson_spike_test.csv", index=False, encoding="utf-8-sig")

sig_spikes = df_poisson[df_poisson["significant"]]
print(f"\n유의미한 스파이크: {len(sig_spikes)}개")
for _, row in sig_spikes.iterrows():
    print(f"  {row['cluster']}: {row['baseline_mean']:.1f} -> {row['observed_2026_03']} ({row['ratio']:.1f}x), p={row['p_value']:.2e}")

# ---- 4. 추세 분석 (Mann-Kendall 간이 구현) ----
def mann_kendall(x):
    """S 통계량 기반 MK 검정 (scipy 없을 때 사용)"""
    n = len(x)
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = x[j] - x[i]
            if diff > 0: s += 1
            elif diff < 0: s -= 1
    var_s = n * (n - 1) * (2 * n + 5) / 18
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    trend = "증가" if z > 0 else "감소"
    return {"S": s, "z": z, "p": p, "trend": trend}

print("\n=== Mann-Kendall 추세 검정 (2025-07 ~ 2026-04) ===")
mk_results = []
for cluster in ts.columns:
    series = ts[cluster].values
    mk = mann_kendall(series)
    mk["cluster"] = cluster
    mk_results.append(mk)
    if mk["p"] < 0.1:
        print(f"  [{cluster}] {mk['trend']} 추세: z={mk['z']:.2f}, p={mk['p']:.3f}")

df_mk = pd.DataFrame(mk_results)
df_mk.to_csv(OUT / "03_mann_kendall_trend.csv", index=False, encoding="utf-8-sig")

ts.to_csv(OUT / "03_timeseries_counts.csv", encoding="utf-8-sig")
ts_zscore.to_csv(OUT / "03_timeseries_zscore.csv", encoding="utf-8-sig")
print("\n저장 완료: 03_*.csv")
