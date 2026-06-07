"""
run_all.py — 전체 통계 분석 파이프라인 재실행
사용: python stats_analysis/run_all.py
"""
import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable
SCRIPTS = [
    "stats_analysis/00_build_dataset.py",
    "stats_analysis/01_cluster_stat_tests.py",
    "stats_analysis/02_brand_product_chi2.py",
    "stats_analysis/03_timeseries_spike.py",
    "stats_analysis/04_cohort_crossanalysis.py",
    "stats_analysis/05_domain_analysis.py",
    "stats_analysis/06_keyword_network.py",
    "stats_analysis/07_insight_drilldown.py",
    "stats_analysis/08_final_visualizations.py",
    "stats_analysis/09_advanced_insights.py",
    "stats_analysis/10_advanced_visualizations.py",
]

BASE = Path(__file__).parent.parent
for script in SCRIPTS:
    print(f"\n{'='*50}\n>>> {script}\n{'='*50}")
    result = subprocess.run(
        [PYTHON, str(BASE / script)],
        cwd=str(BASE),
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
    )
    if result.returncode != 0:
        print(f"[오류] {script} 실패 (exit {result.returncode})")
        sys.exit(result.returncode)

print("\n\n=== 전체 파이프라인 완료 ===")
print("출력물: stats_analysis/data/ (CSV), stats_analysis/output/ (PNG)")
