"""
07_insight_drilldown.py
통계 검정에서 유의미하게 나온 발견들을 raw text로 드릴다운
1. 2026-03 카메라·화면 스파이크 → 실제 문장에서 무슨 단어가 새로 등장했나?
2. 포장 불만 ★3.29 → 가장 격렬한 불만 예시 + 특정 패턴 분석
3. Z Fold7 이전폰 비교 35.8% → 무엇과 비교하나? (Galaxy S/Note/iPhone?)
4. Gmail 유저 AS 불만 집중 → 실제 텍스트 확인
5. 얼리어답터 helpfulness 우세 리뷰 → 어떤 패턴이 공감 유발?
"""
import pandas as pd
import numpy as np
import re
from collections import Counter
from pathlib import Path

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "output"

master = pd.read_csv(DATA / "master_dataset.csv", encoding="utf-8-sig")
main = master[master["cluster_display"] != "기타"].copy()

def extract_nouns(text):
    STOPWORDS = {
        "이게", "이건", "이거", "그거", "그것", "그게", "것", "것도", "것은", "것이",
        "있는", "있어", "없는", "없어", "있고", "없고", "않은", "않아",
        "너무", "정말", "진짜", "조금", "좀", "더", "매우", "아주", "완전",
        "하지만", "그런데", "근데", "그래서", "그리고", "또한",
        "쿠팡", "아이폰", "갤럭시", "삼성", "애플", "폰", "제품", "스마트폰",
        "사용", "이용", "구매", "구입", "생각", "느낌", "경우", "부분", "상태",
        "정도", "기능", "성능", "품질", "가격", "리뷰", "후기",
    }
    _PARTICLES = sorted(
        ["에서", "에게", "이라", "이며", "이고", "이다", "이야", "으로", "로서",
         "이랑", "이나", "라고", "라는", "라면", "부터", "까지", "마저", "조차",
         "에서는", "에서도", "을", "를", "이", "가", "은", "는", "도", "만",
         "과", "와", "에", "의", "로", "에게", "서", "요", "야", "고", "지"],
        key=len, reverse=True,
    )
    words = str(text).split()
    nouns = []
    for word in words:
        w = word.strip(".,!?~ㅠㅋㅎ…")
        for p in _PARTICLES:
            if w.endswith(p) and len(w) > len(p) + 1:
                w = w[:-len(p)]
                break
        tokens = re.findall(r'[가-힣]{2,}', w)
        for tok in tokens:
            if tok not in STOPWORDS:
                nouns.append(tok)
    return nouns

def top_words(texts, n=20, min_count=2):
    words = []
    for t in texts:
        words.extend(extract_nouns(t))
    return Counter(words).most_common(n)

print("=" * 60)
print("[인사이트 1] 2026-03 카메라·화면 스파이크 — 무슨 말이 쏟아졌나?")
print("=" * 60)

for cluster_name in ["카메라·사진", "화면·프라이버시"]:
    sents_mar = main[(main["cluster_display"] == cluster_name) & (main["review_month"] == "2026-03")]["sentence"].tolist()
    sents_base = main[(main["cluster_display"] == cluster_name) & (main["review_month"] < "2026-03")]["sentence"].tolist()
    print(f"\n[{cluster_name}] 2026-03: {len(sents_mar)}문장 vs 이전: {len(sents_base)}문장")

    # 새로 등장한 키워드: 3월에 상대적으로 더 높은 빈도
    mar_cnt = Counter()
    for t in sents_mar:
        mar_cnt.update(extract_nouns(t))
    base_cnt = Counter()
    for t in sents_base:
        base_cnt.update(extract_nouns(t))

    mar_total = max(len(sents_mar), 1)
    base_total = max(len(sents_base), 1)

    # log ratio: 3월에 상대적으로 많은 단어
    log_ratios = {}
    all_words = set(list(mar_cnt.keys()) + list(base_cnt.keys()))
    for w in all_words:
        m_freq = (mar_cnt[w] + 1) / mar_total
        b_freq = (base_cnt[w] + 1) / base_total
        if mar_cnt[w] >= 3:
            log_ratios[w] = np.log(m_freq / b_freq)

    emergent = sorted(log_ratios.items(), key=lambda x: -x[1])[:15]
    print(f"  2026-03 급증 키워드 (log-ratio 상위):")
    for w, lr in emergent:
        print(f"    {w}: {mar_cnt[w]}회 (log-ratio={lr:.2f})")

    print(f"\n  대표 문장 샘플 (2026-03):")
    for sent in sents_mar[:5]:
        print(f"    · {sent[:80]}")

print("\n" + "=" * 60)
print("[인사이트 2] 포장·박스 불만 ★3.29 — 가장 격렬한 불만은?")
print("=" * 60)
packaging = main[main["cluster_display"] == "포장·박스 불만"].copy()
print(f"총 {len(packaging)}문장")
# 별점 1점 리뷰에서
pkg_1star = packaging[packaging["rating"] == 1]
print(f"별점 1점 리뷰의 포장 불만: {len(pkg_1star)}문장")
print("\n별점 1점 포장 불만 Top 키워드:")
print(top_words(pkg_1star["sentence"].tolist(), n=15))

print("\n별점 1점 포장 불만 대표 문장 (helpfulness 순):")
top_help = pkg_1star.dropna(subset=["helpfulTrueCount"]).sort_values("helpfulTrueCount", ascending=False).head(5)
for _, row in top_help.iterrows():
    print(f"  [★1, 공감{int(row['helpfulTrueCount'])}] {str(row['sentence'])[:100]}")

# 포장 패턴 분석
print("\n포장 불만 주요 패턴:")
pattern_keywords = {
    "박스 파손": ["파손", "찌그러", "눌린", "구겨", "깨진", "망가"],
    "뽁뽁이 없음": ["뽁뽁이", "완충", "포장재", "없이", "빈"],
    "중고품 의심": ["중고", "사용한", "개봉흔", "개봉", "스크래치"],
    "사은품 누락": ["사은품", "누락", "없음", "빠져"],
    "재포장 의심": ["재포장", "테이프", "개봉"],
}
for pattern, kws in pattern_keywords.items():
    cnt = sum(
        1 for sent in packaging["sentence"].fillna("")
        if any(kw in sent for kw in kws)
    )
    print(f"  {pattern}: {cnt}건 ({cnt/len(packaging)*100:.1f}%)")

print("\n" + "=" * 60)
print("[인사이트 3] Z Fold7 이전폰 비교 35.8% — 무엇과 비교하나?")
print("=" * 60)
fold7_compare = main[
    (main["product_label"] == "Galaxy Z Fold7") &
    (main["cluster_display"] == "이전폰 비교·교체")
].copy()
print(f"Galaxy Z Fold7 '이전폰 비교' 문장: {len(fold7_compare)}")
print("\n비교 대상 키워드:")
print(top_words(fold7_compare["sentence"].tolist(), n=20))

# 특정 모델 언급 패턴
compare_targets = {
    "갤럭시 S시리즈": ["갤스", "갤럭시s", "s25", "s24", "s23", "s22", "갤럭시 s"],
    "갤럭시 노트": ["노트", "note"],
    "아이폰으로 이탈": ["아이폰", "애플"],
    "폴드 이전 모델": ["폴드5", "폴드6", "fold5", "fold6", "이전 폴드"],
    "플립": ["플립", "flip", "폴더블"],
}
for target, kws in compare_targets.items():
    cnt = sum(
        1 for sent in fold7_compare["sentence"].fillna("").str.lower()
        if any(kw.lower() in sent for kw in kws)
    )
    if cnt > 0:
        print(f"  {target}: {cnt}건 ({cnt/len(fold7_compare)*100:.1f}%)")

print("\n대표 문장:")
for _, row in fold7_compare.head(5).iterrows():
    print(f"  · {str(row['sentence'])[:90]}")

print("\n" + "=" * 60)
print("[인사이트 4] Gmail 유저 AS·품질 불만 집중 — 실제 텍스트 확인")
print("=" * 60)
gmail_main = main[main["domain_group"] == "Gmail"].copy()
naver_main = main[main["domain_group"] == "네이버"].copy()

print(f"Gmail 문장: {len(gmail_main)}, 네이버 문장: {len(naver_main)}")
print("\nGmail 클러스터 분포:")
print((gmail_main["cluster_display"].value_counts() / len(gmail_main) * 100).round(1))

# AS·품질 관련 클러스터에서 Gmail vs 네이버 비교
as_clusters = ["배터리·충전·발열", "화면·프라이버시", "영상·게임 성능"]
for cl in as_clusters:
    g_pct = len(gmail_main[gmail_main["cluster_display"] == cl]) / len(gmail_main) * 100
    n_pct = len(naver_main[naver_main["cluster_display"] == cl]) / len(naver_main) * 100
    print(f"  {cl}: Gmail={g_pct:.1f}%, 네이버={n_pct:.1f}%")

print("\nGmail '배터리' 불만 대표 문장:")
gmail_battery = gmail_main[gmail_main["cluster_display"] == "배터리·충전·발열"]
for _, row in gmail_battery.head(4).iterrows():
    print(f"  · {str(row['sentence'])[:90]}")

print("\n" + "=" * 60)
print("[인사이트 5] 공감 폭발 리뷰 — 어떤 패턴이 높은 helpfulness를 만드나?")
print("=" * 60)
rev_level = master.drop_duplicates(subset=["review_id"]).dropna(subset=["helpfulTrueCount"]).copy()
high_help = rev_level[rev_level["helpfulTrueCount"] >= 10].sort_values("helpfulTrueCount", ascending=False)
print(f"helpfulTrueCount >= 10인 리뷰: {len(high_help)}건")
print(f"  평균 길이: {high_help['content_len'].mean():.0f}자")
print(f"  사진 있는 비율: {high_help['has_photo'].mean()*100:.1f}%")
print(f"  별점 분포: {high_help['rating'].value_counts().sort_index().to_dict()}")
print(f"  브랜드: {high_help['brand'].value_counts().to_dict()}")
print(f"  코호트: {high_help['cohort'].value_counts().to_dict()}")

# 가장 공감받은 리뷰의 불만 클러스터
top_help_rev = high_help.head(20)[["review_id", "helpfulTrueCount", "rating", "content_len", "has_photo", "cohort", "brand"]]
print("\nTop20 공감 리뷰:")
print(top_help_rev.to_string())

# 이 리뷰들의 클러스터 분포
top_ids = set(high_help.head(30)["review_id"].astype(str))
top_clusters = main[main["review_id"].astype(str).isin(top_ids)]["cluster_display"].value_counts()
print("\nTop30 공감 리뷰의 불만 클러스터:")
print((top_clusters / top_clusters.sum() * 100).round(1))

print("\n저장 완료: 07_insight_drilldown.txt (콘솔 출력)")
