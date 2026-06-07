"""
00_build_dataset.py
원본 데이터 통합 + 파생 변수 생성
- review_id 기반 병합 (reviewId 키 사용)
- email domain 추출, 날짜 변환, 코호트 정의
"""
import pandas as pd
import numpy as np
import json
import re
from pathlib import Path

BASE = Path(__file__).parent.parent
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

# --- 1. 리뷰 원본 로드 (utf-8-sig, 2757행) ---
reviews = pd.read_csv(
    BASE / "주빈task2.2/review_with_sentiment.csv",
    encoding="utf-8-sig",
    low_memory=False,
)
print(f"[reviews] {reviews.shape}")

# --- 2. 이메일 도메인 추출 ---
def extract_domain(member_str):
    try:
        s = str(member_str).replace("'", '"')
        obj = json.loads(s)
        email = obj.get("email", "")
        if "@" in str(email):
            return str(email).split("@")[1].replace("*", "").strip()
    except Exception:
        pass
    try:
        m = re.search(r'"email":\s*"([^"]+)"', str(member_str))
        if m:
            email = m.group(1)
            if "@" in email:
                return email.split("@")[1].replace("*", "").strip()
    except Exception:
        pass
    return "unknown"

reviews["email_domain"] = reviews["member"].apply(extract_domain)

def domain_group(d):
    d = str(d).lower()
    if "naver" in d: return "네이버"
    if "gmail" in d or "google" in d: return "Gmail"
    if "kakao" in d or "daum" in d or "hanmail" in d: return "카카오"
    if "nate" in d: return "네이트"
    if d == "unknown": return "unknown"
    return "기타"

reviews["domain_group"] = reviews["email_domain"].apply(domain_group)
print("Domain group dist:", reviews["domain_group"].value_counts().to_dict())

# --- 3. 날짜 변환 ---
reviews["review_date"] = pd.to_datetime(reviews["reviewAt"], unit="ms")
reviews["review_month"] = reviews["review_date"].dt.to_period("M").astype(str)
reviews["review_date_only"] = reviews["review_date"].dt.date

reviews["has_photo"] = reviews["attachments"].apply(
    lambda x: len(str(x)) > 10 if pd.notna(x) else False
)
reviews["content_len"] = reviews["content"].fillna("").apply(len)
reviews["review_word_count"] = reviews["content"].fillna("").apply(lambda x: len(str(x).split()))

# --- 4. 코호트 (제품별 첫 리뷰 날짜 기준) ---
first_review = (
    reviews.groupby("product_name")["review_date"].min().rename("first_review_date")
)
reviews = reviews.merge(first_review, on="product_name", how="left")
reviews["days_since_launch"] = (
    reviews["review_date"] - reviews["first_review_date"]
).dt.days

def cohort_label(d):
    if pd.isna(d): return "unknown"
    d = int(d)
    if d <= 30: return "D+0~30 (얼리어답터)"
    if d <= 90: return "D+31~90 (초기다수)"
    return "D+91+ (후기다수)"

reviews["cohort"] = reviews["days_since_launch"].apply(cohort_label)
print("Cohort dist:", reviews["cohort"].value_counts().to_dict())

# --- 5. 브랜드 추가 ---
reviews["brand"] = reviews["product_name"].apply(
    lambda x: "Apple" if "iphone" in str(x).lower() else "Samsung"
)

# product_label 정리
product_label_map = {
    "iphone_17": "iPhone 17",
    "iphone_17_pro": "iPhone 17 Pro",
    "iphone_17_pro_max": "iPhone 17 Pro Max",
    "galaxy_s26": "Galaxy S26",
    "galaxy_s26_ultra": "Galaxy S26 Ultra",
    "galaxy_z_fold7": "Galaxy Z Fold7",
    "galaxy_z_flip7": "Galaxy Z Flip7",
}
reviews["product_label"] = reviews["product_name"].map(product_label_map).fillna(reviews["product_name"])

# --- 6. 클러스터 데이터 로드 & 병합 ---
cc = pd.read_csv(
    BASE / "output/bertopic_results/cc_clustered.csv",
    encoding="utf-8-sig",
)
print(f"[cc_clustered] {cc.shape}, unique review_ids: {cc['review_id'].nunique()}")

# 메타데이터: reviewId 기준으로 cc에 붙이기
review_meta = reviews[[
    "reviewId", "review_date", "review_month",
    "email_domain", "domain_group",
    "helpfulTrueCount", "helpfulFalseCount", "helpfulCount",
    "has_photo", "content_len", "review_word_count",
    "days_since_launch", "cohort", "brand", "product_label",
    "content",  # 원문 텍스트 (드릴다운 분석용)
]].copy()
review_meta["reviewId"] = review_meta["reviewId"].astype(str)

cc["review_id"] = cc["review_id"].astype(str)

master = cc.merge(
    review_meta.rename(columns={"reviewId": "review_id"}),
    on="review_id",
    how="left",
    suffixes=("", "_meta"),
)
# 접미사 정리: _meta 컬럼 제거 (cc의 brand/product_label 우선)
meta_dupes = [c for c in master.columns if c.endswith("_meta")]
master = master.drop(columns=meta_dupes)
print(f"[master] {master.shape}")
print("Columns:", list(master.columns))
print("domain_group null:", master["domain_group"].isna().sum())
print("cohort dist:", master["cohort"].value_counts().to_dict())
print("review_month range:", master["review_month"].dropna().min(), "~", master["review_month"].dropna().max())

# 클러스터 표시명 (발표용)
CLUSTER_DISPLAY = {
    "[0] 폰을_다시_폰은": "이전폰 비교·교체",
    "[1] 박스_없이_뽁뽁이": "포장·박스 불만",
    "[2] 케이스_케이스를_손으로": "케이스·그립감",
    "[3] 프로_쓰던_기존": "이전 프로 비교",
    "[4] 카메라_사진_카메라가": "카메라·사진",
    "[5] 배송_고민은_바로": "배송 속도·방법",
    "[7] 디자인_크게_변화가": "디자인 변화",
    "[8] 배터리_충전_발열": "배터리·충전·발열",
    "[9] 가격이_가격_부담이": "가격·가성비",
    "[10] 화면_프라이버시_화면이": "화면·프라이버시",
    "[11] 영상_게임_여러": "영상·게임 성능",
}
master["cluster_display"] = master["cluster_name"].map(CLUSTER_DISPLAY).fillna("기타")

master.to_csv(OUT / "master_dataset.csv", index=False, encoding="utf-8-sig")
print("Saved master_dataset.csv")
