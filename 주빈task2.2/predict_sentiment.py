"""
review_for_analysis.json에 대해 CE 모델로 sentiment inference 수행.
출력: task2.2/review_with_sentiment.csv
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── 경로 설정 ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH  = ROOT / "si_dataset" / "review_for_analysis.json"
MODEL_PATH = ROOT / "outputs" / "final_sentiment_model_ce_seed100"
OUT_PATH   = Path(__file__).resolve().parent / "review_with_sentiment.csv"

MAX_LENGTH = 512
BATCH_SIZE = 16

ID2LABEL = {0: "negative", 1: "weak_negative", 2: "weak_positive", 3: "positive"}

# ── 데이터 로드 ────────────────────────────────────────────
def load_df() -> pd.DataFrame:
    with open(DATA_PATH, encoding="utf-8") as f:
        raw = re.sub(r":\s*NaN\b", ": null", f.read())
    df = pd.DataFrame(json.loads(raw))
    df["text"] = (df["title"].fillna("") + " " + df["content"].fillna("")).str.strip()
    return df


# ── Dataset ────────────────────────────────────────────────
class ReviewDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        )

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


# ── Inference ──────────────────────────────────────────────
def predict(df: pd.DataFrame) -> pd.DataFrame:
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()

    dataset = ReviewDataset(df["text"].tolist(), tokenizer)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

    probs_arr = np.vstack(all_probs)          # (N, 4)
    pred_labels = probs_arr.argmax(axis=1)

    df = df.copy()
    df["pred_label"]        = pred_labels
    df["pred_label_string"] = [ID2LABEL[l] for l in pred_labels]
    df["is_negative"]       = pred_labels <= 1   # 0,1 → 불만  /  2,3 → 만족

    for i, name in ID2LABEL.items():
        df[f"prob_{name}"] = probs_arr[:, i]

    return df


# ── Main ───────────────────────────────────────────────────
def main():
    print("데이터 로드 중...")
    df = load_df()
    print(f"총 리뷰: {len(df):,}개")

    print("Inference 시작...")
    df = predict(df)

    # 결과 요약
    print("\n[예측 라벨 분포]")
    print(df["pred_label_string"].value_counts())
    print(f"\n불만 리뷰(is_negative=True): {df['is_negative'].sum():,}개 "
          f"({df['is_negative'].mean()*100:.1f}%)")
    print(f"만족 리뷰(is_negative=False): {(~df['is_negative']).sum():,}개 "
          f"({(~df['is_negative']).mean()*100:.1f}%)")

    # rating 기반 라벨과 비교
    if "rating" in df.columns:
        rating_neg = (df["rating"] <= 2).mean() * 100
        model_neg  = df["is_negative"].mean() * 100
        print(f"\n[rating 기반 저점(1-2점) 비율]: {rating_neg:.1f}%")
        print(f"[모델 기반 불만 비율]:           {model_neg:.1f}%")

    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {OUT_PATH}")


if __name__ == "__main__":
    main()
