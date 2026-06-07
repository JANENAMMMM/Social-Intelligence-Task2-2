"""
리뷰를 kss로 문장 단위 분리 후 ensemble.pt로 sentiment inference.
출력: task2.2/sentence_sentiments.csv
"""

import os
import re
import warnings
warnings.filterwarnings("ignore")

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import kss
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

N_WORKERS = max(1, os.cpu_count() - 2)  # 코어 2개는 OS용으로 남김

HERE          = Path(__file__).resolve().parent
IN_PATH       = HERE / "review_with_sentiment.csv"
ENSEMBLE_PATH = HERE / "ensemble.pt"
OUT_PATH      = HERE / "sentence_sentiments.csv"

BATCH_SIZE = 64
MIN_CHARS  = 15    # 최소 15자
MAX_CHARS  = 300   # 너무 긴 문장 (단락 수준) 제외
MIN_KO_RATIO = 0.3 # 한글 비율 30% 미만 제외


# ── 문장 품질 필터 ─────────────────────────────────────────
def is_valid(text: str) -> bool:
    text = text.strip()

    # 길이 필터
    if len(text) < MIN_CHARS or len(text) > MAX_CHARS:
        return False

    # 한글 비율 필터
    ko_chars = len(re.findall(r'[가-힣]', text))
    if ko_chars / len(text) < MIN_KO_RATIO:
        return False

    # 반복 문자 필터 (ㅡㅡㅡ, ----, ㅋㅋㅋ 등)
    clean = re.sub(r'[\s\W_]', '', text)
    if len(clean) >= 4:
        most_common = max(set(clean), key=clean.count)
        if clean.count(most_common) / len(clean) >= 0.7:
            return False

    return True


# ── kss 문장 분리 ──────────────────────────────────────────
def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []

    try:
        sentences = kss.split_sentences(text, backend="pecab")
    except Exception:
        # kss 실패시 줄바꿈 기준 fallback
        sentences = [s.strip() for s in text.split('\n') if s.strip()]

    return [s.strip() for s in sentences if is_valid(s.strip())]


# ── 병렬 처리용 worker ─────────────────────────────────────
def _process_row(args):
    review_id, text, product_name, rating, pred_label_review = args
    sentences = split_sentences(text)
    return [
        {
            "review_id":         review_id,
            "sentence_idx":      idx,
            "sentence":          sent,
            "product_name":      product_name,
            "rating":            rating,
            "pred_label_review": pred_label_review,
        }
        for idx, sent in enumerate(sentences)
    ]


# ── 문장 분리 + 인덱싱 (병렬) ──────────────────────────────
def build_sentence_df(df: pd.DataFrame) -> pd.DataFrame:
    args = [
        (
            row.get("reviewId") or row.get("review_id"),
            str(row.get("text", "")),
            row.get("product_name", ""),
            row.get("rating", None),
            row.get("pred_label", None),
        )
        for _, row in df.iterrows()
    ]

    print(f"병렬 문장 분리 시작 (workers={N_WORKERS})...")
    with Pool(N_WORKERS) as pool:
        results = pool.map(_process_row, args)

    rows = [row for result in results for row in result]
    return pd.DataFrame(rows)


# ── Dataset ────────────────────────────────────────────────
class SentenceDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


# ── Ensemble Inference ─────────────────────────────────────
def predict(sdf: pd.DataFrame) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ens          = torch.load(ENSEMBLE_PATH, map_location="cpu")
    backbone     = ens["backbone"]
    max_length   = ens["max_length"]
    weights      = ens["prob_weights"]
    id2label     = {int(k): v for k, v in ens["id2label"].items()}
    state_dicts  = ens["member_state_dicts"]
    member_names = ens["member_names"]

    print(f"backbone: {backbone}")
    print(f"members: {member_names}, weights: {[round(w,3) for w in weights]}")

    tokenizer = AutoTokenizer.from_pretrained(backbone)
    dataset   = SentenceDataset(sdf["sentence"].tolist(), tokenizer, max_length)
    loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    member_probs_list = []
    for i, (name, sd) in enumerate(zip(member_names, state_dicts)):
        print(f"  [{i+1}/{len(member_names)}] {name} inference...")
        model = AutoModelForSequenceClassification.from_pretrained(
            backbone, num_labels=len(id2label)
        )
        model.load_state_dict(sd)
        model.to(device)
        model.eval()

        all_probs = []
        with torch.no_grad():
            for batch in loader:
                batch  = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                probs  = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)

        member_probs_list.append(np.vstack(all_probs))
        del model
        torch.cuda.empty_cache()

    ensemble_probs = sum(w * p for w, p in zip(weights, member_probs_list))
    pred_labels    = ensemble_probs.argmax(axis=1)

    sdf = sdf.copy()
    sdf["pred_label"]        = pred_labels
    sdf["pred_label_string"] = [id2label[l] for l in pred_labels]
    sdf["is_negative"]       = pred_labels <= 1

    for i, name in id2label.items():
        sdf[f"prob_{name}"] = ensemble_probs[:, i]

    return sdf


# ── Main ───────────────────────────────────────────────────
def main():
    print("리뷰 로드...")
    df = pd.read_csv(IN_PATH, encoding="utf-8-sig")
    print(f"총 리뷰: {len(df):,}개")

    print("kss 문장 분리 중... (시간이 걸립니다)")
    sdf = build_sentence_df(df)
    print(f"총 문장: {len(sdf):,}개  (리뷰당 평균 {len(sdf)/len(df):.1f}문장)")
    print(f"문장 길이 분포: 평균 {sdf['sentence'].str.len().mean():.0f}자, "
          f"중앙값 {sdf['sentence'].str.len().median():.0f}자")

    print("\nEnsemble inference 시작...")
    sdf = predict(sdf)

    neg = sdf[sdf["is_negative"]]
    print(f"\n[부정 문장]: {len(neg):,}개 ({len(neg)/len(sdf)*100:.1f}%)")
    print(f"[부정 문장 포함 리뷰]: {neg['review_id'].nunique():,}개")

    print("\n[라벨 분포]")
    print(sdf["pred_label_string"].value_counts())

    print("\n[리뷰당 부정 문장 수]")
    print(neg.groupby("review_id").size().value_counts().sort_index().head(10))

    sdf.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {OUT_PATH}")


if __name__ == "__main__":
    main()
