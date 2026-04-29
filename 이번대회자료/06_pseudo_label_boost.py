"""
V5 완료 후 실행: 의사 라벨링(Pseudo-labeling) + DART 모드 부스트
v5 결과 기반으로 고신뢰 테스트 데이터를 훈련 보강 → AUC 추가 향상
"""
import os, warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import lightgbm as lgb

warnings.filterwarnings("ignore")

TARGET  = "임신 성공 여부"
ID_COL  = "ID"
N_FOLDS = 5
SEEDS   = [42, 2024, 777, 1234, 31337]
SEED    = 42

# v5 결과 파일 찾기
import glob
v5_oof_files  = sorted(glob.glob("oof_v5_auc_*.npy"))
v5_test_files = sorted(glob.glob("test_v5_auc_*.npy"))

if not v5_oof_files:
    print("v5 결과 파일 없음. v5 파이프라인 먼저 실행하세요.")
    exit(1)

V5_OOF_FILE  = v5_oof_files[-1]
V5_TEST_FILE = v5_test_files[-1]
V5_AUC = float(V5_OOF_FILE.replace("oof_v5_auc_", "").replace(".npy", ""))

print("=" * 65)
print("V6 Pseudo-labeling + DART Boost")
print(f"v5 기준: {V5_AUC:.5f}")
print("=" * 65)

# ─────── 전처리 (v4/v5 방식) ─────────────────────────────────────
print("\n[1] 데이터 로드 및 전처리...")
train_raw = pd.read_csv("./data/train.csv")
test_raw  = pd.read_csv("./data/test.csv")

COUNT_COLS = [
    "총 시술 횟수", "클리닉 내 총 시술 횟수",
    "IVF 시술 횟수", "DI 시술 횟수",
    "총 임신 횟수", "IVF 임신 횟수", "DI 임신 횟수",
    "총 출산 횟수", "IVF 출산 횟수", "DI 출산 횟수",
]

def preprocess(df):
    df = df.copy()
    for col in COUNT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    df["ever_delivered"] = (df["총 출산 횟수"].fillna(0) > 0).astype(int)
    df["is_FET"]         = (df["해동된 배아 수"].fillna(0) > 0).astype(int)
    return df

def encode_categories(train, test):
    train, test = train.copy(), test.copy()
    obj_cols = [c for c in train.select_dtypes("object").columns if c not in [ID_COL, TARGET]]
    for col in obj_cols:
        train[col] = train[col].astype("category")
        if col in test.columns:
            test[col] = test[col].astype("category")
        all_cats = sorted(
            set(train[col].cat.categories) |
            (set(test[col].cat.categories) if col in test.columns else set()),
            key=str
        )
        train[col] = train[col].cat.set_categories(all_cats)
        if col in test.columns:
            test[col] = test[col].cat.set_categories(all_cats)
    return train, test

train = preprocess(train_raw)
test  = preprocess(test_raw)
train, test = encode_categories(train, test)

X      = train.drop(columns=[ID_COL, TARGET])
y      = train[TARGET].copy()
X_test = test.drop(columns=[ID_COL])

# ─────── 의사 라벨링 ──────────────────────────────────────────────
print("\n[2] 의사 라벨링 설정...")
test_preds = np.load(V5_TEST_FILE)

# 고신뢰 임계값 설정 (보수적으로)
THRESH_POS = 0.80   # 임신 성공 확신 (top ~1%)
THRESH_NEG = 0.08   # 임신 실패 확신 (bottom ~1%)

pseudo_pos_mask = test_preds >= THRESH_POS
pseudo_neg_mask = test_preds <= THRESH_NEG

print(f"  고신뢰 양성(≥{THRESH_POS}): {pseudo_pos_mask.sum():,}건")
print(f"  고신뢰 음성(≤{THRESH_NEG}): {pseudo_neg_mask.sum():,}건")
print(f"  합계: {(pseudo_pos_mask | pseudo_neg_mask).sum():,}건")

# 의사 라벨 데이터프레임 생성
X_pseudo_pos = X_test[pseudo_pos_mask].copy()
X_pseudo_neg = X_test[pseudo_neg_mask].copy()

y_pseudo_pos = pd.Series([1] * pseudo_pos_mask.sum(), index=X_pseudo_pos.index)
y_pseudo_neg = pd.Series([0] * pseudo_neg_mask.sum(), index=X_pseudo_neg.index)

X_augmented = pd.concat([X, X_pseudo_pos, X_pseudo_neg], ignore_index=True)
y_augmented = pd.concat([y, y_pseudo_pos, y_pseudo_neg], ignore_index=True)

print(f"  증강 후 학습 데이터: {X_augmented.shape[0]:,}건 (원본: {len(X):,})")
print(f"  증강 후 타겟 비율: {y_augmented.mean():.4f}")

# ─────── Pseudo-label only (DART 제거: early stopping 없어 너무 느림) ──

# ─────── 의사 라벨 LGBM ──────────────────────────────────────────
print("\n[4] 의사 라벨 LGBM...")

LGBM_BASE = dict(
    n_estimators     = 2000,
    learning_rate    = 0.05,
    num_leaves       = 63,
    min_child_samples= 20,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_alpha        = 0.1,
    reg_lambda       = 0.1,
    verbose          = -1,
    n_jobs           = -1,
)

# 의사 라벨 모델은 원본 검증 데이터에서만 평가
pl_oof  = np.zeros(len(X))  # 원본 훈련셋 OOF
pl_test = np.zeros(len(X_test))

for seed in SEEDS:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    this_oof  = np.zeros(len(X))
    this_test = np.zeros(len(X_test))

    for fold, (tr_i, va_i) in enumerate(skf.split(X, y), 1):
        # 원본 fold 인덱스 (의사 라벨 추가)
        n_orig = len(X)
        n_aug  = len(X_augmented) - n_orig

        # 훈련: 원본 tr_i + 전체 의사 라벨
        X_aug_fold = pd.concat([
            X.iloc[tr_i],
            X_augmented.iloc[n_orig:],  # 의사 라벨 전부
        ], ignore_index=True)
        y_aug_fold = pd.concat([
            y.iloc[tr_i],
            y_augmented.iloc[n_orig:],
        ], ignore_index=True)

        m = LGBMClassifier(**{**LGBM_BASE, "random_state": seed})
        m.fit(
            X_aug_fold, y_aug_fold,
            eval_set=[(X.iloc[va_i], y.iloc[va_i])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        this_oof[va_i] = m.predict_proba(X.iloc[va_i])[:, 1]
        this_test     += m.predict_proba(X_test)[:, 1] / N_FOLDS

    s_auc = roc_auc_score(y, this_oof)
    print(f"  PL Seed {seed:>5}: {s_auc:.5f}")
    pl_oof  += this_oof  / len(SEEDS)
    pl_test += this_test / len(SEEDS)

pl_auc = roc_auc_score(y, pl_oof)
print(f"  의사라벨 LGBM OOF AUC: {pl_auc:.5f}")

# ─────── 앙상블 ───────────────────────────────────────────────────
print("\n[5] 앙상블...")
from scipy.stats import rankdata

def rank_norm(arr):
    return rankdata(arr) / len(arr)

v5_oof  = np.load(V5_OOF_FILE)
v5_test = np.load(V5_TEST_FILE)

ensemble_oof  = 0.6 * rank_norm(v5_oof)  + 0.4 * rank_norm(pl_oof)
ensemble_test = 0.6 * rank_norm(v5_test) + 0.4 * rank_norm(pl_test)
ensemble_auc  = roc_auc_score(y, ensemble_oof)

print(f"  v5 OOF AUC:          {V5_AUC:.5f}")
print(f"  의사라벨 OOF AUC:     {pl_auc:.5f}")
print(f"  앙상블 OOF AUC:       {ensemble_auc:.5f}")
print(f"  v5 대비:              {ensemble_auc - V5_AUC:+.5f}")

# ─────── 저장 ──────────────────────────────────────────────────────
best_auc = max(V5_AUC, pl_auc, ensemble_auc)
if ensemble_auc == best_auc:
    final_preds = ensemble_test
elif pl_auc == best_auc:
    final_preds = pl_test
else:
    final_preds = v5_test

sub = pd.read_csv("./data/sample_submission.csv")
sub["probability"] = final_preds
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
filename  = f"submission_{timestamp}_v6_oof{best_auc:.5f}.csv"
sub.to_csv(filename, index=False)
print(f"\n  저장: {filename}")

np.save(f"oof_v6_auc_{ensemble_auc:.5f}.npy", ensemble_oof)
np.save(f"test_v6_auc_{ensemble_auc:.5f}.npy", final_preds)
