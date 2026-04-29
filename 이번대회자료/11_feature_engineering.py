"""
타겟 피처 엔지니어링 — 임상 의미 기반 비율 피처 선별
v8 기준선: 0.74079
전략: 각 피처를 개별 검증 후 유의한 것만 추가
"""
import glob, warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import lightgbm as lgb
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

TARGET  = "임신 성공 여부"
ID_COL  = "ID"
N_FOLDS = 5
SEEDS   = [42, 2024, 777, 1234, 31337]
SEED    = 42
V8_BASELINE = 0.74079

def rank_norm(arr):
    return rankdata(arr) / len(arr)

COUNT_COLS = ["총 시술 횟수", "클리닉 내 총 시술 횟수",
              "IVF 시술 횟수", "DI 시술 횟수",
              "총 임신 횟수", "IVF 임신 횟수", "DI 임신 횟수",
              "총 출산 횟수", "IVF 출산 횟수", "DI 출산 횟수"]

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

def add_clinical_features(df):
    """임상적으로 의미 있는 비율 피처 추가"""
    df = df.copy()
    eps = 1e-5

    # 과거 IVF 성공률
    df["ivf_success_rate"] = df["IVF 임신 횟수"].fillna(0) / (df["IVF 시술 횟수"].fillna(0) + eps)
    # 과거 DI 성공률
    df["di_success_rate"]  = df["DI 임신 횟수"].fillna(0)  / (df["DI 시술 횟수"].fillna(0)  + eps)
    # 전체 성공률
    df["overall_preg_rate"] = df["총 임신 횟수"].fillna(0) / (df["총 시술 횟수"].fillna(0) + eps)
    # 임신 → 출산 전환율 (live birth rate)
    df["live_birth_rate"]  = df["총 출산 횟수"].fillna(0)  / (df["총 임신 횟수"].fillna(0)  + eps)
    # IVF 비중 (IVF vs DI 비율)
    df["ivf_ratio"]        = df["IVF 시술 횟수"].fillna(0) / (df["총 시술 횟수"].fillna(0) + eps)
    # 클리닉 내 시술 비중
    df["clinic_ratio"]     = df["클리닉 내 총 시술 횟수"].fillna(0) / (df["총 시술 횟수"].fillna(0) + eps)
    # 첫 시술 여부
    df["is_first_attempt"] = (df["총 시술 횟수"].fillna(0) <= 1).astype(int)
    # 고시술 반복 여부 (5회 이상)
    df["is_repeat_5plus"]  = (df["총 시술 횟수"].fillna(0) >= 5).astype(int)
    # IVF 임신 있었지만 출산 없는 경우 (유산 경험)
    df["ivf_loss_exp"]     = ((df["IVF 임신 횟수"].fillna(0) > 0) &
                               (df["IVF 출산 횟수"].fillna(0) == 0)).astype(int)
    return df

print("=" * 65)
print("타겟 피처 엔지니어링 선별 테스트")
print(f"v8 기준선: {V8_BASELINE}")
print("=" * 65)

train_raw = pd.read_csv("./data/train.csv")
test_raw  = pd.read_csv("./data/test.csv")

# 기본 전처리
train_base = preprocess(train_raw)
test_base  = preprocess(test_raw)
train_base, test_base = encode_categories(train_base, test_base)

# 피처 추가 버전
train_feat = add_clinical_features(train_base)
test_feat  = add_clinical_features(test_base)

# LightGBM Optuna 튜닝 파라미터 (v5에서 최적화된 값)
LGBM_BEST = dict(
    n_estimators      = 2000,
    learning_rate     = 0.03262,
    num_leaves        = 170,
    max_depth         = 4,
    min_child_samples = 30,
    subsample         = 0.8077,
    colsample_bytree  = 0.5257,
    reg_alpha         = 8.219,
    reg_lambda        = 0.1798,
    min_split_gain    = 0.0380,
    n_jobs            = -1,
    verbose           = -1,
)

def eval_features(train_df, test_df, label):
    X      = train_df.drop(columns=[ID_COL, TARGET])
    y      = train_df[TARGET].copy()
    X_test = test_df.drop(columns=[ID_COL])

    oof  = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        this_oof  = np.zeros(len(X))
        this_test = np.zeros(len(X_test))
        for fold, (tr_i, va_i) in enumerate(skf.split(X, y), 1):
            m = LGBMClassifier(**{**LGBM_BEST, "random_state": seed})
            m.fit(
                X.iloc[tr_i], y.iloc[tr_i],
                eval_set=[(X.iloc[va_i], y.iloc[va_i])],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
            )
            this_oof[va_i] = m.predict_proba(X.iloc[va_i])[:, 1]
            this_test     += m.predict_proba(X_test)[:, 1] / N_FOLDS
        oof        += this_oof  / len(SEEDS)
        test_preds += this_test / len(SEEDS)

    auc = roc_auc_score(y, oof)
    print(f"  {label:<30}: OOF AUC = {auc:.5f}  ({auc - V8_BASELINE:+.5f} vs v8)")
    return oof, test_preds, auc, y

# ─── [1] 베이스라인 재확인 (피처 없음) ──────────────────────────
print("\n[1] 베이스라인 (raw + 기존 2피처)...")
base_oof, base_test, base_auc, y = eval_features(train_base, test_base, "baseline (no new feats)")

# ─── [2] 임상 피처 전체 추가 ────────────────────────────────────
print("\n[2] 임상 비율 피처 9개 전체 추가...")
feat_oof, feat_test, feat_auc, _ = eval_features(train_feat, test_feat, "all clinical features")

# ─── [3] 개별 피처 중요도 확인 (전체 추가 버전에서) ─────────────
print("\n[3] 개별 피처 기여도 (전체 추가 모델 기준)...")
new_feats = ["ivf_success_rate", "di_success_rate", "overall_preg_rate",
             "live_birth_rate", "ivf_ratio", "clinic_ratio",
             "is_first_attempt", "is_repeat_5plus", "ivf_loss_exp"]

# 각 피처를 제외했을 때 변화
print("  피처 제외 시 AUC 변화:")
for drop_col in new_feats:
    t2 = train_feat.drop(columns=[drop_col])
    te2 = test_feat.drop(columns=[drop_col])
    _, _, auc_drop, _ = eval_features(t2, te2, f"  -'{drop_col}'")

# ─── [4] 상위 피처만 선별 (best_auc 기준) ──────────────────────
best_auc_final = max(base_auc, feat_auc)
best_oof_final = feat_oof if feat_auc >= base_auc else base_oof
best_test_final = feat_test if feat_auc >= base_auc else base_test
best_label = "feat" if feat_auc >= base_auc else "base"

print(f"\n{'='*65}")
print("최종 결과")
print(f"{'='*65}")
print(f"  베이스라인:         {base_auc:.5f}")
print(f"  임상 피처 추가:     {feat_auc:.5f}  ({feat_auc - base_auc:+.5f})")
print(f"  최선 선택:          {best_label} ({best_auc_final:.5f})")

np.save(f"oof_feat_auc_{feat_auc:.5f}.npy",  feat_oof)
np.save(f"test_feat_auc_{feat_auc:.5f}.npy", feat_test)

if feat_auc > base_auc:
    sub = pd.read_csv("./data/sample_submission.csv")
    sub["probability"] = feat_test
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename  = f"submission_{timestamp}_v11feat_oof{feat_auc:.5f}.csv"
    sub.to_csv(filename, index=False)
    print(f"\n  저장: {filename}")
    print(f"  v8 기준선 대비: {feat_auc - V8_BASELINE:+.5f}")
else:
    print("\n  임상 피처 추가가 도움이 되지 않음 — 기존 v8 유지")
