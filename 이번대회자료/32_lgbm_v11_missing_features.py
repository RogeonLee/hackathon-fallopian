"""
v11 — 빠진 핵심 피처 3개 추가 재학습
핵심 발견:
  1. '배아 생성 주요 이유' (13개 값) — train_features에 완전히 빠짐
     배아 저장용 9,192건 → 성공률 0.09% (거의 0)
     기증용 1,108건 → 성공률 0%
     현재 시술용 → 27.4%
     → 이 구분이 없으면 0%짜리 케이스들을 27%로 잘못 예측

  2. '특정 시술 유형' (24개 값) — binary 분해하면서 정보 손실
     BLASTOCYST 포함: 36%, 미포함: 26%
     24개 조합을 하나의 categorical로 사용해야 함

  3. 'real_age_num' — 기증 난자이면 기증자 나이 사용
     정호님 V3의 핵심 피처, 현재 v9/v10에 없음
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from datetime import datetime
import lightgbm as lgb

TARGET = "임신 성공 여부"
ID_COL = "ID"
N_FOLDS = 5
SEEDS   = [42, 2024, 777, 1234, 31337]

ts = datetime.now().strftime("%Y%m%d_%H%M")

print("=" * 70)
print("v11 — 빠진 핵심 피처 추가 재학습")
print("=" * 70)

# ── 데이터 로드 ──────────────────────────────────────────────
train = pd.read_parquet("train_features.parquet")
test  = pd.read_parquet("test_features.parquet")
train_raw = pd.read_parquet("train_clean.parquet")
test_raw  = pd.read_parquet("test_clean.parquet")

y = train[TARGET].values

# ── 빠진 핵심 컬럼 추가 ─────────────────────────────────────
ADD_COLS = [
    "배아 생성 주요 이유",   # ★ 핵심: 배아저장용(0.09%) vs 현재시술용(27.4%)
    "특정 시술 유형",        # ★ 24개 값 그대로 (binary 분해 대신)
    "시술 시기 코드",        # 클리닉 코드 (7개)
    "배란 유도 유형",        # 임상 중요 (4개)
    "불임 원인 - 자궁경부 문제",   # 빠진 불임 원인
    "불임 원인 - 정자 면역학적 요인",
    "불임 원인 - 정자 형태",
]

for col in ADD_COLS:
    if col in train_raw.columns and col not in train.columns:
        train[col] = train_raw[col].values
        test[col]  = test_raw[col].values
        print(f"  추가: {col} (고유값 {train[col].nunique()}개)")
    elif col in train.columns:
        print(f"  이미있음: {col}")
    else:
        print(f"  없음: {col}")

# real_age_num: 기증 난자이면 기증자 나이 사용
DONOR_AGE_MAP = {
    '만20세 이하': 19.0, '만21-25세': 23.0, '만26-30세': 28.0,
    '만31-35세': 33.0, '만36-40세': 38.0, '만41-45세': 43.0,
    '알 수 없음': np.nan,
}
AGE_MAP = {
    '만18-34세': 26.0, '만35-37세': 36.0, '만38-39세': 38.5,
    '만40-42세': 41.0, '만43-44세': 43.5, '만45-50세': 47.5,
    '알 수 없음': np.nan,
}

for df, raw in [(train, train_raw), (test, test_raw)]:
    donor_age = raw['난자 기증자 나이'].map(DONOR_AGE_MAP)
    patient_age = raw['시술 당시 나이'].map(AGE_MAP) if raw['시술 당시 나이'].dtype == object else raw['시술 당시 나이']
    is_donor = (raw['난자_기증'] == 1) if '난자_기증' in raw.columns else (raw['난자 출처'].str.contains('기증', na=False))
    df['real_age_num'] = np.where(is_donor & donor_age.notna(), donor_age, patient_age)
    df['real_age_num'] = df['real_age_num'].astype('float32')

print(f"  추가: real_age_num")

# object → category
for col in ADD_COLS:
    if col in train.columns and train[col].dtype == object:
        all_cats = sorted(set(train[col].dropna().unique()) | set(test[col].dropna().unique()), key=str)
        train[col] = pd.Categorical(train[col], categories=all_cats)
        test[col]  = pd.Categorical(test[col],  categories=all_cats)

# ── 피처 준비 ────────────────────────────────────────────────
drop_cols = [TARGET, ID_COL]
feature_cols = [c for c in train.columns if c not in drop_cols]
X      = train[feature_cols]
X_test = test[[c for c in feature_cols if c in test.columns]]

# test에 없는 컬럼은 NaN으로
for c in feature_cols:
    if c not in X_test.columns:
        X_test[c] = np.nan

X_test = X_test[feature_cols]

cat_cols = [c for c in feature_cols if X[c].dtype.name == 'category']
print(f"\n  총 피처: {len(feature_cols)}개  (categorical: {len(cat_cols)}개)")

# ── LGBM 파라미터 (v9/v10과 동일) ───────────────────────────
LGBM_PARAMS = dict(
    objective        = "binary",
    metric           = "auc",
    n_estimators     = 2000,
    learning_rate    = 0.025,
    num_leaves       = 95,
    min_child_samples= 150,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    verbose          = -1,
)

# ── 5-Seed × 5-Fold 학습 ─────────────────────────────────────
oof_all  = np.zeros(len(y))
test_all = np.zeros(len(X_test))
n_runs   = 0

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)

for seed in SEEDS:
    oof_seed  = np.zeros(len(y))
    test_seed = np.zeros(len(X_test))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx],       y[va_idx]

        model = lgb.LGBMClassifier(
            **LGBM_PARAMS,
            random_state = seed,
            n_jobs       = -1,
        )
        model.fit(
            X_tr, y_tr,
            eval_set    = [(X_va, y_va)],
            callbacks   = [lgb.early_stopping(100, verbose=False),
                           lgb.log_evaluation(-1)],
            categorical_feature = cat_cols,
        )
        oof_seed[va_idx]  = model.predict_proba(X_va)[:, 1]
        test_seed        += model.predict_proba(X_test)[:, 1] / N_FOLDS

    fold_auc = roc_auc_score(y, oof_seed)
    print(f"  seed={seed}: OOF={fold_auc:.5f}")

    oof_all  += oof_seed
    test_all += test_seed
    n_runs   += 1

oof_avg  = oof_all  / n_runs
test_avg = test_all / n_runs

final_auc = roc_auc_score(y, oof_avg)
print(f"\n  [v11] 최종 OOF={final_auc:.5f}")
print(f"  비교: v9/v10 OOF=0.74079")
print(f"  개선: {final_auc - 0.74079:+.5f}")

# 예상 LB 시나리오
print(f"\n  예상 LB (갭 패턴 적용):")
for gap_name, gap in [("LGBM 갭(+0.00143)", 0.00143), ("중간갭(+0.00120)", 0.00120)]:
    lb = final_auc + gap
    flag = " ← 1위(0.74246) 추월!" if lb > 0.74246 else f" ← 1위까지 {0.74246-lb:.5f}"
    print(f"    {gap_name}: {lb:.5f}{flag}")

# OOF/Test 저장
np.save(f"oof_v11_auc_{final_auc:.5f}.npy", oof_avg)
np.save(f"test_v11_auc_{final_auc:.5f}.npy", test_avg)

# 제출 파일
sub = pd.read_csv("./data/sample_submission.csv")
sub["probability"] = test_avg
fname = f"submission_{ts}_v11_missing_feat_oof{final_auc:.5f}.csv"
sub.to_csv(fname, index=False)
print(f"\n  저장: {fname}")
