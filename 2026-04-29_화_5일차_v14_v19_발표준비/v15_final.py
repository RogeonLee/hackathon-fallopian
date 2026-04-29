"""
v15 - v14 기반 최종 개선판
주요 변경:
  1. 특정 시술 유형에서 BLASTOCYST/FER 등 토큰 피처 추출 (성공률 36.1%)
  2. is_elective_SET 피처 (38.0% vs 21.9%)
  3. 횟수형 컬럼("0회"..."6회 이상") → 수치 변환
  4. XGBoost 2종 추가 (알고리즘 다양화)
  5. Nelder-Mead OOF 가중치 최적화
  6. v14와 동일한 LGB 하이퍼파라미터 유지 (검증된 값)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from scipy.optimize import minimize, differential_evolution

# =========================
# 1. 데이터 로드
# =========================
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

target = "임신 성공 여부"

# =========================
# 2. 전처리 + 피처 엔지니어링
# =========================

AGE_MIDPOINT = {
    "만18-34세": 26.0,
    "만35-37세": 36.0,
    "만38-39세": 38.5,
    "만40-42세": 41.0,
    "만43-44세": 43.5,
    "만45-50세": 47.5,
    "알 수 없음": np.nan,
}

DONOR_AGE_MIDPOINT = {
    "만20세 이하": 18.0,
    "만21-25세":   23.0,
    "만26-30세":   28.0,
    "만31-35세":   33.0,
    "알 수 없음":   np.nan,
}

COUNT_COLS = [
    "총 시술 횟수", "클리닉 내 총 시술 횟수",
    "IVF 시술 횟수", "DI 시술 횟수",
    "총 임신 횟수", "IVF 임신 횟수", "DI 임신 횟수",
    "총 출산 횟수", "IVF 출산 횟수", "DI 출산 횟수",
]

PROC_TOKENS = ["BLASTOCYST", "FER", "AH", "ICSI", "IVF"]


def extract_count(val):
    """'0회', '6회 이상' 같은 값 → float"""
    s = str(val).replace("회 이상", "").replace("회", "").strip()
    try:
        return float(s)
    except Exception:
        return np.nan


def preprocess(df):
    df = df.copy()

    # ── 나이 ──────────────────────────────────────────
    if "시술 당시 나이" in df.columns:
        df["age_num"] = df["시술 당시 나이"].map(AGE_MIDPOINT)

    # 기증 난자 사용 시 기증자 나이로 대체
    if "난자 출처" in df.columns and "난자 기증자 나이" in df.columns:
        donor_age_num = df["난자 기증자 나이"].map(DONOR_AGE_MIDPOINT)
        is_donor = df["난자 출처"] == "기증 제공"
        df["real_age_num"] = np.where(is_donor, donor_age_num, df.get("age_num", np.nan))
    elif "age_num" in df.columns:
        df["real_age_num"] = df["age_num"]

    # ── 횟수형 컬럼 수치화 ─────────────────────────────
    for col in COUNT_COLS:
        if col in df.columns:
            df[col + "_n"] = df[col].apply(extract_count)

    # ── 특정 시술 유형 토큰 피처 ───────────────────────
    if "특정 시술 유형" in df.columns:
        for tok in PROC_TOKENS:
            df[f"has_{tok}"] = (
                df["특정 시술 유형"].str.contains(tok, na=False).astype(np.int8)
            )

    # ── 임상 파생 피처 ─────────────────────────────────
    emb_total = df.get("총 생성 배아 수", pd.Series(dtype=float))
    emb_xfer  = df.get("이식된 배아 수",  pd.Series(dtype=float))

    # 선택적 단일 배아 이식: 배아 품질 선별 신호 (38.0% vs 21.9%)
    if "이식된 배아 수" in df.columns and "총 생성 배아 수" in df.columns:
        df["is_elective_SET"] = (
            (emb_xfer == 1) & (emb_total > 1)
        ).astype(np.int8)

    # 배반포 × 선택적 단일이식 시너지
    if "has_BLASTOCYST" in df.columns and "is_elective_SET" in df.columns:
        df["eset_blast"] = (df["has_BLASTOCYST"] & df["is_elective_SET"]).astype(np.int8)

    # 배아 이식 효율
    if "총 생성 배아 수" in df.columns:
        df["embryo_transfer_ratio"] = emb_xfer / (emb_total + 1)
        df["log_n_embryo"] = np.log1p(emb_total)

    # 과거 출산/임신 비율
    birth_n = df.get("총 출산 횟수_n", pd.Series(0.0, index=df.index))
    preg_n  = df.get("총 임신 횟수_n",  pd.Series(0.0, index=df.index))
    df["ever_delivered"]  = (birth_n.fillna(0) > 0).astype(np.int8)
    df["past_birth_rate"] = np.where(preg_n > 0, birth_n / (preg_n + 1e-6), 0.0)

    # 반복 시술 (3회 이상 → 성공률 급락)
    if "총 시술 횟수_n" in df.columns:
        df["repeated_3plus"] = (df["총 시술 횟수_n"].fillna(0) >= 3).astype(np.int8)

    # ICSI 주입 비율
    if "미세주입된 난자 수" in df.columns and "혼합된 난자 수" in df.columns:
        df["icsi_ratio"] = df["미세주입된 난자 수"] / (df["혼합된 난자 수"] + 1)

    # ── 결측 플래그 ────────────────────────────────────
    existing_cols = [c for c in df.columns if "_isnull" not in c]
    for col in existing_cols:
        if df[col].isnull().any():
            df[col + "_isnull"] = df[col].isnull().astype(np.int8)

    # ── 문자열 → 카테고리 코드 ────────────────────────
    for col in df.select_dtypes(include="object").columns:
        if col == "ID":
            continue
        df[col] = pd.Categorical(df[col]).codes.astype(np.int16)

    return df


train = preprocess(train)
test  = preprocess(test)

# train 기준으로 test 컬럼 정렬 (null flag 차이 보정)
features = [c for c in train.columns if c not in [target, "ID"]]
for f in features:
    if f not in test.columns:
        test[f] = 0

X      = train[features]
y      = train[target]
X_test = test[features]

print(f"피처 수: {len(features)}")
print(f"훈련 샘플: {len(X):,}")

# =========================
# 3. 모델 정의
# =========================
# LGB 하이퍼파라미터 — v14와 동일하게 유지 (검증된 값)
LGB_COMMON = dict(n_jobs=-1, verbose=-1)

lgb1 = LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=63,
                      subsample=0.8, colsample_bytree=0.8,
                      random_state=42, **LGB_COMMON)

lgb2 = LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=31,
                      min_child_samples=200, subsample=0.7, colsample_bytree=0.7,
                      random_state=2024, **LGB_COMMON)

lgb3 = LGBMClassifier(n_estimators=2000, learning_rate=0.02, num_leaves=95,
                      subsample=0.9, colsample_bytree=0.9,
                      random_state=777, **LGB_COMMON)

# XGB — v14에 없던 알고리즘 다양화
xgb1 = XGBClassifier(
    n_estimators=2000, learning_rate=0.03, max_depth=7,
    min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
    eval_metric="auc", random_state=42, n_jobs=-1, verbosity=0,
    early_stopping_rounds=150,
)

xgb2 = XGBClassifier(
    n_estimators=2000, learning_rate=0.03, max_depth=6,
    min_child_weight=15, subsample=0.7, colsample_bytree=0.7,
    eval_metric="auc", random_state=2024, n_jobs=-1, verbosity=0,
    early_stopping_rounds=150,
)

models = [
    ("LGB_base",     lgb1, "lgb"),
    ("LGB_shallow",  lgb2, "lgb"),
    ("LGB_deep",     lgb3, "lgb"),
    ("XGB_depth7",   xgb1, "xgb"),
    ("XGB_depth6",   xgb2, "xgb"),
]

# =========================
# 4. CV + OOF 수집
# =========================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

n_models = len(models)
oof_preds  = np.zeros((len(X), n_models))
test_preds = np.zeros((len(X_test), n_models))

for m_idx, (name, model, mtype) in enumerate(models):
    print(f"\n{'='*40}")
    print(f"[{m_idx+1}/{n_models}] {name}")
    print(f"{'='*40}")

    fold_test = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        if mtype == "lgb":
            model.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val)],
                      eval_metric="auc",
                      callbacks=[])
        else:
            model.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val)],
                      verbose=False)

        val_pred = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx, m_idx] = val_pred
        fold_test += model.predict_proba(X_test)[:, 1] / skf.n_splits

        auc = roc_auc_score(y_val, val_pred)
        print(f"  Fold {fold}: {auc:.5f}")

    test_preds[:, m_idx] = fold_test
    oof_auc = roc_auc_score(y, oof_preds[:, m_idx])
    print(f"  ── OOF AUC: {oof_auc:.5f}")

# =========================
# 5. 앙상블 가중치 최적화
# =========================
print("\n앙상블 가중치 최적화 중...")

def neg_auc_weights(w):
    w = np.array(w)
    w = np.clip(w, 0.0, 1.0)
    total = w.sum()
    if total < 1e-9:
        return 0.0
    w = w / total
    blend = (oof_preds * w).sum(axis=1)
    return -roc_auc_score(y, blend)

# 단순 평균 베이스라인
simple_auc = roc_auc_score(y, oof_preds.mean(axis=1))
print(f"단순 평균 OOF AUC: {simple_auc:.5f}")

# Nelder-Mead 다중 시작점 (균등 + 각 모델 독점 + 랜덤 3회)
init_weights = [np.ones(n_models) / n_models]  # 균등
for i in range(n_models):                       # 각 모델 단독
    w = np.zeros(n_models)
    w[i] = 1.0
    init_weights.append(w)
for seed in [0, 1, 2]:                          # 랜덤
    rng = np.random.RandomState(seed)
    init_weights.append(rng.dirichlet(np.ones(n_models)))

best_w   = np.ones(n_models) / n_models
best_val = neg_auc_weights(best_w)

for w0 in init_weights:
    res = minimize(neg_auc_weights, w0, method="Nelder-Mead",
                   options={"maxiter": 10000, "xatol": 1e-7, "fatol": 1e-8})
    if res.fun < best_val:
        best_val = res.fun
        best_w   = res.x

opt_w = np.clip(best_w, 0.0, 1.0)
opt_w /= opt_w.sum()
opt_auc = -best_val
print(f"최적화 OOF AUC : {opt_auc:.5f}")
print(f"최적 가중치    : {dict(zip([m[0] for m in models], opt_w.round(4)))}")

# 가중치가 단순평균보다 나쁘면 단순평균 사용
if simple_auc > opt_auc:
    opt_w   = np.ones(n_models) / n_models
    opt_auc = simple_auc
    print("단순 평균이 더 좋아 단순 평균 사용")

# =========================
# 6. 최종 예측
# =========================
final_raw  = (test_preds * opt_w).sum(axis=1)
rank_norm  = pd.Series(final_raw).rank(pct=True).values
final_pred = 0.5 * final_raw + 0.5 * rank_norm   # Rank Averaging

# =========================
# 7. 제출 파일 + numpy 저장
# =========================
ts    = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
fname = f"v15_submission_oof{opt_auc:.5f}_{ts}.csv"

submission = pd.DataFrame({"ID": test["ID"], "probability": final_pred})
submission.to_csv(fname, index=False)

# 블렌딩용 OOF/test 저장
oof_blend  = (oof_preds * opt_w).sum(axis=1)
test_blend = (test_preds * opt_w).sum(axis=1)
np.save(f"oof_v15_auc_{opt_auc:.5f}.npy",  oof_blend)
np.save(f"test_v15_auc_{opt_auc:.5f}.npy", test_blend)

print(f"\n{'='*50}")
print(f"완료: {fname}")
print(f"최종 OOF AUC: {opt_auc:.5f}")
print(f"OOF numpy  : oof_v15_auc_{opt_auc:.5f}.npy")
print(f"Test numpy : test_v15_auc_{opt_auc:.5f}.npy")
print(f"{'='*50}")
