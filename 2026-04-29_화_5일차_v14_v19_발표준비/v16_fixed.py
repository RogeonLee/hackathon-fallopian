"""
v16 - v14 인코딩 방식 유지 + 검증된 임상 피처만 추가
핵심 수정:
  - LGB: 문자열 컬럼 object 그대로 유지 (v14 방식) → LGB 자체 categorical 처리
  - 추가: has_BLASTOCYST(36.1%), is_elective_SET(38%), eset_blast(43%), 횟수_n 수치화
  - XGB: 별도 encoding (int 코드 변환) — v15와 동일
  - 시드 다양화: LGB 5시드, XGB 3시드
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from scipy.optimize import minimize

# =========================
# 1. 데이터 로드
# =========================
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")
target = "임신 성공 여부"

COUNT_COLS = [
    "총 시술 횟수", "클리닉 내 총 시술 횟수",
    "IVF 시술 횟수", "DI 시술 횟수",
    "총 임신 횟수", "IVF 임신 횟수", "DI 임신 횟수",
    "총 출산 횟수", "IVF 출산 횟수", "DI 출산 횟수",
]
PROC_TOKENS = ["BLASTOCYST", "FER", "AH", "ICSI", "IVF"]

AGE_MIDPOINT = {
    "만18-34세": 26.0, "만35-37세": 36.0, "만38-39세": 38.5,
    "만40-42세": 41.0, "만43-44세": 43.5, "만45-50세": 47.5,
    "알 수 없음": np.nan,
}
DONOR_AGE_MIDPOINT = {
    "만20세 이하": 18.0, "만21-25세": 23.0, "만26-30세": 28.0,
    "만31-35세": 33.0, "알 수 없음": np.nan,
}

def extract_count(val):
    s = str(val).replace("회 이상", "").replace("회", "").strip()
    try: return float(s)
    except: return np.nan


# =========================
# 2-A. LGB용 전처리 (v14 방식 유지 + 새 피처 추가)
# =========================
def preprocess_lgb(df):
    df = df.copy()

    # v14 원본 방식: 첫 번째 숫자 추출 (알 수 없음 → NaN)
    if "시술 당시 나이" in df.columns:
        df["age_num"] = df["시술 당시 나이"].str.extract(r"(\d+)").astype(float)

    # 기증 난자 사용 시 기증자 나이로 대체 (수치)
    if "난자 출처" in df.columns and "난자 기증자 나이" in df.columns:
        donor_num = df["난자 기증자 나이"].map(DONOR_AGE_MIDPOINT)
        is_donor  = df["난자 출처"] == "기증 제공"
        df["real_age_num"] = np.where(is_donor, donor_num, df.get("age_num", np.nan))

    # 횟수 컬럼 수치화 (문자열 버전은 그대로 유지 — LGB가 categorical 처리)
    for col in COUNT_COLS:
        if col in df.columns:
            df[col + "_n"] = df[col].apply(extract_count)

    # ★ 임상 파생 피처 (binary/float — 인코딩 문제 없음)
    if "특정 시술 유형" in df.columns:
        for tok in PROC_TOKENS:
            df[f"has_{tok}"] = (
                df["특정 시술 유형"].str.contains(tok, na=False).astype(np.int8)
            )

    if "이식된 배아 수" in df.columns and "총 생성 배아 수" in df.columns:
        emb_x = df["이식된 배아 수"]; emb_t = df["총 생성 배아 수"]
        df["is_elective_SET"]      = ((emb_x == 1) & (emb_t > 1)).astype(np.int8)
        df["embryo_transfer_ratio"] = emb_x / (emb_t + 1)
        df["log_n_embryo"]          = np.log1p(emb_t)

    if "has_BLASTOCYST" in df.columns and "is_elective_SET" in df.columns:
        df["eset_blast"] = (df["has_BLASTOCYST"] & df["is_elective_SET"]).astype(np.int8)

    birth_n = df.get("총 출산 횟수_n", pd.Series(0.0, index=df.index)).fillna(0)
    preg_n  = df.get("총 임신 횟수_n",  pd.Series(0.0, index=df.index)).fillna(0)
    df["ever_delivered"]  = (birth_n > 0).astype(np.int8)
    df["past_birth_rate"] = np.where(preg_n > 0, birth_n / (preg_n + 1e-6), 0.0)

    if "총 시술 횟수_n" in df.columns:
        df["repeated_3plus"] = (df["총 시술 횟수_n"].fillna(0) >= 3).astype(np.int8)

    if "미세주입된 난자 수" in df.columns and "혼합된 난자 수" in df.columns:
        df["icsi_ratio"] = df["미세주입된 난자 수"] / (df["혼합된 난자 수"] + 1)

    # 결측 플래그 (v14 방식)
    for col in [c for c in df.columns if "_isnull" not in c]:
        if df[col].isnull().any():
            df[col + "_isnull"] = df[col].isnull().astype(np.int8)

    # ★ 문자열 → pd.Categorical dtype (LGB가 진짜 categorical로 처리)
    # .codes(int)로 변환하지 않음 — 그러면 LGB가 수치형으로 잘못 인식
    for col in df.select_dtypes(include="object").columns:
        if col == "ID":
            continue
        df[col] = pd.Categorical(df[col])

    return df


# =========================
# 2-B. XGB용 전처리 (int 코드 변환)
# =========================
def preprocess_xgb(df):
    df = preprocess_lgb(df)   # 임상 피처 포함

    # 문자열 → int 코드 (XGB는 object 컬럼 불가)
    for col in df.select_dtypes(include="object").columns:
        if col == "ID":
            continue
        df[col] = pd.Categorical(df[col]).codes.astype(np.int16)

    return df


# =========================
# 3. 전처리 실행
# =========================
train_lgb = preprocess_lgb(train)
test_lgb  = preprocess_lgb(test)
train_xgb = preprocess_xgb(train)
test_xgb  = preprocess_xgb(test)

feat_lgb = [c for c in train_lgb.columns if c not in [target, "ID"]]
feat_xgb = [c for c in train_xgb.columns if c not in [target, "ID"]]

# test 컬럼 정렬
for f in feat_lgb:
    if f not in test_lgb.columns: test_lgb[f] = 0
for f in feat_xgb:
    if f not in test_xgb.columns: test_xgb[f] = 0

X_lgb  = train_lgb[feat_lgb]; X_lgb_test  = test_lgb[feat_lgb]
X_xgb  = train_xgb[feat_xgb]; X_xgb_test  = test_xgb[feat_xgb]
y      = train[target]

print(f"LGB 피처 수: {len(feat_lgb)},  XGB 피처 수: {len(feat_xgb)}")

# =========================
# 4. 모델 정의 (v14 하이퍼파라미터 + 시드 5종)
# =========================
LGB_COMMON = dict(n_jobs=-1, verbose=-1)
SEEDS_LGB = [42, 2024, 777, 31415, 99999]
SEEDS_XGB = [42, 2024, 777]

models = []

# LGB: base config (leaves=63) — 3 seeds
for s in SEEDS_LGB[:3]:
    models.append(("LGB_base_s" + str(s),
                   LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=63,
                                  subsample=0.8, colsample_bytree=0.8,
                                  random_state=s, **LGB_COMMON),
                   "lgb"))

# LGB: shallow (leaves=31) — 2 seeds
for s in SEEDS_LGB[:2]:
    models.append(("LGB_sh_s" + str(s),
                   LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=31,
                                  min_child_samples=200, subsample=0.7, colsample_bytree=0.7,
                                  random_state=s, **LGB_COMMON),
                   "lgb"))

# XGB — 2 seeds, depth 7
for s in SEEDS_XGB[:2]:
    models.append(("XGB_d7_s" + str(s),
                   XGBClassifier(n_estimators=2000, learning_rate=0.03, max_depth=7,
                                 min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
                                 eval_metric="auc", random_state=s, n_jobs=-1, verbosity=0,
                                 early_stopping_rounds=150),
                   "xgb"))

# =========================
# 5. CV + OOF
# =========================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
n_models = len(models)
oof_preds  = np.zeros((len(y), n_models))
test_preds = np.zeros((len(X_lgb_test), n_models))

for m_idx, (name, model, mtype) in enumerate(models):
    print(f"\n{'='*40}")
    print(f"[{m_idx+1}/{n_models}] {name}")
    is_lgb = (mtype == "lgb")
    X_tr_all  = X_lgb  if is_lgb else X_xgb
    X_te_all  = X_lgb_test if is_lgb else X_xgb_test

    fold_test = np.zeros(len(X_te_all))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr_all, y)):
        X_tr, X_val = X_tr_all.iloc[tr_idx], X_tr_all.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        if is_lgb:
            model.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val)],
                      eval_metric="auc",
                      callbacks=[])
        else:
            model.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val)],
                      verbose=False)

        vp = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx, m_idx] = vp
        fold_test += model.predict_proba(X_te_all)[:, 1] / skf.n_splits
        print(f"  Fold {fold}: {roc_auc_score(y_val, vp):.5f}")

    test_preds[:, m_idx] = fold_test
    print(f"  ── OOF AUC: {roc_auc_score(y, oof_preds[:, m_idx]):.5f}")

# =========================
# 6. 가중치 최적화
# =========================
print("\n최적화 중...")

def neg_auc_w(w):
    w = np.clip(np.array(w), 0, 1)
    s = w.sum()
    if s < 1e-9: return 0.0
    return -roc_auc_score(y, (oof_preds * (w / s)).sum(axis=1))

simple_auc = roc_auc_score(y, oof_preds.mean(axis=1))
print(f"단순 평균 OOF AUC: {simple_auc:.5f}")

init_ws = [np.ones(n_models) / n_models]
for i in range(n_models):
    w = np.zeros(n_models); w[i] = 1.0; init_ws.append(w)
for sd in [0, 7, 42, 99]:
    rng = np.random.RandomState(sd)
    init_ws.append(rng.dirichlet(np.ones(n_models)))

best_w, best_val = np.ones(n_models) / n_models, neg_auc_w(np.ones(n_models) / n_models)
for w0 in init_ws:
    res = minimize(neg_auc_w, w0, method="Nelder-Mead",
                   options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-9})
    if res.fun < best_val:
        best_val, best_w = res.fun, res.x

opt_w = np.clip(best_w, 0, 1); opt_w /= opt_w.sum()
opt_auc = -best_val
if simple_auc > opt_auc:
    opt_w = np.ones(n_models) / n_models; opt_auc = simple_auc
    print("단순 평균 사용")

print(f"최적 OOF AUC: {opt_auc:.5f}")
print("가중치:", dict(zip([m[0] for m in models], opt_w.round(4))))

# =========================
# 7. 제출 파일
# =========================
final_raw  = (test_preds * opt_w).sum(axis=1)
rank_norm  = pd.Series(final_raw).rank(pct=True).values
final_pred = 0.5 * final_raw + 0.5 * rank_norm

ts    = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
fname = f"v16_submission_oof{opt_auc:.5f}_{ts}.csv"

sub = pd.DataFrame({"ID": test["ID"], "probability": final_pred})
sub.to_csv(fname, index=False)

oof_blend = (oof_preds * opt_w).sum(axis=1)
test_blend = final_raw
np.save(f"oof_v16_auc_{opt_auc:.5f}.npy",  oof_blend)
np.save(f"test_v16_auc_{opt_auc:.5f}.npy", test_blend)

print(f"\n{'='*55}")
print(f"완료: {fname}")
print(f"OOF AUC: {opt_auc:.5f}")
print(f"{'='*55}")
