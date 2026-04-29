"""
XGBoost Optuna 튜닝 + 최종 대앙상블
v7 기준선: 0.74076
"""
import glob, warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata
import lightgbm as lgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

TARGET  = "임신 성공 여부"
ID_COL  = "ID"
N_FOLDS = 5
SEEDS   = [42, 2024, 777, 1234, 31337]
SEED    = 42
V7_BASELINE = 0.74076

def rank_norm(arr):
    return rankdata(arr) / len(arr)

# ─── 전처리 ──────────────────────────────────────────────────────
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

print("=" * 65)
print("XGBoost Optuna + 최종 대앙상블")
print(f"v7 기준선: {V7_BASELINE}")
print("=" * 65)

train_raw = pd.read_csv("./data/train.csv")
test_raw  = pd.read_csv("./data/test.csv")
train = preprocess(train_raw)
test  = preprocess(test_raw)
train, test = encode_categories(train, test)

X = train.drop(columns=[ID_COL, TARGET])
y = train[TARGET].copy()
X_test = test.drop(columns=[ID_COL])

# XGB용: category → label-encoded int
X_xgb      = X.copy()
X_test_xgb = X_test.copy()
for col in X_xgb.select_dtypes("category").columns:
    X_xgb[col]      = X_xgb[col].cat.codes.astype(float).replace(-1, np.nan)
    X_test_xgb[col] = X_test_xgb[col].cat.codes.astype(float).replace(-1, np.nan)
X_xgb      = X_xgb.astype(float)
X_test_xgb = X_test_xgb.astype(float)

# ─── XGBoost Optuna 튜닝 ─────────────────────────────────────────
print("\n[1] XGBoost Optuna 튜닝...")
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def xgb_objective(trial):
    params = dict(
        n_estimators          = 2000,
        learning_rate         = trial.suggest_float("lr", 0.01, 0.08, log=True),
        max_depth             = trial.suggest_int("max_depth", 3, 8),
        min_child_weight      = trial.suggest_int("min_child_weight", 1, 20),
        subsample             = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree      = trial.suggest_float("colsample_bytree", 0.4, 1.0),
        colsample_bylevel     = trial.suggest_float("colsample_bylevel", 0.4, 1.0),
        reg_alpha             = trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda            = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        gamma                 = trial.suggest_float("gamma", 0.0, 5.0),
        n_jobs                = -1,
        verbosity             = 0,
        eval_metric           = "auc",
        early_stopping_rounds = 30,
        random_state          = SEED,
    )
    skf3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    aucs = []
    for tr_i, va_i in skf3.split(X_xgb, y):
        m = XGBClassifier(**params)
        m.fit(X_xgb.iloc[tr_i], y.iloc[tr_i],
              eval_set=[(X_xgb.iloc[va_i], y.iloc[va_i])], verbose=False)
        aucs.append(roc_auc_score(y.iloc[va_i], m.predict_proba(X_xgb.iloc[va_i])[:, 1]))
    return np.mean(aucs)

study = optuna.create_study(direction="maximize")
study.optimize(xgb_objective, n_trials=60, show_progress_bar=True)
print(f"  XGB Optuna best 3-fold AUC: {study.best_value:.5f}")
print(f"  Best params: {study.best_params}")

XGB_TUNED = dict(
    n_estimators          = 2000,
    learning_rate         = study.best_params["lr"],
    max_depth             = study.best_params["max_depth"],
    min_child_weight      = study.best_params["min_child_weight"],
    subsample             = study.best_params["subsample"],
    colsample_bytree      = study.best_params["colsample_bytree"],
    colsample_bylevel     = study.best_params["colsample_bylevel"],
    reg_alpha             = study.best_params["reg_alpha"],
    reg_lambda            = study.best_params["reg_lambda"],
    gamma                 = study.best_params["gamma"],
    n_jobs                = -1,
    verbosity             = 0,
    eval_metric           = "auc",
    early_stopping_rounds = 50,
)

# ─── XGBoost Tuned 5-Seed ──────────────────────────────────────
print("\n[2] XGBoost Tuned (5-Seed × 5-Fold)...")
xgbt_oof  = np.zeros(len(X_xgb))
xgbt_test = np.zeros(len(X_test_xgb))

for seed in SEEDS:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    this_oof  = np.zeros(len(X_xgb))
    this_test = np.zeros(len(X_test_xgb))
    for fold, (tr_i, va_i) in enumerate(skf.split(X_xgb, y), 1):
        m = XGBClassifier(**{**XGB_TUNED, "random_state": seed})
        m.fit(X_xgb.iloc[tr_i], y.iloc[tr_i],
              eval_set=[(X_xgb.iloc[va_i], y.iloc[va_i])], verbose=False)
        this_oof[va_i] = m.predict_proba(X_xgb.iloc[va_i])[:, 1]
        this_test     += m.predict_proba(X_test_xgb)[:, 1] / N_FOLDS
    s_auc = roc_auc_score(y, this_oof)
    print(f"  XGBt Seed {seed:>5}: {s_auc:.5f}")
    xgbt_oof  += this_oof  / len(SEEDS)
    xgbt_test += this_test / len(SEEDS)

xgbt_auc = roc_auc_score(y, xgbt_oof)
print(f"  XGB Tuned 앙상블 OOF AUC: {xgbt_auc:.5f}")

# ─── 기존 모델 OOF 로드 ────────────────────────────────────────
print("\n[3] 기존 모델 OOF 로드...")
# v5 OOF (LGBM base + tuned + XGB base + stacking)
v5_oof_f   = sorted(glob.glob("oof_v5_auc_*.npy"))
v5_test_f  = sorted(glob.glob("test_v5_auc_*.npy"))
cb_oof_f   = sorted(glob.glob("oof_cb5seed_auc_*.npy"))
cb_test_f  = sorted(glob.glob("test_cb5seed_auc_*.npy"))

if not v5_oof_f:
    print("  v5 OOF 없음 - v5 파이프라인 먼저 실행하세요")
    exit(1)

v5_oof  = np.load(v5_oof_f[-1])
v5_test = np.load(v5_test_f[-1])
v5_auc  = roc_auc_score(y, v5_oof)

if cb_oof_f:
    cb_oof  = np.load(cb_oof_f[-1])
    cb_test = np.load(cb_test_f[-1])
    cb_auc  = roc_auc_score(y, cb_oof)
    print(f"  v5 OOF AUC:      {v5_auc:.5f}")
    print(f"  CatBoost OOF AUC:{cb_auc:.5f}")
    print(f"  XGB Tuned OOF:   {xgbt_auc:.5f}")

    # 그리드 서치: 최적 3원 블렌딩 가중치
    best_blend_auc = 0
    best_w = (0.5, 0.25, 0.25)
    for w1 in [0.4, 0.5, 0.6, 0.7]:
        for w2 in [0.1, 0.15, 0.2, 0.25, 0.3]:
            w3 = 1 - w1 - w2
            if w3 < 0.05:
                continue
            blend_oof = w1*rank_norm(v5_oof) + w2*rank_norm(cb_oof) + w3*rank_norm(xgbt_oof)
            blend_auc = roc_auc_score(y, blend_oof)
            if blend_auc > best_blend_auc:
                best_blend_auc = blend_auc
                best_w = (w1, w2, w3)

    print(f"\n  최적 가중치: v5={best_w[0]:.2f}, CB={best_w[1]:.2f}, XGBt={best_w[2]:.2f}")
    print(f"  최적 3원 블렌딩 OOF AUC: {best_blend_auc:.5f}")

    final_oof  = best_w[0]*rank_norm(v5_oof)  + best_w[1]*rank_norm(cb_oof)  + best_w[2]*rank_norm(xgbt_oof)
    final_test = best_w[0]*rank_norm(v5_test) + best_w[1]*rank_norm(cb_test) + best_w[2]*rank_norm(xgbt_test)
else:
    # CatBoost 없음 → v5 + XGB 블렌딩
    blend_oof  = 0.7*rank_norm(v5_oof) + 0.3*rank_norm(xgbt_oof)
    blend_test = 0.7*rank_norm(v5_test) + 0.3*rank_norm(xgbt_test)
    best_blend_auc = roc_auc_score(y, blend_oof)
    print(f"  v5 + XGBt OOF AUC: {best_blend_auc:.5f}")
    final_oof, final_test = blend_oof, blend_test

# ─── 스태킹 재학습 (모든 모델 포함) ────────────────────────────
print("\n[4] 최종 스태킹...")
if cb_oof_f:
    stack_oof  = np.column_stack([v5_oof, cb_oof, xgbt_oof])
    stack_test = np.column_stack([v5_test, cb_test, xgbt_test])
else:
    stack_oof  = np.column_stack([v5_oof, xgbt_oof])
    stack_test = np.column_stack([v5_test, xgbt_test])

scaler = StandardScaler()
stack_oof_s  = scaler.fit_transform(stack_oof)
stack_test_s = scaler.transform(stack_test)

meta = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
meta.fit(stack_oof_s, y)

meta_oof   = meta.predict_proba(stack_oof_s)[:, 1]
meta_test  = meta.predict_proba(stack_test_s)[:, 1]
meta_auc   = roc_auc_score(y, meta_oof)
print(f"  스태킹 OOF AUC: {meta_auc:.5f}")

# ─── 최종 비교 ─────────────────────────────────────────────────
all_results = {
    "v5_alone":    roc_auc_score(y, v5_oof),
    "xgbt_alone":  xgbt_auc,
    "blend_3way":  best_blend_auc,
    "stacking":    meta_auc,
}
if cb_oof_f:
    all_results["cb_alone"] = cb_auc

print("\n" + "=" * 65)
print("최종 결과")
print("=" * 65)
best_auc = max(all_results.values())
for name, auc in sorted(all_results.items(), key=lambda x: -x[1]):
    star = " ← 최고!" if auc == best_auc else ""
    print(f"  {name:<14}: {auc:.5f}  ({auc-V7_BASELINE:+.5f} vs v7){star}")

# 최선 제출 파일
if best_auc == meta_auc:
    final_preds = meta_test
elif best_auc == best_blend_auc:
    final_preds = final_test
elif best_auc == xgbt_auc:
    final_preds = xgbt_test
else:
    final_preds = v5_test

sub = pd.read_csv("./data/sample_submission.csv")
sub["probability"] = final_preds
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
filename  = f"submission_{timestamp}_v8_oof{best_auc:.5f}.csv"
sub.to_csv(filename, index=False)
print(f"\n  저장: {filename}")
print(f"  v4 기준선(0.74033) 대비: {best_auc - 0.74033:+.5f}")

np.save(f"oof_v8_auc_{best_auc:.5f}.npy", meta_oof if best_auc == meta_auc else final_oof)
np.save(f"test_v8_auc_{best_auc:.5f}.npy", final_preds)
