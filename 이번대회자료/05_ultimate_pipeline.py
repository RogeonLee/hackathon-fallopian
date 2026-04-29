"""
V5 Ultimate Pipeline — 1등 목표
전략: v4(raw+category) 기반 + Optuna 튜닝 + CatBoost + 멀티시드 스태킹

v4 기준선: OOF AUC 0.74033
목표: OOF AUC 0.745+
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

V4_BASELINE = 0.74033  # 이전 최고 성적

print("=" * 65)
print("V5 Ultimate Pipeline")
print(f"N_FOLDS={N_FOLDS}  SEEDS={SEEDS}")
print(f"v4 기준선: {V4_BASELINE}")
print("=" * 65)

# ──────────────────────── 1. 데이터 로드 ────────────────────────
print("\n[1] 데이터 로드...")
train_raw = pd.read_csv("./data/train.csv")
test_raw  = pd.read_csv("./data/test.csv")
print(f"  train: {train_raw.shape}  test: {test_raw.shape}")

COUNT_COLS = [
    "총 시술 횟수", "클리닉 내 총 시술 횟수",
    "IVF 시술 횟수", "DI 시술 횟수",
    "총 임신 횟수", "IVF 임신 횟수", "DI 임신 횟수",
    "총 출산 횟수", "IVF 출산 횟수", "DI 출산 횟수",
]

# ──────────────────────── 2. 전처리 (v4 방식) ────────────────────
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """v4 검증된 최소 전처리: 횟수 파싱 + 핵심 2개 파생변수"""
    df = df.copy()
    # 횟수 문자열 → 숫자
    for col in COUNT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
            )
    # 핵심 파생변수
    births = df["총 출산 횟수"].fillna(0)
    thawed = df["해동된 배아 수"].fillna(0)
    df["ever_delivered"] = (births > 0).astype(int)
    df["is_FET"]         = (thawed > 0).astype(int)
    return df


def encode_categories(train: pd.DataFrame, test: pd.DataFrame):
    """object 컬럼 → category (LightGBM/CatBoost 네이티브 처리용)"""
    train, test = train.copy(), test.copy()
    obj_cols = [c for c in train.select_dtypes("object").columns
                if c not in [ID_COL, TARGET]]
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


print("[2] 전처리...")
train = preprocess(train_raw)
test  = preprocess(test_raw)
train, test = encode_categories(train, test)

X = train.drop(columns=[ID_COL, TARGET])
y = train[TARGET].copy()
X_test = test.drop(columns=[ID_COL])
print(f"  X: {X.shape}  X_test: {X_test.shape}")
print(f"  카테고리 피처: {X.select_dtypes('category').shape[1]}개")

# ──────────────────────── 3. LGBM 기본 (v4 재현) ────────────────
print("\n[3] LGBM 기본 (5-Seed × 5-Fold)...")

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

def run_lgbm_multiseed(params, seeds, X, y, X_test, tag="LGBM"):
    all_oof  = np.zeros(len(X))
    all_test = np.zeros(len(X_test))
    for seed in seeds:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        this_oof  = np.zeros(len(X))
        this_test = np.zeros(len(X_test))
        for fold, (tr_i, va_i) in enumerate(skf.split(X, y), 1):
            m = LGBMClassifier(**{**params, "random_state": seed})
            m.fit(
                X.iloc[tr_i], y.iloc[tr_i],
                eval_set=[(X.iloc[va_i], y.iloc[va_i])],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            )
            this_oof[va_i] = m.predict_proba(X.iloc[va_i])[:, 1]
            this_test     += m.predict_proba(X_test)[:, 1] / N_FOLDS
        s_auc = roc_auc_score(y, this_oof)
        print(f"  {tag} Seed {seed:>5}: {s_auc:.5f}")
        all_oof  += this_oof  / len(seeds)
        all_test += this_test / len(seeds)
    auc = roc_auc_score(y, all_oof)
    print(f"  {tag} 앙상블 OOF AUC: {auc:.5f}")
    return all_oof, all_test, auc


lgbm_oof, lgbm_test, lgbm_auc = run_lgbm_multiseed(
    LGBM_BASE, SEEDS, X, y, X_test, tag="LGBM_base"
)

# ──────────────────────── 4. Optuna 튜닝 ────────────────────────
print("\n[4] Optuna 하이퍼파라미터 튜닝...")
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = dict(
            n_estimators      = 2000,
            learning_rate     = trial.suggest_float("lr", 0.01, 0.08, log=True),
            num_leaves        = trial.suggest_int("num_leaves", 31, 200),
            max_depth         = trial.suggest_int("max_depth", 4, 10),
            min_child_samples = trial.suggest_int("min_child_samples", 10, 80),
            subsample         = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha         = trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda        = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            min_split_gain    = trial.suggest_float("min_split_gain", 0.0, 1.0),
            verbose = -1, n_jobs = -1, random_state = SEED,
        )
        skf3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        aucs = []
        for tr_i, va_i in skf3.split(X, y):
            m = LGBMClassifier(**params)
            m.fit(
                X.iloc[tr_i], y.iloc[tr_i],
                eval_set=[(X.iloc[va_i], y.iloc[va_i])],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
            )
            aucs.append(roc_auc_score(y.iloc[va_i], m.predict_proba(X.iloc[va_i])[:, 1]))
        return np.mean(aucs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, show_progress_bar=True)
    print(f"  Best 3-fold AUC: {study.best_value:.5f}")
    print(f"  Best params: {study.best_params}")

    LGBM_TUNED = {**LGBM_BASE,
        "learning_rate"    : study.best_params["lr"],
        "num_leaves"       : study.best_params["num_leaves"],
        "max_depth"        : study.best_params["max_depth"],
        "min_child_samples": study.best_params["min_child_samples"],
        "subsample"        : study.best_params["subsample"],
        "colsample_bytree" : study.best_params["colsample_bytree"],
        "reg_alpha"        : study.best_params["reg_alpha"],
        "reg_lambda"       : study.best_params["reg_lambda"],
        "min_split_gain"   : study.best_params["min_split_gain"],
    }
    RUN_TUNED = True
except Exception as e:
    print(f"  Optuna 실패: {e}")
    LGBM_TUNED = LGBM_BASE
    RUN_TUNED = False

# ──────────────────────── 5. LGBM Tuned ────────────────────────
print("\n[5] LGBM Tuned (5-Seed × 5-Fold)...")
lgbm_tuned_oof, lgbm_tuned_test, lgbm_tuned_auc = run_lgbm_multiseed(
    LGBM_TUNED, SEEDS, X, y, X_test, tag="LGBM_tuned"
)

# ──────────────────────── 6. CatBoost ───────────────────────────
print("\n[6] CatBoost (3-Seed × 5-Fold)...")
try:
    from catboost import CatBoostClassifier, Pool

    # CatBoost용: category → string (NaN → "missing")
    X_cb      = X.copy()
    X_test_cb = X_test.copy()
    for col in X_cb.select_dtypes("category").columns:
        X_cb[col]      = X_cb[col].astype(str).replace("nan", "missing")
        X_test_cb[col] = X_test_cb[col].astype(str).replace("nan", "missing")

    cat_feat_names = list(X_cb.select_dtypes(object).columns)

    CB_PARAMS = dict(
        iterations            = 3000,
        learning_rate         = 0.05,
        depth                 = 6,
        l2_leaf_reg           = 3,
        eval_metric           = "AUC",
        early_stopping_rounds = 50,
        verbose               = 0,
        thread_count          = -1,
    )
    CB_SEEDS = [42, 2024, 777]

    cb_oof  = np.zeros(len(X))
    cb_test = np.zeros(len(X_test))

    for seed in CB_SEEDS:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        this_oof  = np.zeros(len(X))
        this_test = np.zeros(len(X_test))
        for fold, (tr_i, va_i) in enumerate(skf.split(X, y), 1):
            m = CatBoostClassifier(**{**CB_PARAMS, "random_seed": seed})
            m.fit(
                X_cb.iloc[tr_i], y.iloc[tr_i],
                eval_set=(X_cb.iloc[va_i], y.iloc[va_i]),
                cat_features=cat_feat_names,
            )
            this_oof[va_i] = m.predict_proba(X_cb.iloc[va_i])[:, 1]
            this_test     += m.predict_proba(X_test_cb)[:, 1] / N_FOLDS
        s_auc = roc_auc_score(y, this_oof)
        print(f"  CB Seed {seed:>5}: {s_auc:.5f}")
        cb_oof  += this_oof  / len(CB_SEEDS)
        cb_test += this_test / len(CB_SEEDS)

    cb_auc = roc_auc_score(y, cb_oof)
    print(f"  CatBoost 앙상블 OOF AUC: {cb_auc:.5f}")

except Exception as e:
    print(f"  CatBoost 오류: {e}")
    cb_oof, cb_test, cb_auc = lgbm_tuned_oof.copy(), lgbm_tuned_test.copy(), lgbm_tuned_auc

# ──────────────────────── 7. XGBoost ────────────────────────────
print("\n[7] XGBoost (3-Seed × 5-Fold)...")
try:
    from xgboost import XGBClassifier

    # XGBoost: category → label-encoded int
    X_xgb      = X.copy()
    X_test_xgb = X_test.copy()
    for col in X_xgb.select_dtypes("category").columns:
        X_xgb[col]      = X_xgb[col].cat.codes.astype(float).replace(-1, np.nan)
        X_test_xgb[col] = X_test_xgb[col].cat.codes.astype(float).replace(-1, np.nan)
    X_xgb      = X_xgb.astype(float)
    X_test_xgb = X_test_xgb.astype(float)

    XGB_PARAMS = dict(
        n_estimators          = 2000,
        learning_rate         = 0.05,
        max_depth             = 6,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        reg_alpha             = 0.1,
        reg_lambda            = 1.0,
        n_jobs                = -1,
        verbosity             = 0,
        eval_metric           = "auc",
        early_stopping_rounds = 50,
    )
    XGB_SEEDS = [42, 2024, 777]

    xgb_oof  = np.zeros(len(X))
    xgb_test = np.zeros(len(X_test))

    for seed in XGB_SEEDS:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        this_oof  = np.zeros(len(X))
        this_test = np.zeros(len(X_test))
        for fold, (tr_i, va_i) in enumerate(skf.split(X, y), 1):
            m = XGBClassifier(**{**XGB_PARAMS, "random_state": seed})
            m.fit(
                X_xgb.iloc[tr_i], y.iloc[tr_i],
                eval_set=[(X_xgb.iloc[va_i], y.iloc[va_i])],
                verbose=False,
            )
            this_oof[va_i] = m.predict_proba(X_xgb.iloc[va_i])[:, 1]
            this_test     += m.predict_proba(X_test_xgb)[:, 1] / N_FOLDS
        s_auc = roc_auc_score(y, this_oof)
        print(f"  XGB Seed {seed:>5}: {s_auc:.5f}")
        xgb_oof  += this_oof  / len(XGB_SEEDS)
        xgb_test += this_test / len(XGB_SEEDS)

    xgb_auc = roc_auc_score(y, xgb_oof)
    print(f"  XGBoost 앙상블 OOF AUC: {xgb_auc:.5f}")

except Exception as e:
    print(f"  XGBoost 오류: {e}")
    xgb_oof, xgb_test, xgb_auc = lgbm_tuned_oof.copy(), lgbm_tuned_test.copy(), lgbm_tuned_auc

# ──────────────────────── 8. 스태킹 앙상블 ──────────────────────
print("\n[8] 스태킹 앙상블...")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata

def rank_norm(arr):
    return rankdata(arr) / len(arr)

# 스택 레이어
stack_oof  = np.column_stack([lgbm_oof, lgbm_tuned_oof, cb_oof, xgb_oof])
stack_test = np.column_stack([lgbm_test, lgbm_tuned_test, cb_test, xgb_test])

scaler = StandardScaler()
stack_oof_s  = scaler.fit_transform(stack_oof)
stack_test_s = scaler.transform(stack_test)

meta = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
meta.fit(stack_oof_s, y)

meta_oof_preds  = meta.predict_proba(stack_oof_s)[:, 1]
meta_test_preds = meta.predict_proba(stack_test_s)[:, 1]
meta_auc = roc_auc_score(y, meta_oof_preds)
print(f"  메타러너 OOF AUC: {meta_auc:.5f}")
print(f"  메타러너 가중치: {meta.coef_[0].round(3)}")

# ──────────────────────── 9. 최종 가중 앙상블 ───────────────────
print("\n[9] 최종 앙상블 구성...")
model_scores = {
    "lgbm_base"  : lgbm_auc,
    "lgbm_tuned" : lgbm_tuned_auc,
    "catboost"   : cb_auc,
    "xgboost"    : xgb_auc,
    "meta"       : meta_auc,
}
print("\n  모델별 OOF AUC:")
for name, auc in model_scores.items():
    diff = auc - V4_BASELINE
    marker = " ▲" if diff > 0 else " ▼"
    print(f"    {name:<14}: {auc:.5f}  ({diff:+.5f} vs v4){marker if diff != 0 else ''}")

# 단순 평균
simple_oof  = (lgbm_oof + lgbm_tuned_oof + cb_oof + xgb_oof + meta_oof_preds) / 5
simple_test = (lgbm_test + lgbm_tuned_test + cb_test + xgb_test + meta_test_preds) / 5
simple_auc  = roc_auc_score(y, simple_oof)

# Rank 기반 가중 앙상블
scores_arr = np.array(list(model_scores.values()))
min_s = scores_arr.min()
w_raw = (scores_arr - min_s + 0.001) ** 2
w = w_raw / w_raw.sum()
print(f"\n  가중치: {dict(zip(model_scores.keys(), w.round(3)))}")

rank_oof  = (w[0]*rank_norm(lgbm_oof) + w[1]*rank_norm(lgbm_tuned_oof) +
             w[2]*rank_norm(cb_oof)   + w[3]*rank_norm(xgb_oof) +
             w[4]*rank_norm(meta_oof_preds))
rank_test = (w[0]*rank_norm(lgbm_test) + w[1]*rank_norm(lgbm_tuned_test) +
             w[2]*rank_norm(cb_test)   + w[3]*rank_norm(xgb_test) +
             w[4]*rank_norm(meta_test_preds))
rank_auc  = roc_auc_score(y, rank_oof)

# 최고 선택
final_configs = {
    "simple_avg": (simple_auc, simple_test),
    "rank_weighted": (rank_auc, rank_test),
    "meta_only": (meta_auc, meta_test_preds),
    "best_single": (max(model_scores.values()),
                    [lgbm_test, lgbm_tuned_test, cb_test, xgb_test, meta_test_preds]
                    [list(model_scores.values()).index(max(model_scores.values()))]),
}
print(f"\n  단순 평균 OOF AUC: {simple_auc:.5f}")
print(f"  Rank 가중 OOF AUC: {rank_auc:.5f}")

best_name  = max(final_configs, key=lambda k: final_configs[k][0])
best_auc   = final_configs[best_name][0]
best_preds = final_configs[best_name][1]
print(f"\n  최선 앙상블: {best_name} → {best_auc:.5f}")
print(f"  v4 대비: {best_auc - V4_BASELINE:+.5f}")

# ──────────────────────── 10. 제출 파일 저장 ────────────────────
print("\n[10] 제출 파일 저장...")
sub = pd.read_csv("./data/sample_submission.csv")
sub["probability"] = best_preds

timestamp = datetime.now().strftime("%Y%m%d_%H%M")
filename  = f"submission_{timestamp}_v5_oof{best_auc:.5f}.csv"
sub.to_csv(filename, index=False)
print(f"  {filename}")

# OOF & test 저장 (이후 앙상블용)
np.save(f"oof_v5_auc_{best_auc:.5f}.npy",  rank_oof)
np.save(f"test_v5_auc_{best_auc:.5f}.npy", best_preds)

# 검증 정보 저장
import json
result_info = {
    "timestamp": timestamp,
    "best_config": best_name,
    "best_oof_auc": float(best_auc),
    "v4_baseline": V4_BASELINE,
    "delta_vs_v4": float(best_auc - V4_BASELINE),
    "all_model_aucs": {k: float(v) for k, v in model_scores.items()},
    "ensemble_aucs": {k: float(v[0]) for k, v in final_configs.items()},
}
with open(f"v5_results_{timestamp}.json", "w", encoding="utf-8") as f:
    json.dump(result_info, f, ensure_ascii=False, indent=2)

# ──────────────────────── 최종 요약 ─────────────────────────────
print("\n" + "=" * 65)
print("V5 최종 결과 요약")
print("=" * 65)
all_results = {**{k: v for k, v in model_scores.items()},
               "simple_avg": simple_auc,
               "rank_weighted": rank_auc}
max_auc = max(all_results.values())
for name, auc in sorted(all_results.items(), key=lambda x: -x[1]):
    star = " ← 최고!" if auc == max_auc else ""
    print(f"  {name:<18}: {auc:.5f}  ({auc-V4_BASELINE:+.5f} vs v4){star}")
print(f"\n  [v4 기준선: {V4_BASELINE}]")
print(f"  최종 제출 파일: {filename}")
