"""
LGBM Optuna 200 trials + 더 넓은 탐색 공간
v8 기준선: 0.74079
이전 v5 best: num_leaves=170, reg_alpha=8.22, max_depth=4
탐색 공간 확장: num_leaves(50~300), max_depth(3~8), reg_alpha(0.01~30)
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
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

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

print("=" * 65)
print("LGBM Optuna 200 trials (확장 탐색)")
print(f"v8 기준선: {V8_BASELINE}")
print("=" * 65)

train_raw = pd.read_csv("./data/train.csv")
test_raw  = pd.read_csv("./data/test.csv")
train = preprocess(train_raw)
test  = preprocess(test_raw)
train, test = encode_categories(train, test)

X = train.drop(columns=[ID_COL, TARGET])
y = train[TARGET].copy()
X_test = test.drop(columns=[ID_COL])

# ─── Optuna 목적함수 ──────────────────────────────────────────
def lgbm_objective(trial):
    params = dict(
        n_estimators      = 3000,
        learning_rate     = trial.suggest_float("lr", 0.005, 0.08, log=True),
        num_leaves        = trial.suggest_int("num_leaves", 31, 300),
        max_depth         = trial.suggest_int("max_depth", 3, 9),
        min_child_samples = trial.suggest_int("min_child_samples", 10, 100),
        subsample         = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.3, 1.0),
        reg_alpha         = trial.suggest_float("reg_alpha", 0.01, 30.0, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
        min_split_gain    = trial.suggest_float("min_split_gain", 0.0, 1.0),
        subsample_freq    = 1,
        n_jobs            = -1,
        verbose           = -1,
    )
    skf3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    aucs = []
    for tr_i, va_i in skf3.split(X, y):
        m = LGBMClassifier(**params, random_state=SEED)
        m.fit(
            X.iloc[tr_i], y.iloc[tr_i],
            eval_set=[(X.iloc[va_i], y.iloc[va_i])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        aucs.append(roc_auc_score(y.iloc[va_i], m.predict_proba(X.iloc[va_i])[:, 1]))
    return np.mean(aucs)

print("\n[1] LGBM Optuna 200 trials...")
study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=SEED))
# v5 best params를 시작점으로 힌트 제공
study.enqueue_trial({
    "lr": 0.03262, "num_leaves": 170, "max_depth": 4,
    "min_child_samples": 30, "subsample": 0.8077, "colsample_bytree": 0.5257,
    "reg_alpha": 8.219, "reg_lambda": 0.1798, "min_split_gain": 0.0380,
})
study.optimize(lgbm_objective, n_trials=200, show_progress_bar=True)
print(f"\n  LGBM Optuna best 3-fold AUC: {study.best_value:.5f}")
print(f"  Best params: {study.best_params}")

LGBM_BEST = dict(
    n_estimators      = 3000,
    learning_rate     = study.best_params["lr"],
    num_leaves        = study.best_params["num_leaves"],
    max_depth         = study.best_params["max_depth"],
    min_child_samples = study.best_params["min_child_samples"],
    subsample         = study.best_params["subsample"],
    colsample_bytree  = study.best_params["colsample_bytree"],
    reg_alpha         = study.best_params["reg_alpha"],
    reg_lambda        = study.best_params["reg_lambda"],
    min_split_gain    = study.best_params["min_split_gain"],
    subsample_freq    = 1,
    n_jobs            = -1,
    verbose           = -1,
)

# ─── 5-Seed × 5-Fold ─────────────────────────────────────────
print("\n[2] LGBM 최적 파라미터 5-Seed × 5-Fold...")
lgbm_oof  = np.zeros(len(X))
lgbm_test = np.zeros(len(X_test))

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
    s_auc = roc_auc_score(y, this_oof)
    print(f"  Seed {seed:>5}: {s_auc:.5f}")
    lgbm_oof  += this_oof  / len(SEEDS)
    lgbm_test += this_test / len(SEEDS)

lgbm_auc = roc_auc_score(y, lgbm_oof)
print(f"  LGBM200 앙상블 OOF: {lgbm_auc:.5f}  ({lgbm_auc - V8_BASELINE:+.5f} vs v8)")
np.save(f"oof_lgbm200_auc_{lgbm_auc:.5f}.npy",  lgbm_oof)
np.save(f"test_lgbm200_auc_{lgbm_auc:.5f}.npy", lgbm_test)

# ─── v8과 블렌딩 ─────────────────────────────────────────────
print("\n[3] v8 + LGBM200 블렌딩...")
v8_oof_f  = sorted(glob.glob("oof_v8_auc_*.npy"))
v8_test_f = sorted(glob.glob("test_v8_auc_*.npy"))

if v8_oof_f:
    v8_oof  = np.load(v8_oof_f[-1])
    v8_test = np.load(v8_test_f[-1])
    v8_auc  = roc_auc_score(y, v8_oof)

    best_blend_auc = 0
    best_w = (0.5, 0.5)
    for w in np.arange(0.3, 0.85, 0.05):
        b = w*rank_norm(lgbm_oof) + (1-w)*rank_norm(v8_oof)
        a = roc_auc_score(y, b)
        if a > best_blend_auc:
            best_blend_auc  = a
            best_w          = (round(w,2), round(1-w,2))
            best_blend_oof  = b
            best_blend_test = w*rank_norm(lgbm_test) + (1-w)*rank_norm(v8_test)

    print(f"  lgbm200={best_w[0]}, v8={best_w[1]} → {best_blend_auc:.5f}  ({best_blend_auc-V8_BASELINE:+.5f})")

    # 전체 모델 로드 후 대앙상블
    cb_oof_f  = sorted(glob.glob("oof_cb5seed_auc_*.npy"))
    v5_oof_f  = sorted(glob.glob("oof_v5_auc_*.npy"))
    xgbt_oof_f = sorted(glob.glob("oof_xgbt_auc_*.npy"))
    cb_oof_f_t  = sorted(glob.glob("oof_cbt_auc_*.npy"))

    all_oofs  = {"lgbm200": lgbm_oof, "v8": v8_oof}
    all_tests = {"lgbm200": lgbm_test, "v8": v8_test}
    if v5_oof_f:
        v5_oof  = np.load(v5_oof_f[-1]); v5_test = np.load(sorted(glob.glob("test_v5_auc_*.npy"))[-1])
        all_oofs["v5"] = v5_oof; all_tests["v5"] = v5_test
    if cb_oof_f:
        cb_oof  = np.load(cb_oof_f[-1]); cb_test = np.load(sorted(glob.glob("test_cb5seed_auc_*.npy"))[-1])
        all_oofs["cb"] = cb_oof; all_tests["cb"] = cb_test

    # 그리드 탐색 (lgbm200 포함)
    grand_best_auc  = best_blend_auc
    grand_best_oof  = best_blend_oof
    grand_best_test = best_blend_test
    grand_best_desc = f"lgbm200={best_w[0]},v8={best_w[1]}"

    from itertools import product as iproduct
    model_names = list(all_oofs.keys())
    n = len(model_names)

    for ws_raw in iproduct(*[np.arange(0.05, 0.7, 0.05)]*n):
        total = sum(ws_raw)
        if total < 0.9 or total > 1.1:
            continue
        ws = np.array(ws_raw) / total
        b  = sum(w*rank_norm(all_oofs[name]) for w, name in zip(ws, model_names))
        a  = roc_auc_score(y, b)
        if a > grand_best_auc:
            grand_best_auc  = a
            grand_best_oof  = b
            grand_best_test = sum(w*rank_norm(all_tests[name]) for w, name in zip(ws, model_names))
            grand_best_desc = " ".join(f"{name}={w:.2f}" for name, w in zip(model_names, ws))

    print(f"\n  대앙상블 최적: {grand_best_desc}")
    print(f"  대앙상블 OOF:  {grand_best_auc:.5f}  ({grand_best_auc - V8_BASELINE:+.5f} vs v8)")

    final_best = grand_best_auc
    final_preds = grand_best_test
    final_oof_  = grand_best_oof
else:
    final_best = lgbm_auc
    final_preds = lgbm_test
    final_oof_  = lgbm_oof

print(f"\n{'='*65}")
print("최종 결과")
print(f"{'='*65}")
print(f"  LGBM200 단독: {lgbm_auc:.5f}")
if v8_oof_f:
    print(f"  최적 대앙상블: {grand_best_auc:.5f}  ({grand_best_auc - V8_BASELINE:+.5f} vs v8)")

sub = pd.read_csv("./data/sample_submission.csv")
sub["probability"] = final_preds
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
filename  = f"submission_{timestamp}_v13lgbm200_oof{final_best:.5f}.csv"
sub.to_csv(filename, index=False)
print(f"\n  저장: {filename}")

np.save(f"oof_v13_auc_{final_best:.5f}.npy",  final_oof_)
np.save(f"test_v13_auc_{final_best:.5f}.npy", final_preds)
