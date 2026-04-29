"""
CatBoost Optuna 튜닝 + 최종 4원 앙상블
v8 기준선: 0.74079
목표: CB OOF 0.741+ → 4원 블렌딩 개선
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

def cb_preprocess(X, X_test):
    """CatBoost 전용: category/object → str, NaN → 'missing'"""
    X_cb      = X.copy()
    X_test_cb = X_test.copy()
    for col in X_cb.select_dtypes("category").columns:
        X_cb[col]      = X_cb[col].astype(object).fillna("missing").astype(str)
        X_test_cb[col] = X_test_cb[col].astype(object).fillna("missing").astype(str)
    for col in X_cb.select_dtypes(object).columns:
        X_cb[col]      = X_cb[col].fillna("missing").astype(str)
        X_test_cb[col] = X_test_cb[col].fillna("missing").astype(str)
    return X_cb, X_test_cb

print("=" * 65)
print("CatBoost Optuna 튜닝 + 4원 대앙상블")
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

X_cb, X_test_cb = cb_preprocess(X, X_test)
cat_feat_names = list(X_cb.select_dtypes(object).columns)
print(f"  카테고리 피처: {len(cat_feat_names)}개")

# ─── CatBoost Optuna 튜닝 ────────────────────────────────────────
print("\n[1] CatBoost Optuna 튜닝 (50 trials × 3-fold)...")
from catboost import CatBoostClassifier

def cb_objective(trial):
    params = dict(
        iterations            = 3000,
        learning_rate         = trial.suggest_float("lr", 0.02, 0.1, log=True),
        depth                 = trial.suggest_int("depth", 4, 8),
        l2_leaf_reg           = trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        random_strength       = trial.suggest_float("random_strength", 0.1, 5.0),
        bagging_temperature   = trial.suggest_float("bagging_temperature", 0.0, 1.0),
        border_count          = trial.suggest_int("border_count", 32, 128),
        eval_metric           = "AUC",
        early_stopping_rounds = 50,
        verbose               = 0,
        thread_count          = -1,
        random_seed           = SEED,
    )
    skf3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    aucs = []
    for tr_i, va_i in skf3.split(X_cb, y):
        m = CatBoostClassifier(**params)
        m.fit(
            X_cb.iloc[tr_i], y.iloc[tr_i],
            eval_set=(X_cb.iloc[va_i], y.iloc[va_i]),
            cat_features=cat_feat_names,
        )
        aucs.append(roc_auc_score(y.iloc[va_i], m.predict_proba(X_cb.iloc[va_i])[:, 1]))
    return np.mean(aucs)

study_cb = optuna.create_study(direction="maximize")
study_cb.optimize(cb_objective, n_trials=50, show_progress_bar=True)
print(f"  CB Optuna best 3-fold AUC: {study_cb.best_value:.5f}")
print(f"  Best params: {study_cb.best_params}")

CB_TUNED = dict(
    iterations            = 3000,
    learning_rate         = study_cb.best_params["lr"],
    depth                 = study_cb.best_params["depth"],
    l2_leaf_reg           = study_cb.best_params["l2_leaf_reg"],
    random_strength       = study_cb.best_params["random_strength"],
    bagging_temperature   = study_cb.best_params["bagging_temperature"],
    border_count          = study_cb.best_params["border_count"],
    eval_metric           = "AUC",
    early_stopping_rounds = 50,
    verbose               = 0,
    thread_count          = -1,
)

# ─── CatBoost Tuned 5-Seed × 5-Fold ────────────────────────────
print("\n[2] CatBoost Tuned (5-Seed × 5-Fold)...")
cbt_oof  = np.zeros(len(X_cb))
cbt_test = np.zeros(len(X_test_cb))

for seed in SEEDS:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    this_oof  = np.zeros(len(X_cb))
    this_test = np.zeros(len(X_test_cb))
    for fold, (tr_i, va_i) in enumerate(skf.split(X_cb, y), 1):
        m = CatBoostClassifier(**{**CB_TUNED, "random_seed": seed})
        m.fit(
            X_cb.iloc[tr_i], y.iloc[tr_i],
            eval_set=(X_cb.iloc[va_i], y.iloc[va_i]),
            cat_features=cat_feat_names,
        )
        this_oof[va_i] = m.predict_proba(X_cb.iloc[va_i])[:, 1]
        this_test     += m.predict_proba(X_test_cb)[:, 1] / N_FOLDS
    s_auc = roc_auc_score(y, this_oof)
    print(f"  CBt Seed {seed:>5}: {s_auc:.5f}")
    cbt_oof  += this_oof  / len(SEEDS)
    cbt_test += this_test / len(SEEDS)

cbt_auc = roc_auc_score(y, cbt_oof)
print(f"  CB Tuned 앙상블 OOF AUC: {cbt_auc:.5f}")
np.save(f"oof_cbt_auc_{cbt_auc:.5f}.npy",  cbt_oof)
np.save(f"test_cbt_auc_{cbt_auc:.5f}.npy", cbt_test)

# ─── 기존 OOF 로드 ────────────────────────────────────────────────
print("\n[3] 기존 모델 OOF 로드...")
v5_oof_f  = sorted(glob.glob("oof_v5_auc_*.npy"))
v5_test_f = sorted(glob.glob("test_v5_auc_*.npy"))
cb_oof_f  = sorted(glob.glob("oof_cb5seed_auc_*.npy"))
cb_test_f = sorted(glob.glob("test_cb5seed_auc_*.npy"))
v8_oof_f  = sorted(glob.glob("oof_v8_auc_*.npy"))  # v8 best (blend3way OOF)

if not v5_oof_f:
    raise FileNotFoundError("v5 OOF 없음")

v5_oof  = np.load(v5_oof_f[-1])
v5_test = np.load(v5_test_f[-1])
v5_auc  = roc_auc_score(y, v5_oof)

cb_oof  = np.load(cb_oof_f[-1])   # original (untuned) CB
cb_test = np.load(cb_test_f[-1])
cb_auc  = roc_auc_score(y, cb_oof)

print(f"  v5 OOF AUC:         {v5_auc:.5f}")
print(f"  CB orig OOF AUC:    {cb_auc:.5f}")
print(f"  CB Tuned OOF AUC:   {cbt_auc:.5f}")

# XGB Tuned OOF: v8 파일에서 역산 (또는 별도 저장본 탐색)
xgbt_oof_f  = sorted(glob.glob("oof_xgbt_auc_*.npy"))
xgbt_test_f = sorted(glob.glob("test_xgbt_auc_*.npy"))
if xgbt_oof_f:
    xgbt_oof  = np.load(xgbt_oof_f[-1])
    xgbt_test = np.load(xgbt_test_f[-1])
    xgbt_auc  = roc_auc_score(y, xgbt_oof)
    print(f"  XGB Tuned OOF AUC:  {xgbt_auc:.5f}")
    use_xgb = True
else:
    print("  XGB Tuned OOF 없음 — v5+CBt 2원 블렌딩만 시도")
    use_xgb = False

# ─── 최적 블렌딩 그리드 서치 ────────────────────────────────────
print("\n[4] 최적 블렌딩 그리드 서치...")

# 사용 가능한 모델 리스트 구성
models_oof  = {"v5": v5_oof,  "cb_orig": cb_oof,  "cbt": cbt_oof}
models_test = {"v5": v5_test, "cb_orig": cb_test, "cbt": cbt_test}
if use_xgb:
    models_oof["xgbt"]  = xgbt_oof
    models_test["xgbt"] = xgbt_test

for name, arr in models_oof.items():
    print(f"    {name:<10}: {roc_auc_score(y, arr):.5f}")

# v5 + CBt 2원 블렌딩
best_2way = 0
best_2w   = (0.7, 0.3)
for w1 in np.arange(0.3, 0.9, 0.05):
    w2 = 1 - w1
    b  = w1*rank_norm(v5_oof) + w2*rank_norm(cbt_oof)
    a  = roc_auc_score(y, b)
    if a > best_2way:
        best_2way = a
        best_2w   = (round(w1, 2), round(w2, 2))
print(f"\n  v5+CBt 2원 최적: v5={best_2w[0]:.2f}, CBt={best_2w[1]:.2f} → {best_2way:.5f}")

# 3원 / 4원 블렌딩
best_blend_auc  = best_2way
best_blend_oof  = best_2w[0]*rank_norm(v5_oof) + best_2w[1]*rank_norm(cbt_oof)
best_blend_test = best_2w[0]*rank_norm(v5_test) + best_2w[1]*rank_norm(cbt_test)
best_blend_name = f"v5={best_2w[0]},CBt={best_2w[1]}"

# 3원: v5 + CBt + cb_orig
for w1 in [0.4, 0.5, 0.6]:
    for w2 in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]:
        for w3 in [0.05, 0.1, 0.15, 0.2]:
            if abs(w1+w2+w3 - 1.0) > 0.001:
                continue
            b = w1*rank_norm(v5_oof) + w2*rank_norm(cbt_oof) + w3*rank_norm(cb_oof)
            a = roc_auc_score(y, b)
            if a > best_blend_auc:
                best_blend_auc  = a
                best_blend_oof  = b
                best_blend_test = w1*rank_norm(v5_test) + w2*rank_norm(cbt_test) + w3*rank_norm(cb_test)
                best_blend_name = f"v5={w1},CBt={w2},CB={w3}"

if use_xgb:
    for w1 in [0.4, 0.45, 0.5, 0.55]:
        for w2 in [0.15, 0.2, 0.25, 0.3, 0.35]:
            for w3 in [0.05, 0.1, 0.15]:
                w4 = round(1 - w1 - w2 - w3, 3)
                if w4 < 0.05 or w4 > 0.4:
                    continue
                b = (w1*rank_norm(v5_oof) + w2*rank_norm(cbt_oof)
                     + w3*rank_norm(cb_oof) + w4*rank_norm(xgbt_oof))
                a = roc_auc_score(y, b)
                if a > best_blend_auc:
                    best_blend_auc  = a
                    best_blend_oof  = b
                    best_blend_test = (w1*rank_norm(v5_test) + w2*rank_norm(cbt_test)
                                       + w3*rank_norm(cb_test) + w4*rank_norm(xgbt_test))
                    best_blend_name = f"v5={w1},CBt={w2},CB={w3},XGBt={w4}"

print(f"  최적 블렌딩: {best_blend_name}")
print(f"  최적 블렌딩 OOF AUC: {best_blend_auc:.5f}")

# ─── 스태킹 ──────────────────────────────────────────────────────
print("\n[5] 스태킹 (LogReg meta)...")
cols_oof  = [v5_oof, cbt_oof, cb_oof]
cols_test = [v5_test, cbt_test, cb_test]
if use_xgb:
    cols_oof.append(xgbt_oof)
    cols_test.append(xgbt_test)
stack_oof_s  = StandardScaler().fit_transform(np.column_stack(cols_oof))
stack_test_s = StandardScaler().fit(np.column_stack(cols_oof)).transform(np.column_stack(cols_test))

scaler = StandardScaler()
stack_oof_s  = scaler.fit_transform(np.column_stack(cols_oof))
stack_test_s = scaler.transform(np.column_stack(cols_test))

meta = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
meta.fit(stack_oof_s, y)
meta_oof  = meta.predict_proba(stack_oof_s)[:, 1]
meta_test = meta.predict_proba(stack_test_s)[:, 1]
meta_auc  = roc_auc_score(y, meta_oof)
print(f"  스태킹 OOF AUC: {meta_auc:.5f}")

# ─── 최종 결과 ──────────────────────────────────────────────────
all_results = {
    "v5_alone":    v5_auc,
    "cb_orig":     cb_auc,
    "cbt_alone":   cbt_auc,
    "blend_best":  best_blend_auc,
    "stacking":    meta_auc,
}

print("\n" + "=" * 65)
print("최종 결과")
print("=" * 65)
best_auc = max(all_results.values())
for name, auc in sorted(all_results.items(), key=lambda x: -x[1]):
    star = " ← 최고!" if auc == best_auc else ""
    print(f"  {name:<14}: {auc:.5f}  ({auc - V8_BASELINE:+.5f} vs v8){star}")
print(f"  최적 블렌드 구성: {best_blend_name}")

# 최선 예측 선택
if best_auc == meta_auc:
    final_preds, final_oof = meta_test, meta_oof
elif best_auc == best_blend_auc:
    final_preds, final_oof = best_blend_test, best_blend_oof
elif best_auc == cbt_auc:
    final_preds, final_oof = cbt_test, cbt_oof
else:
    final_preds, final_oof = v5_test, v5_oof

sub = pd.read_csv("./data/sample_submission.csv")
sub["probability"] = final_preds
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
filename  = f"submission_{timestamp}_v9_oof{best_auc:.5f}.csv"
sub.to_csv(filename, index=False)
print(f"\n  저장: {filename}")
print(f"  v4 기준선(0.74033) 대비: {best_auc - 0.74033:+.5f}")
print(f"  v8 기준선(0.74079) 대비: {best_auc - V8_BASELINE:+.5f}")

np.save(f"oof_v9_auc_{best_auc:.5f}.npy",  final_oof)
np.save(f"test_v9_auc_{best_auc:.5f}.npy", final_preds)
