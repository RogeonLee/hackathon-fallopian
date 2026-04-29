"""
MLP Neural Network 앙상블
v8 기준선: 0.74079
sklearn MLP + 다양한 아키텍처 × 5-seed
"""
import glob, warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from scipy.stats import rankdata

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

def prepare_for_mlp(train, test):
    """MLP용: 카테고리 → ordinal encoding, NaN → -1, StandardScaler"""
    X      = train.drop(columns=[ID_COL, TARGET])
    y      = train[TARGET].copy()
    X_test = test.drop(columns=[ID_COL])

    # object/category → ordinal
    cat_cols = [c for c in X.select_dtypes(["object", "category"]).columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Ordinal encode
    X_cat      = X[cat_cols].astype(str).fillna("missing")
    X_test_cat = X_test[cat_cols].astype(str).fillna("missing") if cat_cols else pd.DataFrame()

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if cat_cols:
        X_cat_arr      = enc.fit_transform(X_cat)
        X_test_cat_arr = enc.transform(X_test_cat)
    else:
        X_cat_arr = X_test_cat_arr = np.empty((len(X), 0))

    X_num      = X[num_cols].fillna(-1).values
    X_test_num = X_test[num_cols].fillna(-1).values

    X_all      = np.hstack([X_num, X_cat_arr])
    X_test_all = np.hstack([X_test_num, X_test_cat_arr])

    return X_all, X_test_all, y

print("=" * 65)
print("MLP Neural Network 앙상블")
print(f"v8 기준선: {V8_BASELINE}")
print("=" * 65)

train_raw = pd.read_csv("./data/train.csv")
test_raw  = pd.read_csv("./data/test.csv")
train = preprocess(train_raw)
test  = preprocess(test_raw)

# 카테고리 정렬 (인코딩 일관성)
obj_cols = [c for c in train.select_dtypes("object").columns if c not in [ID_COL, TARGET]]
for col in obj_cols:
    all_vals = sorted(set(train[col].dropna()) | set(test[col].dropna()), key=str)
    train[col] = pd.Categorical(train[col], categories=all_vals)
    test[col]  = pd.Categorical(test[col],  categories=all_vals)

X_all, X_test_all, y = prepare_for_mlp(train, test)
print(f"  피처 수: {X_all.shape[1]}")

# MLP 아키텍처 3종
ARCHS = {
    "mlp_wide":   dict(hidden_layer_sizes=(512, 256, 128), alpha=1e-3, learning_rate_init=1e-3),
    "mlp_deep":   dict(hidden_layer_sizes=(256, 128, 64, 32), alpha=1e-3, learning_rate_init=1e-3),
    "mlp_res":    dict(hidden_layer_sizes=(256, 256, 128), alpha=1e-4, learning_rate_init=5e-4),
}

mlp_oofs  = {}
mlp_tests = {}

for arch_name, arch_params in ARCHS.items():
    print(f"\n[{arch_name}] 5-Seed × 5-Fold...")
    oof_agg  = np.zeros(len(X_all))
    test_agg = np.zeros(len(X_test_all))

    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        this_oof  = np.zeros(len(X_all))
        this_test = np.zeros(len(X_test_all))

        for fold, (tr_i, va_i) in enumerate(skf.split(X_all, y), 1):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_all[tr_i])
            X_va = scaler.transform(X_all[va_i])
            X_te = scaler.transform(X_test_all)

            m = MLPClassifier(
                **arch_params,
                activation       = "relu",
                solver           = "adam",
                batch_size       = 1024,
                max_iter         = 200,
                early_stopping   = True,
                validation_fraction = 0.1,
                n_iter_no_change = 15,
                random_state     = seed,
            )
            m.fit(X_tr, y.iloc[tr_i])
            this_oof[va_i] = m.predict_proba(X_va)[:, 1]
            this_test     += m.predict_proba(X_te)[:, 1] / N_FOLDS

        s_auc = roc_auc_score(y, this_oof)
        print(f"  Seed {seed:>5}: {s_auc:.5f}")
        oof_agg  += this_oof  / len(SEEDS)
        test_agg += this_test / len(SEEDS)

    auc = roc_auc_score(y, oof_agg)
    print(f"  {arch_name} 앙상블 OOF: {auc:.5f}  ({auc - V8_BASELINE:+.5f} vs v8)")
    mlp_oofs[arch_name]  = oof_agg
    mlp_tests[arch_name] = test_agg
    np.save(f"oof_{arch_name}_auc_{auc:.5f}.npy",  oof_agg)
    np.save(f"test_{arch_name}_auc_{auc:.5f}.npy", test_agg)

# ─── MLP 앙상블 + v8 블렌딩 ────────────────────────────────────
print("\n[MLP 앙상블 블렌딩]")
mlp_ensemble_oof  = np.mean([rank_norm(v) for v in mlp_oofs.values()], axis=0)
mlp_ensemble_test = np.mean([rank_norm(v) for v in mlp_tests.values()], axis=0)
mlp_auc = roc_auc_score(y, mlp_ensemble_oof)
print(f"  MLP 3-arch 앙상블 OOF: {mlp_auc:.5f}")

# v8 OOF 로드 후 블렌딩
v8_oof_f  = sorted(glob.glob("oof_v8_auc_*.npy"))
v8_test_f = sorted(glob.glob("test_v8_auc_*.npy"))

if v8_oof_f:
    v8_oof  = np.load(v8_oof_f[-1])
    v8_test = np.load(v8_test_f[-1])
    v8_auc  = roc_auc_score(y, v8_oof)
    print(f"  v8 OOF: {v8_auc:.5f}")

    best_blend_auc  = 0
    best_w          = (0.9, 0.1)
    for w in np.arange(0.5, 0.95, 0.05):
        b = w*rank_norm(v8_oof) + (1-w)*rank_norm(mlp_ensemble_oof)
        a = roc_auc_score(y, b)
        if a > best_blend_auc:
            best_blend_auc  = a
            best_w          = (round(w,2), round(1-w,2))
            best_blend_oof  = b
            best_blend_test = w*rank_norm(v8_test) + (1-w)*mlp_ensemble_test

    print(f"  최적 v8+MLP 블렌딩: v8={best_w[0]}, MLP={best_w[1]} → {best_blend_auc:.5f}  ({best_blend_auc-V8_BASELINE:+.5f})")
else:
    best_blend_auc = mlp_auc

print(f"\n{'='*65}")
print("최종 결과")
print(f"{'='*65}")
for name, oof in mlp_oofs.items():
    print(f"  {name:<14}: {roc_auc_score(y, oof):.5f}")
print(f"  mlp_ensemble  : {mlp_auc:.5f}")
if v8_oof_f:
    print(f"  v8+MLP blend  : {best_blend_auc:.5f}  ({best_blend_auc-V8_BASELINE:+.5f} vs v8)")

final_best = best_blend_auc if v8_oof_f else mlp_auc
final_preds = best_blend_test if v8_oof_f else mlp_ensemble_test

if final_best > V8_BASELINE:
    sub = pd.read_csv("./data/sample_submission.csv")
    sub["probability"] = final_preds
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename  = f"submission_{timestamp}_v12mlp_oof{final_best:.5f}.csv"
    sub.to_csv(filename, index=False)
    print(f"\n  저장: {filename}")
    print(f"  v8 대비: {final_best - V8_BASELINE:+.5f}")
else:
    print("\n  MLP 블렌딩이 v8을 넘지 못함")
