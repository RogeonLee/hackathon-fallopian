"""
LightGBM DART + GOSS 추가 다양화 앙상블
v8 기준선: 0.74079
- DART boosting: 드롭아웃으로 오버피팅 방지, 다양한 예측 생성
- GOSS: 경사도 기반 샘플링으로 다른 편향 학습
- v5 OOF(LGBM+XGB 스태킹) + CB + XGBt + DART + GOSS 5원 블렌딩
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
print("LightGBM DART + GOSS 다양화 앙상블")
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

def run_lgbm_multiseed(params, label, seeds=SEEDS):
    oof  = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    for seed in seeds:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        this_oof  = np.zeros(len(X))
        this_test = np.zeros(len(X_test))
        for fold, (tr_i, va_i) in enumerate(skf.split(X, y), 1):
            m = LGBMClassifier(**{**params, "random_state": seed})
            m.fit(
                X.iloc[tr_i], y.iloc[tr_i],
                eval_set=[(X.iloc[va_i], y.iloc[va_i])],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
            )
            this_oof[va_i] = m.predict_proba(X.iloc[va_i])[:, 1]
            this_test     += m.predict_proba(X_test)[:, 1] / N_FOLDS
        s_auc = roc_auc_score(y, this_oof)
        print(f"  {label} Seed {seed:>5}: {s_auc:.5f}")
        oof        += this_oof  / len(seeds)
        test_preds += this_test / len(seeds)
    auc = roc_auc_score(y, oof)
    print(f"  {label} 앙상블 OOF AUC: {auc:.5f}")
    return oof, test_preds, auc

# ─── [1] DART boosting ──────────────────────────────────────────
print("\n[1] LGBM DART (5-Seed × 5-Fold)...")
DART_PARAMS = dict(
    n_estimators      = 1000,
    learning_rate     = 0.04,
    num_leaves        = 170,
    max_depth         = 4,
    min_child_samples = 30,
    subsample         = 0.808,
    colsample_bytree  = 0.526,
    reg_alpha         = 8.22,
    reg_lambda        = 0.18,
    boosting_type     = "dart",
    drop_rate         = 0.1,
    skip_drop         = 0.5,
    n_jobs            = -1,
    verbose           = -1,
)
dart_oof, dart_test, dart_auc = run_lgbm_multiseed(DART_PARAMS, "DART")
np.save(f"oof_dart_auc_{dart_auc:.5f}.npy",  dart_oof)
np.save(f"test_dart_auc_{dart_auc:.5f}.npy", dart_test)

# ─── [2] GOSS boosting ──────────────────────────────────────────
print("\n[2] LGBM GOSS (5-Seed × 5-Fold)...")
GOSS_PARAMS = dict(
    n_estimators      = 2000,
    learning_rate     = 0.04,
    num_leaves        = 170,
    max_depth         = 4,
    min_child_samples = 30,
    colsample_bytree  = 0.526,
    reg_alpha         = 8.22,
    reg_lambda        = 0.18,
    boosting_type     = "goss",
    top_rate          = 0.2,
    other_rate        = 0.1,
    n_jobs            = -1,
    verbose           = -1,
)
goss_oof, goss_test, goss_auc = run_lgbm_multiseed(GOSS_PARAMS, "GOSS")
np.save(f"oof_goss_auc_{goss_auc:.5f}.npy",  goss_oof)
np.save(f"test_goss_auc_{goss_auc:.5f}.npy", goss_test)

# ─── [3] 기존 OOF 로드 ──────────────────────────────────────────
print("\n[3] 기존 OOF 로드 및 최종 블렌딩...")
v5_oof_f  = sorted(glob.glob("oof_v5_auc_*.npy"))
v5_test_f = sorted(glob.glob("test_v5_auc_*.npy"))
cb_oof_f  = sorted(glob.glob("oof_cb5seed_auc_*.npy"))
cb_test_f = sorted(glob.glob("test_cb5seed_auc_*.npy"))
cbt_oof_f = sorted(glob.glob("oof_cbt_auc_*.npy"))
cbt_test_f = sorted(glob.glob("test_cbt_auc_*.npy"))

v5_oof  = np.load(v5_oof_f[-1]);  v5_test  = np.load(v5_test_f[-1])
cb_oof  = np.load(cb_oof_f[-1]);  cb_test  = np.load(cb_test_f[-1])
v5_auc  = roc_auc_score(y, v5_oof)
cb_auc  = roc_auc_score(y, cb_oof)

models = {"v5": (v5_oof, v5_test, v5_auc),
          "cb":  (cb_oof,  cb_test,  cb_auc),
          "dart":(dart_oof,dart_test,dart_auc),
          "goss":(goss_oof,goss_test,goss_auc)}

if cbt_oof_f:
    cbt_oof  = np.load(cbt_oof_f[-1]);  cbt_test  = np.load(cbt_test_f[-1])
    cbt_auc  = roc_auc_score(y, cbt_oof)
    models["cbt"] = (cbt_oof, cbt_test, cbt_auc)

for name, (oof, _, auc) in models.items():
    print(f"  {name:<8}: {auc:.5f}")

# ─── 그리드 블렌딩 ─────────────────────────────────────────────
best_auc  = 0
best_combo = {}
model_names = list(models.keys())
model_oofs  = [models[n][0] for n in model_names]
model_tests = [models[n][1] for n in model_names]

# 균일 가중치 블렌딩
uniform_oof = sum(rank_norm(o) for o in model_oofs) / len(model_oofs)
uniform_auc = roc_auc_score(y, uniform_oof)
print(f"\n  균일 가중치 블렌딩: {uniform_auc:.5f}")

# v5를 앵커로 한 가중치 탐색 (v5 비중 0.3~0.6)
step = 0.1
from itertools import product as iproduct

best_blend_oof  = uniform_oof
best_blend_test = sum(rank_norm(t) for t in model_tests) / len(model_tests)
best_blend_auc  = uniform_auc
best_weights    = {n: 1/len(model_names) for n in model_names}

# v5 + (나머지 균일 배분)
for w_v5 in np.arange(0.3, 0.65, 0.05):
    rest_names = [n for n in model_names if n != "v5"]
    n_rest = len(rest_names)
    # 나머지 가중치 균일
    w_rest = (1 - w_v5) / n_rest
    oof_b = w_v5 * rank_norm(models["v5"][0])
    tst_b = w_v5 * rank_norm(models["v5"][1])
    for rn in rest_names:
        oof_b += w_rest * rank_norm(models[rn][0])
        tst_b += w_rest * rank_norm(models[rn][1])
    a = roc_auc_score(y, oof_b)
    if a > best_blend_auc:
        best_blend_auc  = a
        best_blend_oof  = oof_b
        best_blend_test = tst_b
        best_weights    = {"v5": round(w_v5,2)}
        best_weights.update({rn: round(w_rest,3) for rn in rest_names})

# 2-3원 서브셋 탐색
from itertools import combinations
for r in [2, 3]:
    for subset in combinations(model_names, r):
        for ws in iproduct(*[np.arange(0.1, 0.9, 0.1)]*r):
            if abs(sum(ws) - 1.0) > 0.05:
                continue
            ws_norm = np.array(ws) / sum(ws)
            oof_b = sum(w*rank_norm(models[n][0]) for n,w in zip(subset, ws_norm))
            a = roc_auc_score(y, oof_b)
            if a > best_blend_auc:
                best_blend_auc  = a
                best_blend_oof  = oof_b
                best_blend_test = sum(w*rank_norm(models[n][1]) for n,w in zip(subset, ws_norm))
                best_weights    = {n: round(float(w),3) for n,w in zip(subset, ws_norm)}

print(f"  최적 블렌딩 OOF AUC: {best_blend_auc:.5f}")
print(f"  최적 가중치: {best_weights}")

# ─── 스태킹 ──────────────────────────────────────────────────────
print("\n[4] 스태킹...")
all_oofs  = np.column_stack([models[n][0] for n in model_names])
all_tests = np.column_stack([models[n][1] for n in model_names])
scaler = StandardScaler()
s_oof  = scaler.fit_transform(all_oofs)
s_test = scaler.transform(all_tests)
meta = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
meta.fit(s_oof, y)
meta_oof  = meta.predict_proba(s_oof)[:, 1]
meta_test = meta.predict_proba(s_test)[:, 1]
meta_auc  = roc_auc_score(y, meta_oof)
print(f"  스태킹 OOF AUC: {meta_auc:.5f}")

# ─── 최종 결과 ──────────────────────────────────────────────────
all_results = {n: roc_auc_score(y, models[n][0]) for n in model_names}
all_results["blend_best"] = best_blend_auc
all_results["stacking"]   = meta_auc

print("\n" + "=" * 65)
print("최종 결과")
print("=" * 65)
final_best = max(all_results.values())
for name, auc in sorted(all_results.items(), key=lambda x: -x[1]):
    star = " ← 최고!" if auc == final_best else ""
    print(f"  {name:<14}: {auc:.5f}  ({auc - V8_BASELINE:+.5f} vs v8){star}")

if final_best == meta_auc:
    final_preds, final_oof_ = meta_test, meta_oof
elif final_best == best_blend_auc:
    final_preds, final_oof_ = best_blend_test, best_blend_oof
else:
    best_name = max(all_results, key=all_results.get)
    final_preds, final_oof_ = models[best_name][1], models[best_name][0]

sub = pd.read_csv("./data/sample_submission.csv")
sub["probability"] = final_preds
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
filename  = f"submission_{timestamp}_v10_oof{final_best:.5f}.csv"
sub.to_csv(filename, index=False)
print(f"\n  저장: {filename}")
print(f"  v4 기준선 대비: {final_best - 0.74033:+.5f}")
print(f"  v8 기준선 대비: {final_best - V8_BASELINE:+.5f}")

np.save(f"oof_v10_auc_{final_best:.5f}.npy",  final_oof_)
np.save(f"test_v10_auc_{final_best:.5f}.npy", final_preds)
