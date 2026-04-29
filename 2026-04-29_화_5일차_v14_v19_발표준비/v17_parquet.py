"""
v17 - 파이프라인 parquet 피처셋 + is_elective_SET 추가
- train/test_features.parquet (기존 엔지니어링 피처 80개 — LGB 0.740+ 보장)
- is_elective_SET 신규 추가 (38.0% vs 21.9%, parquet에 없는 순수 eSET 신호)
- LGB 5시드 + XGB 2시드 (enable_categorical=True)
- OOF 가중치 최적화 후 기존 모델(ag_raw_10h)과 최종 블렌딩
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from scipy.optimize import minimize
import glob, os

# =========================
# 1. 데이터 로드
# =========================
BASE = os.path.dirname(os.path.abspath(__file__))
NPY_DIR = os.path.join(BASE, "이번대회자료")

train = pd.read_parquet(os.path.join(NPY_DIR, "train_features.parquet"))
test  = pd.read_parquet(os.path.join(NPY_DIR, "test_features.parquet"))
target = "임신 성공 여부"

print(f"Train: {train.shape},  Test: {test.shape}")

# =========================
# 2. is_elective_SET 추가 (parquet에 없는 새 피처)
# =========================
for df in [train, test]:
    if "이식된 배아 수" in df.columns and "총 생성 배아 수" in df.columns:
        emb_x = df["이식된 배아 수"].fillna(0)
        emb_t = df["총 생성 배아 수"].fillna(0)
        df["is_elective_SET"] = ((emb_x == 1) & (emb_t > 1)).astype(np.int8)

# =========================
# 3. 피처/타겟 분리
# =========================
features = [c for c in train.columns if c not in [target, "ID"]]
# test에 없는 컬럼 보정
for f in features:
    if f not in test.columns:
        test[f] = 0

X      = train[features]
y      = train[target]
X_test = test[features]

# object 컬럼 → pd.Categorical (LGB용)
obj_cols = X.select_dtypes(include="object").columns.tolist()
for col in obj_cols:
    X[col]      = pd.Categorical(X[col])
    X_test[col] = pd.Categorical(X_test[col])

print(f"피처 수: {len(features)}")
print(f"dtypes: { {str(k):int(v) for k,v in X.dtypes.value_counts().items()} }")

# XGB용 — Categorical → int codes (enable_categorical 없이도 동작)
def to_xgb(df):
    df = df.copy()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = pd.Categorical(df[col]).codes.astype(np.int16)
    return df

X_xgb      = to_xgb(X)
X_xgb_test = to_xgb(X_test)

# =========================
# 4. 모델 정의
# =========================
LGB_COMMON = dict(n_jobs=-1, verbose=-1)
SEEDS_LGB  = [42, 2024, 777, 31415, 99999]

models = []

# LGB base (leaves=63) × 3시드
for s in SEEDS_LGB[:3]:
    models.append((f"LGB63_s{s}",
                   LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=63,
                                  subsample=0.8, colsample_bytree=0.8,
                                  random_state=s, **LGB_COMMON),
                   "lgb"))

# LGB deep (leaves=95) × 2시드
for s in SEEDS_LGB[:2]:
    models.append((f"LGB95_s{s}",
                   LGBMClassifier(n_estimators=2000, learning_rate=0.025, num_leaves=95,
                                  min_child_samples=150, subsample=0.8, colsample_bytree=0.8,
                                  random_state=s, **LGB_COMMON),
                   "lgb"))

# XGB × 2시드
for s in [42, 2024]:
    models.append((f"XGB_s{s}",
                   XGBClassifier(n_estimators=2000, learning_rate=0.03, max_depth=7,
                                 min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
                                 eval_metric="auc", random_state=s, n_jobs=-1, verbosity=0,
                                 early_stopping_rounds=150),
                   "xgb"))

# =========================
# 5. CV + OOF
# =========================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
n_models   = len(models)
oof_preds  = np.zeros((len(y), n_models))
test_preds = np.zeros((len(X_test), n_models))

for m_idx, (name, model, mtype) in enumerate(models):
    print(f"\n{'='*40}")
    print(f"[{m_idx+1}/{n_models}] {name}")

    is_lgb = (mtype == "lgb")
    Xtr_all = X      if is_lgb else X_xgb
    Xte_all = X_test if is_lgb else X_xgb_test

    fold_test = np.zeros(len(Xte_all))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(Xtr_all, y)):
        Xtr, Xval = Xtr_all.iloc[tr_idx], Xtr_all.iloc[val_idx]
        ytr, yval = y.iloc[tr_idx],        y.iloc[val_idx]

        if is_lgb:
            model.fit(Xtr, ytr,
                      eval_set=[(Xval, yval)],
                      eval_metric="auc",
                      callbacks=[])
        else:
            model.fit(Xtr, ytr,
                      eval_set=[(Xval, yval)],
                      verbose=False)

        vp = model.predict_proba(Xval)[:, 1]
        oof_preds[val_idx, m_idx] = vp
        fold_test += model.predict_proba(Xte_all)[:, 1] / skf.n_splits
        print(f"  Fold {fold}: {roc_auc_score(yval, vp):.5f}")

    test_preds[:, m_idx] = fold_test
    print(f"  ── OOF AUC: {roc_auc_score(y, oof_preds[:, m_idx]):.5f}")

# =========================
# 6. 가중치 최적화
# =========================
print("\n가중치 최적화 중...")

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
opt_auc = max(-best_val, simple_auc)
if -best_val < simple_auc:
    opt_w = np.ones(n_models) / n_models

print(f"최적 OOF AUC: {opt_auc:.5f}")
print("가중치:", {nm: round(float(ww), 4) for nm, ww in zip([m[0] for m in models], opt_w)})

# =========================
# 7. 기존 최고 모델과 재블렌딩
# =========================
prev_models = {
    "ag_raw_10h": ("oof_ag_raw_10h_auc_0.74126.npy", "test_ag_raw_10h_auc_0.74126.npy"),
    "ag8h":       ("oof_ag8h_auc_0.74114.npy",        "test_ag8h_auc_0.74114.npy"),
    "pseudo_v3":  ("oof_pseudo_v3_auc_0.74080.npy",   "test_pseudo_v3_auc_0.74080.npy"),
}

oof_pool  = {"v17": (oof_preds * opt_w).sum(axis=1)}
test_pool = {"v17": (test_preds * opt_w).sum(axis=1)}

for name, (oof_f, test_f) in prev_models.items():
    fp = os.path.join(NPY_DIR, oof_f)
    tp = os.path.join(NPY_DIR, test_f)
    if os.path.exists(fp):
        oof_pool[name]  = np.load(fp)
        test_pool[name] = np.load(tp)
        print(f"  로드: {name} OOF {roc_auc_score(y, oof_pool[name]):.5f}")

nm_list  = list(oof_pool.keys())
oof_mat  = np.stack([oof_pool[n]  for n in nm_list], axis=1)
test_mat = np.stack([test_pool[n] for n in nm_list], axis=1)
n2       = len(nm_list)

def neg_auc_b(w):
    w = np.clip(np.array(w), 0, 1)
    s = w.sum()
    if s < 1e-9: return 0.0
    return -roc_auc_score(y, (oof_mat * (w/s)).sum(axis=1))

b_simple = roc_auc_score(y, oof_mat.mean(axis=1))
print(f"\n블렌딩 단순평균: {b_simple:.5f}")

init2 = [np.ones(n2)/n2]
for i in range(n2):
    w=np.zeros(n2); w[i]=1.0; init2.append(w)
for sd in [0,7,42,99]:
    init2.append(np.random.RandomState(sd).dirichlet(np.ones(n2)))

best_b, best_bv = np.ones(n2)/n2, neg_auc_b(np.ones(n2)/n2)
for w0 in init2:
    res = minimize(neg_auc_b, w0, method="Nelder-Mead",
                   options={"maxiter":20000,"xatol":1e-8,"fatol":1e-9})
    if res.fun < best_bv:
        best_bv, best_b = res.fun, res.x

opt_b = np.clip(best_b, 0, 1); opt_b /= opt_b.sum()
blend_auc = max(-best_bv, b_simple)
print(f"블렌딩 최적 OOF: {blend_auc:.5f}")
print("블렌딩 가중치:", {n:round(float(w),4) for n,w in zip(nm_list,opt_b)})

# =========================
# 8. 제출 파일 저장
# =========================
ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M")

# (A) v17 단독
final_v17 = (test_preds * opt_w).sum(axis=1)
rank_v17  = pd.Series(final_v17).rank(pct=True).values
sub_v17   = pd.DataFrame({"ID": test["ID"],
                           "probability": 0.5*final_v17 + 0.5*rank_v17})
sub_v17.to_csv(os.path.join(NPY_DIR, f"v17_alone_oof{opt_auc:.5f}_{ts}.csv"), index=False)

# (B) 최종 블렌딩
final_blend = (test_mat * opt_b).sum(axis=1)
rank_blend  = pd.Series(final_blend).rank(pct=True).values
sub_blend   = pd.DataFrame({"ID": test["ID"],
                             "probability": 0.5*final_blend + 0.5*rank_blend})
sub_blend.to_csv(os.path.join(NPY_DIR, f"v17_blend_oof{blend_auc:.5f}_{ts}.csv"), index=False)

# numpy 저장 (추가 블렌딩용)
np.save(os.path.join(NPY_DIR, f"oof_v17_auc_{opt_auc:.5f}.npy"), oof_pool["v17"])
np.save(os.path.join(NPY_DIR, f"test_v17_auc_{opt_auc:.5f}.npy"), test_pool["v17"])

print(f"\n{'='*55}")
print(f"[A] v17 단독  : v17_alone_oof{opt_auc:.5f}_{ts}.csv")
print(f"[B] 최종블렌딩: v17_blend_oof{blend_auc:.5f}_{ts}.csv  ← 제출 권장")
print(f"{'='*55}")
