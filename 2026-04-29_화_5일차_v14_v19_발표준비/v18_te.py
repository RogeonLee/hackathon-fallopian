"""
v18 - parquet + OOF Target Encoding + Early Stopping
  TE 대상: 시술 시기 코드(7값) + 특정 시술 유형(24값)
  → 기존 te_lgbm 0.74007 → 여기서 parquet 피처까지 추가로 0.741+ 목표
  → LGB early_stopping(120), XGB early_stopping_rounds=150
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from xgboost import XGBClassifier
from scipy.optimize import minimize
import os

BASE    = os.path.dirname(os.path.abspath(__file__))
NPY_DIR = os.path.join(BASE, "이번대회자료")
DATA_DIR = os.path.join(NPY_DIR, "data")

# =========================
# 1. 데이터 로드
# =========================
train_par = pd.read_parquet(os.path.join(NPY_DIR, "train_features.parquet"))
test_par  = pd.read_parquet(os.path.join(NPY_DIR, "test_features.parquet"))

# TE용 원본 컬럼 (raw CSV에서만 존재)
TE_SRC_COLS = ["ID", "시술 시기 코드", "특정 시술 유형"]
train_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), usecols=TE_SRC_COLS)
test_raw  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),  usecols=TE_SRC_COLS)

target = "임신 성공 여부"

train = train_par.merge(train_raw, on="ID", how="left")
test  = test_par.merge(test_raw,  on="ID", how="left")
print(f"병합 후 Train: {train.shape},  Test: {test.shape}")

# =========================
# 2. is_elective_SET 추가
# =========================
for df in [train, test]:
    emb_x = df["이식된 배아 수"].fillna(0)
    emb_t = df["총 생성 배아 수"].fillna(0)
    df["is_elective_SET"] = ((emb_x == 1) & (emb_t > 1)).astype(np.int8)

# =========================
# 3. 피처 분류
# =========================
TE_COLS   = ["시술 시기 코드", "특정 시술 유형"]   # TE 적용 컬럼 (CV 내부 처리)
BASE_FEATS = [c for c in train.columns
              if c not in [target, "ID"] + TE_COLS]

# object → pd.Categorical (LGB용)
for col in train[BASE_FEATS].select_dtypes(include="object").columns:
    train[col] = pd.Categorical(train[col])
    test[col]  = pd.Categorical(test[col])

y = train[target]
print(f"기본 피처 수: {len(BASE_FEATS)},  TE 컬럼: {TE_COLS}")

# =========================
# 4. 모델 정의
# =========================
LGB_COMMON = dict(n_jobs=-1, verbose=-1)
SEEDS = [42, 2024, 777, 31415, 99999]

models = []
for s in SEEDS[:3]:
    models.append((f"LGB63_s{s}",
                   LGBMClassifier(n_estimators=5000, learning_rate=0.03, num_leaves=63,
                                  subsample=0.8, colsample_bytree=0.8,
                                  random_state=s, **LGB_COMMON), "lgb"))
for s in SEEDS[:2]:
    models.append((f"LGB95_s{s}",
                   LGBMClassifier(n_estimators=5000, learning_rate=0.025, num_leaves=95,
                                  min_child_samples=150, subsample=0.8, colsample_bytree=0.8,
                                  random_state=s, **LGB_COMMON), "lgb"))
for s in [42, 2024]:
    models.append((f"XGB_s{s}",
                   XGBClassifier(n_estimators=2000, learning_rate=0.03, max_depth=7,
                                 min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
                                 eval_metric="auc", random_state=s, n_jobs=-1, verbosity=0,
                                 early_stopping_rounds=150), "xgb"))

# =========================
# 5. CV + OOF TE + 학습
# =========================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
n_models   = len(models)
oof_preds  = np.zeros((len(train), n_models))
test_preds = np.zeros((len(test),  n_models))


def apply_te(col_tr_vals, y_tr_vals, col_val_vals, col_te_vals, global_mean):
    """OOF Target Encoding: train fold 기준 TE 맵 생성, val/test에 적용"""
    te_map = {}
    for code, tgt in zip(col_tr_vals, y_tr_vals):
        if code not in te_map:
            te_map[code] = []
        te_map[code].append(tgt)
    te_map = {k: np.mean(v) for k, v in te_map.items()}

    vec = lambda arr: np.array([te_map.get(v, global_mean) for v in arr])
    return vec(col_tr_vals), vec(col_val_vals), vec(col_te_vals)


for m_idx, (name, model, mtype) in enumerate(models):
    print(f"\n{'='*40}\n[{m_idx+1}/{n_models}] {name}")
    fold_test = np.zeros(len(test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(train, y)):
        Xtr  = train[BASE_FEATS].iloc[tr_idx].copy()
        Xval = train[BASE_FEATS].iloc[val_idx].copy()
        Xte  = test[BASE_FEATS].copy()
        ytr  = y.iloc[tr_idx].values
        yval = y.iloc[val_idx]

        global_mean = ytr.mean()

        # OOF TE 적용
        for col in TE_COLS:
            col_tr_arr  = train[col].iloc[tr_idx].astype(str).values
            col_val_arr = train[col].iloc[val_idx].astype(str).values
            col_te_arr  = test[col].astype(str).values

            te_tr, te_val, te_te = apply_te(
                col_tr_arr, ytr, col_val_arr, col_te_arr, global_mean)

            Xtr[col + "_te"]  = te_tr
            Xval[col + "_te"] = te_val
            Xte[col + "_te"]  = te_te

        # XGB: category → int codes
        if mtype == "xgb":
            for df in [Xtr, Xval, Xte]:
                for c in df.select_dtypes(include=["category", "object"]).columns:
                    df[c] = pd.Categorical(df[c]).codes.astype(np.int16)

        # 학습
        if mtype == "lgb":
            model.fit(Xtr, ytr,
                      eval_set=[(Xval, yval)],
                      callbacks=[early_stopping(120, verbose=False),
                                 log_evaluation(300)])
        else:
            model.fit(Xtr, ytr,
                      eval_set=[(Xval, yval)],
                      verbose=False)

        vp = model.predict_proba(Xval)[:, 1]
        oof_preds[val_idx, m_idx] = vp
        fold_test += model.predict_proba(Xte)[:, 1] / skf.n_splits
        print(f"  Fold {fold}: {roc_auc_score(yval, vp):.5f}")

    test_preds[:, m_idx] = fold_test
    oof_auc = roc_auc_score(y, oof_preds[:, m_idx])
    print(f"  ── OOF AUC: {oof_auc:.5f}")

# =========================
# 6. 가중치 최적화
# =========================
print("\n가중치 최적화 중...")

def neg_auc_w(w):
    w = np.clip(np.array(w), 0, 1)
    s = w.sum()
    if s < 1e-9: return 0.0
    return -roc_auc_score(y, (oof_preds * (w/s)).sum(axis=1))

simple_auc = roc_auc_score(y, oof_preds.mean(axis=1))
print(f"단순 평균 OOF AUC: {simple_auc:.5f}")

init_ws = [np.ones(n_models)/n_models]
for i in range(n_models):
    w = np.zeros(n_models); w[i]=1.0; init_ws.append(w)
for sd in [0, 7, 42, 99]:
    init_ws.append(np.random.RandomState(sd).dirichlet(np.ones(n_models)))

best_w, best_val = np.ones(n_models)/n_models, neg_auc_w(np.ones(n_models)/n_models)
for w0 in init_ws:
    res = minimize(neg_auc_w, w0, method="Nelder-Mead",
                   options={"maxiter":20000, "xatol":1e-8, "fatol":1e-9})
    if res.fun < best_val:
        best_val, best_w = res.fun, res.x

opt_w = np.clip(best_w, 0, 1); opt_w /= opt_w.sum()
opt_auc = max(-best_val, simple_auc)
if -best_val < simple_auc:
    opt_w = np.ones(n_models)/n_models
print(f"최적 OOF AUC: {opt_auc:.5f}")
print("가중치:", {nm: round(float(ww),4)
                 for nm, ww in zip([m[0] for m in models], opt_w)})

# =========================
# 7. 기존 최고 모델과 재블렌딩
# =========================
prev = {
    "ag_raw_10h": ("oof_ag_raw_10h_auc_0.74126.npy", "test_ag_raw_10h_auc_0.74126.npy"),
    "ag8h":       ("oof_ag8h_auc_0.74114.npy",        "test_ag8h_auc_0.74114.npy"),
    "pseudo_v3":  ("oof_pseudo_v3_auc_0.74080.npy",   "test_pseudo_v3_auc_0.74080.npy"),
    "v15_xgb":    ("oof_v15_auc_0.74037.npy",          "test_v15_auc_0.74037.npy"),
}

oof_v18  = (oof_preds  * opt_w).sum(axis=1)
test_v18 = (test_preds * opt_w).sum(axis=1)

oof_pool  = {"v18": oof_v18}
test_pool = {"v18": test_v18}

for nm, (of, tf) in prev.items():
    fp = os.path.join(NPY_DIR, of)
    tp = os.path.join(NPY_DIR, tf)
    if os.path.exists(fp):
        oof_pool[nm]  = np.load(fp)
        test_pool[nm] = np.load(tp)
        print(f"  로드: {nm} OOF {roc_auc_score(y, oof_pool[nm]):.5f}")

nm_list = list(oof_pool.keys())
oof_mat  = np.stack([oof_pool[n]  for n in nm_list], axis=1)
test_mat = np.stack([test_pool[n] for n in nm_list], axis=1)
n2 = len(nm_list)

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
print("블렌딩 가중치:", {n: round(float(w),4) for n,w in zip(nm_list, opt_b)})

# =========================
# 8. 제출 파일 저장
# =========================
ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M")

def make_sub(preds, test_df, fname):
    rank_n = pd.Series(preds).rank(pct=True).values
    final  = 0.5 * preds + 0.5 * rank_n
    pd.DataFrame({"ID": test_df["ID"], "probability": final}).to_csv(fname, index=False)
    print(f"저장: {fname}")

make_sub(test_v18,
         test,
         os.path.join(NPY_DIR, f"v18_alone_oof{opt_auc:.5f}_{ts}.csv"))

final_blend = (test_mat * opt_b).sum(axis=1)
make_sub(final_blend,
         test,
         os.path.join(NPY_DIR, f"v18_blend_oof{blend_auc:.5f}_{ts}.csv"))

np.save(os.path.join(NPY_DIR, f"oof_v18_auc_{opt_auc:.5f}.npy"),  oof_v18)
np.save(os.path.join(NPY_DIR, f"test_v18_auc_{opt_auc:.5f}.npy"), test_v18)

print(f"\n{'='*55}")
print(f"[A] v18 단독  : v18_alone_oof{opt_auc:.5f}_{ts}.csv")
print(f"[B] 최종블렌딩: v18_blend_oof{blend_auc:.5f}_{ts}.csv  ← 제출 권장")
print(f"{'='*55}")
