"""
v19: CatBoost + Pseudo Labeling
- parquet 79 피처 + is_elective_SET (총 80 피처)
- 현재 최강 블렌딩(LB 0.74222) 예측으로 pseudo label 생성 (rank 0.97/0.03)
- CatBoost 3시드 × 5-fold, pseudo는 train fold에만 (OOF 깨끗)
- pseudo sample_weight 0.5
- 완료 후 기존 상위 모델들과 NM 블렌딩
"""

import os, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")

BASE    = os.path.dirname(os.path.abspath(__file__))
NPY_DIR = os.path.join(BASE, "이번대회자료")
DATA_DIR = os.path.join(NPY_DIR, "data")

TARGET = "임신 성공 여부"
SEEDS  = [42, 2024, 777]
N_FOLDS = 5
PSEUDO_WEIGHT = 0.5
TH_POS, TH_NEG = 0.97, 0.03

# =========================
# 1. 데이터 로드
# =========================
train = pd.read_parquet(os.path.join(NPY_DIR, "train_features.parquet"))
test  = pd.read_parquet(os.path.join(NPY_DIR, "test_features.parquet"))

print(f"Train: {train.shape},  Test: {test.shape}")

# =========================
# 2. is_elective_SET 추가
# =========================
for df in [train, test]:
    emb_x = df["이식된 배아 수"].fillna(0)
    emb_t = df["총 생성 배아 수"].fillna(0)
    df["is_elective_SET"] = ((emb_x == 1) & (emb_t > 1)).astype(np.int8)

# =========================
# 3. 피처/타겟 분리
# =========================
features = [c for c in train.columns if c not in [TARGET, "ID"]]
for f in features:
    if f not in test.columns:
        test[f] = 0

X      = train[features].copy()
y      = train[TARGET].values.astype(float)
X_test = test[features].copy()

# int8/Int8 → float32 (CatBoost 호환)
for col in X.select_dtypes(include=["Int8", "Int64"]).columns:
    X[col]      = X[col].astype(float)
    X_test[col] = X_test[col].astype(float)

print(f"피처 수: {len(features)}")

# =========================
# 4. Pseudo Label 생성
# =========================
blend_sub = pd.read_csv(os.path.join(NPY_DIR, "blend_NM_oof0.74140_20260429_1337.csv"))
# test와 순서가 같다고 가정 (ID로 align)
test_prob = test[["ID"]].merge(blend_sub, on="ID", how="left")["probability"].values
test_rank = pd.Series(test_prob).rank(pct=True).values

mask_pos = test_rank >= TH_POS
mask_neg = test_rank <= TH_NEG
n_pos, n_neg = mask_pos.sum(), mask_neg.sum()
print(f"\nPseudo label 생성: pos={n_pos} (rank≥{TH_POS}), neg={n_neg} (rank≤{TH_NEG}), 총={n_pos+n_neg}")

# pseudo 피처: test의 동일 피처 슬라이스
X_pseudo = X_test.iloc[mask_pos | mask_neg].copy()
y_pseudo  = np.where((mask_pos | mask_neg)[mask_pos | mask_neg], 0, 0)  # placeholder
y_pseudo  = np.where(mask_pos[mask_pos | mask_neg], 1, 0).astype(float)

n_orig   = len(X)
n_pseudo = len(X_pseudo)
print(f"원본 train {n_orig} + pseudo {n_pseudo} = 총 {n_orig + n_pseudo}")

# 합친 학습 데이터
X_aug = pd.concat([X, X_pseudo], ignore_index=True)
y_aug = np.concatenate([y, y_pseudo])
sw    = np.concatenate([np.ones(n_orig), np.full(n_pseudo, PSEUDO_WEIGHT)])

# =========================
# 5. CatBoost CV + OOF
# =========================
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
pseudo_idx = np.arange(n_orig, n_orig + n_pseudo)

oof_per_seed  = []
test_per_seed = []

for seed_i, seed in enumerate(SEEDS):
    print(f"\n{'='*50}")
    print(f"[Seed {seed_i+1}/{len(SEEDS)}] random_seed={seed}")

    oof_s  = np.zeros(n_orig)
    test_s = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        # train = 원본 fold + pseudo 전부
        tr_full = np.concatenate([tr_idx, pseudo_idx])

        Xtr  = X_aug.iloc[tr_full]
        ytr  = y_aug[tr_full]
        swtr = sw[tr_full]
        Xval = X_aug.iloc[val_idx]   # val은 원본만
        yval = y_aug[val_idx]

        train_pool = Pool(Xtr, label=ytr, weight=swtr)
        val_pool   = Pool(Xval, label=yval)

        cat = CatBoostClassifier(
            iterations=3000,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=3,
            loss_function="Logloss",
            eval_metric="AUC",
            early_stopping_rounds=150,
            random_seed=seed,
            verbose=0,
            thread_count=-1,
        )
        cat.fit(train_pool, eval_set=val_pool, use_best_model=True)

        vp = cat.predict_proba(Xval)[:, 1]
        oof_s[val_idx] = vp
        test_s += cat.predict_proba(X_test)[:, 1] / N_FOLDS

        fold_auc = roc_auc_score(yval, vp)
        print(f"  Fold {fold}: AUC={fold_auc:.5f}  (best_iter={cat.get_best_iteration()})")

    seed_auc = roc_auc_score(y, oof_s)
    print(f"  ── Seed OOF AUC: {seed_auc:.5f}")
    oof_per_seed.append(oof_s)
    test_per_seed.append(test_s)

# =========================
# 6. 시드 앙상블 (단순 평균)
# =========================
oof_mat  = np.stack(oof_per_seed,  axis=1)   # (n_orig, n_seeds)
test_mat = np.stack(test_per_seed, axis=1)

oof_mean  = oof_mat.mean(axis=1)
test_mean = test_mat.mean(axis=1)
auc_mean  = roc_auc_score(y, oof_mean)
print(f"\n3-seed 평균 OOF AUC: {auc_mean:.5f}")

# Nelder-Mead seed 가중치
def neg_auc_s(w):
    w = np.clip(w, 0, 1); s = w.sum()
    if s < 1e-9: return 0.0
    return -roc_auc_score(y, (oof_mat * (w/s)).sum(axis=1))

ns = len(SEEDS)
best_ws, best_vs = np.ones(ns)/ns, neg_auc_s(np.ones(ns)/ns)
for w0 in [np.ones(ns)/ns] + [np.eye(ns)[i] for i in range(ns)]:
    res = minimize(neg_auc_s, w0, method="Nelder-Mead",
                   options={"maxiter": 10000, "xatol": 1e-8, "fatol": 1e-9})
    if res.fun < best_vs:
        best_vs, best_ws = res.fun, res.x
opt_ws = np.clip(best_ws, 0, 1); opt_ws /= opt_ws.sum()
auc_opt = max(-best_vs, auc_mean)
if -best_vs < auc_mean:
    opt_ws = np.ones(ns)/ns

oof_v19  = (oof_mat  * opt_ws).sum(axis=1)
test_v19 = (test_mat * opt_ws).sum(axis=1)
print(f"최적 seed 가중치 OOF: {auc_opt:.5f}")
print("seed 가중치:", {f"s{s}": round(float(w),4) for s,w in zip(SEEDS, opt_ws)})

# =========================
# 7. 기존 최강 모델들과 블렌딩
# =========================
prev = {
    "ag_raw_10h": ("oof_ag_raw_10h_auc_0.74126.npy", "test_ag_raw_10h_auc_0.74126.npy"),
    "ag8h":       ("oof_ag8h_auc_0.74114.npy",        "test_ag8h_auc_0.74114.npy"),
    "pseudo_v3":  ("oof_pseudo_v3_auc_0.74080.npy",   "test_pseudo_v3_auc_0.74080.npy"),
}

oof_pool  = {"v19": oof_v19}
test_pool = {"v19": test_v19}

for nm, (of, tf) in prev.items():
    fp = os.path.join(NPY_DIR, of)
    tp = os.path.join(NPY_DIR, tf)
    if os.path.exists(fp):
        oof_pool[nm]  = np.load(fp)
        test_pool[nm] = np.load(tp)
        print(f"  로드: {nm} OOF {roc_auc_score(y, oof_pool[nm]):.5f}")

nm_list  = list(oof_pool.keys())
oof_b    = np.stack([oof_pool[n]  for n in nm_list], axis=1)
test_b   = np.stack([test_pool[n] for n in nm_list], axis=1)
n2 = len(nm_list)

def neg_auc_b(w):
    w = np.clip(w, 0, 1); s = w.sum()
    if s < 1e-9: return 0.0
    return -roc_auc_score(y, (oof_b * (w/s)).sum(axis=1))

b_simple = roc_auc_score(y, oof_b.mean(axis=1))
print(f"\n블렌딩 단순평균: {b_simple:.5f}")

init2 = [np.ones(n2)/n2] + [np.eye(n2)[i] for i in range(n2)]
for sd in [0, 7, 42, 99]:
    init2.append(np.random.RandomState(sd).dirichlet(np.ones(n2)))

best_b, best_bv = np.ones(n2)/n2, neg_auc_b(np.ones(n2)/n2)
for w0 in init2:
    res = minimize(neg_auc_b, w0, method="Nelder-Mead",
                   options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-9})
    if res.fun < best_bv:
        best_bv, best_b = res.fun, res.x

opt_b = np.clip(best_b, 0, 1); opt_b /= opt_b.sum()
blend_auc = max(-best_bv, b_simple)
if -best_bv < b_simple:
    opt_b = np.ones(n2)/n2
print(f"블렌딩 최적 OOF: {blend_auc:.5f}")
print("블렌딩 가중치:", {n: round(float(w), 4) for n, w in zip(nm_list, opt_b)})

# =========================
# 8. 저장
# =========================
ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M")

def make_sub(preds, fname):
    rank_n = pd.Series(preds).rank(pct=True).values
    final  = 0.5 * preds + 0.5 * rank_n
    sub = pd.DataFrame({"ID": test["ID"], "probability": final})
    sub.to_csv(fname, index=False)
    print(f"저장: {fname}")

make_sub(test_v19,
         os.path.join(NPY_DIR, f"v19_alone_oof{auc_opt:.5f}_{ts}.csv"))

final_blend = (test_b * opt_b).sum(axis=1)
make_sub(final_blend,
         os.path.join(NPY_DIR, f"v19_blend_oof{blend_auc:.5f}_{ts}.csv"))

np.save(os.path.join(NPY_DIR, f"oof_v19_auc_{auc_opt:.5f}.npy"),  oof_v19)
np.save(os.path.join(NPY_DIR, f"test_v19_auc_{auc_opt:.5f}.npy"), test_v19)

print(f"\n{'='*60}")
print(f"[A] v19 단독   : v19_alone_oof{auc_opt:.5f}_{ts}.csv")
print(f"[B] 최종블렌딩 : v19_blend_oof{blend_auc:.5f}_{ts}.csv  ← 제출 권장")
print(f"v19 기여도(weight): {opt_b[0]:.4f}")
print(f"{'='*60}")
