"""
[작전 D 강화] AutoGluon raw + stack=0 + 10시간
- 효율 피처 제거 (raw + 기본 전처리만)
- num_stack_levels=0 강제 (DyStack 우회로 시간 절약)
- num_bag_folds=8, num_bag_sets=2 → 16-fold equivalent
- KNN 제외 (약체)
"""
import os, sys, glob, warnings, shutil
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

warnings.filterwarnings("ignore")

TARGET = "임신 성공 여부"
ID_COL = "ID"
TIME_LIMIT = 36000  # 10h

def rank_norm(arr): return rankdata(arr) / len(arr)

# ─── 최소 전처리 (v8 스타일, 효율 피처 X) ───────────────────────
COUNT_COLS = ["총 시술 횟수", "클리닉 내 총 시술 횟수",
              "IVF 시술 횟수", "DI 시술 횟수",
              "총 임신 횟수", "IVF 임신 횟수", "DI 임신 횟수",
              "총 출산 횟수", "IVF 출산 횟수", "DI 출산 횟수"]

def preprocess_raw(df):
    df = df.copy()
    for col in COUNT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    # 핵심 robust binary flag만 추가 (이미 v9에 있는 것들)
    df["ever_delivered"] = (df["총 출산 횟수"].fillna(0) > 0).astype("int8")
    df["is_FET"]         = (df["해동된 배아 수"].fillna(0) > 0).astype("int8")
    df["is_first_attempt"] = (df["총 시술 횟수"].fillna(0) == 0).astype("int8")
    return df

print("=" * 75)
print("[작전 D 강화] AG raw + stack=0 + 10h (효율 피처 X, DyStack 우회)")
print("=" * 75)

train = pd.read_csv("./data/train.csv")
test  = pd.read_csv("./data/test.csv")
train = preprocess_raw(train)
test  = preprocess_raw(test)

train_ag = train.drop(columns=[ID_COL])
test_ag  = test.drop(columns=[ID_COL])
y = train[TARGET].copy()

print(f"  Train: {train_ag.shape}, Test: {test_ag.shape}")

model_path = "./autogluon_models_raw_10h"
if os.path.exists(model_path):
    shutil.rmtree(model_path)

from autogluon.tabular import TabularPredictor

print(f"\n[1] AutoGluon best_quality stack=0, bag=8x2, time={TIME_LIMIT}s ...")
predictor = TabularPredictor(
    label       = TARGET,
    eval_metric = "roc_auc",
    path        = model_path,
    verbosity   = 2,
).fit(
    train_data            = train_ag,
    presets               = "best_quality",
    time_limit            = TIME_LIMIT,
    num_bag_folds         = 8,
    num_bag_sets          = 2,
    num_stack_levels      = 0,
    dynamic_stacking      = False,    # DyStack 우회 (시간 절약)
    excluded_model_types  = ["KNN"],
)

print("\n[2] 리더보드...")
lb = predictor.leaderboard(silent=True)
print(lb[["model", "score_val", "pred_time_val"]].head(20).to_string())
best_val_auc = lb["score_val"].max()
print(f"\n  Best val AUC: {best_val_auc:.5f}")

print("\n[3] OOF 추출...")
oof_proba = predictor.predict_proba_oof(as_multiclass=False)
oof_arr = (oof_proba.values if hasattr(oof_proba, 'values') else np.array(oof_proba)).ravel()
ag_oof_auc = roc_auc_score(y, oof_arr)
print(f"  AG raw 10h OOF AUC: {ag_oof_auc:.5f}")
np.save(f"oof_ag_raw_10h_auc_{ag_oof_auc:.5f}.npy", oof_arr)

print("\n[4] 테스트 예측...")
test_proba = predictor.predict_proba(test_ag, as_multiclass=False)
ag_test = (test_proba.values if hasattr(test_proba, 'values') else np.array(test_proba)).ravel()
np.save(f"test_ag_raw_10h_auc_{ag_oof_auc:.5f}.npy", ag_test)

# v8과 페어 + 그리드 서치 (즉시 검증)
print("\n[5] v8 + ag_raw_10h 페어 블렌드...")
v8_oof  = np.load(sorted(glob.glob("oof_v8_auc_*.npy"))[-1])
v8_test = np.load(sorted(glob.glob("test_v8_auc_*.npy"))[-1])

best_w, best_a = 0.5, 0
for w in np.arange(0.05, 0.96, 0.01):
    a = roc_auc_score(y, w*rank_norm(oof_arr) + (1-w)*rank_norm(v8_oof))
    if a > best_a: best_a, best_w = a, round(w, 2)
print(f"  최적 w_ag_raw={best_w}: OOF={best_a:.5f}")

sub_base = pd.read_csv("./data/sample_submission.csv")
ts = datetime.now().strftime("%Y%m%d_%H%M")

# AG raw 단독
sub = sub_base.copy(); sub["probability"] = ag_test
sub.to_csv(f"submission_{ts}_ag_raw_10h_alone_oof{ag_oof_auc:.5f}.csv", index=False)

# v8 페어
blend = best_w*rank_norm(ag_test) + (1-best_w)*rank_norm(v8_test)
sub = sub_base.copy(); sub["probability"] = blend
fname = f"submission_{ts}_ag_raw_10h_v8_w{best_w}_oof{best_a:.5f}.csv"
sub.to_csv(fname, index=False)
print(f"  저장: {fname}")

# ag8h와 NM 블렌드 (다른 시드 다양성)
print("\n[6] 전체 OOF 풀 NM 블렌드 (ag_raw_10h 추가)...")
from scipy.optimize import minimize

candidates = [
    ("v8",         "oof_v8_auc_*.npy",         "test_v8_auc_*.npy"),
    ("v9",         "oof_v9_auc_*.npy",         "test_v9_auc_*.npy"),
    ("ag8h",       "oof_ag8h_auc_*.npy",       "test_ag8h_auc_*.npy"),
    ("agv10full",  "oof_agv10full_auc_*.npy",  "test_agv10full_auc_*.npy"),
    ("ag_raw_10h", "oof_ag_raw_10h_auc_*.npy", "test_ag_raw_10h_auc_*.npy"),
]
pools = []
for tag, op, tp in candidates:
    of = sorted(glob.glob(op))
    tf = sorted(glob.glob(tp))
    if of and tf:
        o = np.load(of[-1]); t = np.load(tf[-1])
        try:
            a = roc_auc_score(y, o)
            pools.append((tag, rank_norm(o), rank_norm(t), a))
            print(f"  - {tag}: OOF={a:.5f}")
        except Exception as e:
            print(f"  - {tag} 스킵: {e}")

oof_mat  = np.stack([p[1] for p in pools], axis=1)
test_mat = np.stack([p[2] for p in pools], axis=1)
names    = [p[0] for p in pools]
K = len(names)

def neg_auc(x):
    w = np.exp(x - x.max()); w = w / w.sum()
    return -roc_auc_score(y, oof_mat @ w)

best_neg, best_w_full = 1.0, None
for seed in range(8):
    rng = np.random.default_rng(seed)
    x0  = rng.normal(size=K) * 0.5
    res = minimize(neg_auc, x0, method="Nelder-Mead",
                   options={"xatol":1e-4, "fatol":1e-6, "maxiter":2000})
    if res.fun < best_neg:
        best_neg = res.fun
        best_w_full = np.exp(res.x - res.x.max()); best_w_full /= best_w_full.sum()

best_oof_full = -best_neg
print(f"\n  ★ NM 최적 OOF: {best_oof_full:.5f}")
for n, w in sorted(zip(names, best_w_full), key=lambda x: -x[1]):
    print(f"    {n:>12s}: {w:.4f}")

blend_test = test_mat @ best_w_full
sub = sub_base.copy(); sub["probability"] = blend_test
fname_nm = f"submission_{ts}_ag_raw_10h_NMblend_oof{best_oof_full:.5f}.csv"
sub.to_csv(fname_nm, index=False)
print(f"  저장: {fname_nm}")

print(f"\n{'='*75}")
print(f"  [작전 D 강화 결과]")
print(f"  AG raw 10h 단독 OOF      : {ag_oof_auc:.5f}")
print(f"  v8 페어 OOF              : {best_a:.5f}")
print(f"  NM 전체 블렌드 OOF       : {best_oof_full:.5f}")
print(f"  v8 갭(+0.00129) 가산 시 LB 추정: {best_oof_full + 0.00129:.5f}")
print(f"  1위 0.74246 대비: {0.74246 - (best_oof_full + 0.00129):+.5f}")
print(f"{'='*75}")
