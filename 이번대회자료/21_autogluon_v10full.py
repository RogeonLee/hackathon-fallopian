"""
[옵션 C] AutoGluon best_quality 8시간 풀런 + v10 효율 피처
+ 모든 기존 OOF로 scipy Nelder-Mead 가중치 최적화 블렌딩
"""
import os, sys, glob, warnings, shutil
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fe_v10 import fe_v10

warnings.filterwarnings("ignore")

TARGET     = "임신 성공 여부"
ID_COL     = "ID"
V8_LB      = 0.74208
LEAD_LB    = 0.74237
TIME_LIMIT = 28800   # 8시간

def rank_norm(arr): return rankdata(arr) / len(arr)

print("=" * 75)
print("[옵션 C] AutoGluon best_quality 8h + v10 효율 피처 + Nelder-Mead 블렌드")
print(f"  v8 LB: {V8_LB} / 1위 LB: {LEAD_LB}")
print("=" * 75)

train = pd.read_csv("./data/train.csv")
test  = pd.read_csv("./data/test.csv")

train_fe = fe_v10(train)
test_fe  = fe_v10(test)
new_cols = [c for c in train_fe.columns if c not in train.columns]
print(f"\n  신규 피처 {len(new_cols)}개 추가, Train={train_fe.shape}, Test={test_fe.shape}")

train_ag = train_fe.drop(columns=[ID_COL])
test_ag  = test_fe.drop(columns=[ID_COL])
y = train[TARGET].copy()

model_path = "./autogluon_models_v10full"
if os.path.exists(model_path):
    shutil.rmtree(model_path)

from autogluon.tabular import TabularPredictor

print(f"\n[1] AutoGluon best_quality (time_limit={TIME_LIMIT}s, bag=8, stack=2)...")
predictor = TabularPredictor(
    label       = TARGET,
    eval_metric = "roc_auc",
    path        = model_path,
    verbosity   = 2,
).fit(
    train_data       = train_ag,
    presets          = "best_quality",
    time_limit       = TIME_LIMIT,
    num_bag_folds    = 8,
    num_bag_sets     = 1,
    num_stack_levels = 2,
)

print("\n[2] 리더보드...")
lb = predictor.leaderboard(silent=True)
print(lb[["model", "score_val", "pred_time_val"]].head(20).to_string())
best_val_auc = lb["score_val"].max()
print(f"\n  Best val AUC: {best_val_auc:.5f}")

print("\n[3] OOF 예측 추출...")
oof_proba = predictor.predict_proba_oof(as_multiclass=False)
oof_arr = (oof_proba.values if hasattr(oof_proba, 'values') else np.array(oof_proba)).ravel()
ag_oof_auc = roc_auc_score(y, oof_arr)
print(f"  AG v10full OOF AUC: {ag_oof_auc:.5f}")
np.save(f"oof_agv10full_auc_{ag_oof_auc:.5f}.npy", oof_arr)

print("\n[4] 테스트 예측...")
test_proba = predictor.predict_proba(test_ag, as_multiclass=False)
ag_test = (test_proba.values if hasattr(test_proba, 'values') else np.array(test_proba)).ravel()
np.save(f"test_agv10full_auc_{ag_oof_auc:.5f}.npy", ag_test)

# ─────────────────────────────────────────────────
# [5] 모든 OOF 풀로 Nelder-Mead 가중치 최적화
# ─────────────────────────────────────────────────
print("\n[5] 기존 OOF 풀 로드...")
candidates = [
    ("v8",        "oof_v8_auc_*.npy",        "test_v8_auc_*.npy"),
    ("v9",        "oof_v9_auc_*.npy",        "test_v9_auc_*.npy"),
    ("v5",        "oof_v5_auc_*.npy",        "test_v5_auc_*.npy"),
    ("ag8h",      "oof_ag8h_auc_*.npy",      "test_ag8h_auc_*.npy"),
    ("agv10quick","oof_agv10quick_auc_*.npy","test_agv10quick_auc_*.npy"),
    ("v17b",      "oof_pseudo_auc_*.npy",    "test_pseudo_auc_*.npy"),
    ("v13",       "oof_v13_auc_*.npy",       "test_v13_auc_*.npy"),
    ("meta",      "oof_meta_lgbm_auc_*.npy", "test_meta_lgbm_auc_*.npy"),
    ("cb5seed",   "oof_cb5seed_auc_*.npy",   "test_cb5seed_auc_*.npy"),
]
pools = []  # [(name, oof_rank, test_rank, oof_auc)]
for tag, op, tp in candidates:
    of = sorted(glob.glob(op))
    tf = sorted(glob.glob(tp))
    if not of or not tf:
        continue
    o = np.load(of[-1])
    t = np.load(tf[-1])
    if o.shape[0] != y.shape[0] or t.shape[0] == 0:
        continue
    a = roc_auc_score(y, o)
    pools.append((tag, rank_norm(o), rank_norm(t), a))
    print(f"  - {tag}: OOF AUC={a:.5f}")

# v10 full OOF 추가
pools.append(("agv10full", rank_norm(oof_arr), rank_norm(ag_test), ag_oof_auc))
print(f"  - agv10full: OOF AUC={ag_oof_auc:.5f}  (★ 신규)")

oof_mat  = np.stack([p[1] for p in pools], axis=1)  # (N, K)
test_mat = np.stack([p[2] for p in pools], axis=1)
names    = [p[0] for p in pools]
K = len(names)
print(f"\n  Pool size: {K} models")

# 객관함수: -AUC of softmax-weighted blend
def neg_auc(x):
    w = np.exp(x - x.max())
    w = w / w.sum()
    blend = oof_mat @ w
    return -roc_auc_score(y, blend)

# 시드별 최적 + 베스트 선택
print("\n[6] Nelder-Mead 가중치 최적화 (multi-start)...")
best_neg, best_w = 1.0, None
for seed in range(8):
    rng = np.random.default_rng(seed)
    x0  = rng.normal(size=K) * 0.5
    res = minimize(neg_auc, x0, method="Nelder-Mead",
                   options={"xatol":1e-4, "fatol":1e-6, "maxiter":2000})
    if res.fun < best_neg:
        best_neg = res.fun
        best_w   = np.exp(res.x - res.x.max()); best_w /= best_w.sum()

best_oof_auc = -best_neg
print(f"\n  ★ 최적 블렌드 OOF AUC: {best_oof_auc:.5f}")
for n, w in sorted(zip(names, best_w), key=lambda x: -x[1]):
    print(f"     {n:>12s}: {w:.4f}")

# 단순 비교용: agv10full 단독
print(f"\n  단독 OOF: agv10full={ag_oof_auc:.5f}, ag8h={pools[[i for i,n in enumerate(names) if n=='ag8h'][0]][3] if 'ag8h' in names else 'N/A'}")

# 제출 파일 생성
sub_base = pd.read_csv("./data/sample_submission.csv")
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

blend_test = test_mat @ best_w
sub = sub_base.copy()
sub["probability"] = blend_test
fname_blend = f"submission_{timestamp}_v10full_NMblend_oof{best_oof_auc:.5f}.csv"
sub.to_csv(fname_blend, index=False)

# top3 가중치만 사용한 보수적 블렌드도 저장
top3_idx = np.argsort(-best_w)[:3]
w3 = best_w[top3_idx]; w3 /= w3.sum()
blend3 = test_mat[:, top3_idx] @ w3
oof3   = oof_mat[:,  top3_idx] @ w3
auc3   = roc_auc_score(y, oof3)
sub3 = sub_base.copy()
sub3["probability"] = blend3
fname_top3 = f"submission_{timestamp}_v10full_top3_oof{auc3:.5f}.csv"
sub3.to_csv(fname_top3, index=False)
print(f"\n  Top3 모델: {[names[i] for i in top3_idx]} → OOF {auc3:.5f}")

# agv10full 단독
sub_alone = sub_base.copy()
sub_alone["probability"] = ag_test
fname_alone = f"submission_{timestamp}_agv10full_alone_oof{ag_oof_auc:.5f}.csv"
sub_alone.to_csv(fname_alone, index=False)

# v8과 단순 그리드 블렌드(보험용)
v8_idx = next((i for i,n in enumerate(names) if n=="v8"), None)
if v8_idx is not None:
    v8o, v8t = oof_mat[:,v8_idx], test_mat[:,v8_idx]
    bw, ba = 0.5, 0
    for w in np.arange(0.05, 0.96, 0.05):
        a = roc_auc_score(y, w*rank_norm(oof_arr) + (1-w)*v8o)
        if a > ba: ba, bw = a, round(w, 2)
    blend_v8 = bw*rank_norm(ag_test) + (1-bw)*v8t
    sub_v8 = sub_base.copy()
    sub_v8["probability"] = blend_v8
    fname_v8 = f"submission_{timestamp}_v10full_v8blend_w{bw}_oof{ba:.5f}.csv"
    sub_v8.to_csv(fname_v8, index=False)
    print(f"\n  v8 페어 블렌드: w_v10={bw}, OOF={ba:.5f}")

print(f"\n{'='*75}")
print(f"  [옵션 C 최종 요약]")
print(f"  AG v10full 단독 OOF      : {ag_oof_auc:.5f}")
print(f"  Nelder-Mead 블렌드 OOF   : {best_oof_auc:.5f}")
print(f"  Top3 보수적 블렌드 OOF   : {auc3:.5f}")
print(f"  v8 LB(0.74208) 갭 +0.00129 가산 시 LB 추정:")
print(f"    NM   : {best_oof_auc + 0.00129:.5f}  (1위 0.74237 대비 {0.74237 - (best_oof_auc + 0.00129):+.5f})")
print(f"    Top3 : {auc3 + 0.00129:.5f}")
print(f"  제출 파일:")
print(f"    {fname_blend}")
print(f"    {fname_top3}")
if v8_idx is not None: print(f"    {fname_v8}")
print(f"    {fname_alone}")
print(f"{'='*75}")
