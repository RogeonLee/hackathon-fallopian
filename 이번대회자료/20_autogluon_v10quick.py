"""
[옵션 A] AutoGluon medium_quality 30분 + v10 효율 피처
효율 피처가 OOF에 도움 되는지 빠르게 검증.
"""
import os, sys, glob, warnings, shutil
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fe_v10 import fe_v10

warnings.filterwarnings("ignore")

TARGET     = "임신 성공 여부"
ID_COL     = "ID"
V8_LB      = 0.74208
LEAD_LB    = 0.74237
TIME_LIMIT = 1800   # 30분

def rank_norm(arr): return rankdata(arr) / len(arr)

print("=" * 72)
print("[옵션 A] AutoGluon medium_quality + v10 효율 피처 — 30분 검증")
print(f"  v8 LB: {V8_LB} / 1위 LB: {LEAD_LB} / 격차: {LEAD_LB - V8_LB:+.5f}")
print("=" * 72)

train = pd.read_csv("./data/train.csv")
test  = pd.read_csv("./data/test.csv")

train_fe = fe_v10(train)
test_fe  = fe_v10(test)
new_cols = [c for c in train_fe.columns if c not in train.columns]
print(f"\n  신규 피처 {len(new_cols)}개 추가")
print(f"  Train: {train_fe.shape}, Test: {test_fe.shape}")

train_ag = train_fe.drop(columns=[ID_COL])
test_ag  = test_fe.drop(columns=[ID_COL])
y = train[TARGET].copy()

model_path = "./autogluon_models_v10quick"
if os.path.exists(model_path):
    shutil.rmtree(model_path)

from autogluon.tabular import TabularPredictor

print(f"\n[1] AutoGluon medium_quality (time_limit={TIME_LIMIT}s, bag=5)...")
predictor = TabularPredictor(
    label       = TARGET,
    eval_metric = "roc_auc",
    path        = model_path,
    verbosity   = 2,
).fit(
    train_data       = train_ag,
    presets          = "medium_quality",
    time_limit       = TIME_LIMIT,
    num_bag_folds    = 5,
    num_bag_sets     = 1,
    num_stack_levels = 0,
)

print("\n[2] 모델 리더보드...")
lb = predictor.leaderboard(silent=True)
print(lb[["model", "score_val", "pred_time_val"]].head(15).to_string())
best_val_auc = lb["score_val"].max()
print(f"\n  Best val AUC: {best_val_auc:.5f}")

# OOF 추출
print("\n[3] OOF 예측 추출...")
has_oof = False
ag_oof_auc = best_val_auc
oof_arr = None
try:
    oof_proba = predictor.predict_proba_oof(as_multiclass=False)
    oof_arr = (oof_proba.values if hasattr(oof_proba, 'values') else np.array(oof_proba)).ravel()
    ag_oof_auc = roc_auc_score(y, oof_arr)
    print(f"  AutoGluon OOF AUC: {ag_oof_auc:.5f}")
    np.save(f"oof_agv10quick_auc_{ag_oof_auc:.5f}.npy", oof_arr)
    has_oof = True
except Exception as e:
    print(f"  OOF 실패: {e}")

# 테스트 예측
print("\n[4] 테스트 예측...")
test_proba = predictor.predict_proba(test_ag, as_multiclass=False)
ag_test = (test_proba.values if hasattr(test_proba, 'values') else np.array(test_proba)).ravel()
np.save(f"test_agv10quick_auc_{ag_oof_auc:.5f}.npy", ag_test)

# 블렌딩 분석
if has_oof:
    sub_base = pd.read_csv("./data/sample_submission.csv")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    print("\n[5] 기존 OOF 로드 + 블렌딩 그리드 서치...")
    pools = {}  # {name: (oof, test)}
    for tag, oof_pat, test_pat in [
        ("v8",   "oof_v8_auc_*.npy",        "test_v8_auc_*.npy"),
        ("ag8h", "oof_ag8h_auc_*.npy",      "test_ag8h_auc_*.npy"),
        ("v17b", "oof_pseudo_auc_*.npy",    "test_pseudo_auc_*.npy"),
        ("v9",   "oof_v9_auc_*.npy",        "test_v9_auc_*.npy"),
    ]:
        of = sorted(glob.glob(oof_pat))
        tf = sorted(glob.glob(test_pat))
        if of and tf:
            o = np.load(of[-1])
            t = np.load(tf[-1])
            try:
                a = roc_auc_score(y, o)
                pools[tag] = (o, t)
                print(f"  - {tag}: OOF AUC={a:.5f}, files=({of[-1]}, {tf[-1]})")
            except Exception as e:
                print(f"  - {tag} 스킵: {e}")
    pools["agv10q"] = (oof_arr, ag_test)

    # 페어 블렌드 (agv10q × 각각)
    print("\n[6] agv10q × 각 모델 페어 블렌드...")
    for tag, (o, t) in pools.items():
        if tag == "agv10q":
            continue
        best_w, best_a = 0.5, 0
        for w in np.arange(0.05, 0.96, 0.05):
            b = w*rank_norm(oof_arr) + (1-w)*rank_norm(o)
            a = roc_auc_score(y, b)
            if a > best_a:
                best_a, best_w = a, round(w, 2)
        print(f"  agv10q × {tag}: best OOF={best_a:.5f}  (w_v10={best_w})")

    # 3-way 블렌드: agv10q + v8 + ag8h (가장 강한 셋)
    if {"v8", "ag8h"}.issubset(pools):
        print("\n[7] 3-way 블렌드 (v8 + ag8h + agv10q)...")
        v8o, v8t = pools["v8"]
        a8o, a8t = pools["ag8h"]
        best_3, best_w3 = 0, (0.33, 0.33, 0.34)
        for w1 in np.arange(0.05, 0.91, 0.05):
            for w2 in np.arange(0.05, 0.96 - w1, 0.05):
                w3 = 1 - w1 - w2
                if w3 < 0.05:
                    continue
                b = w1*rank_norm(v8o) + w2*rank_norm(a8o) + w3*rank_norm(oof_arr)
                a = roc_auc_score(y, b)
                if a > best_3:
                    best_3, best_w3 = a, (round(w1,2), round(w2,2), round(w3,2))
        w1, w2, w3 = best_w3
        print(f"  3-way 최적 OOF AUC: {best_3:.5f}  (v8={w1}, ag8h={w2}, agv10q={w3})")
        blend_test = w1*rank_norm(v8t) + w2*rank_norm(a8t) + w3*rank_norm(ag_test)
        sub = sub_base.copy()
        sub["probability"] = blend_test
        fname = f"submission_{timestamp}_3way_v10quick_oof{best_3:.5f}.csv"
        sub.to_csv(fname, index=False)
        print(f"  저장: {fname}")

    # 4-way: + v17b
    if {"v8", "ag8h", "v17b"}.issubset(pools):
        print("\n[8] 4-way 블렌드 (v8 + ag8h + v17b + agv10q)...")
        v8o, v8t = pools["v8"]
        a8o, a8t = pools["ag8h"]
        pso, pst = pools["v17b"]

        best_4, best_w4 = 0, (0.25, 0.25, 0.25, 0.25)
        # 0.05 step grid (≈ 1771 조합)
        for w1 in np.arange(0.05, 0.86, 0.05):
            for w2 in np.arange(0.05, 0.91 - w1, 0.05):
                for w3 in np.arange(0.05, 0.96 - w1 - w2, 0.05):
                    w4 = 1 - w1 - w2 - w3
                    if w4 < 0.05:
                        continue
                    b = (w1*rank_norm(v8o) + w2*rank_norm(a8o)
                         + w3*rank_norm(pso) + w4*rank_norm(oof_arr))
                    a = roc_auc_score(y, b)
                    if a > best_4:
                        best_4, best_w4 = a, (round(w1,2), round(w2,2), round(w3,2), round(w4,2))
        w1, w2, w3, w4 = best_w4
        print(f"  4-way 최적 OOF AUC: {best_4:.5f}  (v8={w1}, ag8h={w2}, v17b={w3}, agv10q={w4})")
        blend_test = (w1*rank_norm(v8t) + w2*rank_norm(a8t)
                      + w3*rank_norm(pst) + w4*rank_norm(ag_test))
        sub = sub_base.copy()
        sub["probability"] = blend_test
        fname = f"submission_{timestamp}_4way_v10quick_oof{best_4:.5f}.csv"
        sub.to_csv(fname, index=False)
        print(f"  저장: {fname}")

# AG 단독
sub_alone = pd.read_csv("./data/sample_submission.csv")
sub_alone["probability"] = ag_test
ts = datetime.now().strftime("%Y%m%d_%H%M")
sub_alone.to_csv(f"submission_{ts}_agv10quick_alone_oof{ag_oof_auc:.5f}.csv", index=False)

print(f"\n{'='*72}")
print(f"  [옵션 A 결과 요약]")
print(f"  AG v10quick 단독 OOF: {ag_oof_auc:.5f}")
print(f"  v8 LB 갭(+0.00129) 가산 시 최고 추정 LB: 위 OOF + 0.00129")
print(f"  (참고) 1위 LB 0.74237 — 격차 = 1위 - (OOF + 0.00129)")
print(f"{'='*72}")
