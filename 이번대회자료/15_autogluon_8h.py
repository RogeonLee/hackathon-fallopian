"""
AutoGluon 8시간 풀런 + OOF 추출 + v8 최적 블렌딩
v8 기준선 LB: 0.74208
AG 2h val AUC: 0.74108
"""
import glob, warnings, shutil
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

warnings.filterwarnings("ignore")

TARGET  = "임신 성공 여부"
ID_COL  = "ID"
V8_LB   = 0.74208

def rank_norm(arr): return rankdata(arr) / len(arr)

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

print("=" * 65)
print("AutoGluon 8시간 풀런")
print(f"v8 LB 기준선: {V8_LB}")
print("=" * 65)

train_raw = pd.read_csv("./data/train.csv")
test_raw  = pd.read_csv("./data/test.csv")
train = preprocess(train_raw)
test  = preprocess(test_raw)

train_ag = train.drop(columns=[ID_COL])
test_ag  = test.drop(columns=[ID_COL])
y = train[TARGET].copy()

print(f"  Train: {train_ag.shape}, Test: {test_ag.shape}")

# 기존 모델 폴더 삭제 후 새로 학습
model_path = "./autogluon_models_8h"
if __import__("os").path.exists(model_path):
    shutil.rmtree(model_path)
    print("  기존 모델 폴더 삭제 완료")

from autogluon.tabular import TabularPredictor

print("\n[1] AutoGluon best_quality 8시간 학습...")
predictor = TabularPredictor(
    label       = TARGET,
    eval_metric = "roc_auc",
    path        = model_path,
    verbosity   = 2,
).fit(
    train_data     = train_ag,
    presets        = "best_quality",
    time_limit     = 28800,           # 8시간
    num_bag_folds  = 8,               # 8-fold bagging (더 안정적인 OOF)
    num_bag_sets   = 1,
    num_stack_levels = 2,             # 2단계 스태킹
)

# ─── 리더보드 출력 ──────────────────────────────────────────────
print("\n[2] 모델 리더보드...")
lb = predictor.leaderboard(silent=True)
print(lb[["model", "score_val", "pred_time_val"]].head(20).to_string())
best_val_auc = lb["score_val"].max()
print(f"\n  Best val AUC: {best_val_auc:.5f}")

# ─── OOF 예측 추출 ──────────────────────────────────────────────
print("\n[3] OOF 예측 추출...")
try:
    oof_proba = predictor.predict_proba_oof(as_multiclass=False)
    if hasattr(oof_proba, 'values'):
        oof_arr = oof_proba.values.ravel()
    else:
        oof_arr = np.array(oof_proba).ravel()
    ag_oof_auc = roc_auc_score(y, oof_arr)
    print(f"  AutoGluon OOF AUC: {ag_oof_auc:.5f}")
    np.save(f"oof_ag8h_auc_{ag_oof_auc:.5f}.npy", oof_arr)
    has_oof = True
except Exception as e:
    print(f"  OOF 추출 실패: {e}")
    ag_oof_auc = best_val_auc
    has_oof = False

# ─── 테스트 예측 ──────────────────────────────────────────────
print("\n[4] 테스트 예측...")
test_proba = predictor.predict_proba(test_ag, as_multiclass=False)
if hasattr(test_proba, 'values'):
    ag_test = test_proba.values.ravel()
else:
    ag_test = np.array(test_proba).ravel()
np.save(f"test_ag8h_auc_{ag_oof_auc:.5f}.npy", ag_test)
print(f"  test range: [{ag_test.min():.4f}, {ag_test.max():.4f}]")

# ─── v8과 최적 블렌딩 ─────────────────────────────────────────
print("\n[5] v8 + AG8h 블렌딩...")
v8_oof_f  = sorted(glob.glob("oof_v8_auc_*.npy"))
v8_test_f = sorted(glob.glob("test_v8_auc_*.npy"))

sub_base = pd.read_csv("./data/sample_submission.csv")
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

if v8_oof_f and has_oof:
    v8_oof  = np.load(v8_oof_f[-1])
    v8_test = np.load(v8_test_f[-1])
    v8_auc  = roc_auc_score(y, v8_oof)
    print(f"  v8 OOF AUC:  {v8_auc:.5f}")
    print(f"  AG8h OOF AUC: {ag_oof_auc:.5f}")

    # 최적 블렌딩 그리드 서치 (OOF 기반)
    best_blend_auc  = 0
    best_w          = 0.5
    for w in np.arange(0.05, 0.96, 0.05):
        b = w*rank_norm(ag_test_oof := oof_arr) + (1-w)*rank_norm(v8_oof)
        # OOF 블렌딩
        b_oof = w*rank_norm(oof_arr) + (1-w)*rank_norm(v8_oof)
        a = roc_auc_score(y, b_oof)
        if a > best_blend_auc:
            best_blend_auc = a
            best_w = round(w, 2)

    print(f"\n  최적 블렌딩 OOF AUC: {best_blend_auc:.5f}  (AG={best_w}, v8={1-best_w})")

    # 다양한 비율로 제출 파일 생성
    for w_ag in [0.3, 0.4, 0.5, 0.6, 0.7, best_w]:
        w_ag = round(w_ag, 2)
        blend_test = w_ag*rank_norm(ag_test) + (1-w_ag)*rank_norm(v8_test)
        sub = sub_base.copy()
        sub["probability"] = blend_test
        fname = f"submission_{timestamp}_ag8h_w{w_ag}_oofAUC{best_blend_auc:.5f}.csv"
        sub.to_csv(fname, index=False)

    print(f"  제출 파일 저장 완료 (ag8h_w0.3~0.7 + best_w={best_w})")

else:
    # OOF 없으면 고정 비율로 저장
    v8_test = np.load(v8_test_f[-1]) if v8_test_f else None
    for w_ag in [0.3, 0.4, 0.5, 0.6, 0.7]:
        blend_test = w_ag*rank_norm(ag_test) + (1-w_ag)*rank_norm(v8_test) if v8_test is not None else ag_test
        sub = sub_base.copy()
        sub["probability"] = blend_test
        sub.to_csv(f"submission_{timestamp}_ag8h_w{w_ag}_val{best_val_auc:.5f}.csv", index=False)

# AG 단독
sub_ag = sub_base.copy()
sub_ag["probability"] = ag_test
sub_ag.to_csv(f"submission_{timestamp}_ag8h_alone_val{best_val_auc:.5f}.csv", index=False)

print(f"\n{'='*65}")
print(f"  AG8h best val AUC:  {best_val_auc:.5f}")
if has_oof:
    print(f"  AG8h OOF AUC:       {ag_oof_auc:.5f}")
    print(f"  최적 블렌딩 OOF:    {best_blend_auc:.5f}")
print(f"  v8 LB 기준선 대비: +{best_val_auc - V8_LB:+.5f} (val 기준)")
print(f"{'='*65}")
