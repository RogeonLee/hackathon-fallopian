"""
AutoGluon TabularPredictor
v8 기준선: 0.74079
전략: best_quality 프리셋으로 자동 모델 탐색 + 스태킹 + 앙상블
"""
import glob, warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

warnings.filterwarnings("ignore")

TARGET  = "임신 성공 여부"
ID_COL  = "ID"
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

print("=" * 65)
print("AutoGluon TabularPredictor")
print(f"v8 기준선: {V8_BASELINE}")
print("=" * 65)

train_raw = pd.read_csv("./data/train.csv")
test_raw  = pd.read_csv("./data/test.csv")

train = preprocess(train_raw)
test  = preprocess(test_raw)

train_ag = train.drop(columns=[ID_COL])
test_ag  = test.drop(columns=[ID_COL])

print(f"  Train shape: {train_ag.shape}")
print(f"  Test  shape: {test_ag.shape}")

from autogluon.tabular import TabularPredictor

# ─── [1] best_quality 프리셋 (시간 제한 2시간) ──────────────────
print("\n[1] AutoGluon best_quality (time_limit=7200s)...")
predictor = TabularPredictor(
    label    = TARGET,
    eval_metric = "roc_auc",
    path     = "./autogluon_models",
    verbosity = 2,
).fit(
    train_data   = train_ag,
    presets      = "best_quality",
    time_limit   = 7200,
)

print("\n[2] 리더보드 확인...")
lb = predictor.leaderboard(silent=True)
print(lb[["model", "score_val", "pred_time_val"]].to_string())

# ─── OOF 예측 ────────────────────────────────────────────────
print("\n[3] OOF 예측 생성...")
oof_preds = predictor.predict_proba(train_ag, as_multiclass=False)
# AutoGluon은 label 포함한 train_ag에서 OOF를 계산할 수 없으므로
# Val score를 사용하거나 수동 CV를 해야 함
# predict_proba on train_ag는 학습 데이터 예측 (OOF 아님)
# leaderboard의 score_val이 OOF 스코어에 가장 가까움

best_model_score = lb["score_val"].max()
print(f"  Best model val AUC: {best_model_score:.5f}")

# ─── 테스트 예측 ─────────────────────────────────────────────
print("\n[4] 테스트 예측...")
test_preds_ag = predictor.predict_proba(test_ag, as_multiclass=False)
if hasattr(test_preds_ag, 'values'):
    test_preds_ag = test_preds_ag.values
test_preds_ag = np.array(test_preds_ag).ravel()

print(f"  test preds shape: {test_preds_ag.shape}, range: [{test_preds_ag.min():.4f}, {test_preds_ag.max():.4f}]")

# ─── v8과 블렌딩 ─────────────────────────────────────────────
print("\n[5] v8 + AutoGluon 블렌딩...")
v8_test_f = sorted(glob.glob("test_v8_auc_*.npy"))
v8_oof_f  = sorted(glob.glob("oof_v8_auc_*.npy"))

if v8_test_f:
    v8_test = np.load(v8_test_f[-1])
    v8_oof  = np.load(v8_oof_f[-1])

    # v8 OOF + AG val score로 블렌딩 비율 추정 (실제 OOF 없으므로 근사)
    # 블렌딩은 테스트 예측에만 적용
    best_blend_auc  = best_model_score  # 추정값
    final_blend_test = {}

    for w in np.arange(0.1, 0.9, 0.05):
        b = w * rank_norm(test_preds_ag) + (1-w) * rank_norm(v8_test)
        final_blend_test[round(w,2)] = b

    # 최적 w는 나중에 LB 제출로 확인 — 지금은 0.3 AG + 0.7 v8 저장
    for w_ag in [0.2, 0.3, 0.4, 0.5]:
        blend = w_ag*rank_norm(test_preds_ag) + (1-w_ag)*rank_norm(v8_test)
        sub = pd.read_csv("./data/sample_submission.csv")
        sub["probability"] = blend
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        fname = f"submission_{timestamp}_v14ag_w{w_ag}_valAUC{best_model_score:.5f}.csv"
        sub.to_csv(fname, index=False)
        print(f"  저장: {fname}")
else:
    sub = pd.read_csv("./data/sample_submission.csv")
    sub["probability"] = test_preds_ag
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    fname = f"submission_{timestamp}_v14ag_valAUC{best_model_score:.5f}.csv"
    sub.to_csv(fname, index=False)
    print(f"  저장: {fname}")

# AutoGluon 단독 제출도 저장
sub_ag = pd.read_csv("./data/sample_submission.csv")
sub_ag["probability"] = test_preds_ag
sub_ag.to_csv(f"submission_{datetime.now().strftime('%Y%m%d_%H%M')}_v14ag_alone.csv", index=False)

np.save("test_autogluon.npy", test_preds_ag)

print(f"\n{'='*65}")
print(f"  AutoGluon best val AUC: {best_model_score:.5f}")
print(f"  v8 기준선 대비: {best_model_score - V8_BASELINE:+.5f} (val 기준 추정)")
print(f"{'='*65}")
print("\n  ※ 실제 개선 여부는 Dacon 제출 후 LB 점수로 확인 필요")
