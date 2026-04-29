"""
[전략 3] AG 내부 상위 모델 추출 + 정밀 블렌딩
AG8h의 상위 LGBM/CatBoost 모델들을 개별 추출하여 v8과 최적 조합
"""
import glob, warnings
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

warnings.filterwarnings("ignore")
TARGET, ID_COL = "임신 성공 여부", "ID"
V8_BASELINE = 0.74079

def rank_norm(a): return rankdata(a) / len(a)

COUNT_COLS = ["총 시술 횟수","클리닉 내 총 시술 횟수","IVF 시술 횟수","DI 시술 횟수",
              "총 임신 횟수","IVF 임신 횟수","DI 임신 횟수","총 출산 횟수","IVF 출산 횟수","DI 출산 횟수"]

def preprocess(df):
    df = df.copy()
    for col in COUNT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    df["ever_delivered"] = (df["총 출산 횟수"].fillna(0) > 0).astype(int)
    df["is_FET"]         = (df["해동된 배아 수"].fillna(0) > 0).astype(int)
    return df

print("=" * 65)
print("[전략 3] AG 내부 상위 모델 추출 + 정밀 블렌딩")
print("=" * 65)

train_raw = pd.read_csv("./data/train.csv")
test_raw  = pd.read_csv("./data/test.csv")
train = preprocess(train_raw)
test  = preprocess(test_raw)
y = train[TARGET].copy()

train_ag = train.drop(columns=[ID_COL])
test_ag  = test.drop(columns=[ID_COL])

from autogluon.tabular import TabularPredictor

model_path = "./autogluon_models_8h"
print(f"\n[1] AG8h 모델 로드...")
predictor = TabularPredictor.load(model_path)

lb = predictor.leaderboard(silent=True)
print(lb[["model","score_val"]].head(15).to_string())

# 상위 모델들 개별 예측 추출
print("\n[2] 상위 모델 개별 테스트 예측 추출...")
top_models = lb.head(10)["model"].tolist()

model_test_preds = {}
model_oof_preds  = {}
for model_name in top_models:
    try:
        # 테스트 예측
        tp = predictor.predict_proba(test_ag, model=model_name, as_multiclass=False)
        tp = np.array(tp).ravel()
        # OOF 예측
        op = predictor.predict_proba_oof(model=model_name, as_multiclass=False)
        op = np.array(op).ravel()
        auc = roc_auc_score(y, op)
        model_test_preds[model_name] = tp
        model_oof_preds[model_name]  = op
        print(f"  {model_name:<35}: OOF AUC = {auc:.5f}")
    except Exception as e:
        print(f"  {model_name}: 실패 ({e})")

# 기존 OOF 로드
v8_oof  = np.load(sorted(glob.glob("oof_v8_auc_*.npy"))[-1])
v8_test = np.load(sorted(glob.glob("test_v8_auc_*.npy"))[-1])
ag_oof  = np.load(sorted(glob.glob("oof_ag8h_auc_*.npy"))[-1])
ag_test = np.load(sorted(glob.glob("test_ag8h_auc_*.npy"))[-1])
v8_auc  = roc_auc_score(y, v8_oof)
ag_auc  = roc_auc_score(y, ag_oof)

print(f"\n  v8 OOF: {v8_auc:.5f}, AG8h OOF: {ag_auc:.5f}")

# 모든 예측 통합 후 최적 블렌딩
print("\n[3] 정밀 블렌딩 탐색...")
all_oofs  = {"v8": v8_oof, "ag": ag_oof, **model_oof_preds}
all_tests = {"v8": v8_test,"ag": ag_test, **model_test_preds}

# AG WeightedEnsemble이 이미 최적이지만, v8과의 혼합 비율 정밀 탐색
# 상위 3개 모델 + v8 조합 탐색
top3 = list(model_oof_preds.keys())[:3]
print(f"  탐색 모델: {top3}")

best_auc = roc_auc_score(y, 0.75*rank_norm(ag_oof) + 0.25*rank_norm(v8_oof))
best_test = 0.75*rank_norm(ag_test) + 0.25*rank_norm(v8_test)
best_desc = "ag=0.75,v8=0.25 (기준)"
print(f"  기준 (ag0.75+v8): {best_auc:.5f}")

# v8 + ag + top1 모델 3원 블렌딩
if top3:
    for w_v8 in np.arange(0.1, 0.4, 0.1):
        for w_ag in np.arange(0.4, 0.8, 0.1):
            for w_t1 in np.arange(0.05, 0.3, 0.05):
                if abs(w_v8+w_ag+w_t1-1.0) > 0.02: continue
                ws = np.array([w_v8,w_ag,w_t1]); ws /= ws.sum()
                b = (ws[0]*rank_norm(v8_oof) + ws[1]*rank_norm(ag_oof)
                     + ws[2]*rank_norm(model_oof_preds[top3[0]]))
                a = roc_auc_score(y, b)
                if a > best_auc:
                    best_auc  = a
                    best_test = (ws[0]*rank_norm(v8_test) + ws[1]*rank_norm(ag_test)
                                 + ws[2]*rank_norm(model_test_preds[top3[0]]))
                    best_desc = f"v8={ws[0]:.2f},ag={ws[1]:.2f},{top3[0][:8]}={ws[2]:.2f}"

print(f"  최적: {best_desc} → OOF {best_auc:.5f}  ({best_auc-0.74119:+.5f} vs AG8h blend)")

sub = pd.read_csv("./data/sample_submission.csv")
sub["probability"] = best_test
ts = datetime.now().strftime("%Y%m%d_%H%M")
fname = f"submission_{ts}_v18agmodels_oof{best_auc:.5f}.csv"
sub.to_csv(fname, index=False)
print(f"  저장: {fname}")

np.save(f"oof_v18_auc_{best_auc:.5f}.npy", best_test)
print(f"\n{'='*65}")
print(f"  최적 블렌딩 OOF: {best_auc:.5f}")
print(f"  AG8h blend 대비: {best_auc-0.74119:+.5f}")
print(f"{'='*65}")
