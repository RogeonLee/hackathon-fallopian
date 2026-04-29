"""
[작전 B] Adversarial Validation
- train(label=0) vs test(label=1) 이진 분류 → AUC
- AUC > 0.55면 distribution shift 존재
- LightGBM feature importance로 top shift features 식별
- 처리 후보: 제거 / sample weighting / quantile binning
"""
import warnings, json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

TARGET = "임신 성공 여부"
ID_COL = "ID"

COUNT_COLS = ["총 시술 횟수", "클리닉 내 총 시술 횟수",
              "IVF 시술 횟수", "DI 시술 횟수",
              "총 임신 횟수", "IVF 임신 횟수", "DI 임신 횟수",
              "총 출산 횟수", "IVF 출산 횟수", "DI 출산 횟수"]

def preprocess(df):
    df = df.copy()
    for col in COUNT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    return df

print("=" * 75)
print("[작전 B] Adversarial Validation — train vs test distribution shift")
print("=" * 75)

train = preprocess(pd.read_csv("./data/train.csv"))
test  = preprocess(pd.read_csv("./data/test.csv"))

X_tr = train.drop(columns=[ID_COL, TARGET])
X_te = test.drop(columns=[ID_COL])

# 두 데이터셋 컬럼 정렬 (train의 컬럼 기준)
common = [c for c in X_tr.columns if c in X_te.columns]
X_tr = X_tr[common]
X_te = X_te[common]
print(f"  공통 피처: {len(common)}개")

# 결합
X = pd.concat([X_tr, X_te], axis=0, ignore_index=True)
y_adv = np.concatenate([np.zeros(len(X_tr)), np.ones(len(X_te))])

# object 컬럼은 category로
for c in X.columns:
    if X[c].dtype == 'object':
        X[c] = X[c].astype('category')

print(f"  합쳐진 shape: {X.shape}")
print(f"  train 비율: {1 - y_adv.mean():.4f}, test 비율: {y_adv.mean():.4f}")

# ─── 5-fold CV로 adversarial AUC ────────────────────────────────
print("\n[1] 5-fold CV adversarial AUC 측정...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []
importances = np.zeros(X.shape[1])

cat_features = [c for c in X.columns if X[c].dtype.name == 'category']

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y_adv), 1):
    model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05,
        num_leaves=63, max_depth=-1,
        min_child_samples=50, colsample_bytree=0.8,
        objective='binary', metric='auc',
        random_state=42, verbose=-1,
    )
    model.fit(
        X.iloc[tr_idx], y_adv[tr_idx],
        eval_set=[(X.iloc[va_idx], y_adv[va_idx])],
        categorical_feature=cat_features,
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )
    pred = model.predict_proba(X.iloc[va_idx])[:, 1]
    auc = roc_auc_score(y_adv[va_idx], pred)
    auc_scores.append(auc)
    importances += model.feature_importances_
    print(f"  fold {fold}: AUC = {auc:.5f}")

mean_auc = np.mean(auc_scores)
print(f"\n  ★ 평균 adversarial AUC: {mean_auc:.5f}")
print(f"     해석: {'분포 차이 존재 (>0.55)' if mean_auc > 0.55 else '분포 거의 동일 (<0.55)'}")
print(f"     {'✅ shift 처리 시도 가치 있음' if mean_auc > 0.55 else '⚠️ shift 거의 없음 - 처리 무익'}")

# ─── 피처 중요도 분석 ────────────────────────────────────────────
print("\n[2] Top shift 피처 (feature importance 기준)...")
imp_df = pd.DataFrame({
    'feature': X.columns,
    'importance': importances / 5,
}).sort_values('importance', ascending=False)
print(imp_df.head(25).to_string(index=False))

# ─── 학습 데이터에 test와 가까운 sample weight 부여 가능 ──────────
print("\n[3] Sample weight 후보 계산 (전체 train 대상)...")
# 전체 데이터로 모델 다시 학습 → train 각 sample이 test로 분류될 확률 → weight
final_model = lgb.LGBMClassifier(
    n_estimators=500, learning_rate=0.05,
    num_leaves=63, min_child_samples=50,
    objective='binary', metric='auc',
    random_state=42, verbose=-1,
)
final_model.fit(X, y_adv, categorical_feature=cat_features)
train_pred = final_model.predict_proba(X.iloc[:len(X_tr)])[:, 1]
print(f"  train의 'test 같음' 확률 분포: mean={train_pred.mean():.4f}, "
      f"median={np.median(train_pred):.4f}, max={train_pred.max():.4f}")

# weight = pred / (1 - pred), normalize
weights = train_pred / (1 - train_pred + 1e-6)
weights = weights / weights.mean()
weights = np.clip(weights, 0.1, 5.0)
print(f"  weight (clip [0.1, 5.0]): mean={weights.mean():.4f}, "
      f"std={weights.std():.4f}, min={weights.min():.4f}, max={weights.max():.4f}")

np.save("adv_train_weights.npy", weights)
print(f"  저장: adv_train_weights.npy")

# 결과 요약 저장
result = {
    "adv_auc_mean": float(mean_auc),
    "adv_auc_per_fold": [float(a) for a in auc_scores],
    "shift_exists": bool(mean_auc > 0.55),
    "top_shift_features": imp_df.head(30).to_dict(orient='records'),
    "weight_stats": {
        "mean": float(weights.mean()),
        "median": float(np.median(weights)),
        "std": float(weights.std()),
        "min": float(weights.min()),
        "max": float(weights.max()),
    },
}
with open("adv_validation_result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(f"  저장: adv_validation_result.json")

print(f"\n{'='*75}")
print(f"  [Adv Val 결론]")
print(f"  Adversarial AUC: {mean_auc:.5f}")
if mean_auc > 0.55:
    print(f"  → distribution shift 존재")
    print(f"  → top {min(10, len(imp_df))} shift 피처 제거 후 재학습 권장")
    print(f"  → 또는 adv_train_weights.npy로 sample weighting")
else:
    print(f"  → 분포 거의 동일, 추가 처리 불필요")
print(f"{'='*75}")
