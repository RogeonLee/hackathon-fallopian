"""
[전략 1] AG OOF를 메타 피처로 LGBM 재학습
AG의 앙상블 지식을 새 피처로 추가 → LGBM이 AG의 패턴을 활용
"""
import glob, warnings
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import lightgbm as lgb
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
TARGET, ID_COL = "임신 성공 여부", "ID"
N_FOLDS, SEEDS, SEED = 5, [42, 2024, 777, 1234, 31337], 42
V8_LB = 0.74208

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

def encode_categories(train, test):
    train, test = train.copy(), test.copy()
    for col in [c for c in train.select_dtypes("object").columns if c not in [ID_COL, TARGET]]:
        train[col] = train[col].astype("category")
        if col in test.columns: test[col] = test[col].astype("category")
        cats = sorted(set(train[col].cat.categories) | (set(test[col].cat.categories) if col in test.columns else set()), key=str)
        train[col] = train[col].cat.set_categories(cats)
        if col in test.columns: test[col] = test[col].cat.set_categories(cats)
    return train, test

print("=" * 65)
print("[전략 1] AG OOF 메타 피처 LGBM")
print("=" * 65)

train_raw = pd.read_csv("./data/train.csv")
test_raw  = pd.read_csv("./data/test.csv")
train = preprocess(train_raw); test = preprocess(test_raw)
train, test = encode_categories(train, test)
X = train.drop(columns=[ID_COL, TARGET]); y = train[TARGET].copy()
X_test = test.drop(columns=[ID_COL])

# AG8h OOF 로드
ag_oof_f  = sorted(glob.glob("oof_ag8h_auc_*.npy"))
ag_test_f = sorted(glob.glob("test_ag8h_auc_*.npy"))
if not ag_oof_f:
    raise FileNotFoundError("AG8h OOF 없음 — 15번 먼저 실행 필요")
ag_oof  = np.load(ag_oof_f[-1])
ag_test_pred = np.load(ag_test_f[-1])
print(f"  AG8h OOF AUC: {roc_auc_score(y, ag_oof):.5f}")

# v5, v8, CB OOF도 로드해서 메타 피처 세트 구성
meta_oofs  = {"ag": ag_oof}
meta_tests = {"ag": ag_test_pred}
for pat, name in [("oof_v5_auc_*.npy","v5"), ("oof_cb5seed_auc_*.npy","cb"),
                  ("oof_cbt_auc_*.npy","cbt"), ("oof_v8_auc_*.npy","v8")]:
    fs = sorted(glob.glob(pat))
    if fs:
        meta_oofs[name]  = np.load(fs[-1])
        meta_tests[name] = np.load(sorted(glob.glob(pat.replace("oof_","test_")))[-1])
        print(f"  {name} OOF AUC: {roc_auc_score(y, meta_oofs[name]):.5f}")

# 메타 피처 추가
X_meta      = X.copy()
X_test_meta = X_test.copy()
for name, arr in meta_oofs.items():
    X_meta[f"meta_{name}"]      = arr
    X_test_meta[f"meta_{name}"] = meta_tests[name]
print(f"  메타 피처 {len(meta_oofs)}개 추가 → 총 피처: {X_meta.shape[1]}")

# LGBM 학습 (v5 최적 파라미터)
LGBM_PARAMS = dict(
    n_estimators=3000, learning_rate=0.03262, num_leaves=170, max_depth=4,
    min_child_samples=30, subsample=0.8077, colsample_bytree=0.5257,
    reg_alpha=8.219, reg_lambda=0.1798, min_split_gain=0.0380,
    subsample_freq=1, n_jobs=-1, verbose=-1,
)

print("\n[2] LGBM + 메타피처 5-Seed × 5-Fold...")
meta_oof_pred  = np.zeros(len(X_meta))
meta_test_pred = np.zeros(len(X_test_meta))

for seed in SEEDS:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    this_oof = np.zeros(len(X_meta)); this_test = np.zeros(len(X_test_meta))
    for fold, (tr_i, va_i) in enumerate(skf.split(X_meta, y), 1):
        m = LGBMClassifier(**{**LGBM_PARAMS, "random_state": seed})
        m.fit(X_meta.iloc[tr_i], y.iloc[tr_i],
              eval_set=[(X_meta.iloc[va_i], y.iloc[va_i])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        this_oof[va_i]  = m.predict_proba(X_meta.iloc[va_i])[:, 1]
        this_test      += m.predict_proba(X_test_meta)[:, 1] / N_FOLDS
    s = roc_auc_score(y, this_oof)
    print(f"  Seed {seed:>5}: {s:.5f}")
    meta_oof_pred  += this_oof  / len(SEEDS)
    meta_test_pred += this_test / len(SEEDS)

meta_auc = roc_auc_score(y, meta_oof_pred)
print(f"  메타 LGBM OOF AUC: {meta_auc:.5f}  ({meta_auc-0.74119:+.5f} vs AG8h blend)")
np.save(f"oof_meta_lgbm_auc_{meta_auc:.5f}.npy", meta_oof_pred)
np.save(f"test_meta_lgbm_auc_{meta_auc:.5f}.npy", meta_test_pred)

# v8 블렌딩
v8_oof  = meta_oofs["v8"]; v8_test = meta_tests["v8"]
best_auc, best_w = 0, 0.5
for w in np.arange(0.1, 0.95, 0.05):
    a = roc_auc_score(y, w*rank_norm(meta_oof_pred) + (1-w)*rank_norm(v8_oof))
    if a > best_auc: best_auc, best_w = a, round(w,2)

final_test = best_w*rank_norm(meta_test_pred) + (1-best_w)*rank_norm(v8_test)
print(f"\n  최적 블렌딩: meta={best_w}, v8={1-best_w} → OOF {best_auc:.5f}")

sub = pd.read_csv("./data/sample_submission.csv")
sub["probability"] = final_test
ts = datetime.now().strftime("%Y%m%d_%H%M")
fname = f"submission_{ts}_v16meta_oof{best_auc:.5f}.csv"
sub.to_csv(fname, index=False)
print(f"  저장: {fname}")
print(f"\n{'='*65}")
print(f"  메타 LGBM OOF:  {meta_auc:.5f}")
print(f"  최적 블렌딩 OOF: {best_auc:.5f}")
print(f"  AG8h blend 대비: {best_auc-0.74119:+.5f}")
print(f"{'='*65}")
