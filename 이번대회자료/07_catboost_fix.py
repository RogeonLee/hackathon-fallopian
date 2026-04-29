"""
CatBoost 단독 실행 + v5 앙상블 개선
NaN → "missing" 처리로 오류 수정
"""
import glob, warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

warnings.filterwarnings("ignore")

TARGET  = "임신 성공 여부"
ID_COL  = "ID"
N_FOLDS = 5
SEED    = 42

def rank_norm(arr):
    return rankdata(arr) / len(arr)

# ─── 전처리 (v4/v5 방식) ─────────────────────────────────────────
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

def encode_categories(train, test):
    train, test = train.copy(), test.copy()
    obj_cols = [c for c in train.select_dtypes("object").columns if c not in [ID_COL, TARGET]]
    for col in obj_cols:
        train[col] = train[col].astype("category")
        if col in test.columns:
            test[col] = test[col].astype("category")
        all_cats = sorted(
            set(train[col].cat.categories) |
            (set(test[col].cat.categories) if col in test.columns else set()),
            key=str
        )
        train[col] = train[col].cat.set_categories(all_cats)
        if col in test.columns:
            test[col] = test[col].cat.set_categories(all_cats)
    return train, test

print("CatBoost 단독 실행 (NaN → 'missing' 수정)")
train_raw = pd.read_csv("./data/train.csv")
test_raw  = pd.read_csv("./data/test.csv")

train = preprocess(train_raw)
test  = preprocess(test_raw)
train, test = encode_categories(train, test)

X = train.drop(columns=[ID_COL, TARGET])
y = train[TARGET].copy()
X_test = test.drop(columns=[ID_COL])

# CatBoost용 전처리: category → object → fillna → str
X_cb      = X.copy()
X_test_cb = X_test.copy()
for col in X_cb.select_dtypes("category").columns:
    # astype(object) 먼저: NaN을 float NaN으로 꺼낸 후 fillna
    X_cb[col]      = X_cb[col].astype(object).fillna("missing").astype(str)
    X_test_cb[col] = X_test_cb[col].astype(object).fillna("missing").astype(str)

# object 타입 NaN도 처리
for col in X_cb.select_dtypes(object).columns:
    X_cb[col]      = X_cb[col].fillna("missing").astype(str)
    X_test_cb[col] = X_test_cb[col].fillna("missing").astype(str)

cat_feat_names = list(X_cb.select_dtypes(object).columns)
print(f"  카테고리 피처: {len(cat_feat_names)}개")

from catboost import CatBoostClassifier

CB_PARAMS = dict(
    iterations            = 3000,
    learning_rate         = 0.05,
    depth                 = 6,
    l2_leaf_reg           = 3,
    eval_metric           = "AUC",
    early_stopping_rounds = 50,
    verbose               = 0,
    thread_count          = -1,
)
CB_SEEDS = [42, 2024, 777, 1234, 31337]

cb_oof  = np.zeros(len(X))
cb_test = np.zeros(len(X_test))

for seed in CB_SEEDS:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    this_oof  = np.zeros(len(X))
    this_test = np.zeros(len(X_test))
    for fold, (tr_i, va_i) in enumerate(skf.split(X, y), 1):
        m = CatBoostClassifier(**{**CB_PARAMS, "random_seed": seed})
        m.fit(
            X_cb.iloc[tr_i], y.iloc[tr_i],
            eval_set=(X_cb.iloc[va_i], y.iloc[va_i]),
            cat_features=cat_feat_names,
        )
        this_oof[va_i] = m.predict_proba(X_cb.iloc[va_i])[:, 1]
        this_test     += m.predict_proba(X_test_cb)[:, 1] / N_FOLDS
    s_auc = roc_auc_score(y, this_oof)
    print(f"  CB Seed {seed:>5}: {s_auc:.5f}")
    cb_oof  += this_oof  / len(CB_SEEDS)
    cb_test += this_test / len(CB_SEEDS)

cb_auc = roc_auc_score(y, cb_oof)
print(f"  CatBoost 앙상블 OOF AUC: {cb_auc:.5f}")

# v5 결과와 블렌딩
v5_oof_files  = sorted(glob.glob("oof_v5_auc_*.npy"))
v5_test_files = sorted(glob.glob("test_v5_auc_*.npy"))

if v5_oof_files:
    v5_oof  = np.load(v5_oof_files[-1])
    v5_test = np.load(v5_test_files[-1])
    v5_auc  = roc_auc_score(y, v5_oof)
    print(f"\n  v5 OOF AUC:  {v5_auc:.5f}")
    print(f"  CB OOF AUC:  {cb_auc:.5f}")

    # 블렌딩 (v5는 여러 모델 앙상블, CB는 단독 모델 — 적절한 가중치)
    blend_oof  = 0.7 * rank_norm(v5_oof)  + 0.3 * rank_norm(cb_oof)
    blend_test = 0.7 * rank_norm(v5_test) + 0.3 * rank_norm(cb_test)
    blend_auc  = roc_auc_score(y, blend_oof)

    # 동일 가중치 블렌딩도 시험
    equal_oof  = 0.5 * rank_norm(v5_oof)  + 0.5 * rank_norm(cb_oof)
    equal_test = 0.5 * rank_norm(v5_test) + 0.5 * rank_norm(cb_test)
    equal_auc  = roc_auc_score(y, equal_oof)

    print(f"\n  블렌딩 OOF AUC:")
    print(f"    v5 only:    {v5_auc:.5f}")
    print(f"    70% v5 + 30% CB: {blend_auc:.5f}")
    print(f"    50% v5 + 50% CB: {equal_auc:.5f}")

    best_auc = max(v5_auc, cb_auc, blend_auc, equal_auc)
    if equal_auc == best_auc:
        final_test, final_name = equal_test, "50-50blend"
    elif blend_auc == best_auc:
        final_test, final_name = blend_test, "70-30blend"
    elif cb_auc == best_auc:
        final_test, final_name = cb_test, "catboost"
    else:
        final_test, final_name = v5_test, "v5"
else:
    final_test, final_name, best_auc = cb_test, "catboost", cb_auc

sub = pd.read_csv("./data/sample_submission.csv")
sub["probability"] = final_test
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
filename  = f"submission_{timestamp}_v7_oof{best_auc:.5f}.csv"
sub.to_csv(filename, index=False)
print(f"\n  저장: {filename} ({final_name})")

np.save(f"oof_cb5seed_auc_{cb_auc:.5f}.npy", cb_oof)
np.save(f"test_cb5seed_auc_{cb_auc:.5f}.npy", cb_test)

print(f"\n  v4 기준선 대비: {best_auc - 0.74033:+.5f}")
