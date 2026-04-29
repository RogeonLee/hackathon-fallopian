"""
[전략 2] Pseudo-labeling — AG8h 고신뢰도 테스트 예측을 학습 데이터에 추가
THRESH_POS=0.82, THRESH_NEG=0.07 (보수적 설정)
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
THRESH_POS, THRESH_NEG = 0.82, 0.07

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
print("[전략 2] Pseudo-labeling (AG8h 고신뢰도)")
print(f"  POS threshold: {THRESH_POS}, NEG threshold: {THRESH_NEG}")
print("=" * 65)

train_raw = pd.read_csv("./data/train.csv")
test_raw  = pd.read_csv("./data/test.csv")
train = preprocess(train_raw); test = preprocess(test_raw)
train, test = encode_categories(train, test)

# AG8h 테스트 예측 로드 (블렌딩 예측 사용 — 더 안정적)
ag_test_f = sorted(glob.glob("test_ag8h_auc_*.npy"))
v8_test_f = sorted(glob.glob("test_v8_auc_*.npy"))
if not ag_test_f: raise FileNotFoundError("AG8h test 없음")

ag_test_pred = np.load(ag_test_f[-1])
v8_test_pred = np.load(v8_test_f[-1])
# 블렌딩 예측 사용 (더 안정적인 pseudo-label)
blend_test = 0.75*rank_norm(ag_test_pred) + 0.25*rank_norm(v8_test_pred)

# 고신뢰도 pseudo-label 선택
test_ids    = test_raw[ID_COL].values
mask_pos    = blend_test >= THRESH_POS
mask_neg    = blend_test <= THRESH_NEG
n_pos, n_neg = mask_pos.sum(), mask_neg.sum()
print(f"  Pseudo-label 후보: 양성 {n_pos}개, 음성 {n_neg}개 (총 {n_pos+n_neg}개 / {len(blend_test)}개)")

if n_pos + n_neg < 100:
    print("  임계값을 낮춤...")
    THRESH_POS, THRESH_NEG = 0.78, 0.09
    mask_pos = blend_test >= THRESH_POS
    mask_neg = blend_test <= THRESH_NEG
    n_pos, n_neg = mask_pos.sum(), mask_neg.sum()
    print(f"  재설정 후: 양성 {n_pos}개, 음성 {n_neg}개")

# Pseudo-labeled test 데이터 구성
pseudo_test = test_raw[mask_pos | mask_neg].copy()
pseudo_test = preprocess(pseudo_test)
pseudo_test[TARGET] = 0
pseudo_test.loc[mask_pos[mask_pos | mask_neg], TARGET] = 1

# 학습 데이터에 pseudo-label 추가
train_aug = pd.concat([train_raw, pseudo_test], ignore_index=True)
train_aug = preprocess(train_aug)
train_aug, test_enc = encode_categories(train_aug, test_raw.copy())
train_aug_enc = preprocess(train_aug)

# 실제 학습용 데이터
X_aug      = train_aug.drop(columns=[ID_COL, TARGET])
y_aug      = train_aug[TARGET].copy()
X_orig     = X_aug.iloc[:len(train_raw)]   # 원본 인덱스
y_orig     = y_aug.iloc[:len(train_raw)]
test_proc  = preprocess(test_raw)
_, test_enc2 = encode_categories(train_raw.copy(), test_raw.copy())
X_test     = test_enc2.drop(columns=[ID_COL])

# 카테고리 정렬
_, test_final = encode_categories(train_aug, test_raw)
X_aug_final = train_aug.drop(columns=[ID_COL, TARGET])
X_test_final = test_final.drop(columns=[ID_COL])

print(f"  증강 학습 데이터: {len(X_aug_final)}행 (원본 {len(train_raw)} + pseudo {n_pos+n_neg})")

LGBM_PARAMS = dict(
    n_estimators=3000, learning_rate=0.03262, num_leaves=170, max_depth=4,
    min_child_samples=30, subsample=0.8077, colsample_bytree=0.5257,
    reg_alpha=8.219, reg_lambda=0.1798, min_split_gain=0.0380,
    subsample_freq=1, n_jobs=-1, verbose=-1,
)

print("\n[2] Pseudo-label LGBM 5-Seed × 5-Fold (원본 데이터만 OOF 평가)...")
pl_oof  = np.zeros(len(train_raw))
pl_test = np.zeros(len(X_test_final))

for seed in SEEDS:
    # pseudo-label은 항상 학습에 포함, OOF는 원본 데이터만 평가
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    this_oof  = np.zeros(len(train_raw))
    this_test = np.zeros(len(X_test_final))
    orig_idx  = np.arange(len(train_raw))
    pseudo_idx = np.arange(len(train_raw), len(X_aug_final))

    for fold, (tr_i, va_i) in enumerate(skf.split(orig_idx, y_orig), 1):
        # 학습: 원본 학습 폴드 + 전체 pseudo
        tr_combined = np.concatenate([orig_idx[tr_i], pseudo_idx])
        m = LGBMClassifier(**{**LGBM_PARAMS, "random_state": seed})
        m.fit(X_aug_final.iloc[tr_combined], y_aug.iloc[tr_combined],
              eval_set=[(X_aug_final.iloc[orig_idx[va_i]], y_orig.iloc[va_i])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        this_oof[va_i]  = m.predict_proba(X_aug_final.iloc[orig_idx[va_i]])[:, 1]
        this_test      += m.predict_proba(X_test_final)[:, 1] / N_FOLDS

    s = roc_auc_score(y_orig, this_oof)
    print(f"  Seed {seed:>5}: {s:.5f}")
    pl_oof  += this_oof  / len(SEEDS)
    pl_test += this_test / len(SEEDS)

pl_auc = roc_auc_score(y_orig, pl_oof)
print(f"  Pseudo-label LGBM OOF AUC: {pl_auc:.5f}  ({pl_auc-0.74119:+.5f} vs AG8h blend)")
np.save(f"oof_pseudo_auc_{pl_auc:.5f}.npy", pl_oof)
np.save(f"test_pseudo_auc_{pl_auc:.5f}.npy", pl_test)

# v8 + AG8h + pseudo 블렌딩
v8_oof  = np.load(sorted(glob.glob("oof_v8_auc_*.npy"))[-1])
v8_test_arr = np.load(v8_test_f[-1])
ag_oof  = np.load(sorted(glob.glob("oof_ag8h_auc_*.npy"))[-1])

best_auc = 0; best_combo = None; best_test = None
for w_pl, w_ag, w_v8 in [(0.4,0.4,0.2),(0.5,0.3,0.2),(0.5,0.4,0.1),
                          (0.6,0.3,0.1),(0.4,0.5,0.1),(0.3,0.6,0.1),
                          (0.6,0.4,0.0),(0.7,0.3,0.0),(1.0,0.0,0.0)]:
    b = w_pl*rank_norm(pl_oof) + w_ag*rank_norm(ag_oof) + w_v8*rank_norm(v8_oof)
    a = roc_auc_score(y_orig, b)
    if a > best_auc:
        best_auc = a
        best_combo = (w_pl, w_ag, w_v8)
        best_test  = w_pl*rank_norm(pl_test) + w_ag*rank_norm(ag_test_pred) + w_v8*rank_norm(v8_test_arr)

print(f"\n  최적 블렌딩: pl={best_combo[0]}, ag={best_combo[1]}, v8={best_combo[2]} → OOF {best_auc:.5f}")
sub = pd.read_csv("./data/sample_submission.csv")
sub["probability"] = best_test
ts = datetime.now().strftime("%Y%m%d_%H%M")
fname = f"submission_{ts}_v17pseudo_oof{best_auc:.5f}.csv"
sub.to_csv(fname, index=False)
print(f"  저장: {fname}")
print(f"\n{'='*65}")
print(f"  Pseudo LGBM OOF:   {pl_auc:.5f}")
print(f"  최적 블렌딩 OOF:   {best_auc:.5f}")
print(f"  AG8h blend 대비:   {best_auc-0.74119:+.5f}")
print(f"{'='*65}")
