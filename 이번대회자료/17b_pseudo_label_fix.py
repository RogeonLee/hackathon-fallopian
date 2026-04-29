"""
[전략 2 수정] Pseudo-labeling — 카테고리 불일치 수정
핵심 수정: train+pseudo 합친 뒤 encode_categories 재적용
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
N_FOLDS, SEEDS = 5, [42, 2024, 777, 1234, 31337]
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

def encode_all(train_df, test_df):
    """train+augmented, test 동시에 카테고리 정렬"""
    obj_cols = [c for c in train_df.select_dtypes("object").columns if c not in [ID_COL, TARGET]]
    for col in obj_cols:
        all_vals = sorted(
            set(train_df[col].dropna().astype(str)) |
            set(test_df[col].dropna().astype(str)),
            key=str
        )
        train_df[col] = pd.Categorical(train_df[col].astype(str), categories=all_vals)
        test_df[col]  = pd.Categorical(test_df[col].astype(str),  categories=all_vals)
    return train_df, test_df

print("=" * 65)
print("[전략 2] Pseudo-labeling (수정)")
print("=" * 65)

train_raw = pd.read_csv("./data/train.csv")
test_raw  = pd.read_csv("./data/test.csv")

# AG8h + v8 블렌딩 예측으로 고신뢰도 샘플 선택
ag_test  = np.load(sorted(glob.glob("test_ag8h_auc_*.npy"))[-1])
v8_test  = np.load(sorted(glob.glob("test_v8_auc_*.npy"))[-1])
blend_pred = 0.75*rank_norm(ag_test) + 0.25*rank_norm(v8_test)

mask_pos = blend_pred >= THRESH_POS
mask_neg = blend_pred <= THRESH_NEG
n_pos, n_neg = mask_pos.sum(), mask_neg.sum()
print(f"  Pseudo-label: 양성 {n_pos}개, 음성 {n_neg}개 (총 {n_pos+n_neg}개)")

# Pseudo-label 데이터 구성
pseudo_rows = test_raw[mask_pos | mask_neg].copy()
pseudo_rows[TARGET] = np.where(mask_pos[mask_pos | mask_neg], 1, 0)

# train + pseudo 합치기 (raw 상태에서)
train_aug_raw = pd.concat([train_raw, pseudo_rows], ignore_index=True)
n_orig = len(train_raw)

# 전처리 (수치형 변환)
train_aug = preprocess(train_aug_raw)
test_proc = preprocess(test_raw)

# 카테고리 통일 인코딩 (train_aug와 test 동시)
train_aug, test_proc = encode_all(train_aug, test_proc)

X_aug  = train_aug.drop(columns=[ID_COL, TARGET])
y_aug  = train_aug[TARGET].copy()
X_test = test_proc.drop(columns=[ID_COL])

# 원본 / pseudo 인덱스
orig_idx   = np.arange(n_orig)
pseudo_idx = np.arange(n_orig, len(X_aug))
y_orig     = y_aug.iloc[:n_orig]

print(f"  학습 데이터: {n_orig} + {len(pseudo_idx)} = {len(X_aug)}행")

LGBM_PARAMS = dict(
    n_estimators=3000, learning_rate=0.03262, num_leaves=170, max_depth=4,
    min_child_samples=30, subsample=0.8077, colsample_bytree=0.5257,
    reg_alpha=8.219, reg_lambda=0.1798, min_split_gain=0.0380,
    subsample_freq=1, n_jobs=-1, verbose=-1,
)

print("\n[2] Pseudo LGBM 5-Seed × 5-Fold...")
pl_oof  = np.zeros(n_orig)
pl_test = np.zeros(len(X_test))

for seed in SEEDS:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    this_oof  = np.zeros(n_orig)
    this_test = np.zeros(len(X_test))
    for fold, (tr_i, va_i) in enumerate(skf.split(orig_idx, y_orig), 1):
        tr_combined = np.concatenate([orig_idx[tr_i], pseudo_idx])
        m = LGBMClassifier(**{**LGBM_PARAMS, "random_state": seed})
        m.fit(
            X_aug.iloc[tr_combined], y_aug.iloc[tr_combined],
            eval_set=[(X_aug.iloc[orig_idx[va_i]], y_orig.iloc[va_i])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
        )
        this_oof[va_i]  = m.predict_proba(X_aug.iloc[orig_idx[va_i]])[:, 1]
        this_test      += m.predict_proba(X_test)[:, 1] / N_FOLDS
    s = roc_auc_score(y_orig, this_oof)
    print(f"  Seed {seed:>5}: {s:.5f}")
    pl_oof  += this_oof  / len(SEEDS)
    pl_test += this_test / len(SEEDS)

pl_auc = roc_auc_score(y_orig, pl_oof)
print(f"  Pseudo LGBM OOF AUC: {pl_auc:.5f}  ({pl_auc-0.74119:+.5f} vs AG8h blend)")
np.save(f"oof_pseudo_auc_{pl_auc:.5f}.npy", pl_oof)
np.save(f"test_pseudo_auc_{pl_auc:.5f}.npy", pl_test)

# 블렌딩
ag_oof  = np.load(sorted(glob.glob("oof_ag8h_auc_*.npy"))[-1])
v8_oof  = np.load(sorted(glob.glob("oof_v8_auc_*.npy"))[-1])
v8_t    = np.load(sorted(glob.glob("test_v8_auc_*.npy"))[-1])

best_auc, best_combo, best_test_out = 0, None, None
for w_pl, w_ag, w_v8 in [
    (0.5,0.4,0.1),(0.6,0.3,0.1),(0.4,0.5,0.1),
    (0.6,0.4,0.0),(0.7,0.3,0.0),(0.5,0.5,0.0),
    (0.8,0.2,0.0),(1.0,0.0,0.0),(0.4,0.4,0.2),
]:
    b = w_pl*rank_norm(pl_oof) + w_ag*rank_norm(ag_oof) + w_v8*rank_norm(v8_oof)
    a = roc_auc_score(y_orig, b)
    if a > best_auc:
        best_auc = a; best_combo = (w_pl,w_ag,w_v8)
        best_test_out = w_pl*rank_norm(pl_test) + w_ag*rank_norm(ag_test) + w_v8*rank_norm(v8_t)

print(f"\n  최적: pl={best_combo[0]},ag={best_combo[1]},v8={best_combo[2]} → OOF {best_auc:.5f}")

sub = pd.read_csv("./data/sample_submission.csv")
sub["probability"] = best_test_out
ts = datetime.now().strftime("%Y%m%d_%H%M")
fname = f"submission_{ts}_v17bpseudo_oof{best_auc:.5f}.csv"
sub.to_csv(fname, index=False)
print(f"  저장: {fname}")
print(f"\n{'='*65}")
print(f"  Pseudo OOF:        {pl_auc:.5f}")
print(f"  최적 블렌딩 OOF:   {best_auc:.5f}  ({best_auc-0.74119:+.5f} vs AG8h blend)")
print(f"{'='*65}")
