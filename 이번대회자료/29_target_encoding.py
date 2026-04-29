"""
[Categorical Encoding 재실험]
- 5-fold OOF Target Encoding (smoothing 10 / 30 / 100)
- Frequency encoding
- Combination encoding (시술유형 × 시술시기, 나이 × 시술유형)
- Hierarchical encoding (나이그룹별 평균 success)
- LGBM 학습 → 다른 종류 신호 OOF
"""
import glob, warnings
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import lightgbm as lgb

warnings.filterwarnings("ignore")
TARGET, ID_COL = "임신 성공 여부", "ID"
N_FOLDS = 5
SEEDS   = [42, 2024, 777]

def rn(a): return rankdata(a) / len(a)

COUNT_COLS = ["총 시술 횟수","클리닉 내 총 시술 횟수","IVF 시술 횟수","DI 시술 횟수",
              "총 임신 횟수","IVF 임신 횟수","DI 임신 횟수","총 출산 횟수","IVF 출산 횟수","DI 출산 횟수"]

def preprocess(df):
    df = df.copy()
    for col in COUNT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    df["ever_delivered"]   = (df["총 출산 횟수"].fillna(0) > 0).astype(int)
    df["is_FET"]           = (df["해동된 배아 수"].fillna(0) > 0).astype(int)
    df["is_first_attempt"] = (df["총 시술 횟수"].fillna(0) == 0).astype(int)
    return df

# Target Encoding 함수 (smoothing)
def fit_te(train_col, y, smoothing=30):
    """train col → mean target with smoothing"""
    global_mean = y.mean()
    counts = train_col.groupby(train_col).count() if hasattr(train_col, 'groupby') else None
    df = pd.DataFrame({'col': train_col.values, 'y': y.values})
    grouped = df.groupby('col')['y'].agg(['mean', 'count'])
    smooth = (grouped['mean']*grouped['count'] + global_mean*smoothing) / (grouped['count'] + smoothing)
    return smooth.to_dict(), global_mean

def transform_te(col, mapping, global_mean):
    return col.map(mapping).fillna(global_mean).astype('float32')

def fit_freq(train_col):
    """frequency encoding"""
    counts = train_col.value_counts(dropna=False).to_dict()
    return counts

def transform_freq(col, mapping):
    return col.map(mapping).fillna(0).astype('float32')

print("="*78)
print("[작전 H] Categorical Encoding 재실험")
print("="*78)

train_raw = pd.read_csv("./data/train.csv")
test_raw  = pd.read_csv("./data/test.csv")
y_orig = train_raw[TARGET].astype(int)
train_p = preprocess(train_raw)
test_p  = preprocess(test_raw)

# 카테고리 컬럼 식별
cat_cols = [c for c in train_p.select_dtypes("object").columns if c not in [ID_COL, TARGET]]
print(f"\n  카테고리 컬럼 ({len(cat_cols)}개): {cat_cols[:10]}...")

# ─── Frequency Encoding ────────────────────────────────────
print("\n[1] Frequency Encoding...")
combined = pd.concat([train_p[cat_cols], test_p[cat_cols]], axis=0)
freq_features_train = pd.DataFrame(index=train_p.index)
freq_features_test  = pd.DataFrame(index=test_p.index)
for c in cat_cols:
    fmap = combined[c].value_counts(dropna=False).to_dict()
    freq_features_train[f'freq_{c}'] = train_p[c].map(fmap).fillna(0).astype('float32')
    freq_features_test[f'freq_{c}']  = test_p[c].map(fmap).fillna(0).astype('float32')
print(f"  freq features: {freq_features_train.shape[1]}개")

# ─── Combination Encoding ──────────────────────────────────
print("\n[2] Combination Encoding (top pairs)...")
combo_pairs = [
    ("시술 유형", "시술 시기 코드"),
    ("시술 유형", "시술 당시 나이"),
    ("시술 시기 코드", "시술 당시 나이"),
    ("배란 유도 유형", "시술 유형"),
    ("난자 출처", "정자 출처"),
]
combo_train = pd.DataFrame(index=train_p.index)
combo_test  = pd.DataFrame(index=test_p.index)
for c1, c2 in combo_pairs:
    if c1 in cat_cols and c2 in cat_cols:
        new_col = f"{c1}_X_{c2}"
        combo_train[new_col] = train_p[c1].astype(str) + "_" + train_p[c2].astype(str)
        combo_test[new_col]  = test_p[c1].astype(str)  + "_" + test_p[c2].astype(str)
print(f"  combo features: {combo_train.shape[1]}개")

# ─── 5-fold OOF Target Encoding (smoothing 30) ──────────────
print("\n[3] 5-fold OOF TE (smoothing=30) for cat + combo")
te_train = pd.DataFrame(index=train_p.index)
te_test  = pd.DataFrame(index=test_p.index)

all_cat = list(cat_cols) + list(combo_train.columns)

# combine for TE: train_p[cat] + combo_train, test_p[cat] + combo_test
df_tr_te = pd.concat([train_p[cat_cols].reset_index(drop=True),
                      combo_train.reset_index(drop=True)], axis=1)
df_te_te = pd.concat([test_p[cat_cols].reset_index(drop=True),
                      combo_test.reset_index(drop=True)], axis=1)

for c in all_cat:
    df_tr_te[c] = df_tr_te[c].astype(str)
    df_te_te[c] = df_te_te[c].astype(str)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
for c in all_cat:
    oof_te = np.zeros(len(train_p))
    for tr_idx, va_idx in skf.split(np.arange(len(train_p)), y_orig):
        mapping, gm = fit_te(df_tr_te[c].iloc[tr_idx], y_orig.iloc[tr_idx], smoothing=30)
        oof_te[va_idx] = df_tr_te[c].iloc[va_idx].map(mapping).fillna(gm)
    te_train[f'te30_{c}'] = oof_te.astype('float32')
    # test: 전체 train으로
    mapping_full, gm = fit_te(df_tr_te[c], y_orig, smoothing=30)
    te_test[f'te30_{c}'] = df_te_te[c].map(mapping_full).fillna(gm).astype('float32')

print(f"  TE features: {te_train.shape[1]}개")

# ─── Hierarchical Encoding (나이그룹 mean) ──────────────────
print("\n[4] Hierarchical Encoding (나이그룹별 평균)")
# 나이 + 시술유형 평균 (이미 combo로 일부 커버됨, 추가 안 함)
# 임시로 패스

# ─── LGBM 학습 (raw + freq + TE) ─────────────────────────
print("\n[5] LGBM 학습 (raw + freq + te 조합, 5-fold × 3-seed)")
# raw 피처
def encode_native(train_df, test_df, cat_cols):
    for col in cat_cols:
        all_vals = sorted(set(train_df[col].dropna().astype(str)) | set(test_df[col].dropna().astype(str)), key=str)
        train_df[col] = pd.Categorical(train_df[col].astype(str), categories=all_vals)
        test_df[col]  = pd.Categorical(test_df[col].astype(str),  categories=all_vals)
    return train_df, test_df

base_train, base_test = encode_native(train_p.copy(), test_p.copy(), cat_cols)

# 합치기: raw + freq + TE
X_tr = pd.concat([
    base_train.drop(columns=[ID_COL, TARGET]).reset_index(drop=True),
    freq_features_train.reset_index(drop=True),
    te_train.reset_index(drop=True),
], axis=1)
X_te = pd.concat([
    base_test.drop(columns=[ID_COL]).reset_index(drop=True),
    freq_features_test.reset_index(drop=True),
    te_test.reset_index(drop=True),
], axis=1)
print(f"  최종 X_tr shape: {X_tr.shape}")

cat_for_lgbm = [c for c in X_tr.columns if X_tr[c].dtype.name == 'category']

LGBM_PARAMS = dict(
    n_estimators=2000, learning_rate=0.03,
    num_leaves=63, max_depth=-1,
    min_child_samples=50, colsample_bytree=0.8, subsample=0.8,
    reg_alpha=0.1, reg_lambda=0.1,
    objective='binary', metric='auc', verbose=-1,
)

oof_te_lgbm = np.zeros(len(y_orig))
test_te_lgbm = np.zeros(len(X_te))
n_models = 0

for seed in SEEDS:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y_orig)):
        m = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=seed)
        m.fit(
            X_tr.iloc[tr_idx], y_orig.iloc[tr_idx],
            eval_set=[(X_tr.iloc[va_idx], y_orig.iloc[va_idx])],
            categorical_feature=cat_for_lgbm,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )
        oof_te_lgbm[va_idx] += m.predict_proba(X_tr.iloc[va_idx])[:, 1]
        test_te_lgbm += m.predict_proba(X_te)[:, 1]
        n_models += 1
        print(f"    seed={seed}, fold={fold+1}: best_iter={m.best_iteration_}")

oof_te_lgbm  /= len(SEEDS)
test_te_lgbm /= n_models
auc_te = roc_auc_score(y_orig, oof_te_lgbm)
print(f"\n  ★ TE LGBM OOF AUC: {auc_te:.5f}")

np.save(f"oof_te_lgbm_auc_{auc_te:.5f}.npy", oof_te_lgbm)
np.save(f"test_te_lgbm_auc_{auc_te:.5f}.npy", test_te_lgbm)

# ─── 다른 OOF와 페어 그리드 + 상관계수 ──────────────────────
print("\n[6] 기존 OOF와 페어 블렌드 + 상관계수 분석")
for tag, op, tp in [("v8","oof_v8_auc_*.npy","test_v8_auc_*.npy"),
                    ("ag_raw_10h","oof_ag_raw_10h_auc_*.npy","test_ag_raw_10h_auc_*.npy"),
                    ("pseudo_v3","oof_pseudo_v3_auc_*.npy","test_pseudo_v3_auc_*.npy")]:
    of = sorted(glob.glob(op)); tf = sorted(glob.glob(tp))
    if not of: continue
    o = np.load(of[-1]); t = np.load(tf[-1])
    corr = np.corrcoef(rn(oof_te_lgbm), rn(o))[0, 1]
    bw, ba = 0.5, 0
    for w in np.arange(0.05, 0.96, 0.01):
        a = roc_auc_score(y_orig, w*rn(oof_te_lgbm) + (1-w)*rn(o))
        if a > ba: ba, bw = a, round(w, 2)
    print(f"  TE_LGBM × {tag:>12s}: corr={corr:.4f}, OOF={ba:.5f}, w_te={bw}")

sub_base = pd.read_csv("./data/sample_submission.csv")
ts = datetime.now().strftime("%Y%m%d_%H%M")
sub = sub_base.copy(); sub["probability"] = test_te_lgbm
sub.to_csv(f"submission_{ts}_te_lgbm_alone_oof{auc_te:.5f}.csv", index=False)

print(f"\n{'='*78}")
print(f"  [TE LGBM 결과]")
print(f"  단독 OOF: {auc_te:.5f}")
print(f"  v8(0.74079) 대비: {auc_te-0.74079:+.5f}")
print(f"{'='*78}")
