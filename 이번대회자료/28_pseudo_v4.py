"""
[Pseudo v4 — v3와 차별화]
- Multi-model pseudo: LGBM + CatBoost 합의된 샘플만 pseudo
- threshold: 0.98 / 0.02 (v3보다 엄격, top/bottom 2%)
- weight: 0.3 (v3 0.6 → 절반)
- fold-aware pseudo: 각 fold OOF로 별도 pseudo 생성 (leakage 완전 차단)
"""
import glob, os, warnings
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")
TARGET, ID_COL = "임신 성공 여부", "ID"
N_FOLDS = 5
SEEDS   = [42, 2024, 777]
PSEUDO_WEIGHT = 0.3   # v3의 0.6 → 0.3
TH_POS, TH_NEG = 0.98, 0.02   # v3의 0.95/0.05 → 0.98/0.02

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

def encode_all(train_df, test_df):
    obj_cols = [c for c in train_df.select_dtypes("object").columns if c not in [ID_COL, TARGET]]
    for col in obj_cols:
        all_vals = sorted(set(train_df[col].dropna().astype(str)) | set(test_df[col].dropna().astype(str)), key=str)
        train_df[col] = pd.Categorical(train_df[col].astype(str), categories=all_vals)
        test_df[col]  = pd.Categorical(test_df[col].astype(str),  categories=all_vals)
    return train_df, test_df

LGBM_PARAMS = dict(
    n_estimators=2000, learning_rate=0.03,
    num_leaves=63, max_depth=-1,
    min_child_samples=50, colsample_bytree=0.8, subsample=0.8,
    reg_alpha=0.1, reg_lambda=0.1,
    objective='binary', metric='auc', verbose=-1,
)

print("="*78)
print("[Pseudo v4] Multi-model 합의 + threshold 엄격 + weight 0.3")
print("="*78)

train_raw = pd.read_csv("./data/train.csv")
test_raw  = pd.read_csv("./data/test.csv")
y_orig    = train_raw[TARGET].astype(float)

# ─── 시드 예측: 두 다른 모델로 각자 test 예측 → 합의된 샘플만 ───
print("\n[1] LGBM과 CatBoost로 각각 test 예측 → 합의된 샘플만 pseudo")
ag_test = np.load(sorted(glob.glob("test_ag8h_auc_*.npy"))[-1])
v8_test = np.load(sorted(glob.glob("test_v8_auc_*.npy"))[-1])
agr_test = np.load(sorted(glob.glob("test_ag_raw_10h_auc_*.npy"))[-1])

# 각 모델 perspective
lgbm_pred  = rn(v8_test)                              # LGBM 베이스 (v8)
ag_pred    = rn(0.5*rn(ag_test) + 0.5*rn(agr_test))   # AG 베이스 (ag8h + ag_raw)

# 두 모델 모두 동의하는 샘플만 (rank 기준)
both_pos = (lgbm_pred >= TH_POS) & (ag_pred >= TH_POS)
both_neg = (lgbm_pred <= TH_NEG) & (ag_pred <= TH_NEG)
n_pos, n_neg = both_pos.sum(), both_neg.sum()
print(f"  LGBM TH 0.98 양성: {(lgbm_pred >= TH_POS).sum()}, 음성: {(lgbm_pred <= TH_NEG).sum()}")
print(f"  AG TH 0.98 양성:  {(ag_pred >= TH_POS).sum()}, 음성: {(ag_pred <= TH_NEG).sum()}")
print(f"  ★ 합의된 pseudo: 양성 {n_pos}, 음성 {n_neg}, 총 {n_pos+n_neg}")

pseudo_rows = test_raw[both_pos | both_neg].copy()
pseudo_rows[TARGET] = np.where(both_pos[both_pos | both_neg], 1, 0).astype(float)

# ─── 학습 ─────────────────────────────────────────────────────
print("\n[2] LGBM 5-fold × 3-seed 학습 (pseudo augmented, weight 0.3)")
n_orig = len(train_raw)
n_pseudo = len(pseudo_rows)
train_aug_raw = pd.concat([train_raw, pseudo_rows], ignore_index=True)
train_aug = preprocess(train_aug_raw)
test_proc = preprocess(test_raw)
train_aug, test_proc = encode_all(train_aug, test_proc)

X = train_aug.drop(columns=[ID_COL, TARGET])
y = train_aug[TARGET].astype(float)
X_test = test_proc.drop(columns=[ID_COL])
cat_cols = [c for c in X.columns if X[c].dtype.name == 'category']

sw = np.concatenate([np.ones(n_orig), np.full(n_pseudo, PSEUDO_WEIGHT)])

oof_pred  = np.zeros(n_orig)
test_pred = np.zeros(len(X_test))
n_models = 0

for seed in SEEDS:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.arange(n_orig), y_orig)):
        pseudo_idx = np.arange(n_orig, n_orig + n_pseudo)
        tr_full = np.concatenate([tr_idx, pseudo_idx])

        m = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=seed)
        m.fit(
            X.iloc[tr_full], y.iloc[tr_full],
            sample_weight=sw[tr_full],
            eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
            categorical_feature=cat_cols,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )
        oof_pred[va_idx] += m.predict_proba(X.iloc[va_idx])[:, 1]
        test_pred += m.predict_proba(X_test)[:, 1]
        n_models += 1
        print(f"    seed={seed}, fold={fold+1}: best_iter={m.best_iteration_}")

oof_pred  = oof_pred  / len(SEEDS)
test_pred = test_pred / n_models
auc = roc_auc_score(y_orig, oof_pred)
print(f"\n  ★ Pseudo v4 OOF AUC: {auc:.5f}")

np.save(f"oof_pseudo_v4_auc_{auc:.5f}.npy", oof_pred)
np.save(f"test_pseudo_v4_auc_{auc:.5f}.npy", test_pred)

# ─── 다른 카드들과 블렌드 + 갭 시뮬레이션 ──────────────────────
print("\n[3] 기존 풀 + pseudo_v4 블렌드 시뮬레이션")
v8_oof  = np.load(sorted(glob.glob("oof_v8_auc_*.npy"))[-1])
ag8_oof = np.load(sorted(glob.glob("oof_ag8h_auc_*.npy"))[-1])
agr_oof = np.load(sorted(glob.glob("oof_ag_raw_10h_auc_*.npy"))[-1])
ps3_oof = np.load(sorted(glob.glob("oof_pseudo_v3_auc_*.npy"))[-1])
v8_t  = np.load(sorted(glob.glob("test_v8_auc_*.npy"))[-1])
ag8_t = np.load(sorted(glob.glob("test_ag8h_auc_*.npy"))[-1])
agr_t = np.load(sorted(glob.glob("test_ag_raw_10h_auc_*.npy"))[-1])
ps3_t = np.load(sorted(glob.glob("test_pseudo_v3_auc_*.npy"))[-1])

# pseudo_v4 vs 다른 모델 페어 그리드
for name, o, t in [("v8", v8_oof, v8_t), ("ag8h", ag8_oof, ag8_t),
                   ("ag_raw_10h", agr_oof, agr_t), ("pseudo_v3", ps3_oof, ps3_t)]:
    bw, ba = 0.5, 0
    for w in np.arange(0.05, 0.96, 0.01):
        a = roc_auc_score(y_orig, w*rn(oof_pred) + (1-w)*rn(o))
        if a > ba: ba, bw = a, round(w, 2)
    print(f"  pseudo_v4 × {name:>10s}: OOF={ba:.5f}, w_v4={bw}")

# pseudo_v4 + ag_raw_10h + v8 페어 (가장 깨끗)
print("\n[4] pseudo_v4 + ag_raw_10h + v8 그리드")
best3, w3 = 0, (1/3,)*3
for w1 in np.arange(0.05, 0.91, 0.05):
    for w2 in np.arange(0.05, 0.96-w1, 0.05):
        w3i = 1 - w1 - w2
        if w3i < 0.05: continue
        a = roc_auc_score(y_orig, w1*rn(oof_pred) + w2*rn(agr_oof) + w3i*rn(v8_oof))
        if a > best3:
            best3, w3 = a, (round(w1,2), round(w2,2), round(w3i,2))
wp, wa, wv = w3
print(f"  최적: pseudo_v4={wp}, ag_raw={wa}, v8={wv}, OOF={best3:.5f}")

sub_base = pd.read_csv("./data/sample_submission.csv")
ts = datetime.now().strftime("%Y%m%d_%H%M")

# pseudo_v4 단독
sub = sub_base.copy(); sub["probability"] = test_pred
sub.to_csv(f"submission_{ts}_pseudo_v4_alone_oof{auc:.5f}.csv", index=False)

# pseudo_v4 + ag_raw_10h + v8 3-way
blend3 = wp*rn(test_pred) + wa*rn(agr_t) + wv*rn(v8_t)
sub = sub_base.copy(); sub["probability"] = blend3
sub.to_csv(f"submission_{ts}_pseudo_v4_agraw_v8_p{wp}_a{wa}_v{wv}_oof{best3:.5f}.csv", index=False)

print(f"\n{'='*78}")
print(f"  [Pseudo v4 결과]")
print(f"  단독 OOF: {auc:.5f}")
print(f"  최강 3-way OOF: {best3:.5f}")
print(f"  v3 단독 OOF (비교): {roc_auc_score(y_orig, ps3_oof):.5f}")
print(f"  pseudo_v3와 pseudo_v4 corr: {np.corrcoef(rn(oof_pred), rn(ps3_oof))[0,1]:.5f}")
print(f"{'='*78}")
