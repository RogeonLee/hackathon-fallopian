"""
[작전 A] Pseudo Labeling v3 강화판
- Threshold rank 0.95/0.05 (라운드 1) → 0.92/0.08 (라운드 2, 점진 확대)
- Iterative 2라운드
- Pseudo sample weight 0.6 (확신도 보정)
- LGBM 5-seed × 5-fold (raw + native categorical)
- Seed 예측: 현재 LB 최강 0.74218 (ag8h × v8, w0.73)

핵심: pseudo는 항상 training set에만 들어감 (val/OOF는 original만)
"""
import glob, os, warnings
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
SEEDS   = [42, 2024, 777]   # 시간 절약 위해 3-seed (필요시 5-seed 확장)
PSEUDO_WEIGHT = 0.6

def rank_norm(a): return rankdata(a) / len(a)

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
        all_vals = sorted(
            set(train_df[col].dropna().astype(str)) |
            set(test_df[col].dropna().astype(str)),
            key=str
        )
        train_df[col] = pd.Categorical(train_df[col].astype(str), categories=all_vals)
        test_df[col]  = pd.Categorical(test_df[col].astype(str),  categories=all_vals)
    return train_df, test_df

LGBM_PARAMS = dict(
    n_estimators=3000, learning_rate=0.02,
    num_leaves=63, max_depth=-1,
    min_child_samples=50, colsample_bytree=0.8, subsample=0.8,
    reg_alpha=0.1, reg_lambda=0.1,
    objective='binary', metric='auc',
    verbose=-1,
)

def make_pseudo_rows(test_raw, test_pred_rank, thresh_pos, thresh_neg):
    """rank percentile 기준 pseudo 행 생성"""
    mask_pos = test_pred_rank >= thresh_pos
    mask_neg = test_pred_rank <= thresh_neg
    n_pos, n_neg = mask_pos.sum(), mask_neg.sum()
    pseudo_rows = test_raw[mask_pos | mask_neg].copy()
    pseudo_rows[TARGET] = np.where(mask_pos[mask_pos | mask_neg], 1, 0).astype(float)
    print(f"  Pseudo: pos={n_pos}, neg={n_neg}, total={n_pos+n_neg}")
    return pseudo_rows

def train_pseudo_lgbm(train_raw, test_raw, pseudo_rows, y_orig, seeds=SEEDS):
    """pseudo는 항상 training fold에만 들어가도록 학습"""
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

    # sample_weight: original=1.0, pseudo=PSEUDO_WEIGHT
    sw = np.concatenate([np.ones(n_orig), np.full(n_pseudo, PSEUDO_WEIGHT)])

    oof_pred  = np.zeros(n_orig)
    test_pred = np.zeros(len(X_test))

    n_models = 0
    for seed in seeds:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(np.arange(n_orig), y_orig)):
            # training set = train_fold + pseudo_indices
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

    oof_pred  = oof_pred  / len(seeds)            # OOF는 시드 평균 (각 fold 1회씩)
    test_pred = test_pred / n_models               # test는 모든 모델 평균
    auc = roc_auc_score(y_orig, oof_pred)
    return oof_pred, test_pred, auc

# ─────────────────────────────────────────────────────
print("=" * 75)
print("[작전 A] Pseudo Labeling v3 강화판 (iterative 2라운드)")
print("=" * 75)

train_raw = pd.read_csv("./data/train.csv")
test_raw  = pd.read_csv("./data/test.csv")
y_orig    = train_raw[TARGET].astype(float)

# 시드 예측: 현재 LB 0.74218 (ag8h × v8 페어, w0.73)
ag_test = np.load(sorted(glob.glob("test_ag8h_auc_*.npy"))[-1])
v8_test = np.load(sorted(glob.glob("test_v8_auc_*.npy"))[-1])
seed_pred = 0.73 * rank_norm(ag_test) + 0.27 * rank_norm(v8_test)
seed_pred_rank = rank_norm(seed_pred)
print(f"\n  Seed pred: ag8h(0.73) + v8(0.27)")

# ─── 라운드 1: threshold 0.95 / 0.05 ──────────────────────────
print("\n" + "─"*75)
print("[라운드 1] Threshold 0.95 / 0.05")
print("─"*75)
pseudo_r1 = make_pseudo_rows(test_raw, seed_pred_rank, 0.95, 0.05)
print("  학습 시작...")
oof_r1, test_r1, auc_r1 = train_pseudo_lgbm(train_raw, test_raw, pseudo_r1, y_orig)
print(f"\n  ★ Round 1 OOF AUC: {auc_r1:.5f}")

# ─── 라운드 2: 새 OOF + test로 더 정확한 pseudo 생성 ───────────
print("\n" + "─"*75)
print("[라운드 2] 라운드 1 결과 + ag8h+v8 블렌드로 pseudo 재생성, threshold 0.92/0.08")
print("─"*75)
# 라운드 2 시드: round1 test 예측을 추가 블렌딩
seed_pred_r2 = 0.5 * rank_norm(test_r1) + 0.3 * rank_norm(ag_test) + 0.2 * rank_norm(v8_test)
seed_pred_r2_rank = rank_norm(seed_pred_r2)
pseudo_r2 = make_pseudo_rows(test_raw, seed_pred_r2_rank, 0.92, 0.08)
print("  학습 시작...")
oof_r2, test_r2, auc_r2 = train_pseudo_lgbm(train_raw, test_raw, pseudo_r2, y_orig)
print(f"\n  ★ Round 2 OOF AUC: {auc_r2:.5f}")

# ─── 두 라운드 평균 (앙상블) ──────────────────────────────────
print("\n" + "─"*75)
print("[라운드 1 × 라운드 2 평균]")
print("─"*75)
oof_avg  = 0.5 * oof_r1  + 0.5 * oof_r2
test_avg = 0.5 * test_r1 + 0.5 * test_r2
auc_avg = roc_auc_score(y_orig, oof_avg)
print(f"  ★ 평균 OOF AUC: {auc_avg:.5f}")

# rank 평균도
oof_rk  = 0.5 * rank_norm(oof_r1)  + 0.5 * rank_norm(oof_r2)
test_rk = 0.5 * rank_norm(test_r1) + 0.5 * rank_norm(test_r2)
auc_rk = roc_auc_score(y_orig, oof_rk)
print(f"  ★ Rank 평균 OOF AUC: {auc_rk:.5f}")

# 더 좋은 쪽 채택
if auc_rk >= auc_avg:
    oof_final, test_final, auc_final = oof_rk, test_rk, auc_rk
    print(f"  → Rank 평균 채택 ({auc_rk:.5f} ≥ {auc_avg:.5f})")
else:
    oof_final, test_final, auc_final = oof_avg, test_avg, auc_avg
    print(f"  → 단순 평균 채택 ({auc_avg:.5f} > {auc_rk:.5f})")

# ─── 저장 ────────────────────────────────────────────────────
np.save(f"oof_pseudo_v3_auc_{auc_final:.5f}.npy", oof_final)
np.save(f"test_pseudo_v3_auc_{auc_final:.5f}.npy", test_final)
np.save(f"oof_pseudo_v3_r1_auc_{auc_r1:.5f}.npy", oof_r1)
np.save(f"test_pseudo_v3_r1_auc_{auc_r1:.5f}.npy", test_r1)
np.save(f"oof_pseudo_v3_r2_auc_{auc_r2:.5f}.npy", oof_r2)
np.save(f"test_pseudo_v3_r2_auc_{auc_r2:.5f}.npy", test_r2)

# ─── 즉시 블렌드 분석 ────────────────────────────────────────
print("\n" + "─"*75)
print("[기존 OOF 풀 + pseudo_v3 NM 블렌드]")
print("─"*75)
from scipy.optimize import minimize

candidates = [
    ("v8",         "oof_v8_auc_*.npy",         "test_v8_auc_*.npy"),
    ("v9",         "oof_v9_auc_*.npy",         "test_v9_auc_*.npy"),
    ("ag8h",       "oof_ag8h_auc_*.npy",       "test_ag8h_auc_*.npy"),
    ("agv10full",  "oof_agv10full_auc_*.npy",  "test_agv10full_auc_*.npy"),
]
pools = []
for tag, op, tp in candidates:
    of = sorted(glob.glob(op))
    tf = sorted(glob.glob(tp))
    if of and tf:
        o = np.load(of[-1]); t = np.load(tf[-1])
        try:
            a = roc_auc_score(y_orig, o)
            pools.append((tag, rank_norm(o), rank_norm(t), a))
            print(f"  - {tag}: OOF={a:.5f}")
        except: pass

pools.append(("pseudo_v3", rank_norm(oof_final), rank_norm(test_final), auc_final))
print(f"  - pseudo_v3 (★신규): OOF={auc_final:.5f}")

oof_mat  = np.stack([p[1] for p in pools], axis=1)
test_mat = np.stack([p[2] for p in pools], axis=1)
names    = [p[0] for p in pools]
K = len(names)

def neg_auc(x):
    w = np.exp(x - x.max()); w = w / w.sum()
    return -roc_auc_score(y_orig, oof_mat @ w)

best_neg, best_w = 1.0, None
for seed in range(8):
    rng = np.random.default_rng(seed)
    x0  = rng.normal(size=K) * 0.5
    res = minimize(neg_auc, x0, method="Nelder-Mead",
                   options={"xatol":1e-4, "fatol":1e-6, "maxiter":2000})
    if res.fun < best_neg:
        best_neg = res.fun
        best_w = np.exp(res.x - res.x.max()); best_w /= best_w.sum()

best_oof = -best_neg
print(f"\n  ★ NM 최적 OOF: {best_oof:.5f}")
for n, w in sorted(zip(names, best_w), key=lambda x: -x[1]):
    print(f"    {n:>12s}: {w:.4f}")

sub_base = pd.read_csv("./data/sample_submission.csv")
ts = datetime.now().strftime("%Y%m%d_%H%M")

# pseudo_v3 단독
sub = sub_base.copy(); sub["probability"] = test_final
sub.to_csv(f"submission_{ts}_pseudo_v3_alone_oof{auc_final:.5f}.csv", index=False)

# NM 블렌드 (pseudo 포함)
blend = test_mat @ best_w
sub = sub_base.copy(); sub["probability"] = blend
fname_nm = f"submission_{ts}_pseudo_v3_NMblend_oof{best_oof:.5f}.csv"
sub.to_csv(fname_nm, index=False)

# pseudo + ag8h + v8 페어 그리드 (안전 변형)
print("\n[pseudo_v3 + ag8h + v8 3-way 그리드]")
v8_oof  = np.load(sorted(glob.glob("oof_v8_auc_*.npy"))[-1])
ag8_oof = np.load(sorted(glob.glob("oof_ag8h_auc_*.npy"))[-1])
v8_t  = np.load(sorted(glob.glob("test_v8_auc_*.npy"))[-1])
ag8_t = np.load(sorted(glob.glob("test_ag8h_auc_*.npy"))[-1])
best3, w3 = 0, (1/3, 1/3, 1/3)
for w1 in np.arange(0.05, 0.91, 0.05):
    for w2 in np.arange(0.05, 0.96-w1, 0.05):
        ww3 = 1 - w1 - w2
        if ww3 < 0.05: continue
        a = roc_auc_score(y_orig, w1*rank_norm(v8_oof) + w2*rank_norm(ag8_oof) + ww3*rank_norm(oof_final))
        if a > best3:
            best3, w3 = a, (round(w1,2), round(w2,2), round(ww3,2))
print(f"  최적 3-way OOF={best3:.5f} (v8={w3[0]}, ag8h={w3[1]}, pseudo={w3[2]})")
blend3 = w3[0]*rank_norm(v8_t) + w3[1]*rank_norm(ag8_t) + w3[2]*rank_norm(test_final)
sub = sub_base.copy(); sub["probability"] = blend3
sub.to_csv(f"submission_{ts}_pseudo_v3_3way_v8_ag8h_oof{best3:.5f}.csv", index=False)

print(f"\n{'='*75}")
print(f"  [pseudo v3 결과 요약]")
print(f"  R1 OOF (0.95/0.05) : {auc_r1:.5f}")
print(f"  R2 OOF (0.92/0.08) : {auc_r2:.5f}")
print(f"  평균 OOF (best)    : {auc_final:.5f}")
print(f"  NM 블렌드 OOF      : {best_oof:.5f}")
print(f"  3-way OOF          : {best3:.5f}")
print(f"  v8 갭 +0.00129 가산 시 LB 추정: NM={best_oof+0.00129:.5f}, 3way={best3+0.00129:.5f}")
print(f"  1위 0.74246 대비: NM={0.74246-(best_oof+0.00129):+.5f}")
print(f"{'='*75}")
