"""
[최종 결정] 모든 OOF 풀 (8종) NM 블렌드 + 갭 시뮬레이션
- v8, v9, ag8h, ag_raw_10h, agv10full, pseudo_v3, pseudo_v4, te_lgbm
- 마지막 1발 제출 카드 결정 보조
"""
import glob
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from scipy.optimize import minimize

TARGET = "임신 성공 여부"
def rn(a): return rankdata(a) / len(a)

train = pd.read_csv("./data/train.csv")
y = train[TARGET].values
sub_base = pd.read_csv("./data/sample_submission.csv")
ts = datetime.now().strftime("%Y%m%d_%H%M")

def load(pat):
    f = sorted(glob.glob(pat))
    return np.load(f[-1]) if f else None

models = {}
for tag, op, tp in [
    ("v8",         "oof_v8_auc_*.npy",         "test_v8_auc_*.npy"),
    ("v9",         "oof_v9_auc_*.npy",         "test_v9_auc_*.npy"),
    ("ag8h",       "oof_ag8h_auc_*.npy",       "test_ag8h_auc_*.npy"),
    ("ag_raw_10h", "oof_ag_raw_10h_auc_*.npy", "test_ag_raw_10h_auc_*.npy"),
    ("agv10full",  "oof_agv10full_auc_*.npy",  "test_agv10full_auc_*.npy"),
    ("pseudo_v3",  "oof_pseudo_v3_auc_*.npy",  "test_pseudo_v3_auc_*.npy"),
    ("pseudo_v4",  "oof_pseudo_v4_auc_*.npy",  "test_pseudo_v4_auc_*.npy"),
    ("te_lgbm",    "oof_te_lgbm_auc_*.npy",    "test_te_lgbm_auc_*.npy"),
]:
    o = load(op); t = load(tp)
    if o is not None and t is not None:
        models[tag] = (o, t, roc_auc_score(y, o))

print("="*78)
print("[최종 결정] 모든 OOF 풀 통합 분석")
print("="*78)
print(f"\n  단독 OOF:")
for n, (o, t, a) in models.items():
    print(f"    {n:>12s}: {a:.5f}")

# ─── 상관계수 매트릭스 ────────────────────────────────────
print("\n[1] 상관계수 매트릭스 (OOF rank, 다양성 검증)")
names = list(models.keys())
n = len(names)
ranks = {k: rn(v[0]) for k, v in models.items()}
print("        " + " ".join(f"{n[:8]:>9s}" for n in names))
for i, ni in enumerate(names):
    row = " ".join(f"{np.corrcoef(ranks[ni], ranks[nj])[0,1]:9.4f}" for nj in names)
    print(f"  {ni[:8]:>8s} {row}")

# ─── NM 블렌드 (전체 풀) ──────────────────────────────────
print("\n[2] NM 블렌드 (전체 8종 풀)")
oof_mat  = np.stack([rn(models[k][0]) for k in names], axis=1)
test_mat = np.stack([rn(models[k][1]) for k in names], axis=1)
K = len(names)

def neg_auc(x):
    w = np.exp(x - x.max()); w = w / w.sum()
    return -roc_auc_score(y, oof_mat @ w)

best_neg, best_w = 1.0, None
for seed in range(20):
    rng = np.random.default_rng(seed)
    res = minimize(neg_auc, rng.normal(size=K)*0.5, method="Nelder-Mead",
                   options={"xatol":1e-5, "fatol":1e-7, "maxiter":3000})
    if res.fun < best_neg:
        best_neg = res.fun
        best_w = np.exp(res.x - res.x.max()); best_w /= best_w.sum()
nm_full_auc = -best_neg
print(f"  NM full OOF={nm_full_auc:.5f}")
for n, w in sorted(zip(names, best_w), key=lambda x: -x[1]):
    print(f"    {n:>12s}: {w:.4f}")
blend_full = test_mat @ best_w
sub = sub_base.copy(); sub["probability"] = blend_full
fname_full = f"submission_{ts}_NMfull8_oof{nm_full_auc:.5f}.csv"
sub.to_csv(fname_full, index=False)

# ─── NM 블렌드 (효율, pseudo 제외, 깨끗한 풀) ─────────────────
print("\n[3] NM safe (효율/pseudo 제외, raw + AG만)")
safe_keys = ["v8", "v9", "ag8h", "ag_raw_10h", "te_lgbm"]
safe_keys = [k for k in safe_keys if k in models]
oof_s  = np.stack([rn(models[k][0]) for k in safe_keys], axis=1)
test_s = np.stack([rn(models[k][1]) for k in safe_keys], axis=1)
Ks = len(safe_keys)

def neg_auc_s(x):
    w = np.exp(x - x.max()); w = w / w.sum()
    return -roc_auc_score(y, oof_s @ w)

best_neg, best_w_s = 1.0, None
for seed in range(20):
    rng = np.random.default_rng(seed)
    res = minimize(neg_auc_s, rng.normal(size=Ks)*0.5, method="Nelder-Mead",
                   options={"xatol":1e-5, "fatol":1e-7, "maxiter":3000})
    if res.fun < best_neg:
        best_neg = res.fun
        best_w_s = np.exp(res.x - res.x.max()); best_w_s /= best_w_s.sum()
nm_safe_auc = -best_neg
print(f"  NM safe OOF={nm_safe_auc:.5f}")
for n, w in sorted(zip(safe_keys, best_w_s), key=lambda x: -x[1]):
    print(f"    {n:>12s}: {w:.4f}")
blend_safe = test_s @ best_w_s
sub = sub_base.copy(); sub["probability"] = blend_safe
fname_safe = f"submission_{ts}_NMsafe5_raw_oof{nm_safe_auc:.5f}.csv"
sub.to_csv(fname_safe, index=False)

# ─── 모든 후보 카드 LB 갭 시나리오 ──────────────────────────
print("\n[4] 모든 후보 LB 갭 시나리오")
print(f"  검증 갭:")
print(f"    v8 (raw, AG 0%):              +0.00129 → LB 0.74208")
print(f"    ag8h+v8 (AG 73%):             +0.00099 → LB 0.74218")
print(f"    NM safe 4-way (AG 72%+ps28%): -0.00081 → LB 0.74218")
print(f"  → AG 들어간 카드 LB 천장 = 0.74218")
print(f"  → pseudo는 LB 무익 (train fit만)")
print(f"  → 1위 0.74246 = 우리와 다른 종류의 신호 보유")
print()

# 모든 미제출 후보 정리
candidates = [
    ("ag_raw_10h × v8 (w=0.81) ★",      0.74129, "AG 81%, pseudo 0%, eff 0%"),
    ("ag_raw_10h 단독",                   0.74126, "AG 100%, pseudo 0%, eff 0%"),
    ("ag_raw_10h × pseudo (w=0.69)",      0.74136, "AG 69%+ps 31%, eff 0%"),
    ("3-way agraw+ps+v8",                 0.74135, "AG 65%+ps 25%+v8 10%, eff 0%"),
    ("NM safe 5-way (te 포함)",           nm_safe_auc, "v8+v9+ag8h+agraw+te"),
    ("NM full 8-way (모두)",              nm_full_auc, "전체 풀"),
    ("(검증) ag8h+v8 → 0.74218",          0.74119, "AG 73%, 검증됨"),
    ("(검증) NM safe 4way → 0.74218",     0.74137, "AG 72%+ps28%, 검증됨"),
]
print(f"{'후보':<55s} {'OOF':>8s} {'갭+0.00129':>11s} {'갭+0.00099':>11s} {'갭-0.00081':>11s}")
print("-"*102)
for name, oof, _ in candidates:
    lb129 = oof + 0.00129
    lb099 = oof + 0.00099
    lb_neg = oof - 0.00081
    star = ""
    if lb099 > 0.74246: star = "★확실 추월"
    elif lb129 > 0.74246: star = "△가능 추월"
    print(f"{name:<55s} {oof:8.5f} {lb129:11.5f} {lb099:11.5f} {lb_neg:11.5f}  {star}")

print(f"\n  1위 LB: 0.74246 / 우리 LB: 0.74218 / 격차: -0.00028")
