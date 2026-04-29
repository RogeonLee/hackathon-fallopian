"""
[Pseudo v3 안전 변형] 효율 피처 비중 단계별 + 갭 시뮬레이션
- 가장 안전: pseudo_v3 + v8 (둘 다 raw LGBM)
- 균형: pseudo_v3 + ag8h (+ 약간의 v8)
- 공격: pseudo_v3 + ag8h + agv10full (NM 결과)
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

v8_oof, v8_t   = load("oof_v8_auc_*.npy"), load("test_v8_auc_*.npy")
v9_oof, v9_t   = load("oof_v9_auc_*.npy"), load("test_v9_auc_*.npy")
ag8_oof, ag8_t = load("oof_ag8h_auc_*.npy"), load("test_ag8h_auc_*.npy")
agf_oof, agf_t = load("oof_agv10full_auc_*.npy"), load("test_agv10full_auc_*.npy")
ps_oof, ps_t   = load("oof_pseudo_v3_auc_*.npy"), load("test_pseudo_v3_auc_*.npy")

print("="*75)
print("Pseudo v3 안전 블렌드 + 갭 시뮬레이션")
print("="*75)
print(f"\n  단독 OOF:")
for n, o in [("v8",v8_oof),("v9",v9_oof),("ag8h",ag8_oof),("agv10full",agf_oof),("pseudo_v3",ps_oof)]:
    if o is not None:
        print(f"    {n:>10s}: {roc_auc_score(y, o):.5f}")

# ─── [1] pseudo + v8 페어 (둘 다 raw LGBM, 가장 안전) ──────────
print("\n[1] pseudo + v8 페어 (효율 피처 0%, AG 0%) — 가장 보수적")
bw, ba = 0.5, 0
for w in np.arange(0.05, 0.96, 0.01):
    a = roc_auc_score(y, w*rn(ps_oof) + (1-w)*rn(v8_oof))
    if a > ba: ba, bw = a, round(w, 2)
print(f"  최적 OOF={ba:.5f}  (w_pseudo={bw}, w_v8={1-bw:.2f})")
blend = bw*rn(ps_t) + (1-bw)*rn(v8_t)
sub = sub_base.copy(); sub["probability"] = blend
fname1 = f"submission_{ts}_pseudo_v8_w{bw}_oof{ba:.5f}.csv"
sub.to_csv(fname1, index=False)
print(f"  저장: {fname1}")
oof_pv8, test_pv8 = bw*rn(ps_oof) + (1-bw)*rn(v8_oof), blend

# ─── [2] pseudo + ag8h 페어 (다양성 강화) ─────────────────────
print("\n[2] pseudo + ag8h 페어 (효율 피처 0%, ag8h ON)")
bw, ba = 0.5, 0
for w in np.arange(0.05, 0.96, 0.01):
    a = roc_auc_score(y, w*rn(ps_oof) + (1-w)*rn(ag8_oof))
    if a > ba: ba, bw = a, round(w, 2)
print(f"  최적 OOF={ba:.5f}  (w_pseudo={bw}, w_ag8h={1-bw:.2f})")
blend = bw*rn(ps_t) + (1-bw)*rn(ag8_t)
sub = sub_base.copy(); sub["probability"] = blend
fname2 = f"submission_{ts}_pseudo_ag8h_w{bw}_oof{ba:.5f}.csv"
sub.to_csv(fname2, index=False)
print(f"  저장: {fname2}")

# ─── [3] pseudo + ag8h + v8 그리드 (3-way, v8 비중 ↑) ─────────
print("\n[3] pseudo + ag8h + v8 (3-way, 효율 피처 0%, v8 다양화)")
best3, w3 = 0, (0.33, 0.33, 0.34)
for wp in np.arange(0.05, 0.91, 0.05):
    for wa in np.arange(0.05, 0.96-wp, 0.05):
        wv = 1 - wp - wa
        if wv < 0.05: continue
        a = roc_auc_score(y, wp*rn(ps_oof) + wa*rn(ag8_oof) + wv*rn(v8_oof))
        if a > best3:
            best3, w3 = a, (round(wp,2), round(wa,2), round(wv,2))
wp, wa, wv = w3
print(f"  최적 OOF={best3:.5f}  (pseudo={wp}, ag8h={wa}, v8={wv})")
blend = wp*rn(ps_t) + wa*rn(ag8_t) + wv*rn(v8_t)
sub = sub_base.copy(); sub["probability"] = blend
fname3 = f"submission_{ts}_pseudo_ag8h_v8_p{wp}_a{wa}_v{wv}_oof{best3:.5f}.csv"
sub.to_csv(fname3, index=False)
print(f"  저장: {fname3}")

# ─── [4] NM (이미 24번 만든것) - 참고용 OOF만 다시 계산 ─────────
print("\n[4] NM 풀 (참고)")
pools = []
for tag, o, t in [("v8",v8_oof,v8_t),("v9",v9_oof,v9_t),("ag8h",ag8_oof,ag8_t),
                  ("agv10full",agf_oof,agf_t),("pseudo_v3",ps_oof,ps_t)]:
    if o is not None and t is not None:
        pools.append((tag, rn(o), rn(t), roc_auc_score(y, o)))

oof_mat  = np.stack([p[1] for p in pools], axis=1)
test_mat = np.stack([p[2] for p in pools], axis=1)
names    = [p[0] for p in pools]
K = len(names)

def neg_auc(x):
    w = np.exp(x - x.max()); w = w / w.sum()
    return -roc_auc_score(y, oof_mat @ w)

best_neg, best_w = 1.0, None
for seed in range(8):
    rng = np.random.default_rng(seed)
    res = minimize(neg_auc, rng.normal(size=K)*0.5, method="Nelder-Mead",
                   options={"xatol":1e-4, "fatol":1e-6, "maxiter":2000})
    if res.fun < best_neg:
        best_neg = res.fun
        best_w = np.exp(res.x - res.x.max()); best_w /= best_w.sum()

best_oof = -best_neg
print(f"  NM OOF={best_oof:.5f}")
for n, w in sorted(zip(names, best_w), key=lambda x: -x[1]):
    print(f"    {n:>10s}: {w:.4f}")

# ─── [5] 효율 피처 제외 NM (agv10full 빼기) ────────────────────
print("\n[5] NM (agv10full 제외, 효율 피처 0%)")
pools_safe = [p for p in pools if p[0] != "agv10full"]
oof_mat_s  = np.stack([p[1] for p in pools_safe], axis=1)
test_mat_s = np.stack([p[2] for p in pools_safe], axis=1)
names_s    = [p[0] for p in pools_safe]
Ks = len(names_s)

def neg_auc_s(x):
    w = np.exp(x - x.max()); w = w / w.sum()
    return -roc_auc_score(y, oof_mat_s @ w)

best_neg_s, best_w_s = 1.0, None
for seed in range(8):
    rng = np.random.default_rng(seed)
    res = minimize(neg_auc_s, rng.normal(size=Ks)*0.5, method="Nelder-Mead",
                   options={"xatol":1e-4, "fatol":1e-6, "maxiter":2000})
    if res.fun < best_neg_s:
        best_neg_s = res.fun
        best_w_s = np.exp(res.x - res.x.max()); best_w_s /= best_w_s.sum()
best_oof_s = -best_neg_s
print(f"  NM(safe) OOF={best_oof_s:.5f}")
for n, w in sorted(zip(names_s, best_w_s), key=lambda x: -x[1]):
    print(f"    {n:>10s}: {w:.4f}")
blend_s = test_mat_s @ best_w_s
sub = sub_base.copy(); sub["probability"] = blend_s
fname5 = f"submission_{ts}_pseudo_NMsafe_noeff_oof{best_oof_s:.5f}.csv"
sub.to_csv(fname5, index=False)
print(f"  저장: {fname5}")

# ─── [6] 갭 시뮬레이션 표 ─────────────────────────────────────
print("\n" + "="*75)
print("[6] LB 갭 시나리오 (검증된 데이터 기반)")
print("="*75)
print(f"  검증된 갭:")
print(f"    v8 (raw, AG 0%, 효율 0%):                  +0.00129")
print(f"    옵션 A 3-way (ag8h 75%, 효율 5%):           +0.00099")
print(f"    ag8h+v8 페어 (ag8h 73%, 효율 0%):           +0.00099")
print(f"  → ag8h 비중 70%+ 들어가면 갭 약 +0.00099")
print(f"  → 효율 피처 영향은 미세 (5% → 갭 변화 거의 없음)")
print()
print(f"{'후보':<60s} {'OOF':>8s} {'AG비중':>8s} {'LB(갭+0.00129)':>13s} {'LB(갭+0.00099)':>13s}")
print("-"*108)

# AG 비중 추정 함수
def ag_weight(blend_components):
    """blend_components: list of (name, weight)"""
    return sum(w for n, w in blend_components if n in ['ag8h', 'agv10full'])

candidates_full = [
    (f"pseudo+v8 페어 (raw 100%)",                                     ba_pv8 := roc_auc_score(y, oof_pv8), 0.0),
    (f"pseudo+ag8h 페어",                                              roc_auc_score(y, bw*rn(ps_oof) + (1-bw)*rn(ag8_oof)), 0.5),
    (f"pseudo+ag8h+v8 (p={wp},a={wa},v={wv})",                        best3, wa),
    (f"NM safe (effort 제외, ag8h+pseudo+v8+v9)",                     best_oof_s, dict(zip(names_s, best_w_s)).get('ag8h', 0)),
    (f"NM full (agv10full 포함, 24번 NM)",                             best_oof, dict(zip(names, best_w)).get('ag8h', 0) + dict(zip(names, best_w)).get('agv10full', 0)),
    (f"(참고) ag8h+v8 (LB 0.74218 검증)",                              0.74119, 0.73),
    (f"(참고) v8 단독 (LB 0.74208 검증)",                              0.74079, 0.0),
]

for name, oof, agw in candidates_full:
    lb129 = oof + 0.00129
    lb099 = oof + 0.00099
    star = "★1위 추월(0.74246)" if max(lb129, lb099) > 0.74246 else ""
    print(f"{name:<60s} {oof:8.5f} {agw:8.2f} {lb129:13.5f} {lb099:13.5f}  {star}")

print(f"\n  1위 LB: 0.74246")
print(f"  현재 최고 LB: 0.74218 (ag8h+v8 페어)")
