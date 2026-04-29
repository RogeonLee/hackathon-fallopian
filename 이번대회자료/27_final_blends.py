"""
[최종 블렌드] AG raw 10h 완료 후, 모든 OOF 풀로 최강 카드 생성
- ag_raw_10h (stack=0, bag=16) + pseudo_v3 페어 (가장 깨끗한 다양성)
- ag_raw_10h + v8 + pseudo 3-way
- 모든 NM 변형
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
agr_oof, agr_t = load("oof_ag_raw_10h_auc_*.npy"), load("test_ag_raw_10h_auc_*.npy")

print("="*78)
print("[최종 블렌드] 모든 OOF 풀 통합 (AG raw 10h + pseudo_v3 추가)")
print("="*78)
print(f"\n  단독 OOF:")
for n, o in [("v8",v8_oof),("v9",v9_oof),("ag8h",ag8_oof),("agv10full",agf_oof),
             ("pseudo_v3",ps_oof),("ag_raw_10h ★신규",agr_oof)]:
    if o is not None:
        print(f"    {n:>20s}: {roc_auc_score(y, o):.5f}")

# ─── [1] ag_raw_10h × pseudo 페어 (★ 깨끗한 다양성) ──────────
print("\n[1] ag_raw_10h × pseudo_v3 페어 (효율 0%, 두 stack=0 베이스, 깨끗)")
bw, ba = 0.5, 0
for w in np.arange(0.05, 0.96, 0.01):
    a = roc_auc_score(y, w*rn(agr_oof) + (1-w)*rn(ps_oof))
    if a > ba: ba, bw = a, round(w, 2)
print(f"  최적 OOF={ba:.5f}  (w_agraw={bw}, w_pseudo={1-bw:.2f})")
blend = bw*rn(agr_t) + (1-bw)*rn(ps_t)
sub = sub_base.copy(); sub["probability"] = blend
fname = f"submission_{ts}_FINAL_agraw_pseudo_w{bw}_oof{ba:.5f}.csv"
sub.to_csv(fname, index=False)
print(f"  저장: {fname}")

# ─── [2] ag_raw_10h × v8 페어 (이미 만들어짐) ────────────────
print("\n[2] ag_raw_10h × v8 페어 (raw 베이스만)")
bw2, ba2 = 0.5, 0
for w in np.arange(0.05, 0.96, 0.01):
    a = roc_auc_score(y, w*rn(agr_oof) + (1-w)*rn(v8_oof))
    if a > ba2: ba2, bw2 = a, round(w, 2)
print(f"  최적 OOF={ba2:.5f}  (w_agraw={bw2})")

# ─── [3] ag_raw_10h + pseudo + v8 3-way ───────────────────
print("\n[3] ag_raw_10h + pseudo_v3 + v8 3-way (효율 0%, ag8h 0%)")
best3, w3 = 0, (1/3,)*3
for wa in np.arange(0.05, 0.91, 0.05):
    for wp in np.arange(0.05, 0.96-wa, 0.05):
        wv = 1 - wa - wp
        if wv < 0.05: continue
        a = roc_auc_score(y, wa*rn(agr_oof) + wp*rn(ps_oof) + wv*rn(v8_oof))
        if a > best3:
            best3, w3 = a, (round(wa,2), round(wp,2), round(wv,2))
wa, wp, wv = w3
print(f"  최적 OOF={best3:.5f} (agraw={wa}, pseudo={wp}, v8={wv})")
blend3 = wa*rn(agr_t) + wp*rn(ps_t) + wv*rn(v8_t)
sub = sub_base.copy(); sub["probability"] = blend3
fname3 = f"submission_{ts}_FINAL_agraw_pseudo_v8_a{wa}_p{wp}_v{wv}_oof{best3:.5f}.csv"
sub.to_csv(fname3, index=False)
print(f"  저장: {fname3}")

# ─── [4] ag_raw_10h + ag8h + pseudo + v8 (효율 제외) ─────────
print("\n[4] NM 4-way (ag_raw, ag8h, pseudo, v8 — 효율 제외)")
pools_safe = [
    ("ag_raw_10h", rn(agr_oof), rn(agr_t)),
    ("ag8h",       rn(ag8_oof), rn(ag8_t)),
    ("pseudo_v3",  rn(ps_oof),  rn(ps_t)),
    ("v8",         rn(v8_oof),  rn(v8_t)),
]
oof_mat_s  = np.stack([p[1] for p in pools_safe], axis=1)
test_mat_s = np.stack([p[2] for p in pools_safe], axis=1)
names_s    = [p[0] for p in pools_safe]

def neg_auc_s(x):
    w = np.exp(x - x.max()); w = w / w.sum()
    return -roc_auc_score(y, oof_mat_s @ w)

best_neg, best_w = 1.0, None
for seed in range(12):
    rng = np.random.default_rng(seed)
    res = minimize(neg_auc_s, rng.normal(size=len(names_s))*0.5, method="Nelder-Mead",
                   options={"xatol":1e-5, "fatol":1e-7, "maxiter":3000})
    if res.fun < best_neg:
        best_neg = res.fun
        best_w = np.exp(res.x - res.x.max()); best_w /= best_w.sum()
nm_safe_auc = -best_neg
print(f"  NM safe OOF={nm_safe_auc:.5f}")
for n, w in sorted(zip(names_s, best_w), key=lambda x: -x[1]):
    print(f"    {n:>12s}: {w:.4f}")
blend_safe = test_mat_s @ best_w
sub = sub_base.copy(); sub["probability"] = blend_safe
fname4 = f"submission_{ts}_FINAL_NMsafe_4way_oof{nm_safe_auc:.5f}.csv"
sub.to_csv(fname4, index=False)
print(f"  저장: {fname4}")

# ─── [5] NM 풀 (모든 OOF 포함, 효율 포함) ──────────────────
print("\n[5] NM full (효율 포함, 6모델)")
pools_full = pools_safe + [
    ("agv10full", rn(agf_oof), rn(agf_t)),
    ("v9",        rn(v9_oof),  rn(v9_t)),
]
oof_mat_f  = np.stack([p[1] for p in pools_full], axis=1)
test_mat_f = np.stack([p[2] for p in pools_full], axis=1)
names_f    = [p[0] for p in pools_full]

def neg_auc_f(x):
    w = np.exp(x - x.max()); w = w / w.sum()
    return -roc_auc_score(y, oof_mat_f @ w)

best_neg, best_w_f = 1.0, None
for seed in range(12):
    rng = np.random.default_rng(seed)
    res = minimize(neg_auc_f, rng.normal(size=len(names_f))*0.5, method="Nelder-Mead",
                   options={"xatol":1e-5, "fatol":1e-7, "maxiter":3000})
    if res.fun < best_neg:
        best_neg = res.fun
        best_w_f = np.exp(res.x - res.x.max()); best_w_f /= best_w_f.sum()
nm_full_auc = -best_neg
print(f"  NM full OOF={nm_full_auc:.5f}")
for n, w in sorted(zip(names_f, best_w_f), key=lambda x: -x[1]):
    print(f"    {n:>12s}: {w:.4f}")
blend_full = test_mat_f @ best_w_f
sub = sub_base.copy(); sub["probability"] = blend_full
fname5 = f"submission_{ts}_FINAL_NMfull_6way_oof{nm_full_auc:.5f}.csv"
sub.to_csv(fname5, index=False)
print(f"  저장: {fname5}")

# ─── [6] 갭 시나리오 표 ──────────────────────────────────
print("\n" + "="*78)
print("[6] 최종 LB 갭 시나리오 (검증된 데이터 기반)")
print("="*78)
print(f"  검증 갭:")
print(f"    v8 (raw, AG 0%):        +0.00129 → LB 0.74208")
print(f"    옵션 A 3-way (ag8h 75%): +0.00099 → LB 0.74217")
print(f"    ag8h+v8 (ag8h 73%):     +0.00099 → LB 0.74218")
print(f"  → ag8h(stack=2) 70%+면 갭 +0.00099")
print(f"  → ag_raw_10h(stack=0)는 ag8h보다 갭이 더 클 가능성 ↑ (덜 fit)")
print()
print(f"{'후보':<55s} {'OOF':>8s} {'갭+0.00129':>11s} {'갭+0.00115':>11s} {'갭+0.00099':>11s}")
print("-"*102)

# AG 비중 추정
agraw_w_pair  = bw
agraw_w_3way  = wa
candidates = [
    (f"ag_raw_10h 단독",                                                  roc_auc_score(y, agr_oof)),
    (f"ag_raw_10h × v8 (w={bw2})",                                        ba2),
    (f"ag_raw_10h × pseudo (w={agraw_w_pair})  ★",                       ba),
    (f"ag_raw_10h+pseudo+v8 (a={agraw_w_3way},p={wp},v={wv})  ★",        best3),
    (f"NM safe 4-way (효율 제외)  ★",                                     nm_safe_auc),
    (f"NM full 6-way (효율 포함)",                                         nm_full_auc),
    (f"(검증) ag8h+v8 → LB 0.74218",                                       0.74119),
]
for name, oof in candidates:
    lb129 = oof + 0.00129
    lb115 = oof + 0.00115
    lb099 = oof + 0.00099
    star = ""
    if lb115 > 0.74246: star = "★1위 추월 (중간 갭)"
    elif lb129 > 0.74246: star = "△1위 추월 (best 갭)"
    print(f"{name:<55s} {oof:8.5f} {lb129:11.5f} {lb115:11.5f} {lb099:11.5f}  {star}")

print(f"\n  1위 LB: 0.74246")
print(f"  현재 LB: 0.74218 (ag8h+v8)")
