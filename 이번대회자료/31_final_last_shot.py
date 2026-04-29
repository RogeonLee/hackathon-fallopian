"""
[마지막 2번 제출 카드] ag_raw_10h + v10 직접 페어 & DE 최적화
- 핵심 근거:
    ag_raw_10h: OOF 0.74126, LB 0.74228, 갭 +0.00102 (단일 OOF 최강)
    v10        : OOF 0.74077, LB 0.74222, 갭 +0.00143 (순수 LGBM, 갭 최강)
  → 이 두 모델을 블렌딩하면 갭이 +0.00115~0.00130으로 올라올 수 있음
  → OOF 0.74128 × 갭 0.00120 → 예상 LB 0.74248 → 1위(0.74246) 추월 가능!
- 아직 이 페어를 직접 제출한 기록 없음 (기존은 agraw+v8, agraw+pseudo만)
"""
import glob
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from scipy.optimize import minimize, differential_evolution

TARGET = "임신 성공 여부"
def rn(a): return rankdata(a) / len(a)

train = pd.read_csv("./data/train.csv")
y = train[TARGET].values
sub_base = pd.read_csv("./data/sample_submission.csv")
ts = datetime.now().strftime("%Y%m%d_%H%M")

def load(pat):
    f = sorted(glob.glob(pat))
    return np.load(f[-1]) if f else None

agr_oof = load("oof_ag_raw_10h_auc_*.npy")
agr_t   = load("test_ag_raw_10h_auc_*.npy")
v10_oof = load("oof_v10_auc_*.npy")
v10_t   = load("test_v10_auc_*.npy")
v9_oof  = load("oof_v9_auc_*.npy")
v9_t    = load("test_v9_auc_*.npy")
v8_oof  = load("oof_v8_auc_*.npy")
v8_t    = load("test_v8_auc_*.npy")
ag8_oof = load("oof_ag8h_auc_*.npy")
ag8_t   = load("test_ag8h_auc_*.npy")

print("=" * 78)
print("[마지막 한 방] ag_raw_10h × v10 직접 페어 + 3-way 탐색")
print("=" * 78)
print(f"\n  모델 OOF (검증된 LB 갭):")
print(f"    ag_raw_10h : {roc_auc_score(y, agr_oof):.5f}  [LB 0.74228, 갭 +0.00102]")
print(f"    v10        : {roc_auc_score(y, v10_oof):.5f}  [LB 0.74222, 갭 +0.00143]")
print(f"    v9         : {roc_auc_score(y, v9_oof):.5f}  [LB 0.74221, 갭 +0.00142]")
print(f"    v8         : {roc_auc_score(y, v8_oof):.5f}  [LB 0.74208, 갭 +0.00129]")
print(f"    ag8h       : {roc_auc_score(y, ag8_oof):.5f}  [갭 ~+0.00099]")

# ─── [1] ag_raw_10h × v10 직접 페어 (★ 핵심 카드) ────────────────────
print("\n[1] ag_raw_10h × v10 직접 페어 (★ 아직 제출 안 한 조합)")
best_w, best_auc = 0.5, 0
for w in np.arange(0.05, 0.96, 0.01):
    a = roc_auc_score(y, w * rn(agr_oof) + (1-w) * rn(v10_oof))
    if a > best_auc:
        best_auc, best_w = a, round(w, 2)
blend_agr_v10 = best_w * rn(agr_t) + (1-best_w) * rn(v10_t)
print(f"  최적: w_agraw={best_w:.2f}, w_v10={1-best_w:.2f}")
print(f"  OOF={best_auc:.5f}")
print(f"  예상 LB 시나리오:")
for gap_name, gap in [("갭+0.00102(agraw수준)", 0.00102),
                       ("갭+0.00115(중간)", 0.00115),
                       ("갭+0.00130(v10 방향)", 0.00130),
                       ("갭+0.00143(v10수준)", 0.00143)]:
    lb = best_auc + gap
    flag = " ← 1위 추월!" if lb > 0.74246 else ""
    print(f"    {gap_name}: {lb:.5f}{flag}")

sub = sub_base.copy(); sub["probability"] = blend_agr_v10
fname1 = f"submission_{ts}_FINAL_agraw_v10_w{best_w}_oof{best_auc:.5f}.csv"
sub.to_csv(fname1, index=False)
print(f"  저장: {fname1}")

# ─── [2] ag_raw_10h × v9 직접 페어 ───────────────────────────────────
print("\n[2] ag_raw_10h × v9 직접 페어")
best_w9, best_auc9 = 0.5, 0
for w in np.arange(0.05, 0.96, 0.01):
    a = roc_auc_score(y, w * rn(agr_oof) + (1-w) * rn(v9_oof))
    if a > best_auc9:
        best_auc9, best_w9 = a, round(w, 2)
blend_agr_v9 = best_w9 * rn(agr_t) + (1-best_w9) * rn(v9_t)
print(f"  최적: w_agraw={best_w9:.2f}, w_v9={1-best_w9:.2f}")
print(f"  OOF={best_auc9:.5f}")
sub = sub_base.copy(); sub["probability"] = blend_agr_v9
fname2 = f"submission_{ts}_FINAL_agraw_v9_w{best_w9}_oof{best_auc9:.5f}.csv"
sub.to_csv(fname2, index=False)
print(f"  저장: {fname2}")

# ─── [3] 3-way: ag_raw_10h + v10 + v9 ────────────────────────────────
print("\n[3] 3-way: ag_raw_10h + v10 + v9 (순수 LGBM 갭 살리기)")
best3, w3 = 0, (0.6, 0.2, 0.2)
for wa in np.arange(0.4, 0.91, 0.05):
    for w10 in np.arange(0.05, 0.61 - wa + 0.4, 0.05):
        w9_ = round(1 - wa - w10, 2)
        if w9_ < 0.05 or w9_ > 0.55: continue
        a = roc_auc_score(y, wa * rn(agr_oof) + w10 * rn(v10_oof) + w9_ * rn(v9_oof))
        if a > best3:
            best3, w3 = a, (round(wa,2), round(w10,2), round(w9_,2))
wa3, w10_3, w9_3 = w3
blend3 = wa3 * rn(agr_t) + w10_3 * rn(v10_t) + w9_3 * rn(v9_t)
print(f"  최적: agraw={wa3}, v10={w10_3}, v9={w9_3}")
print(f"  OOF={best3:.5f}")
print(f"  예상 LB:")
for gap_name, gap in [("갭+0.00102", 0.00102), ("갭+0.00115", 0.00115), ("갭+0.00130", 0.00130)]:
    lb = best3 + gap
    flag = " ← 1위 추월!" if lb > 0.74246 else ""
    print(f"    {gap_name}: {lb:.5f}{flag}")
sub = sub_base.copy(); sub["probability"] = blend3
fname3 = f"submission_{ts}_FINAL_agraw_v10_v9_{wa3}_{w10_3}_{w9_3}_oof{best3:.5f}.csv"
sub.to_csv(fname3, index=False)
print(f"  저장: {fname3}")

# ─── [4] DE 최적화: ag_raw_10h + v10 + v9 + v8 4-way ────────────────
print("\n[4] Differential Evolution: agraw + v10 + v9 + v8 (순수 갭 극대화)")
oof_pool = np.stack([rn(agr_oof), rn(v10_oof), rn(v9_oof), rn(v8_oof)], axis=1)
test_pool = np.stack([rn(agr_t), rn(v10_t), rn(v9_t), rn(v8_t)], axis=1)
names_de = ["ag_raw_10h", "v10", "v9", "v8"]

def neg_auc_de(w):
    w = np.array(w); w = w / w.sum()
    return -roc_auc_score(y, oof_pool @ w)

result_de = differential_evolution(
    neg_auc_de,
    bounds=[(0.0, 1.0)] * 4,
    maxiter=300,
    seed=42,
    workers=1,
    tol=1e-7,
    mutation=(0.5, 1.5),
    popsize=20,
)
w_de = np.array(result_de.x); w_de = w_de / w_de.sum()
de_auc = -result_de.fun
print(f"  DE 최적 OOF={de_auc:.5f}")
for n, w in sorted(zip(names_de, w_de), key=lambda x: -x[1]):
    print(f"    {n:>12s}: {w:.4f}")
blend_de = test_pool @ w_de
sub = sub_base.copy(); sub["probability"] = blend_de
fname4 = f"submission_{ts}_FINAL_DE4way_agraw_v10_v9_v8_oof{de_auc:.5f}.csv"
sub.to_csv(fname4, index=False)
print(f"  저장: {fname4}")

# ─── [5] NM 최적화: 동일 4종 (비교용) ────────────────────────────────
print("\n[5] Nelder-Mead 4-way (동일 풀, DE와 비교)")
def neg_auc_nm(x):
    w = np.exp(x - x.max()); w = w / w.sum()
    return -roc_auc_score(y, oof_pool @ w)

best_neg_nm, best_w_nm = 1.0, None
for seed in range(30):
    rng = np.random.default_rng(seed)
    res = minimize(neg_auc_nm, rng.normal(size=4)*0.5, method="Nelder-Mead",
                   options={"xatol":1e-6, "fatol":1e-8, "maxiter":5000})
    if res.fun < best_neg_nm:
        best_neg_nm = res.fun
        best_w_nm = np.exp(res.x - res.x.max()); best_w_nm /= best_w_nm.sum()
nm_auc = -best_neg_nm
print(f"  NM 4-way OOF={nm_auc:.5f}")
for n, w in sorted(zip(names_de, best_w_nm), key=lambda x: -x[1]):
    print(f"    {n:>12s}: {w:.4f}")
blend_nm = test_pool @ best_w_nm
sub = sub_base.copy(); sub["probability"] = blend_nm
fname5 = f"submission_{ts}_FINAL_NM4way_agraw_v10_v9_v8_oof{nm_auc:.5f}.csv"
sub.to_csv(fname5, index=False)
print(f"  저장: {fname5}")

# ─── 최종 비교 ────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("[최종 비교] 모든 후보 카드")
print("=" * 78)
print(f"{'카드':<50s} {'OOF':>8s} {'갭+0.00102':>12s} {'갭+0.00115':>12s} {'갭+0.00143':>12s}")
print("-" * 100)

all_cards = [
    ("(현재 최고) ag_raw_10h 단독", 0.74126),
    (f"[NEW] agraw×v10 (w={best_w})", best_auc),
    (f"[NEW] agraw×v9 (w={best_w9})", best_auc9),
    (f"[NEW] agraw+v10+v9 3way", best3),
    (f"[NEW] DE 4way (agraw+v10+v9+v8)", de_auc),
    (f"[NEW] NM 4way (agraw+v10+v9+v8)", nm_auc),
]
for name, oof in all_cards:
    lb102 = oof + 0.00102
    lb115 = oof + 0.00115
    lb143 = oof + 0.00143
    flag = ""
    if lb102 > 0.74246: flag = " ★1위확실"
    elif lb115 > 0.74246: flag = " △1위가능"
    print(f"{name:<50s} {oof:8.5f} {lb102:12.5f} {lb115:12.5f} {lb143:12.5f}  {flag}")

print(f"\n  1위: 0.74246 / 우리: 0.74228 / 격차: -0.00018")
print(f"\n★ 추천 제출 우선순위:")
print(f"  1순위: OOF 최고이며 v10 갭 효과 기대되는 카드")
print(f"  2순위: 현재 0.74228 유지 (자동 best 규칙 적용)")
