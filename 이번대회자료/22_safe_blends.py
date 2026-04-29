"""
[안전 블렌드 생성기]
옵션 A 3-way 결과 (LB 0.74217, OOF→LB 갭 +0.00099로 축소) 분석.
- agv10quick(5%)도 갭 축소에 기여 → effiency 피처 + AG의 train fit 위험 가능성
- 옵션 C NM(agv10full=28%) 위험 큼 — 안전 변형 만들기

생성:
1. ag8h + v8 페어 블렌드 (효율 피처 0%) - 가장 안전
2. ag8h + v8 + agv10full(10%) 보수 NM - agv10full 비중 축소
3. ag8h + v8 + agv10full(15%) 중간 NM
4. v9(LGBM 6-way 스타일) + ag8h 등 raw 베이스 블렌드 후보
"""
import glob, os
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

TARGET = "임신 성공 여부"
def rn(a): return rankdata(a) / len(a)

print("="*70)
print("안전 블렌드 생성기 — 효율 피처 비중 축소 변형")
print("="*70)

train = pd.read_csv("./data/train.csv")
y = train[TARGET].values
sub_base = pd.read_csv("./data/sample_submission.csv")
ts = datetime.now().strftime("%Y%m%d_%H%M")

# 핵심 OOF 로드
def load(pat):
    f = sorted(glob.glob(pat))
    return np.load(f[-1]) if f else None

v8_oof  = load("oof_v8_auc_*.npy");      v8_t  = load("test_v8_auc_*.npy")
v9_oof  = load("oof_v9_auc_*.npy");      v9_t  = load("test_v9_auc_*.npy")
v5_oof  = load("oof_v5_auc_*.npy");      v5_t  = load("test_v5_auc_*.npy")
ag8_oof = load("oof_ag8h_auc_*.npy");    ag8_t = load("test_ag8h_auc_*.npy")
agf_oof = load("oof_agv10full_auc_*.npy"); agf_t = load("test_agv10full_auc_*.npy")
agq_oof = load("oof_agv10quick_auc_*.npy"); agq_t = load("test_agv10quick_auc_*.npy")

models = {"v8":v8_oof, "v9":v9_oof, "v5":v5_oof, "ag8h":ag8_oof, "agv10full":agf_oof, "agv10quick":agq_oof}
print("\n[모델별 OOF AUC]")
for n, o in models.items():
    if o is not None:
        print(f"  {n:>12s}: {roc_auc_score(y, o):.5f}")

# ─── [1] ag8h × v8 순수 페어 (효율 피처 0%) ───────────────────
print("\n[1] ag8h × v8 페어 (효율 피처 0%, 가장 안전)")
best_w, best_a = 0.5, 0
for w in np.arange(0.05, 0.96, 0.01):
    a = roc_auc_score(y, w*rn(ag8_oof) + (1-w)*rn(v8_oof))
    if a > best_a: best_a, best_w = a, round(w,2)
print(f"  최적 OOF AUC = {best_a:.5f}  (w_ag8h={best_w}, w_v8={1-best_w:.2f})")
blend = best_w*rn(ag8_t) + (1-best_w)*rn(v8_t)
sub = sub_base.copy(); sub["probability"] = blend
fname1 = f"submission_{ts}_safe_ag8h_v8_w{best_w}_oof{best_a:.5f}.csv"
sub.to_csv(fname1, index=False)
print(f"  저장: {fname1}")

# ─── [2~4] agv10full 비중 단계별 (10%, 15%, 20%) ──────────────
print("\n[2-4] ag8h + v8 + agv10full(비중 단계별)")
ag8_v8_oof = best_w*rn(ag8_oof) + (1-best_w)*rn(v8_oof)
ag8_v8_t   = best_w*rn(ag8_t)   + (1-best_w)*rn(v8_t)

for w_agf in [0.10, 0.15, 0.20, 0.25, 0.28]:
    blend_oof = (1-w_agf)*ag8_v8_oof + w_agf*rn(agf_oof)
    blend_t   = (1-w_agf)*ag8_v8_t   + w_agf*rn(agf_t)
    a = roc_auc_score(y, blend_oof)
    sub = sub_base.copy(); sub["probability"] = blend_t
    fname = f"submission_{ts}_safe_v10full_w{w_agf}_oof{a:.5f}.csv"
    sub.to_csv(fname, index=False)
    print(f"  agv10full={w_agf:.2f}: OOF={a:.5f}  → {fname}")

# ─── [5] v9 단독 + 기존 블렌드 시도 ───────────────────────────
print("\n[5] v9 + ag8h 페어 (raw LGBM 베이스)")
if v9_oof is not None and ag8_oof is not None:
    bw, ba = 0.5, 0
    for w in np.arange(0.05, 0.96, 0.01):
        a = roc_auc_score(y, w*rn(ag8_oof) + (1-w)*rn(v9_oof))
        if a > ba: ba, bw = a, round(w,2)
    print(f"  최적 OOF = {ba:.5f}  (w_ag8h={bw}, w_v9={1-bw:.2f})")
    blend = bw*rn(ag8_t) + (1-bw)*rn(v9_t)
    sub = sub_base.copy(); sub["probability"] = blend
    fname = f"submission_{ts}_safe_ag8h_v9_w{bw}_oof{ba:.5f}.csv"
    sub.to_csv(fname, index=False)
    print(f"  저장: {fname}")

# ─── [6] v8 + v9 + ag8h 3-way (효율 피처 없음) ─────────────────
print("\n[6] v8 + v9 + ag8h 3-way (raw 베이스)")
b3, w3 = 0, (1/3, 1/3, 1/3)
for w1 in np.arange(0.05, 0.91, 0.05):
    for w2 in np.arange(0.05, 0.96-w1, 0.05):
        ww3 = 1 - w1 - w2
        if ww3 < 0.05: continue
        a = roc_auc_score(y, w1*rn(v8_oof) + w2*rn(v9_oof) + ww3*rn(ag8_oof))
        if a > b3: b3, w3 = a, (round(w1,2), round(w2,2), round(ww3,2))
print(f"  최적 OOF = {b3:.5f}  (v8={w3[0]}, v9={w3[1]}, ag8h={w3[2]})")
blend = w3[0]*rn(v8_t) + w3[1]*rn(v9_t) + w3[2]*rn(ag8_t)
sub = sub_base.copy(); sub["probability"] = blend
fname6 = f"submission_{ts}_safe_3raw_v8_v9_ag8h_oof{b3:.5f}.csv"
sub.to_csv(fname6, index=False)
print(f"  저장: {fname6}")

# ─── [7] LB 갭 시뮬레이션 표 ────────────────────────────────────
print("\n" + "="*70)
print("[7] LB 갭 시뮬레이션 (옵션 A 3-way로 검증된 갭 +0.00099)")
print(f"  v8 OOF→LB 갭(검증): +0.00129")
print(f"  옵션 A 3-way 갭(검증): +0.00099 ← 효율 피처/AG 영향")
print(f"  보수 가정 갭: +0.00080")
print("="*70)
print(f"{'후보':<55s} {'OOF':>8s} {'LB(갭+0.00099)':>13s} {'LB(갭+0.00080)':>13s}")
print("-"*95)
candidates = [
    (f"ag8h × v8 (w={best_w}) [효율 피처 0%]", best_a),
    ("agv10full 10%",          0.0),
    ("agv10full 15%",          0.0),
    ("agv10full 20%",          0.0),
    ("v8+v9+ag8h 3-way [raw]", b3),
    ("(참고) 옵션 C NM 블렌드", 0.74123),
    ("(참고) 옵션 A 3-way (제출완료)", 0.74118),
]
# 동적 OOF 다시 계산
for w_agf in [0.10, 0.15, 0.20]:
    a = roc_auc_score(y, (1-w_agf)*ag8_v8_oof + w_agf*rn(agf_oof))
    candidates[1 if w_agf==0.10 else 2 if w_agf==0.15 else 3] = (
        candidates[1 if w_agf==0.10 else 2 if w_agf==0.15 else 3][0], a)

for name, oof in candidates:
    lb99 = oof + 0.00099
    lb80 = oof + 0.00080
    star = "★1위 추월" if lb99 > 0.74237 else ""
    print(f"{name:<55s} {oof:8.5f} {lb99:13.5f} {lb80:13.5f}  {star}")

print(f"\n  1위 LB: 0.74237")
print(f"  v8 LB(현재): 0.74208 (갭 +0.00129 검증됨)")
print(f"  옵션 A 3-way LB: 0.74217 (갭 +0.00099 검증됨)")
