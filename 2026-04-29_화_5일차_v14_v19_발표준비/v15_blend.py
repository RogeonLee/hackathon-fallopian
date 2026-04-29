"""
v15_blend.py — v15 완료 후 실행
v15 OOF/test + 기존 최고 모델들 재블렌딩
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
import glob, os

DATA_DIR = "."           # 데이터 디렉토리에서 실행
NPY_DIR  = "."           # 현재 디렉토리 (data/)
PREV_DIR = ".."          # 기존 npy 파일 위치 (이번대회자료/)

# =========================
# 1. 레이블 로드
# =========================
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")
y     = train["임신 성공 여부"].values

# =========================
# 2. v15 파일 자동 탐색 (data/ 디렉토리)
# =========================
v15_oof_files  = sorted(glob.glob(f"{NPY_DIR}/oof_v15_auc_*.npy"))
v15_test_files = sorted(glob.glob(f"{NPY_DIR}/test_v15_auc_*.npy"))

if not v15_oof_files:
    raise FileNotFoundError("v15 OOF 파일 없음. v15_final.py를 먼저 실행하세요.")

oof_v15  = np.load(v15_oof_files[-1])
test_v15 = np.load(v15_test_files[-1])
v15_auc  = roc_auc_score(y, oof_v15)
print(f"v15 OOF AUC: {v15_auc:.5f}  [{v15_oof_files[-1]}]")

# =========================
# 3. 기존 최고 모델 로드
# =========================
# OOF 0.741+ 모델만 선별
candidates = {
    "ag_raw_10h":  ("oof_ag_raw_10h_auc_0.74126.npy",  "test_ag_raw_10h_auc_0.74126.npy"),
    "ag8h":        ("oof_ag8h_auc_0.74114.npy",         "test_ag8h_auc_0.74114.npy"),
    "pseudo_v3":   ("oof_pseudo_v3_auc_0.74080.npy",    "test_pseudo_v3_auc_0.74080.npy"),
    "v9":          ("oof_v9_auc_0.74079.npy",            "test_v9_auc_0.74079.npy"),
    "v8":          ("oof_v8_auc_0.74079.npy",            "test_v8_auc_0.74079.npy"),
    "agv10full":   ("oof_agv10full_auc_0.74100.npy",    "test_agv10full_auc_0.74100.npy"),
}

oof_pool  = {"v15": oof_v15}
test_pool = {"v15": test_v15}

for name, (oof_f, test_f) in candidates.items():
    oof_path  = os.path.join(PREV_DIR, oof_f)
    test_path = os.path.join(PREV_DIR, test_f)
    if os.path.exists(oof_path) and os.path.exists(test_path):
        oof_pool[name]  = np.load(oof_path)
        test_pool[name] = np.load(test_path)
        auc = roc_auc_score(y, oof_pool[name])
        print(f"  {name}: OOF {auc:.5f}")

model_names = list(oof_pool.keys())
oof_mat  = np.stack([oof_pool[n]  for n in model_names], axis=1)
test_mat = np.stack([test_pool[n] for n in model_names], axis=1)

print(f"\n총 {len(model_names)}개 모델 블렌딩: {model_names}")

# =========================
# 4. 상관관계 확인
# =========================
corr = pd.DataFrame(oof_mat, columns=model_names).corr()
print("\nOOF 상관관계:")
print(corr.round(4))

# =========================
# 5. Nelder-Mead 가중치 최적화
# =========================
def neg_auc_w(w):
    w = np.clip(np.array(w), 0.0, 1.0)
    s = w.sum()
    if s < 1e-9:
        return 0.0
    w = w / s
    return -roc_auc_score(y, (oof_mat * w).sum(axis=1))

n = len(model_names)
simple_auc = roc_auc_score(y, oof_mat.mean(axis=1))
print(f"\n단순 평균 OOF AUC: {simple_auc:.5f}")

init_ws = [np.ones(n) / n]
for i in range(n):
    w = np.zeros(n); w[i] = 1.0
    init_ws.append(w)
for seed in [0, 7, 42]:
    rng = np.random.RandomState(seed)
    init_ws.append(rng.dirichlet(np.ones(n)))

best_w, best_val = np.ones(n) / n, neg_auc_w(np.ones(n) / n)
for w0 in init_ws:
    res = minimize(neg_auc_w, w0, method="Nelder-Mead",
                   options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-9})
    if res.fun < best_val:
        best_val, best_w = res.fun, res.x

opt_w = np.clip(best_w, 0.0, 1.0)
opt_w /= opt_w.sum()
opt_auc = -best_val
print(f"최적화 OOF AUC: {opt_auc:.5f}")
print("최적 가중치:")
for nm, ww in sorted(zip(model_names, opt_w), key=lambda x: -x[1]):
    print(f"  {nm}: {ww:.4f}")

use_w = opt_w if opt_auc >= simple_auc else np.ones(n) / n
use_auc = max(opt_auc, simple_auc)

# =========================
# 6. 다양한 v15 비중 시도
# =========================
print("\n=== v15 비중별 AUC ===")
v15_idx = model_names.index("v15")
best_ratio_auc = 0
best_ratio = None

for ratio in np.arange(0.3, 0.8, 0.05):
    w_try = use_w.copy()
    w_try[v15_idx] = ratio
    # 나머지는 기존 비율 유지하되 합이 1이 되도록
    rest_sum = 1.0 - ratio
    rest_w = use_w.copy()
    rest_w[v15_idx] = 0
    if rest_w.sum() > 1e-9:
        rest_w = rest_w / rest_w.sum() * rest_sum
    w_try = rest_w; w_try[v15_idx] = ratio
    auc_t = roc_auc_score(y, (oof_mat * w_try).sum(axis=1))
    marker = " ← best" if auc_t > best_ratio_auc else ""
    print(f"  v15 {ratio:.2f}: {auc_t:.5f}{marker}")
    if auc_t > best_ratio_auc:
        best_ratio_auc = auc_t
        best_ratio = ratio

print(f"\n최적 v15 비중: {best_ratio:.2f}  OOF {best_ratio_auc:.5f}")

# =========================
# 7. 최종 제출 파일 생성
# =========================
# (a) Nelder-Mead 최적 가중치
final_raw_nm  = (test_mat * opt_w).sum(axis=1)
rank_nm       = pd.Series(final_raw_nm).rank(pct=True).values
final_nm      = 0.5 * final_raw_nm + 0.5 * rank_nm

# (b) 최적 v15 비중
w_best_ratio = use_w.copy()
w_best_ratio[v15_idx] = 0
if w_best_ratio.sum() > 1e-9:
    w_best_ratio = w_best_ratio / w_best_ratio.sum() * (1 - best_ratio)
w_best_ratio[v15_idx] = best_ratio
final_raw_r   = (test_mat * w_best_ratio).sum(axis=1)
rank_r        = pd.Series(final_raw_r).rank(pct=True).values
final_r       = 0.5 * final_raw_r + 0.5 * rank_r

ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M")

sub_nm = pd.DataFrame({"ID": test["ID"], "probability": final_nm})
fname_nm = f"../blend_NM_oof{opt_auc:.5f}_{ts}.csv"
sub_nm.to_csv(fname_nm, index=False)

sub_r = pd.DataFrame({"ID": test["ID"], "probability": final_r})
fname_r = f"../blend_ratio{best_ratio:.2f}_oof{best_ratio_auc:.5f}_{ts}.csv"
sub_r.to_csv(fname_r, index=False)

print(f"\n{'='*55}")
print(f"[A] Nelder-Mead:  {fname_nm}")
print(f"[B] 비중 최적:    {fname_r}")
print(f"{'='*55}")
print("→ OOF 높은 파일 제출 권장")
