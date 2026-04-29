"""
난임 환자 임신 성공 여부 예측 — LightGBM 베이스라인 (ROC-AUC)
팀: 8조 팔로피안
평가 산식: ROC-AUC
제출 형식: ID, probability

실행:
    python baseline_lgbm_auc.py

환경:
    python >= 3.10, pandas, numpy, scikit-learn, lightgbm
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


# ──────────────────────────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────────────────────────
SEED = 2026
N_SPLITS = 5
TARGET_COL = "임신 성공 여부"
ID_COL = "ID"

# 대회 규정상 최종 제출 코드에는 '/data' 경로를 포함하도록 안내됨.
# 로컬 개발 시에는 아래 LOCAL_DATA_DIR 를 사용하고, 제출 시 DATA_DIR 를 "/data"로 변경.
LOCAL_DATA_DIR = Path(__file__).resolve().parent.parent / "이번대회자료" / "data"
DATA_DIR = Path(os.environ.get("DATA_DIR", LOCAL_DATA_DIR))
OUT_DIR = Path(__file__).resolve().parent / "submissions"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
# 데이터 I/O
# ──────────────────────────────────────────────────────────────────────
def read_tables(data_dir: Path):
    train = pd.read_csv(data_dir / "train.csv", encoding="utf-8-sig")
    test = pd.read_csv(data_dir / "test.csv", encoding="utf-8-sig")
    submit = pd.read_csv(data_dir / "sample_submission.csv", encoding="utf-8-sig")
    return train, test, submit


# ──────────────────────────────────────────────────────────────────────
# 도메인 규칙 기반 결측 처리 (train/test 각 행 내부 연산만 사용 → leakage 아님)
# ──────────────────────────────────────────────────────────────────────
def apply_domain_rules(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 이식된 배아가 없으면 배아 이식 경과일은 의미 없음 → 0
    mask_no_transfer = df["이식된 배아 수"].fillna(0).eq(0)
    df.loc[mask_no_transfer, "배아 이식 경과일"] = 0

    # 해동 배아/난자 0 → 해동 경과일 0
    mask_no_thaw_emb = df["해동된 배아 수"].fillna(0).eq(0)
    df.loc[mask_no_thaw_emb, "배아 해동 경과일"] = 0

    mask_no_thaw_ova = df["해동 난자 수"].fillna(0).eq(0)
    df.loc[mask_no_thaw_ova, "난자 해동 경과일"] = 0

    # 혼합된 난자 0 → 혼합 경과일 0
    mask_no_mix = df["혼합된 난자 수"].fillna(0).eq(0)
    df.loc[mask_no_mix, "난자 혼합 경과일"] = 0

    # 시술 유형이 DI면 배아·난자 관련 수량은 구조적으로 0
    if "시술 유형" in df.columns:
        mask_di = df["시술 유형"].astype(str).eq("DI")
        di_zero_cols = [
            "총 생성 배아 수", "미세주입된 난자 수", "미세주입에서 생성된 배아 수",
            "이식된 배아 수", "미세주입 배아 이식 수", "저장된 배아 수",
            "미세주입 후 저장된 배아 수", "해동된 배아 수", "해동 난자 수",
            "수집된 신선 난자 수", "저장된 신선 난자 수",
            "혼합된 난자 수", "파트너 정자와 혼합된 난자 수", "기증자 정자와 혼합된 난자 수",
            "난자 채취 경과일", "난자 해동 경과일", "난자 혼합 경과일",
            "배아 이식 경과일", "배아 해동 경과일",
        ]
        for c in di_zero_cols:
            if c in df.columns:
                df.loc[mask_di, c] = df.loc[mask_di, c].fillna(0)

    # 최초 시도(임신 경험 없음) → 경과 연수는 음수 sentinel로 보존
    if "총 임신 횟수" in df.columns and "임신 시도 또는 마지막 임신 경과 연수" in df.columns:
        mask_first = df["총 임신 횟수"].astype(str).str.strip().eq("0회")
        df.loc[mask_first, "임신 시도 또는 마지막 임신 경과 연수"] = -1

    return df


# ──────────────────────────────────────────────────────────────────────
# 행 내부 파생변수 (leakage-safe: 같은 row 안의 컬럼만 사용)
# ──────────────────────────────────────────────────────────────────────
def add_row_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def safe_div(a, b):
        a = df[a].astype(float)
        b = df[b].astype(float).replace(0, np.nan)
        return (a / b).fillna(0.0)

    df["fertilize_yield"] = safe_div("미세주입에서 생성된 배아 수", "미세주입된 난자 수")
    df["transfer_density"] = safe_div("이식된 배아 수", "총 생성 배아 수")
    df["freeze_ratio"] = safe_div("저장된 배아 수", "총 생성 배아 수")
    df["icsi_in_transfer"] = safe_div("미세주입 배아 이식 수", "이식된 배아 수")

    # 배아 이식 stage flag
    df["blast_transfer"] = (df["배아 이식 경과일"].fillna(0) >= 5).astype(int)
    df["fet_transfer"] = (df["해동된 배아 수"].fillna(0) > 0).astype(int)

    # 불임 원인 요약
    cause_cols = [c for c in df.columns if c.startswith("불임 원인 - ")]
    if cause_cols:
        df["cause_count"] = df[cause_cols].fillna(0).astype(int).sum(axis=1)

    # 정합성 이상 flag
    df["flag_transfer_gt_total"] = (
        df["이식된 배아 수"].fillna(0) > df["총 생성 배아 수"].fillna(0)
    ).astype(int)

    return df


# ──────────────────────────────────────────────────────────────────────
# 횟수형 문자열 컬럼 ("0회", "1회", "6회 이상") 정수화
# ──────────────────────────────────────────────────────────────────────
def parse_count_strings(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c].astype(str).str.extract(r"(\d+)")[0]
        df[c] = pd.to_numeric(s, errors="coerce")
    return df


# ──────────────────────────────────────────────────────────────────────
# 전처리 총괄
# ──────────────────────────────────────────────────────────────────────
COUNT_STR_COLS = [
    "총 시술 횟수", "클리닉 내 총 시술 횟수",
    "IVF 시술 횟수", "DI 시술 횟수",
    "총 임신 횟수", "IVF 임신 횟수", "DI 임신 횟수",
    "총 출산 횟수", "IVF 출산 횟수", "DI 출산 횟수",
]

CATEGORICAL_CANDIDATES = [
    "시술 시기 코드", "시술 당시 나이", "시술 유형", "특정 시술 유형",
    "배란 유도 유형", "배아 생성 주요 이유",
    "난자 출처", "정자 출처", "난자 기증자 나이", "정자 기증자 나이",
]


def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame):
    # 1) 도메인 결측 규칙
    train_df = apply_domain_rules(train_df)
    test_df = apply_domain_rules(test_df)

    # 2) 횟수 문자열 → 숫자
    train_df = parse_count_strings(train_df, COUNT_STR_COLS)
    test_df = parse_count_strings(test_df, COUNT_STR_COLS)

    # 3) 행 내부 파생변수
    train_df = add_row_features(train_df)
    test_df = add_row_features(test_df)

    # 4) 범주형은 LightGBM category dtype으로 (native handling)
    for c in CATEGORICAL_CANDIDATES:
        if c in train_df.columns:
            train_df[c] = train_df[c].astype("category")
            test_df[c] = test_df[c].astype("category")
            # train/test 카테고리 정렬 (unseen 안전)
            all_cats = pd.api.types.union_categoricals(
                [train_df[c], test_df[c]]
            ).categories
            train_df[c] = train_df[c].cat.set_categories(all_cats)
            test_df[c] = test_df[c].cat.set_categories(all_cats)

    # 5) 학습용 X/y 분리
    feature_cols = [c for c in train_df.columns if c not in (ID_COL, TARGET_COL)]
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL].astype(int).to_numpy()
    X_test = test_df[feature_cols]
    return X_train, y_train, X_test, feature_cols


# ──────────────────────────────────────────────────────────────────────
# 학습: StratifiedKFold + LightGBM + OOF
# ──────────────────────────────────────────────────────────────────────
def train_kfold(X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame,
                categorical_features: list[str], n_splits: int = N_SPLITS, seed: int = SEED):
    params = dict(
        objective="binary",
        metric="auc",
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=200,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=5,
        lambda_l2=1.0,
        verbose=-1,
        seed=seed,
    )

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X), dtype=np.float64)
    test_pred = np.zeros(len(X_test), dtype=np.float64)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y), start=1):
        t0 = time.time()
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        dtrain = lgb.Dataset(X_tr, y_tr, categorical_feature=categorical_features)
        dvalid = lgb.Dataset(X_va, y_va, categorical_feature=categorical_features, reference=dtrain)

        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=5000,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=200, verbose=False),
                lgb.log_evaluation(period=200),
            ],
        )

        oof[va_idx] = booster.predict(X_va, num_iteration=booster.best_iteration)
        test_pred += booster.predict(X_test, num_iteration=booster.best_iteration) / n_splits

        auc = roc_auc_score(y_va, oof[va_idx])
        print(f"[fold {fold}] best_iter={booster.best_iteration}  valid AUC={auc:.5f}  ({time.time()-t0:.1f}s)")

    oof_auc = roc_auc_score(y, oof)
    print(f"\n[OOF] AUC = {oof_auc:.5f}")
    return oof, test_pred, oof_auc


# ──────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────
def main():
    print(f"[info] DATA_DIR = {DATA_DIR}")
    train_df, test_df, submit_df = read_tables(DATA_DIR)
    print(f"[info] train shape = {train_df.shape}, test shape = {test_df.shape}")
    print(f"[info] target positive rate = {train_df[TARGET_COL].mean():.4f}")

    X_train, y_train, X_test, feature_cols = preprocess(train_df, test_df)

    cat_features = [c for c in CATEGORICAL_CANDIDATES if c in feature_cols]
    print(f"[info] n_features = {len(feature_cols)}  n_categorical = {len(cat_features)}")

    oof, test_pred, oof_auc = train_kfold(X_train, y_train, X_test, cat_features)

    # 제출 파일
    submit_df = submit_df.copy()
    submit_df["probability"] = test_pred
    tag = f"lgbm_auc_{oof_auc:.5f}".replace(".", "p")
    out_path = OUT_DIR / f"submission_{tag}.csv"
    submit_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[done] submission saved → {out_path}")

    # OOF 저장 (추후 블렌딩용)
    oof_path = OUT_DIR / f"oof_{tag}.npy"
    np.save(oof_path, oof)
    print(f"[done] oof saved → {oof_path}")


if __name__ == "__main__":
    main()
