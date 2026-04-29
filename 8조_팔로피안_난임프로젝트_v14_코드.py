"""
8조_팔로피안_난임프로젝트_v14_코드.py

대회 규칙 준수형 v14 재현 코드
- 외부 데이터 사용 없음
- test 기반 통계/후처리 없음
- 기존 제출 csv(v9/v10/holdout/jt/AG) 재사용 없음
- train.csv / test.csv / sample_submission.csv만 사용
- Target Encoding은 fold train 내부에서만 fit
- Label Encoding도 fold train 기준으로만 fit
"""

import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

DATA_DIR = Path("./data")
OUT_DIR = Path("./outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "임신 성공 여부"
ID_COL = "ID"

age_map = {
    "만18-34세": 30.0,
    "만35-37세": 36.0,
    "만38-39세": 38.5,
    "만40-42세": 41.0,
    "만43-44세": 43.5,
    "만45-50세": 47.5,
    "알 수 없음": np.nan,
}

donor_age_map = {
    "만20세 이하": 19.0,
    "만21-25세": 23.0,
    "만26-30세": 28.0,
    "만31-35세": 33.0,
    "만36-40세": 38.0,
    "만41-45세": 43.0,
    "알 수 없음": np.nan,
}

count_map = {
    "0회": 0.0, "1회": 1.0, "2회": 2.0, "3회": 3.0,
    "4회": 4.0, "5회": 5.0, "6회 이상": 6.0,
    0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0, 6: 6.0,
}

count_cols = [
    "총 시술 횟수", "클리닉 내 총 시술 횟수", "IVF 시술 횟수", "DI 시술 횟수",
    "총 임신 횟수", "IVF 임신 횟수", "DI 임신 횟수",
    "총 출산 횟수", "IVF 출산 횟수", "DI 출산 횟수",
]

pipeline_cols = [
    "총 생성 배아 수", "미세주입된 난자 수", "미세주입에서 생성된 배아 수",
    "이식된 배아 수", "미세주입 배아 이식 수", "저장된 배아 수",
    "미세주입 후 저장된 배아 수", "해동된 배아 수", "해동 난자 수",
    "수집된 신선 난자 수", "저장된 신선 난자 수", "혼합된 난자 수",
    "파트너 정자와 혼합된 난자 수", "기증자 정자와 혼합된 난자 수",
]

proc_tokens = ["ICSI", "IVF", "BLASTOCYST", "AH", "Unknown", "FER"]


def safe_auc(y, pred, name):
    auc = roc_auc_score(y, pred)
    print(f"{name:35s} AUC = {auc:.6f}")
    return auc


def add_v14_features(df, variant="base"):
    """test 통계값 없이 row 단위로만 생성하는 도메인 피처."""
    df = df.copy()
    eps = 1e-6

    for c in count_cols:
        if c in df.columns:
            if df[c].dtype == "object":
                df[c] = df[c].map(count_map)
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
            df[f"{c}_is_6plus"] = (df[c] == 6.0).astype("int8")

    for c in pipeline_cols:
        if c in df.columns:
            # 구조적 결측: 시술 미진행/해당 없음 → 0 처리
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("float32")

    if "시술 당시 나이" in df.columns:
        df["age_num"] = df["시술 당시 나이"].map(age_map).astype("float32")
    else:
        df["age_num"] = np.nan

    if "난자 기증자 나이" in df.columns:
        df["egg_donor_age_num"] = df["난자 기증자 나이"].map(donor_age_map).astype("float32")
    else:
        df["egg_donor_age_num"] = np.nan

    if "난자 출처" in df.columns:
        is_donor_egg = df["난자 출처"].astype(str).eq("기증 제공")
    else:
        is_donor_egg = np.zeros(len(df), dtype=bool)

    df["real_age_num"] = df["age_num"]
    df.loc[is_donor_egg, "real_age_num"] = df.loc[is_donor_egg, "egg_donor_age_num"]
    df["real_age_num"] = df["real_age_num"].fillna(df["age_num"]).astype("float32")

    if "특정 시술 유형" in df.columns:
        s = df["특정 시술 유형"].fillna("").astype(str)
        for tok in proc_tokens:
            df[f"has_{tok}"] = s.str.contains(tok, regex=False).astype("int8")

    embryo = df.get("총 생성 배아 수", pd.Series(0, index=df.index)).astype("float32")
    transfer = df.get("이식된 배아 수", pd.Series(0, index=df.index)).astype("float32")
    stored = df.get("저장된 배아 수", pd.Series(0, index=df.index)).astype("float32")
    fresh = df.get("수집된 신선 난자 수", pd.Series(0, index=df.index)).astype("float32")
    mixed = df.get("혼합된 난자 수", pd.Series(0, index=df.index)).astype("float32")
    icsi_egg = df.get("미세주입된 난자 수", pd.Series(0, index=df.index)).astype("float32")
    icsi_emb = df.get("미세주입에서 생성된 배아 수", pd.Series(0, index=df.index)).astype("float32")

    if "총 생성 배아 수" in df.columns:
        df["총 생성 배아 수"] = np.maximum(embryo, transfer + stored).astype("float32")
        embryo = df["총 생성 배아 수"]

    if {"미세주입에서 생성된 배아 수", "미세주입된 난자 수"}.issubset(df.columns):
        df["미세주입에서 생성된 배아 수"] = np.minimum(icsi_emb, icsi_egg).astype("float32")
        icsi_emb = df["미세주입에서 생성된 배아 수"]

    df["egg_to_embryo_rate"] = (embryo / (fresh + 1)).astype("float32")
    df["mixed_to_embryo_rate"] = (embryo / (mixed + 1)).astype("float32")
    df["transfer_rate"] = (transfer / (embryo + 1)).astype("float32")
    df["stored_rate"] = (stored / (embryo + 1)).astype("float32")
    df["icsi_success_rate"] = (icsi_emb / (icsi_egg + 1)).astype("float32")
    df["embryo_utilization_rate"] = ((transfer + stored) / (embryo + 1)).astype("float32")
    df["total_pipeline_efficiency"] = ((transfer + stored) / (fresh + 1)).astype("float32")

    df["log_embryo"] = np.log1p(embryo).astype("float32")
    df["age_x_log_embryo"] = (df["real_age_num"] * df["log_embryo"]).astype("float32")
    df["age_normalized_efficiency"] = (df["egg_to_embryo_rate"] * (df["real_age_num"] / 35.0)).astype("float32")

    if {"총 임신 횟수", "총 시술 횟수"}.issubset(df.columns):
        df["past_preg_rate"] = (df["총 임신 횟수"] / (df["총 시술 횟수"] + eps)).astype("float32")

    if {"총 출산 횟수", "총 임신 횟수"}.issubset(df.columns):
        df["past_birth_rate"] = (df["총 출산 횟수"] / (df["총 임신 횟수"] + eps)).astype("float32")
        df["miscarriage_like_count"] = (df["총 임신 횟수"] - df["총 출산 횟수"]).clip(lower=0).astype("float32")

    thawed = df.get("해동된 배아 수", pd.Series(0, index=df.index)).astype("float32")
    df["is_FET"] = (thawed > 0).astype("int8")
    df["fresh_egg_FET_adj"] = fresh.astype("float32")
    df.loc[df["is_FET"] == 1, "fresh_egg_FET_adj"] = -1
    df["fet_x_embryo"] = (df["is_FET"] * embryo).astype("float32")
    df["fet_x_transfer"] = (df["is_FET"] * transfer).astype("float32")
    df["fet_efficiency"] = np.where(df["is_FET"] == 1, transfer / (thawed + 1), -1).astype("float32")

    male_cols = [c for c in ["불임 원인 - 남성 요인", "남성 주 불임 원인", "남성 부 불임 원인"] if c in df.columns]
    male_factor = np.zeros(len(df), dtype=np.int8)
    for c in male_cols:
        male_factor = np.maximum(male_factor, pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int).values)

    df["male_factor"] = male_factor
    df["is_ICSI"] = (icsi_egg > 0).astype("int8")
    df["icsi_male_match"] = ((df["male_factor"] == 1) & (df["is_ICSI"] == 1)).astype("int8")
    df["icsi_male_mismatch"] = (df["male_factor"] != df["is_ICSI"]).astype("int8")
    df["icsi_failure_count"] = (icsi_egg - icsi_emb).clip(lower=0).astype("float32")

    df["age_group_simple"] = pd.cut(
        df["real_age_num"],
        bins=[0, 34, 37, 39, 42, 100],
        labels=["young", "mid35", "late38", "early40", "old"]
    ).astype("object")

    df["embryo_group"] = pd.cut(
        embryo,
        bins=[-1, 0, 3, 7, 15, 999],
        labels=["zero", "low", "mid", "high", "very_high"]
    ).astype("object")

    df["cycle_type"] = np.where(df["is_FET"] == 1, "FET", "Fresh").astype(object)

    if variant == "no_ratio_noise":
        drop_cols = [
            "egg_to_embryo_rate",
            "total_pipeline_efficiency",
            "age_normalized_efficiency",
            "age_x_log_embryo",
        ]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    elif variant == "fet_focus":
        df["fresh_efficiency_only"] = np.where(df["is_FET"] == 0, embryo / (fresh + 1), -1).astype("float32")

    for c in df.columns:
        if df[c].dtype.kind in "fc":
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)

    return df


def fit_transform_fold(X_train, X_valid, X_test, y_train, te_cols=None, smoothing=25):
    """
    Data Leakage 방지:
    - Target Encoding: fold train으로만 fit
    - Label Encoding: fold train category만 mapping
    - Median imputation: fold train median만 사용
    """
    X_train = X_train.copy()
    X_valid = X_valid.copy()
    X_test = X_test.copy()

    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    if te_cols is None:
        te_cols = cat_cols[:]

    global_mean = float(np.mean(y_train))

    for c in te_cols:
        if c not in X_train.columns:
            continue

        tr_key = X_train[c].astype("string").fillna("__MISSING__")
        va_key = X_valid[c].astype("string").fillna("__MISSING__")
        te_key = X_test[c].astype("string").fillna("__MISSING__")

        stat = pd.DataFrame({"key": tr_key, "y": y_train}).groupby("key")["y"].agg(["mean", "count"])
        smooth = (stat["mean"] * stat["count"] + global_mean * smoothing) / (stat["count"] + smoothing)

        X_train[f"{c}_te"] = tr_key.map(smooth).fillna(global_mean).astype("float32")
        X_valid[f"{c}_te"] = va_key.map(smooth).fillna(global_mean).astype("float32")
        X_test[f"{c}_te"] = te_key.map(smooth).fillna(global_mean).astype("float32")

    for c in cat_cols:
        tr = X_train[c].astype("string").fillna("__MISSING__")
        mapping = {v: i for i, v in enumerate(pd.Index(tr.unique()))}
        X_train[c] = tr.map(mapping).astype("int32")
        X_valid[c] = X_valid[c].astype("string").fillna("__MISSING__").map(mapping).fillna(-1).astype("int32")
        X_test[c] = X_test[c].astype("string").fillna("__MISSING__").map(mapping).fillna(-1).astype("int32")

    med = X_train.median(numeric_only=True)
    X_train = X_train.fillna(med).fillna(0)
    X_valid = X_valid.fillna(med).fillna(0)
    X_test = X_test.fillna(med).fillna(0)

    return X_train, X_valid, X_test


def run_lgb_variant(train, test, sample, variant="base", n_splits=5, seeds=(42,), run_name="base"):
    print("\n" + "=" * 80)
    print(f"[run] {run_name} | variant={variant} | folds={n_splits} | seeds={seeds}")
    print("=" * 80)

    train_fe = add_v14_features(train, variant=variant)
    test_fe = add_v14_features(test, variant=variant)

    X = train_fe.drop(columns=[TARGET, ID_COL], errors="ignore")
    y_local = train_fe[TARGET].astype(int).reset_index(drop=True)
    X_test = test_fe.drop(columns=[ID_COL], errors="ignore")

    te_cols = [
        c for c in [
            "시술 시기 코드", "시술 당시 나이", "시술 유형", "특정 시술 유형",
            "배란 유도 유형", "난자 출처", "정자 출처",
            "age_group_simple", "embryo_group", "cycle_type",
        ] if c in X.columns
    ]

    all_oofs, all_preds, names = [], [], []

    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof = np.zeros(len(X), dtype=np.float32)
        pred = np.zeros(len(X_test), dtype=np.float32)

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y_local), 1):
            print(f"{run_name} | seed={seed} | fold={fold}/{n_splits}")

            X_tr_raw, X_va_raw = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y_local.iloc[tr_idx], y_local.iloc[va_idx]

            X_tr, X_va, X_te = fit_transform_fold(
                X_tr_raw, X_va_raw, X_test, y_tr.values,
                te_cols=te_cols, smoothing=25
            )

            model = lgb.LGBMClassifier(
                n_estimators=1800,
                learning_rate=0.022,
                num_leaves=72,
                max_depth=-1,
                subsample=0.88,
                colsample_bytree=0.82,
                reg_alpha=0.20,
                reg_lambda=1.60,
                min_child_samples=45,
                objective="binary",
                random_state=seed + fold,
                verbosity=-1,
                n_jobs=-1,
            )

            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(150, verbose=False)]
            )

            oof[va_idx] = model.predict_proba(X_va)[:, 1]
            pred += model.predict_proba(X_te)[:, 1] / n_splits

            print(f"  fold_auc = {roc_auc_score(y_va, oof[va_idx]):.6f}")

            del X_tr, X_va, X_te, model
            gc.collect()

        auc = roc_auc_score(y_local, oof)
        print(f"{run_name}_seed{seed} OOF AUC = {auc:.6f}")

        all_oofs.append(oof)
        all_preds.append(pred)
        names.append(f"{run_name}_seed{seed}")

    return names, all_oofs, all_preds


def main():
    print("=" * 80)
    print("[1] Load data")
    print("=" * 80)

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

    print("train :", train.shape)
    print("test  :", test.shape)
    print("sample:", sample.shape)

    y = train[TARGET].astype(int).reset_index(drop=True)

    if QUICK_MODE:
        configs = [
            ("base", 5, [42], "v14_base_5f"),
            ("no_ratio_noise", 5, [2024], "v14_no_ratio"),
        ]
    else:
        configs = [
            ("base", 5, [42, 2024], "v14_base_5f"),
            ("base", 7, [777], "v14_base_7f"),
            ("no_ratio_noise", 5, [42], "v14_no_ratio"),
            ("fet_focus", 5, [2024], "v14_fet_focus"),
        ]

    all_names, all_oofs, all_preds = [], [], []

    for variant, nfold, seeds, name in configs:
        names, oofs, preds = run_lgb_variant(
            train, test, sample,
            variant=variant,
            n_splits=nfold,
            seeds=seeds,
            run_name=name,
        )
        all_names.extend(names)
        all_oofs.extend(oofs)
        all_preds.extend(preds)

    print("\n" + "=" * 80)
    print("[2] Internal ensemble")
    print("=" * 80)

    O = np.vstack(all_oofs).T
    P = np.vstack(all_preds)

    for i, name in enumerate(all_names):
        safe_auc(y, O[:, i], name)

    def neg_auc_weight(w):
        w = np.maximum(np.asarray(w), 0)
        if w.sum() == 0:
            w = np.ones_like(w)
        w = w / w.sum()
        return -roc_auc_score(y, O.dot(w))

    res = minimize(
        neg_auc_weight,
        np.ones(O.shape[1]) / O.shape[1],
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-7, "fatol": 1e-9},
    )

    weights = np.maximum(res.x, 0)
    weights = weights / weights.sum()

    internal_oof = O.dot(weights)
    internal_pred = P.T.dot(weights)

    safe_auc(y, internal_oof, "v14_internal_ensemble")

    print("\n[weights]")
    for name, w in sorted(zip(all_names, weights), key=lambda x: -x[1]):
        print(f"{name:30s}: {w:.5f}")

    np.save(OUT_DIR / "v14_internal_oof.npy", internal_oof)
    np.save(OUT_DIR / "v14_internal_pred.npy", internal_pred)

    sub = sample.copy()
    sub["probability"] = np.clip(internal_pred, 1e-7, 1 - 1e-7)

    internal_path = OUT_DIR / "v14_internal_only.csv"
    final_path = OUT_DIR / "8조_팔로피안_난임프로젝트_v14.csv"

    sub.to_csv(internal_path, index=False)
    sub.to_csv(final_path, index=False)

    print("\n" + "=" * 80)
    print("[FINAL]")
    print(f"saved: {internal_path}")
    print(f"saved: {final_path}")
    print("=" * 80)
    print(sub.head())
    print(sub["probability"].describe())


if __name__ == "__main__":
    main()
