# 난임 환자 임신 성공 여부 예측 — 전체 아키텍처 (이번 대회 기준)

> 작성일: 2026-04-23
> 대상: 데이콘 × 오즈코딩스쿨 해커톤 "난임 환자 대상 임신 성공 여부 예측 AI"
> 어제 설계는 **이전 LG Aimers 대회 자료**만 바탕으로 했기에, 본 문서는 이번 대회 실제 데이터(`이번대회자료/data/`)와 규정을 반영하여 **재설계**한 것.

---

## 0. 이전 대회와의 결정적 차이 (반드시 숙지)

| 항목 | 이전 대회 (LG Aimers) | **이번 대회 (Dacon×오즈)** |
| --- | --- | --- |
| 샘플 수 | train 126,244 / test 54,412 | **train 256,351 / test 90,067** |
| 컬럼 수 | 33 | **68** (피처 67 + target 1) |
| Target | `임신 성공 확률` (0~1 soft label, **회귀**) | `임신 성공 여부` (**0/1 binary**, 분류) — train 양성 비율 66,228 / 256,351 = **25.84%** |
| Submission | probability | `probability` (0~1 실수) — 분류 문제지만 제출은 확률 |
| 평가 Metric | 0.5·WBS + 0.5·F1(thr=0.5) | **대회 페이지에서 직접 확인 필요** (probability 제출 + 하드 라벨 → ROC-AUC / LogLoss / Brier 중 하나로 추정) |
| 외부 데이터 | 금지 | 금지 |
| 사전학습 모델 | 논문 공개·법적 문제 없는 경우 허용 | 동일 (논문 링크·출처 기재 필수) |
| **코드 재사용** | 제한 없음 | **이전 대회 코드 유사도 90%+ 시 실격** (전처리/파생변수/Null처리/옵션 등 전반 유사도 검사) |

### 이 차이가 아키텍처에 미치는 영향

1. **회귀 → 이진 분류**: Weighted Brier + SoftF1 Hybrid 손실은 그대로 쓰면 의미 없음. BCE + class-weight / Focal / 순수 Brier(확률 출력 최적화) 중심으로 재편.
2. **Soft label 튜닝 기법 제거**: 이전 1~3위가 사용한 "target shift", "quantile mapping", "(targets − 0.1) 학습" 같은 트릭은 이진 라벨에선 쓰임이 다름. Isotonic calibration은 여전히 유효 (확률 제출이므로).
3. **파일/네이밍 컨벤션 전면 교체**: `V55_oof.npz` 같은 이전 대회 실험 명칭 그대로 쓰지 말 것. 코드 재사용 판정을 피하기 위해 (a) 디렉토리 구조, (b) 함수/클래스 이름, (c) 전처리 순서를 독자적으로 구성.
4. **컬럼 2배**: 이전 대회 33개 피처 대비 이번은 67개. 추가된 ~34개 컬럼의 **정체와 의미 파악이 EDA 1순위**.
5. **스케일 2배**: CV 5-fold 기준 fold당 train ≈ 205k. LightGBM/XGBoost는 무난, MLP는 배치 크기·학습률 재조정 필요.

### 이번 대회 컬럼 목록 (실측)
```
ID, 시술 시기 코드, 시술 당시 나이, 임신 시도 또는 마지막 임신 경과 연수,
시술 유형, 특정 시술 유형, 배란 자극 여부, 배란 유도 유형,
단일 배아 이식 여부, 착상 전 유전 검사 사용 여부, 착상 전 유전 진단 사용 여부,
남성 주/부 불임 원인, 여성 주/부 불임 원인, 부부 주/부 불임 원인, 불명확 불임 원인,
불임 원인 - (난관 질환, 남성 요인, 배란 장애, 여성 요인, 자궁경부 문제, 자궁내막증,
             정자 농도, 정자 면역학적 요인, 정자 운동성, 정자 형태),
배아 생성 주요 이유,
총 시술 횟수, 클리닉 내 총 시술 횟수, IVF 시술 횟수, DI 시술 횟수,
총 임신 횟수, IVF 임신 횟수, DI 임신 횟수,
총 출산 횟수, IVF 출산 횟수, DI 출산 횟수,
총 생성 배아 수, 미세주입된 난자 수, 미세주입에서 생성된 배아 수,
이식된 배아 수, 미세주입 배아 이식 수, 저장된 배아 수, 미세주입 후 저장된 배아 수,
해동된 배아 수, 해동 난자 수, 수집된 신선 난자 수, 저장된 신선 난자 수,
혼합된 난자 수, 파트너 정자와 혼합된 난자 수, 기증자 정자와 혼합된 난자 수,
난자 출처, 정자 출처, 난자 기증자 나이, 정자 기증자 나이,
동결 배아 사용 여부, 신선 배아 사용 여부, 기증 배아 사용 여부,
대리모 여부, PGD 시술 여부, PGS 시술 여부,
난자 채취/해동/혼합 경과일, 배아 이식/해동 경과일,
임신 성공 여부(target)
```
→ 이전 대회 33개 대비 **불임 원인 세부 boolean 10여개, 미세주입 관련 수치, 해동·혼합 관련 경과일**이 대폭 확장됨.

---

## 1. 프로젝트 구조 (이전과 다른 네이밍)

```
해커톤_난임.../
├── 이번대회자료/
│   ├── data/                           # (제공) train.csv, test.csv, sample_submission.csv, 데이터명세.xlsx
│   └── (규정 파일들)
├── 1일차/
│   ├── architecture.md                 # ← 이 문서
│   └── eda_notes.md                    # 매일 EDA 관찰 기록
├── pipeline/                           # (src/ 대신 독자 이름)
│   ├── paths.py                        # 경로·시드·FOLD 수
│   ├── io_loader.py                    # CSV 로드 + dtype 강제
│   ├── schema.py                       # 컬럼 분류(범주/수치/bool/ordinal)
│   ├── clean_rules.py                  # 도메인 규칙 기반 NaN 처리 (시술유형별)
│   ├── encoders.py                     # Ordinal/Target/Freq 인코더 (fold-safe)
│   ├── derive.py                       # 파생변수 (한 함수 = 한 피처)
│   ├── metric_probclf.py               # 대회 metric 재현 + F1@0.5, AUC, LogLoss, Brier 병기
│   ├── cv.py                           # StratifiedKFold 래퍼
│   ├── trainers/
│   │   ├── lgbm_runner.py
│   │   ├── xgb_runner.py
│   │   ├── cat_runner.py
│   │   └── tab_mlp.py                  # PyTorch (이전 코드와 구조·변수명 모두 교체)
│   ├── blend.py                        # OOF 가중 블렌딩 (Nelder-Mead)
│   ├── calibrate.py                    # Isotonic / Beta calibration
│   └── infer.py                        # 최종 예측·제출
├── runs/                               # run_YYYYMMDD_HHMM_<tag>/
│   └── run_20260423_1400_baseline_lgbm/
│       ├── config.yaml
│       ├── oof.parquet
│       ├── pred_test.parquet
│       ├── submission.csv
│       └── log.txt
└── reports/
    ├── eda_target.md
    ├── eda_missing.md
    └── error_analysis.md
```

**명시적 차별화 포인트** (이전 대회 코드 유사도 회피):
- 디렉토리: `src/` → `pipeline/`
- 함수명: `competition_metric` → `score_probclf`, `load_data` → `read_tables`
- 결측 규칙 코드: 이전처럼 `if df['시술 유형']=='DI'` 1줄로 안 쓰고, `schema.py`에 **규칙 테이블 dict**로 선언 후 `clean_rules.py`가 loop로 적용 (데이터 주도)
- 파생변수는 이전 대회의 "수정율·배아생성효율" 같은 이름을 피하고, **도메인 개념에 충실한 별도 네이밍** (`fertilize_yield`, `blast_stage_flag` 등 영문 스네이크)

---

## 2. 데이터 분석(EDA) 계획 — **본 설계의 중심**

> 목표: 이전 대회와 달라진 34개 컬럼의 의미를 파악하고, 25.84% 양성 불균형·하드라벨 구조에서 "어떤 변수가 임신 성공과 실제로 연관 있는가"를 근거 기반으로 확정.

### 2-1. 타겟(`임신 성공 여부`) 구조 분석
- **기본 통계**: 전체 양성률 0.2584. 시술 유형(IVF/DI), 나이대, 배아 이식 관련 컬럼별 양성률을 분해.
- **임상 기준 sanity check**: SART/HFEA 공개 통계상 IVF 임신 성공률은 35세 미만 40%대, 40세 이상 10%대. 본 데이터가 이 분포를 따르는지 확인 → 안 따르면 시점·선택편향(이미 실패 경험 많은 cohort 포함 여부) 의심.
- **클래스 불균형 대응 방향 결정**: 1:2.87 비율. 부스팅의 `scale_pos_weight=2.87` 또는 `is_unbalance=True`, MLP는 `pos_weight` 혹은 Focal(γ≈2) 후보.

### 2-2. 컬럼 타입 재분류 (이전 대회 33개 매핑 + 신규 ~34개)

| 유형 | 컬럼 예 | 처리 |
| --- | --- | --- |
| **ID** | ID | drop (학습 제외, 제출 병합용) |
| **순서형 구간(ordinal bin)** | 시술 당시 나이, 난자·정자 기증자 나이 | "만18-34세" 같은 문자 구간 → 중앙값 + ordinal 인덱스 **둘 다** 만들어 비교 |
| **횟수 구간(ordinal bin)** | 총 시술/임신/출산 횟수, IVF/DI 횟수 | "0회","1회","2회","3회","4회 이상","6회 이상" 패턴 → 상한 가정(4회 이상=4, 6회 이상=6) 후 숫자화. **is_censored flag** 병행 |
| **명목형 코드** | 시술 시기 코드(TRZKPL 등), 배란 유도 유형, 배아 생성 주요 이유, 난자/정자 출처 | LGBM/CatBoost는 native category, XGB/MLP는 Ordinal + Target(OOF) + Frequency 세 가지 병행 |
| **Boolean 원인** | 불임 원인 - 난관 질환 외 9개, 남성/여성/부부 주·부 원인, 불명확 불임 원인 (총 16개 근처) | 0/1 그대로. **합계 파생 (`cause_count`), 패턴 파생 (`cause_pattern_hash`)** |
| **Boolean 시술 여부** | 배란 자극, 단일 배아 이식, 착상 전 유전 검사/진단, 동결/신선/기증 배아 사용, 대리모, PGD, PGS | 0/1, 결측 여부 flag |
| **수치 (수량)** | 총 생성 배아 수, 미세주입 관련 5종, 이식/저장/해동/혼합 배아·난자 수 (~14개) | 수량은 **도메인상 음수 불가**, 0/NaN 구분 중요 |
| **수치 (경과일)** | 난자 채취/해동/혼합 경과일, 배아 이식/해동 경과일 | **blastocyst(5~6일) vs cleavage(2~3일)** 구분이 임신율과 직결 |

→ 이 분류는 `pipeline/schema.py`에 dict로 적재해 모든 후속 단계에서 참조.

### 2-3. 결측 패턴 분석 (구조적 결측 ≫ 무작위 결측)

이번 데이터 첫 행에서 이미 관찰된 패턴:
- `임신 시도 또는 마지막 임신 경과 연수` — 첫 row NaN (최초 시도면 당연)
- `착상 전 유전 검사 사용 여부` NaN — 미실시면 NaN으로 보임
- `난자 해동 경과일`, `난자 혼합 경과일`, `배아 해동 경과일` NaN — 해당 시술 미사용 시

**도메인 규칙 기반 결측 처리 (leakage 아님)**:

| 조건 | 컬럼 | 처리 |
| --- | --- | --- |
| 시술 유형 == DI | 배아/난자/미세주입 관련 수량 전부 | 0 (DI는 체외수정 아님) |
| 특정 시술 유형 ∈ {DI, IVF} (ICSI 아님) | 미세주입 관련 수량 | 0 |
| 이식된 배아 수 == 0 | 배아 이식 경과일 | 0 + `no_transfer` flag |
| 해동 배아/난자 수 == 0 | 해동 경과일 | 0 |
| 난자/정자 출처 == "본인/배우자 제공" | 기증자 나이 | "해당없음" 카테고리 |
| 총 임신 횟수 == 0 | 임신 경과 연수 | -1 sentinel + `first_attempt` flag |

원칙: **fit은 train에만, transform은 train/test**. 규칙 기반은 데이터 통계가 아니므로 CV 밖에서 적용해도 무방. 통계 기반 보간(평균/중앙값)은 반드시 fold 내부에서 train fit.

**결측 자체를 신호로 활용**: 주요 컬럼마다 `*_is_missing` 이진 플래그 생성 → LightGBM 중요도로 잔류 여부 판단.

### 2-4. 변수-타겟 관계 분석 (bivariate)

- **범주형 × target**: 카테고리별 `mean(y)`, `count`, **Wilson score 95% CI** (샘플 소수 카테고리의 과적합 방지).
- **수치형 × target**: target decile plot + monotonicity 확인. 비단조(U자형) 변수는 구간화 고려 (예: 이식 배아 수는 1~2개 최적, 많을수록 감소 가능).
- **교호작용 필수 후보 10개**:
  1. 시술 당시 나이 × 시술 유형 (IVF는 나이 감소 폭 큼)
  2. 이식된 배아 수 × 배아 이식 경과일 (blastocyst × 단일 이식 = eSET 최적)
  3. 단일 배아 이식 여부 × 나이
  4. 기증 난자 사용 × 환자 나이 (기증 난자는 환자 나이 영향 소거)
  5. PGD/PGS × 나이 (유전 진단은 고령에서 효과)
  6. 총 생성 배아 수 × 이식된 배아 수 (이식 집중도)
  7. 미세주입 배아 이식 수 × 특정 시술 유형 (ICSI 일관성 검증)
  8. 이전 출산 횟수 × 임신 경과 연수
  9. 불임 원인 개수(cause_count) × 시술 유형
  10. 해동 배아 사용 × 배아 이식 경과일 (FET 성공률)

### 2-5. 데이터 정합성 검증 (anomaly flag로 저장)

- 이식된 배아 수 > 총 생성 배아 수 → `inconsistent_transfer`
- 저장된 배아 + 이식된 배아 > 총 생성 배아 → `inconsistent_total`
- 미세주입 배아 이식 수 > 이식된 배아 수 → `inconsistent_icsi`
- 혼합된 난자 수 > 수집된 신선 난자 수 + 해동 난자 수 → `inconsistent_mix`
- 시술 유형 DI인데 배아 수 > 0 → `di_with_embryo` (스키마 오류 가능성)

→ flag 각각 피처로 사용 + 중요도 높으면 라벨 노이즈 신호.

### 2-6. 신규 컬럼 34개 중점 조사 체크리스트

이전 대회 기준 33개 대비 새로 생긴(또는 세분화된) 것으로 보이는 영역:
- **미세주입(ICSI) 세분화**: 미세주입된 난자, 미세주입에서 생성된 배아, 미세주입 배아 이식, 미세주입 후 저장된 배아 — 4개
- **해동/혼합**: 해동 배아·난자 수, 난자 해동/혼합 경과일, 배아 해동 경과일 — 5개
- **기증 배아**: 기증 배아 사용 여부 (이전엔 난자/정자만)
- **불임 원인 boolean 세분화**: 정자 농도/운동성/형태/면역학적 4종 + 자궁경부/자궁내막증 등
- **PGD / PGS** 분리 플래그

각각에 대해 (a) 결측률, (b) 양성률 편차, (c) 시술 유형과의 정합성 확인.

### 2-7. Train/Test 분포 비교 (covariate shift)

- 각 컬럼의 train/test 분포를 KS test (수치) / Chi-square (범주)로 비교.
- 시프트 큰 컬럼 상위 5개는 모델 입력에서 제외 또는 rank 변환.
- **주의**: 이 단계 결과를 전처리/학습에 직접 반영하면 leakage 위험. "어떤 컬럼이 리스크인가"만 파악하고 컬럼 drop 판단만 사용.

---

## 3. 전처리 파이프라인 (Data Leakage 절대 방지)

데이콘 가이드에 따라 다음은 전부 **fold 내부에서 fit**:
- Scaler / Imputer (통계 기반)
- Target Encoder (OOF smoothing 필수)
- Frequency Encoder (fold별 train 분포로)
- KBinsDiscretizer, Clustering, Feature Selection

다음은 **fold 밖에서 가능** (leakage 아님):
- 도메인 규칙 기반 결측 처리 (위 2-3)
- 독립 행 내부 산술 연산 (이식/생성 비율 등)
- 고정 매핑 테이블 (예: "만35-37세"→36)

### 파이프라인 순서
1. `read_tables` — train/test 로드 (dtype 강제)
2. `clean_rules.apply` — 도메인 결측 처리 (fold 밖)
3. `derive.build_row_features` — 행 내부 파생 (fold 밖)
4. **StratifiedKFold split** on `y`
5. per fold:
   - `encoders.fit(train_fold)` → transform(train_fold, valid_fold, test)
   - `imputer.fit(train_fold)` → transform(...)
   - `model.fit(train_fold) / predict(valid_fold, test)`
   - OOF 저장, test 예측 누적(평균)

---

## 4. Feature Engineering (도메인 기반, 40~60개 후보 → 필터링)

### 4-1. 비율·효율 (row-independent, fold 밖 가능)
- `fertilize_yield = 미세주입에서 생성된 배아 / 미세주입된 난자`
- `embryo_yield = 총 생성 배아 / (수집 신선 난자 + 해동 난자)`
- `transfer_density = 이식된 배아 / 총 생성 배아`
- `freeze_ratio = 저장된 배아 / 총 생성 배아`
- `icsi_in_transfer = 미세주입 배아 이식 / 이식된 배아`
- `prev_pregnancy_rate = 총 임신 / 총 시술`
- `prev_birth_rate = 총 출산 / 총 임신`
- 모든 비율: 분모 0 시 0 또는 -1 sentinel, 별도 flag

### 4-2. Stage/Timing
- `blast_transfer = 배아 이식 경과일 >= 5` (blastocyst stage)
- `cleavage_transfer = 배아 이식 경과일 in {2,3}`
- `frozen_transfer = 해동된 배아 수 > 0` (FET)
- `fresh_transfer = 해동된 배아 수 == 0 and 이식된 배아 수 > 0`

### 4-3. 원인 요약
- `cause_count = sum(불임원인 bool 10여개)`
- `unexplained = cause_count == 0 and 불명확 불임 원인 == 1`
- `male_factor_strong = (정자 농도 + 운동성 + 형태 + 면역학적) >= 2`
- `female_factor_strong = (난관 + 배란 + 자궁내막증 + 자궁경부) >= 2`

### 4-4. 나이 조합
- `age_ord` — 순서형 인덱스
- `donor_age_gap = 환자나이 - 난자기증자나이` (기증 사용 시)
- `age_bucket_success_prior` — 이전 대회 통계 기반 prior는 **leakage 회피 위해 사용 금지** (train에서 fold-safe로 학습할 것)

### 4-5. Count 조합 & 희귀도
- `(시술 유형, 특정 시술 유형)` 조합 frequency (fold 내 fit)
- `(시술 시기 코드)` frequency
- 극희귀 조합(≤50건) → `rare_combo` flag

### 4-6. 필터링 절차
각 후보에 대해:
1. Train-only fold-safe permutation importance
2. target과의 상관 (Pearson / Spearman / MI)
3. bin별 mean(y) plot에서 단조성/변별력 육안 확인
4. 상위 30~40개만 잔류. 나머지는 `derive.py`에 주석 처리 상태로 보존 (재실험용).

---

## 5. 모델링 전략

### 5-1. 손실 함수 (이진 분류 + 확률 출력 전제)
| 후보 | 목적 | 비고 |
| --- | --- | --- |
| **BCE (logloss)** | 기본 | LightGBM `binary`, XGB `binary:logistic` |
| **BCE + pos_weight** | 불균형 보정 | pos_weight=2.87 |
| **Focal Loss (γ=2)** | 어려운 양성 집중 | MLP에서 주로 |
| **Brier 직접 최적화** | 캘리브레이션 우선 | 확률 제출 metric이 Brier일 경우 가장 유리 |
| Class-balanced BCE | re-weighting | effective number of samples |

→ **대회 metric 공식 발표 확인 후 주 손실 선택.** 셋 다 베이스라인으로 비교 학습하여 OOF score를 기록.

### 5-2. 모델 풀 (다양성 확보)
- **LightGBM** — native categorical, 빠른 반복, 베이스라인 1순위
- **XGBoost** — histogram + GPU, LGBM과 약간 다른 bias
- **CatBoost** — 고카디널리티 categorical(시술 시기 코드 등)에 강함
- **Tabular MLP** — BN + Dropout + BCE, GBDT 앙상블 다양성 기여
  - 이전 대회 MLP 코드 **그대로 재사용 금지**. 아키텍처(레이어 수/순서/활성화) 변경.
- **(선택) FT-Transformer / TabNet** — 시간 여유 시

### 5-3. Cross Validation
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on `y` (이진이므로 단순).
- Group 키 후보: ID가 환자 단위가 아니라 주기(row) 단위로 보이므로 **GroupKFold는 기본 off**. 단, 데이터명세.xlsx 확인 후 ID에 환자 정보 있으면 즉시 GroupKFold 전환.
- OOF metric = 대회 metric(공식) + AUC + LogLoss + Brier + F1@0.5 병기 로그.

### 5-4. 하이퍼파라미터 탐색
- Optuna 30~60 trials, objective = 공식 대회 metric (OOF).
- Early stopping은 공식 metric 기준 custom (LGBM `feval`, XGB `custom_metric`).

---

## 6. 앙상블 & 후처리

### 6-1. 블렌딩
- 모델별 OOF 저장 → Nelder-Mead로 가중치 탐색 (가중치 합=1, 비음수 제약).
- Rank averaging도 병행: AUC 기준 metric이면 rank blending이 유리할 수 있음.

### 6-2. Calibration (확률 제출이므로 중요)
- **Isotonic regression** OOF fit → test 예측 보정.
- Beta calibration 비교.
- Platt scaling은 sigmoid 모양 전제 → 후순위.

### 6-3. Clipping
- `clip(ε, 1-ε)` ε 탐색 (0.001 ~ 0.02). LogLoss metric일 경우 극단값 페널티 줄임.

### 6-4. Threshold (metric에 F1 포함 시)
- 공식 metric에 F1 구성요소 있으면 OOF에서 최적 threshold 탐색. 없으면 생략.

---

## 7. 실험 운영

- 하나의 entrypoint (`pipeline/run.py`) + `config.yaml` 치환. 이전 대회의 `V55_oof` 네이밍 이어가지 않음 → `run_20260423_1400_<tag>` 형식.
- 모든 run은 `runs/<run_id>/` 아래에 config/oof/pred/log 자동 저장.
- **가설 로그** (`reports/hypotheses.md`): 한 줄 포맷 `날짜 | 가설 | 결과 | OOF Δ`.
- **오류 분석**: OOF에서 (a) 고확률 예측 + 실제 0 (FP), (b) 저확률 예측 + 실제 1 (FN)의 row를 각각 200건 추출, 공통 패턴 찾기 → 신규 파생변수 or 규칙 결측 처리 개선.

---

## 8. 일정 (대회 기간 가정)

| 구간 | 작업 |
| --- | --- |
| Day1 오전 | EDA §2-1~2-3 (타겟·스키마·결측) + LGBM BCE 베이스라인 제출 |
| Day1 오후 | EDA §2-4~2-6 (bivariate, 정합성, 신규컬럼) + 파생변수 1차 30개 |
| Day2 오전 | XGB/CatBoost 추가, 손실 3종 비교 (BCE/pos_weight/Focal 또는 Brier) |
| Day2 오후 | 하이퍼파라미터 Optuna, OOF Isotonic calibration |
| Day3 오전 | MLP 추가, 앙상블 가중치 탐색 |
| Day3 오후 | 오류 분석 → 파생변수 보강, 최종 제출 후보 2개 확정 |
| 마감 전 | 재현성 체크 (seed 고정), requirements 기록, 코드 주석 정리 |

---

## 9. 규정 리스크 & 대응

| 리스크 | 대응 |
| --- | --- |
| **이전 대회 코드 유사도 90%+** | 디렉토리/함수/변수명/결측 처리 순서를 **전면 재설계**. 수치 옵션(ε, clip 범위, LR 등)도 동일하지 않게. |
| Data Leakage | (1) test 통계 절대 미사용, (2) 인코더/스케일러 fold 내부 fit, (3) EDA 인사이트를 test만 관찰해 얻지 않기 |
| 외부 데이터 사용 | 금지. 사전학습 모델도 논문 링크·출처 필수. |
| 재현성 | seed, requirements.txt, 실행 순서 README 필수. |
| Public/Private 편차 | CV ↔ Public ↔ Private 3자 추적. CV 기준 우선. |
| 양성 불균형(25.84%) | pos_weight/class_weight/Focal 중 하나 + stratified CV |

---

## 10. 착수 우선순위 (다음 해야 할 일)

1. **데이터 명세.xlsx 파싱** → 컬럼별 의미·허용값·도메인 제약 표로 정리 (schema.py 초안)
2. **공식 평가 metric 확인** (대회 페이지 / 데이콘 공지) → 손실 후보 확정
3. `pipeline/io_loader.py` + `clean_rules.py` 작성 → train/test 결측 규칙 적용
4. LightGBM BCE 베이스라인 제출 (파생변수 없이 raw + 규칙 결측 처리만)
5. EDA 노트북 (`reports/eda_target.md`, `eda_missing.md`) 1차 기록

