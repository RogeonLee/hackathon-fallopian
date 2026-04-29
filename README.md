# 🥚 난임 환자 임신 성공 여부 예측 AI — 8조 팔로리안

> **데이콘 × 오즈코딩스쿨 해커톤** — 난임 환자의 임신 성공 여부 이진 분류
> 기간: 2026-04-23 ~ 2026-04-29 (6일)
> 팀: 8조 팔로리안 (로건님 · 정호님 · 태준님)

---

## 📊 대회 개요

| 항목 | 내용 |
|---|---|
| **데이터** | train 256,351 × 69 / test 90,067 × 68 |
| **타깃** | 임신 성공 여부 (이진), 양성 비율 26% |
| **평가 지표** | ROC-AUC |
| **제출 룰** | Public LB 자동 best 선택 |
| **1위 LB** | 0.74246 (외부 팀) |
| **8조 최종 Public LB** | **0.74218** (자동 best) |

---

## 🏆 핵심 결과

| 모델 | OOF AUC | Public LB | 비고 |
|---|---|---|---|
| 정호 V3 (5-seed × 5-fold LGBM) | 0.74045 | **0.74201** | 한때 Public 1위 |
| 로건 v8 (3원 blend: v5+CB+XGBt) | 0.74079 | 0.74208 | 베이스 갱신 |
| 로건 v9 (6-way 앙상블) | — | 0.74221 | 가이드 헤더 기준 |
| 로건 ag8h+v8 페어 | 0.74119 | **0.74218** | 자동 best 채택 |
| 로건 NM safe 4-way | 0.74137 | 0.74218 | 천장 확인 |

→ **데이터 정보 한계로 OOF 0.74119대 천장 도달.** 모든 강한 모델 상관계수 0.99+ → 알고리즘 다양화의 한계.

---

## 📁 폴더 구조 (작업 일자별)

```
hackathon-fallopian/
│
├── 📅 2026-04-23_수_1일차_초기세팅/
│   ├── architecture.md         # 시스템 설계
│   ├── baseline_lgbm_auc.py    # LGBM 베이스라인
│   ├── oof_lgbm_auc_0p73974.npy
│   └── 8조_팔로피안_대회일지.txt
│
├── 📅 2026-04-24_목_2일차_도메인학습_V3/
│   ├── 난임_도메인_가이드.pdf   # IVF 의학 도메인 학습 자료
│   ├── 데이터전처리.md
│   ├── AH_03_신정호_프로젝트3_v3.ipynb  # 정호 V3 (LB 0.74201)
│   ├── final_features.json
│   └── 2일차기록.html
│
├── 📅 2026-04-27_일_3일차_v9_v10전략/
│   ├── 8조_팔로리안_v9_가이드북.pdf
│   ├── 8조_팔로리안_v10_고도화_전략_리포트.pdf
│   └── 8조_팔로리안_난임프로젝트_v9.ipynb
│
├── 📅 2026-04-28_월_4일차_팀자료수합/
│   ├── 8조_팔로리안_통합기록.md      # 전체 작업 통합 기록
│   ├── 발표자료_종합분석.md
│   ├── 발표자료_슬라이드개요.md
│   ├── 아이디어_회의_최종전략.md
│   ├── 해커톤_난임_정호님/           # 4개 LLM 대화록
│   └── 해커톤_난임_태준님/           # Claude/Gemini 작업 기록
│
├── 📅 2026-04-29_화_5일차_v14_v19_발표준비/
│   ├── 8조_팔로리안_발표_PPT.md      # Marp 발표 슬라이드
│   ├── v15_blend.py / v15_final.py
│   ├── v16_fixed.py
│   ├── v17_parquet.py / v17b_earlystop.py
│   ├── v18_te.py (Target Encoding 강화)
│   ├── v19_catboost_pseudo.py
│   └── 8조_팔로피안_난임프로젝트_v14/v15.ipynb
│
├── 🔧 이번대회자료/                  # 메인 작업 디렉토리
│   ├── data/                         # (gitignore — 데이콘 다운로드 필요)
│   ├── 01~30_*.py                    # 작전 스크립트 30개
│   ├── fe_v10.py                     # v10 효율 피처 모듈
│   ├── oof_*.npy / test_*.npy        # 모델별 OOF/예측
│   └── submission_*.csv              # 제출 후보 50+개
│
├── 📤 submissions/                   # 주요 제출 파일
│   ├── 04-24_submission_초기.csv
│   ├── 04-27_v9_LB0.74221.csv
│   ├── 04-29_v14.csv
│   └── 04-29_v15.csv
│
├── README.md (이 파일)
└── .gitignore
```

---

## 🚀 사용법 (재현 가이드)

### 1. 환경 설정

```bash
git clone https://github.com/RogeonLee/hackathon-fallopian.git
cd hackathon-fallopian

# Python 3.12 venv 권장
python3.12 -m venv .venv
source .venv/bin/activate

# 핵심 의존성
pip install lightgbm==4.3.0 xgboost==2.0.3 catboost==1.2.5
pip install scikit-learn pandas numpy scipy
pip install autogluon.tabular  # AG 작전용
```

### 2. 데이터 준비

데이콘 대회 페이지에서 데이터 다운로드 후:

```bash
mkdir -p 이번대회자료/data
# train.csv, test.csv, sample_submission.csv를 위 경로에 배치
```

### 3. 베이스라인 실행

```bash
# 정호 V3 노트북 (LB 0.74201 재현)
jupyter notebook "2026-04-24_목_2일차_도메인학습_V3/AH_03_신정호_프로젝트3_v3.ipynb"

# 로건 v9 6-way 앙상블 (LB 0.74221)
jupyter notebook "2026-04-27_일_3일차_v9_v10전략/8조_팔로리안_난임프로젝트_v9.ipynb"
```

### 4. 고급 작전 실행 (AutoGluon)

```bash
cd 이번대회자료

# AG raw stack=0 10시간 학습 (단일 OOF 최강 0.74126)
python 25_ag_raw_stack0_10h.py

# Pseudo Labeling v3
python 24_pseudo_v3.py

# 최종 NM 블렌드
python 30_final_decision.py
```

---

## 🔑 핵심 발견

### ✅ 효과적이었던 것

1. **raw 데이터 + LightGBM native categorical** — 베이스 최강
2. **5-seed × 5-fold bagging** — 가장 일관된 +0.0005 효과 (정호 V3 핵심)
3. **AutoGluon best_quality 8h+ + DyStack 우회** — 단일 모델 OOF 최강
4. **Rank Normalization + Nelder-Mead 가중치 최적화** — 블렌드의 정수
5. **Adversarial Validation** — 1시간으로 distribution shift 가설 빠르게 검증

### ❌ 효과 없었던 것 (시행착오)

| 시도 | 결과 |
|---|---|
| DART/GOSS boosting | OOF 0.7391/0.7402, 약함 |
| CatBoost Optuna 튜닝 | default보다 낮음 |
| LogReg meta 스태킹 | 블렌딩보다 ↓ |
| MLP variants | tabular엔 약함 (0.7367~) |
| **v10 효율 피처 20개** | **OOF 부풀림, LB 갭 축소 (train fit)** |
| Pseudo v3/v4 | corr 0.99875, 다양성 0 |
| Target Encoding | corr 0.99+, 다양성 0 |
| AutoGluon 스태킹 | DyStack가 자동 OFF |
| Optuna 200 trials | 개선 0 |

### 📐 메타 발견 (다음 대회 자산)

1. **OOF→LB 갭은 모델 종류별 시스템적**
   - v8 raw: +0.00129, AG 추가: +0.00099, pseudo 추가: -0.00081
2. **상관계수 0.99+ = 알고리즘 다양화 한계** = 데이터 정보 한계
3. **OOF 향상 ≠ LB 향상** — train fit 카드 식별 필수
4. **자동 best public 룰 = 분산 최대화 전략**
5. **추정을 데이터로 검증하는 습관** — 3번 빗나간 패턴은 가설 정정 신호

---

## 👥 팀 구성

| 멤버 | 트랙 | 핵심 기여 |
|---|---|---|
| **로건님** | LGBM + AutoGluon + 블렌딩 | v8 베이스(LB 0.74208), AG 깊이, NM 블렌드 |
| **정호님** | 4개 LLM 협업 (Claude/젠스파크/Gemini/ChatGPT) | V3 5-seed bagging (LB 0.74201, 한때 1위) |
| **태준님** | Claude/Gemini 분석 | 작업 기록 정리 |

---

## 📚 참고 문서

- **`2026-04-28_월_4일차_팀자료수합/8조_팔로리안_통합기록.md`** — 전체 작업 통합 기록
- **`2026-04-29_화_5일차_v14_v19_발표준비/8조_팔로리안_발표_PPT.md`** — Marp 발표 슬라이드
- **`2026-04-27_일_3일차_v9_v10전략/8조_팔로리안_v9_가이드북.pdf`** — v9 6-way 앙상블 가이드
- **`2026-04-27_일_3일차_v9_v10전략/8조_팔로리안_v10_고도화_전략_리포트.pdf`** — v10 고도화 전략

---

## 📝 라이선스 및 주의사항

- **대회 데이터는 데이콘 페이지에서 직접 다운로드** (재배포 금지, gitignore 처리)
- **본 코드는 학습/연구 목적**
- 8조 팔로리안의 작업 결과물

---

*마지막 업데이트: 2026-04-29*
