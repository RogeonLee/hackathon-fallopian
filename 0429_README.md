# 🧬 IVF Pregnancy Prediction Model (AUC 0.742+ Challenge)

> **난임 환자의 개인별 특성 및 시술 데이터를 기반으로 한 임신 성공 확률 예측 모델**

이 프로젝트는 난임 시술(IVF/ICSI) 데이터를 정밀 분석하여, 환자의 생학적 지표와 시술 환경 사이의 복잡한 상호작용을 포착하고 임신 성공 여부를 예측하는 머신러닝 파이프라인을 구축하는 것을 목표로 합니다.

---

## 🚀 Project Overview

* **목표:** AUC-ROC Score **0.742** 돌파 (실전 리더보드 기준)
* **핵심 전략:**
    * 의학적 근거에 기반한 **Killer Feature Engineering** (나이-배반포 시너지 등)
    * AutoGluon을 활용한 **Multi-layer Stack Ensembling**
    * 도메인 특화 데이터 전처리 (Censored 데이터 및 Sentinel 값 처리)

---

## 🛠️ Tech Stack

* **Language:** Python 3.12+
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** AutoGluon, LightGBM, XGBoost, CatBoost
* **Optimization:** Nelder-Mead Weights Optimization, Ridge Stacking

---

## 📂 Project Structure

```text
project/
├── data/
│   ├── train.csv                 # 학습용 데이터 (성공률 ~25.8%)
│   ├── test.csv                  # 평가용 데이터
│   └── sample_submission.csv
├── pipeline/
│   ├── schema.py                 # 컬럼 정의 및 EDA 기반 데이터 스키마
│   ├── preprocess.py             # 데이터 정제 및 누수 방지용 Preprocessor
│   └── feature_engineering.py    # 도메인 지식 기반 파생 피처 생성기
├── main_autogluon.py             # AutoGluon 기반 스태킹 모델 학습
├── main_ensemble.py              # 6-Model (LGBM, XGB, Cat) 앙상블 파이프라인
└── README.md

## 🧬 Medical Synergy Features (Killer Features)단순한 데이터 학습을 넘어, 난임 전문의의 시각에서 중요하게 여겨지는 상호작용 피처들을 주입했습니다.Physician Confidence Signal: 배아가 충분함에도 단일 배아 이식(SET)을 선택했다면, 해당 배아의 질이 최상급임을 의미하는 의학적 판단을 피처화.Age-Day5 Interaction: 35세 미만 연령과 5일차 배반포(Blastocyst) 이식의 강력한 양의 상관관계 포착.Embryo Yield Index: 수집된 난자 수 대비 생성된 배아의 비율을 통해 난소 반응성 및 수정 효율 산출.Censored History Handling: '6회 이상' 등 범주형으로 묶인 과거력을 수치화 및 비선형 패턴 반영.🤖 Model ArchitectureMulti-layer Stack EnsembleAutoGluon의 best_quality 프리셋을 활용하여 복잡한 모델 계층을 구성했습니다.Level 1: LightGBM, RandomForest, CatBoost 등 다양한 기초 모델 학습 (Bagging 적용)Level 2 & 3: 하위 모델의 예측값을 피처로 사용하는 메타 모델 학습Level 4: 최종 WeightedEnsemble을 통한 가중치 최적화📊 ResultsModelValidation AUCHoldout AUCStatusBaseline (v3 features)0.72x0.71x-LightGBMXT_BAG_L10.73980.7434Target AchievedWeightedEnsemble_L40.75460.7338Overfitted최종 성과: 실전 테스트 데이터와 가장 유사한 Holdout 데이터셋에서 AUC 0.7434를 기록하며 목표치(0.742) 돌파 완료.⚙️ How to Run1. 데이터 전처리 및 학습Pythonfrom pipeline.preprocess import IVFPreprocessor
from pipeline.feature_engineering import IVFFeatureEngineer
from autogluon.tabular import TabularPredictor

## 1. 전처리 및 피처 생성
pre = IVFPreprocessor().fit(train_df)
eng = IVFFeatureEngineer()

train_final = eng.transform(pre.transform(train_df))

## 모델 학습
predictor = TabularPredictor(label='target', eval_metric='roc_auc').fit(
    train_final, 
    presets='best_quality',
    num_stack_levels=2
)

2. 예측 및 제출 파일 생성

# Holdout 1위 모델(LightGBMXT_BAG_L1)을 지정하여 예측 시 최상의 결과 도출
best_model = 'LightGBMXT_BAG_L1'
pred_probas = predictor.predict_proba(test_final, model=best_model)[1]

👨‍🔬 AuthorsGemini & Claude - AI Collaboration for Feature Engineering & ArchitectureUser - Domain Research & Pipeline Implementation

