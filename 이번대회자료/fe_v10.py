"""
v10 전략 PDF 기반 피처 엔지니어링 모듈 — 핵심: '나이 대비 배아 생성 효율'

PDF의 col_X 표기를 한국 컬럼명으로 매핑하고,
v9 노트북의 age_map(중위값) 방식을 사용해 age_num을 생성한다.
"""
import numpy as np
import pandas as pd

AGE_MAP = {
    '만18-34세': 26.0,
    '만35-37세': 36.0,
    '만38-39세': 38.5,
    '만40-42세': 41.0,
    '만43-44세': 43.5,
    '만45-50세': 47.5,
    '알 수 없음': np.nan,
}

DONOR_AGE_MAP = {
    '만20세 이하': 19.0,
    '만21-25세': 23.0,
    '만26-30세': 28.0,
    '만31-35세': 33.0,
    '만36-40세': 38.0,
    '만41-45세': 43.0,
    '알 수 없음': np.nan,
}

COUNT_COLS = [
    '총 시술 횟수', '클리닉 내 총 시술 횟수', 'IVF 시술 횟수', 'DI 시술 횟수',
    '총 임신 횟수', 'IVF 임신 횟수', 'DI 임신 횟수',
    '총 출산 횟수', 'IVF 출산 횟수', 'DI 출산 횟수',
]
COUNT_MAP = {'0회':0.0, '1회':1.0, '2회':2.0, '3회':3.0, '4회':4.0, '5회':5.0, '6회 이상':6.0}

# v10 PDF 컬럼 매핑 (한국 컬럼명)
COL_TOTAL_EMBRYO  = '총 생성 배아 수'
COL_ICSI_EGG      = '미세주입된 난자 수'
COL_ICSI_EMBRYO   = '미세주입에서 생성된 배아 수'
COL_TRANSFER      = '이식된 배아 수'
COL_STORED        = '저장된 배아 수'
COL_THAWED        = '해동된 배아 수'
COL_FRESH_EGG     = '수집된 신선 난자 수'
COL_MIXED         = '혼합된 난자 수'
COL_DONOR_SPERM   = '기증자 정자와 혼합된 난자 수'
COL_AGE_CAT       = '시술 당시 나이'

PIPELINE_NUM_COLS = [
    COL_TOTAL_EMBRYO, COL_ICSI_EGG, COL_ICSI_EMBRYO, COL_TRANSFER, COL_STORED,
    COL_THAWED, COL_FRESH_EGG, COL_MIXED, COL_DONOR_SPERM,
    '미세주입 배아 이식 수', '미세주입 후 저장된 배아 수',
    '해동 난자 수', '저장된 신선 난자 수', '파트너 정자와 혼합된 난자 수',
]


def fe_v10(df):
    """v10 전략 피처 엔지니어링. 원본 df는 변형되지 않음."""
    df = df.copy()

    # [0] 횟수 컬럼 숫자화
    for c in COUNT_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).map(COUNT_MAP).astype('float32')

    # [1] 나이 수치화
    df['age_num'] = df[COL_AGE_CAT].map(AGE_MAP).astype('float32')
    if '난자 기증자 나이' in df.columns:
        df['egg_donor_age_num'] = df['난자 기증자 나이'].map(DONOR_AGE_MAP).astype('float32')
    if '정자 기증자 나이' in df.columns:
        df['sperm_donor_age_num'] = df['정자 기증자 나이'].map(DONOR_AGE_MAP).astype('float32')

    # [2] 파이프라인 컬럼 결측 → 0
    for c in PIPELINE_NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype('float32')

    # [3] 논리 정합성 보정
    df[COL_TOTAL_EMBRYO] = np.maximum(
        df[COL_TOTAL_EMBRYO],
        df[COL_TRANSFER] + df[COL_STORED]
    ).astype('float32')
    df[COL_ICSI_EMBRYO] = np.minimum(
        df[COL_ICSI_EMBRYO],
        df[COL_ICSI_EGG]
    ).astype('float32')

    # [4] 단계별 전환율 (v10 PDF 핵심)
    df['egg_to_embryo_rate']        = (df[COL_TOTAL_EMBRYO]  / (df[COL_FRESH_EGG] + 1)).astype('float32')
    df['icsi_success_rate']         = (df[COL_ICSI_EMBRYO]   / (df[COL_ICSI_EGG]  + 1)).astype('float32')
    df['embryo_utilization_rate']   = ((df[COL_TRANSFER] + df[COL_STORED]) / (df[COL_TOTAL_EMBRYO] + 1)).astype('float32')
    df['total_pipeline_efficiency'] = ((df[COL_TRANSFER] + df[COL_STORED]) / (df[COL_FRESH_EGG]    + 1)).astype('float32')

    # [5] 도메인 전략 플래그
    df['is_FET']            = (df[COL_THAWED] > 0).astype('int8')
    df['is_split_strategy'] = ((df[COL_MIXED] > 0) & (df[COL_ICSI_EGG] > 0)).astype('int8')
    df['is_donor_sperm']    = (df[COL_DONOR_SPERM] > 0).astype('int8')
    df['ever_delivered']    = (df['총 출산 횟수'].fillna(0) > 0).astype('int8')

    # [6] ★ 핵심 피처: 나이 대비 배아 생성 효율 (3가지 변형)
    # v1: PDF 코드 정의 — 효율 × (나이 / 35)
    df['age_normalized_efficiency'] = (df['egg_to_embryo_rate'] * (df['age_num'] / 35.0)).astype('float32')
    # v2: PDF 본문 정의 — (총 생성 배아 수) / (환자 나이 - 18)  (최소가임연령 18 가정, 0방지 +1)
    df['embryo_per_fertile_year']   = (df[COL_TOTAL_EMBRYO] / (df['age_num'] - 18.0 + 1.0)).astype('float32')
    # v3: 고령 가중 — 나이가 많을수록 효율 높을 때 가산점
    df['age_weighted_embryo_eff']   = (df['egg_to_embryo_rate'] * (df['age_num'] - 25.0).clip(lower=0)).astype('float32')
    # v4: log 스케일 — 한계효용 체감 반영
    df['log_age_efficiency']        = np.log1p(df['age_normalized_efficiency']).astype('float32')

    # [7] 동결 주기(FET) 노이즈 제거 — 신선 난자 관련 컬럼을 -1로 마킹
    fet_mask = df['is_FET'] == 1
    for c in [COL_FRESH_EGG, '저장된 신선 난자 수']:
        if c in df.columns:
            new_col = f'{c}_FETnull'
            df[new_col] = df[c].astype('float32').copy()
            df.loc[fet_mask, new_col] = -1.0

    # [8] 남성요인-ICSI 정합성
    has_male = '불임 원인 - 남성 요인' in df.columns
    has_proc = '특정 시술 유형' in df.columns
    if has_male and has_proc:
        male_factor = pd.to_numeric(df['불임 원인 - 남성 요인'], errors='coerce').fillna(0).astype('int8')
        has_ICSI = df['특정 시술 유형'].fillna('').astype(str).str.contains('ICSI', regex=False).astype('int8')
        df['is_male_icsi_mismatch'] = (male_factor != has_ICSI).astype('int8')
        df['male_factor_no_icsi']   = ((male_factor == 1) & (has_ICSI == 0)).astype('int8')
        df['icsi_no_male_factor']   = ((male_factor == 0) & (has_ICSI == 1)).astype('int8')

    return df


if __name__ == '__main__':
    # 샘플 검증
    train = pd.read_csv('./data/train.csv', nrows=2000)
    fe = fe_v10(train)
    new = [c for c in fe.columns if c not in train.columns]
    print(f'추가 피처 {len(new)}개:')
    for c in new:
        print(f'  - {c}: dtype={fe[c].dtype}, NaN={fe[c].isna().sum()}, range=[{fe[c].min():.3f}, {fe[c].max():.3f}]')
