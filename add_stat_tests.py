"""
두 노트북에 통계 검증 셀을 추가하는 스크립트.
- topic2_analysis.ipynb: 6개 셀 추가 (행태/브랜드/서베이/BERTopic/불만/Poisson)
- topic3_analysis.ipynb: cell 14 수정 + 1개 셀 추가 (BERTopic×코호트)
"""
import json, sys, io

def lines(src: str) -> list:
    """소스 문자열을 Jupyter source 리스트로 변환"""
    rows = src.split('\n')
    return [r + '\n' for r in rows[:-1]] + ([rows[-1]] if rows[-1] else [])

def make_code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": lines(src)}

def make_md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": lines(src)}

# ─────────────────────────────────────────────────────────────
#  TOPIC 2 새 셀 코드
# ─────────────────────────────────────────────────────────────

T2_BEHAVIOR_STAT = '''\
# ── [통계검증] 2-3. 리뷰 행태 지표별 그룹 차이 검정 ─────────────
from scipy import stats as _st
from scipy.stats import chi2_contingency
from itertools import combinations

_G4 = ['네이버', 'Gmail', '카카오', '네이트']
_df4 = df[df['domain_group'].isin(_G4)].copy()
_pairs = list(combinations(_G4, 2))
_bonf = len(_pairs)

print("=" * 70)
print("리뷰 작성 행태 지표별 그룹 간 통계 검정")
print("=" * 70)

# ── (1) 리뷰 길이 — Kruskal-Wallis + 사후검정 ──────────────
_slen = [_df4[_df4['domain_group']==g]['content_len'].dropna().values for g in _G4]
_H, _p = _st.kruskal(*_slen)
print(f"\\n[1] 리뷰 길이 (content_len)")
print(f"    Kruskal-Wallis: H={_H:.3f}, p={_p:.4f} → {'유의 ★' if _p<0.05 else '비유의'}")
print(f"    {'그룹1':6s} vs {'그룹2':6s} | 중앙값1  중앙값2  p(보정)   r      유의")
for g1, g2 in _pairs:
    s1 = _df4[_df4['domain_group']==g1]['content_len'].dropna().values
    s2 = _df4[_df4['domain_group']==g2]['content_len'].dropna().values
    u, pm = _st.mannwhitneyu(s1, s2, alternative='two-sided')
    pa = min(pm * _bonf, 1.0)
    r = abs(1 - 2*u / (len(s1)*len(s2)))
    print(f"    {g1:6s} vs {g2:6s} | {int(s1.mean()):5d}    {int(s2.mean()):5d}   {pa:.4f}   {r:.3f}  {'★' if pa<0.05 else ''}")

# ── (2) 도움돼요 수 — Kruskal-Wallis + 사후검정 ─────────────
_shelp = [_df4[_df4['domain_group']==g]['helpfulTrueCount'].dropna().values for g in _G4]
_H2, _p2 = _st.kruskal(*_shelp)
print(f"\\n[2] 도움돼요 수 (helpfulTrueCount)")
print(f"    Kruskal-Wallis: H={_H2:.3f}, p={_p2:.4f} → {'유의 ★' if _p2<0.05 else '비유의'}")
for g1, g2 in _pairs:
    s1 = _df4[_df4['domain_group']==g1]['helpfulTrueCount'].dropna().values
    s2 = _df4[_df4['domain_group']==g2]['helpfulTrueCount'].dropna().values
    u, pm = _st.mannwhitneyu(s1, s2, alternative='two-sided')
    pa = min(pm * _bonf, 1.0)
    r = abs(1 - 2*u / (len(s1)*len(s2)))
    print(f"    {g1:6s} vs {g2:6s} | mean1={s1.mean():.2f}  mean2={s2.mean():.2f}  p_adj={pa:.4f}  r={r:.3f}  {'★' if pa<0.05 else ''}")

# ── (3) 사진 첨부율 — Chi-square ─────────────────────────────
_ct_photo = pd.crosstab(_df4['domain_group'], _df4['has_photo']).reindex(_G4)
_chi2, _p3, _dof, _exp = chi2_contingency(_ct_photo)
_n3 = _ct_photo.sum().sum()
_v3 = (_chi2 / (_n3 * (min(_ct_photo.shape)-1))) ** 0.5
print(f"\\n[3] 사진 첨부율 — Chi-square")
print(f"    χ²={_chi2:.3f}, df={_dof}, p={_p3:.4f}, Cramér's V={_v3:.3f}")
print(f"    → {'유의 ★ (그룹별 사진 부착 비율 차이 존재)' if _p3<0.05 else '비유의'}")

# ── (4) 상세 리뷰율(>200자) — Chi-square ─────────────────────
_ct_det = pd.crosstab(_df4['domain_group'], _df4['is_detailed']).reindex(_G4)
_chi2d, _p4, _dofd, _expd = chi2_contingency(_ct_det)
_n4 = _ct_det.sum().sum()
_v4 = (_chi2d / (_n4 * (min(_ct_det.shape)-1))) ** 0.5
print(f"\\n[4] 상세 리뷰율 (>200자) — Chi-square")
print(f"    χ²={_chi2d:.3f}, df={_dofd}, p={_p4:.4f}, Cramér's V={_v4:.3f}")
print(f"    → {'유의 ★ (그룹별 상세 리뷰 비율 차이 존재)' if _p4<0.05 else '비유의'}")

print("\\n[해석] r = rank-biserial 효과 크기 (0.1=소, 0.3=중, 0.5=대)")
print("       Cramér's V: 0.1=소, 0.3=중, 0.5=대")
'''

T2_BRAND_STAT = '''\
# ── [통계검증] 2-4. 브랜드 선호도 독립성 검정 ───────────────────
from scipy.stats import chi2_contingency
import numpy as np

_G4 = ['네이버', 'Gmail', '카카오', '네이트']
_dfb = df[df['domain_group'].isin(_G4)].copy()

print("=" * 70)
print("브랜드 선호도 × 도메인 그룹 통계 검정")
print("=" * 70)

# ── (1) 브랜드 × 도메인 독립성 Chi-square ────────────────────
_ct_br = pd.crosstab(_dfb['domain_group'], _dfb['brand']).reindex(_G4)
_chi2, _p, _dof, _exp = chi2_contingency(_ct_br)
_n = _ct_br.sum().sum()
_v = (_chi2 / (_n * (min(_ct_br.shape)-1))) ** 0.5
print(f"\\n[1] 브랜드 × 도메인 그룹 독립성 (Chi-square)")
print(f"    χ²={_chi2:.3f}, df={_dof}, p={_p:.4f}, Cramér's V={_v:.3f}")
print(f"    → {'브랜드 선호도 그룹 간 차이 유의 ★' if _p<0.05 else '차이 없음'}")
print("\\n    빈도표:")
print(_ct_br.to_string())
print("\\n    비율 (%):")
print((_ct_br.div(_ct_br.sum(axis=1), axis=0)*100).round(1).to_string())

# ── (2) 각 그룹 내 삼성 > 애플 유의성 (이항검정) ─────────────
from scipy.stats import binomtest
print(f"\\n[2] 각 그룹 내 삼성 우위 유의성 (이항검정, H0: 50-50)")
print(f"    {'그룹':6s}  Samsung  비율    p-value  유의")
for g in _G4:
    sub = _dfb[_dfb['domain_group']==g]['brand']
    n_sam = (sub == 'Samsung').sum()
    n_tot = len(sub)
    res = binomtest(n_sam, n_tot, p=0.5, alternative='greater')
    print(f"    {g:6s}  {n_sam:4d}/{n_tot:<4d}  {n_sam/n_tot*100:.1f}%   {res.pvalue:.4f}  {'★' if res.pvalue<0.05 else ''}")

print("\\n[해석] 이항검정 H0: 삼성 = 애플 (50%); H1: 삼성 > 50% (단측)")
print("       모든 그룹에서 삼성 비율이 유의미하게 과반인지 확인")
'''

T2_SURVEY_STAT = '''\
# ── [통계검증] 2-5. Survey 만족도 차원별 Kruskal-Wallis ────────
from scipy import stats as _st
from itertools import combinations

_G4 = ['네이버', 'Gmail', '카카오', '네이트']

print("=" * 70)
print("Survey 만족도 차원별 그룹 간 Kruskal-Wallis 검정")
print("=" * 70)
print("(sv2, TOP_QUESTIONS 변수는 2-5 셀에서 정의됨)")

_sv4 = sv2[sv2['domain_group'].isin(_G4)].copy()

for _q in TOP_QUESTIONS:
    _qdata = _sv4[_sv4['question'] == _q]
    _samps = [_qdata[_qdata['domain_group']==g]['score'].dropna().values for g in _G4]
    _samps_v = [s for s in _samps if len(s) >= 5]
    if len(_samps_v) < 2:
        continue
    _H, _p = _st.kruskal(*_samps_v)
    _means = {g: _qdata[_qdata['domain_group']==g]['score'].mean() for g in _G4}
    _max_g = max(_means, key=_means.get)
    _min_g = min(_means, key=_means.get)
    _diff  = _means[_max_g] - _means[_min_g]
    sig_label = '유의 ★' if _p < 0.05 else '비유의'
    print(f"\\n[{_q[:25]}]")
    print(f"    H={_H:.3f}, p={_p:.4f} → {sig_label}  |  최고: {_max_g}({_means[_max_g]:.3f}) 최저: {_min_g}({_means[_min_g]:.3f}) Δ={_diff:.3f}")
    if _p < 0.05:
        _pairs = list(combinations(_G4, 2))
        _bonf  = len(_pairs)
        for g1, g2 in _pairs:
            s1 = _qdata[_qdata['domain_group']==g1]['score'].dropna().values
            s2 = _qdata[_qdata['domain_group']==g2]['score'].dropna().values
            if len(s1)<5 or len(s2)<5:
                continue
            u, pm = _st.mannwhitneyu(s1, s2, alternative='two-sided')
            pa = min(pm * _bonf, 1.0)
            if pa < 0.05:
                r = abs(1 - 2*u / (len(s1)*len(s2)))
                print(f"    사후: {g1} vs {g2}  p_adj={pa:.4f}  r={r:.3f} ★")

print("\\n[해석] 점수 범위 0~1 (0=부정, 0.5=중립, 1=긍정)")
'''

T2_BERTOPIC_STAT = '''\
# ── [통계검증] 2-11. BERTopic 토픽 분포 × 도메인 그룹 Chi-square ─
from scipy.stats import chi2_contingency, fisher_exact
import numpy as np

_G4 = ['네이버', 'Gmail', '카카오', '네이트']

print("=" * 70)
print("BERTopic 토픽 분포 × 도메인 그룹 통계 검정")
print("=" * 70)

# topic_cross: 아웃라이어 제외 카운트 (셀 2-11에서 정의됨)
_ct = topic_cross.reindex([g for g in _G4 if g in topic_cross.index], fill_value=0)
_chi2, _p, _dof, _exp = chi2_contingency(_ct)
_n = _ct.sum().sum()
_v = (_chi2 / (_n * (min(_ct.shape)-1))) ** 0.5
print(f"\\n[1] 전체 토픽 × 도메인 그룹 (아웃라이어 제외)")
print(f"    χ²={_chi2:.3f}, df={_dof}, p={_p:.4f}, Cramér's V={_v:.3f}")
print(f"    → {'도메인별 토픽 분포 차이 유의 ★' if _p<0.05 else '차이 없음'}")
print(f"    최소 기대빈도={_exp.min():.1f} {'⚠ <5 → 해석 주의' if _exp.min()<5 else '(OK)'}")

# ── (2) A/S 불만 토픽 Gmail vs 나머지 — Fisher's Exact ───────
# A/S 관련 키워드로 토픽 식별
_as_kw = {'보상', '불량', '교환', '센터', '고객', '수리', '결함', '반품'}
_as_tid = None
for _tid in valid_topics:
    _tw = {w for w, _ in topic_model.get_topic(_tid)}
    if _tw & _as_kw:
        _as_tid = _tid
        break

if _as_tid is not None:
    print(f"\\n[2] A/S·불만 토픽(T{_as_tid}) Gmail vs 비-Gmail — Fisher's Exact")
    _dft = df[df['domain_group'].isin(_G4)].copy()
    _dft['_is_as'] = (_dft['bert_topic'] == _as_tid).astype(int)
    _gm_as = _dft[_dft['domain_group']=='Gmail']['_is_as'].sum()
    _gm_n  = (_dft['domain_group']=='Gmail').sum()
    _ot_as = _dft[_dft['domain_group']!='Gmail']['_is_as'].sum()
    _ot_n  = (_dft['domain_group']!='Gmail').sum()
    _ct2   = np.array([[_gm_as, _gm_n-_gm_as], [_ot_as, _ot_n-_ot_as]])
    _or, _pfe = fisher_exact(_ct2, alternative='greater')
    print(f"    Gmail: {_gm_as}/{_gm_n} ({_gm_as/_gm_n*100:.1f}%)")
    print(f"    기타 : {_ot_as}/{_ot_n} ({_ot_as/_ot_n*100:.1f}%)")
    print(f"    OR={_or:.3f}, p={_pfe:.4f} → {'Gmail A/S 비율 유의하게 높음 ★' if _pfe<0.05 else '비유의'}")
    print("\\n    그룹별 A/S 토픽 비율:")
    for g in _G4:
        sub = _dft[_dft['domain_group']==g]
        _c = sub['_is_as'].sum(); _tot = len(sub)
        print(f"    {g:6s}: {_c}/{_tot} ({_c/_tot*100:.1f}%)")
else:
    print("\\n[2] A/S 토픽 자동 식별 실패 — BERTopic 실행 결과에 따라 T2 등 직접 확인")
    # bert_topic == 2 로 시도
    _dft = df[df['domain_group'].isin(_G4)].copy()
    for _tid_try in valid_topics:
        _gm_as = (_dft[_dft['domain_group']=='Gmail']['bert_topic'] == _tid_try).sum()
        _ot_as = (_dft[_dft['domain_group']!='Gmail']['bert_topic'] == _tid_try).sum()
        _gm_n  = (_dft['domain_group']=='Gmail').sum()
        _ot_n  = (_dft['domain_group']!='Gmail').sum()
        _gm_pct = _gm_as/_gm_n*100
        _ot_pct = _ot_as/_ot_n*100
        if _gm_pct > _ot_pct * 1.5 and _gm_as >= 3:
            _ct2 = np.array([[_gm_as, _gm_n-_gm_as], [_ot_as, _ot_n-_ot_as]])
            _or, _pfe = fisher_exact(_ct2, alternative='greater')
            print(f"    T{_tid_try} Gmail우세 토픽: {_gm_pct:.1f}% vs {_ot_pct:.1f}%  OR={_or:.3f} p={_pfe:.4f} {'★' if _pfe<0.05 else ''}")
'''

T2_COMPLAINT_STAT = '''\
# ── [통계검증] 2-12. 불만 카테고리 분포 검정 ────────────────────
from scipy.stats import chi2_contingency, fisher_exact
import numpy as np

_G4 = ['네이버', 'Gmail', '카카오', '네이트']

print("=" * 70)
print("불만 카테고리 분포 × 도메인 그룹 통계 검정")
print("=" * 70)
print("(neg_df, COMPLAINT_CATS 변수는 2-12 셀에서 정의됨)")

# 부정 리뷰에서 카테고리별 이진 변수 생성 (리뷰 단위)
_neg4 = neg_df[neg_df['domain_group'].isin(_G4)].copy()
for _cat, _kws in COMPLAINT_CATS.items():
    _neg4[_cat] = _neg4['content'].apply(
        lambda t: 1 if isinstance(t, str) and any(k in t for k in _kws) else 0)

print(f"\\n분석 대상: 부정 리뷰 (1-2점) — 그룹별 n")
for g in _G4:
    print(f"  {g:6s}: {(_neg4['domain_group']==g).sum()}건")

# ── (1) 카테고리별 이진 Chi-square (그룹 × 유/무) ───────────
print("\\n[1] 카테고리별 그룹 간 발생률 차이 (Chi-square)")
print(f"    {'카테고리':<14}  χ²      p        V     유의  ⚠")
for _cat in COMPLAINT_CATS.keys():
    _ct = pd.crosstab(_neg4['domain_group'], _neg4[_cat]).reindex(_G4, fill_value=0)
    if _ct.shape[1] < 2:
        continue
    _chi2, _p, _dof, _exp = chi2_contingency(_ct)
    _n = _ct.sum().sum()
    _v = (_chi2 / (_n * (min(_ct.shape)-1))) ** 0.5
    _warn = "⚠ 소표본" if _exp.min() < 5 else ""
    print(f"    {_cat:<14}  {_chi2:5.2f}  {_p:.4f}  {_v:.3f}  {'★' if _p<0.05 else ''}  {_warn}")

# ── (2) 핵심 클레임 검증: Gmail A/S 불만 높음? ──────────────
print("\\n[2] Gmail A/S·품질 불만 비율 vs 네이버 — Fisher's Exact")
_cat_as = '🔧 A/S/품질'
_gm = _neg4[_neg4['domain_group']=='Gmail']
_nv = _neg4[_neg4['domain_group']=='네이버']
_gm_n, _gm_y = len(_gm), _gm[_cat_as].sum()
_nv_n, _nv_y = len(_nv), _nv[_cat_as].sum()
_ct_as = np.array([[_gm_y, _gm_n-_gm_y], [_nv_y, _nv_n-_nv_y]])
_or, _pfe = fisher_exact(_ct_as, alternative='greater')
print(f"    Gmail: {_gm_y}/{_gm_n} ({_gm_y/_gm_n*100:.1f}%) vs 네이버: {_nv_y}/{_nv_n} ({_nv_y/_nv_n*100:.1f}%)")
print(f"    Fisher's Exact (단측): OR={_or:.3f}, p={_pfe:.4f} → {'Gmail A/S 비율 유의하게 높음 ★' if _pfe<0.05 else '비유의'}")

# ── (3) 배송/포장 불만 전그룹 공통 최다 검증 ────────────────
print("\\n[3] 배송/포장 불만이 모든 그룹에서 1위인지 확인")
_cat_ship = '📦 배송/포장'
for g in _G4:
    sub = _neg4[_neg4['domain_group']==g]
    n = len(sub)
    if n == 0:
        continue
    rates = {c: sub[c].mean()*100 for c in COMPLAINT_CATS.keys()}
    top1 = max(rates, key=rates.get)
    print(f"  {g:6s} (n={n}): 1위={top1} ({rates[top1]:.1f}%), 배송={rates[_cat_ship]:.1f}%")
'''

T2_POISSON_STAT = '''\
# ── [통계검증] 2-14. Poisson 회귀 계수 유의성 (statsmodels GLM) ──
import statsmodels.api as sm
import numpy as np

print("=" * 70)
print("Poisson 회귀 계수 유의성 — statsmodels GLM (비정규화)")
print("=" * 70)
print("(sklearn은 p-value 미제공 → statsmodels로 동일 변수 재적합)")
print("주의: 정규화 없음(alpha=0) — 계수 크기는 sklearn과 다를 수 있음")

# 회귀 데이터 동일 준비 (reg_df, feature_cols는 2-14 셀에서 정의됨)
_X_sm = sm.add_constant(reg_df[feature_cols].astype(float))
_y_sm = reg_df['helpfulTrueCount'].values.astype(float)

_glm = sm.GLM(_y_sm, _X_sm, family=sm.families.Poisson())
_res = _glm.fit()

_params = _res.params
_pvals  = _res.pvalues
_ci     = _res.conf_int()

print(f"\\n{'변수':<20}  계수      IRR     95% CI [lo,  hi]   p-value  유의")
print("-" * 80)
for _feat in feature_cols:
    _coef = _params[_feat]
    _irr  = np.exp(_coef)
    _lo   = np.exp(_ci.loc[_feat, 0])
    _hi   = np.exp(_ci.loc[_feat, 1])
    _p    = _pvals[_feat]
    _lbl  = label_map.get(_feat, _feat)
    _sig  = '★★★' if _p<0.001 else ('★★' if _p<0.01 else ('★' if _p<0.05 else ''))
    print(f"{_lbl:<20}  {_coef:+.3f}  {_irr:.3f}x  [{_lo:.3f}, {_hi:.3f}]   {_p:.4f}  {_sig}")

print(f"\\n로그-우도: {_res.llf:.1f}  |  AIC: {_res.aic:.1f}  |  df_resid: {_res.df_resid:.0f}")
print("\\n[해석] IRR>1 → 도움돼요 증가, IRR<1 → 감소")
print("       95% CI가 1을 포함하지 않으면 통계적으로 유의")
print("       ★★★ p<0.001, ★★ p<0.01, ★ p<0.05")
'''

# ─────────────────────────────────────────────────────────────
#  TOPIC 3 추가 코드
# ─────────────────────────────────────────────────────────────

T3_HELP_POSTHOC = '''\

# ── [통계검증] Helpfulness 코호트 간 사후검정 (Bonferroni) ─────
from itertools import combinations as _comb
import numpy as np

_pairs_c = list(_comb(COHORT_ORDER, 2))
_bonf_c  = len(_pairs_c)

print("\\n[사후검정] Helpfulness 코호트 쌍별 Mann-Whitney U + Bonferroni")
print(f"{'코호트1':<20} vs {'코호트2':<20} | U-stat    p(보정)   r      유의")
print("-" * 80)
for c1, c2 in _pairs_c:
    s1 = df_valid[df_valid['cohort']==c1]['helpfulTrueCount'].dropna().values
    s2 = df_valid[df_valid['cohort']==c2]['helpfulTrueCount'].dropna().values
    if len(s1)<5 or len(s2)<5:
        continue
    u, pm = stats.mannwhitneyu(s1, s2, alternative='two-sided')
    pa = min(pm * _bonf_c, 1.0)
    r  = abs(1 - 2*u / (len(s1)*len(s2)))
    _sig = '★★★' if pa<0.001 else ('★★' if pa<0.01 else ('★' if pa<0.05 else ''))
    print(f"{c1:<20} vs {c2:<20} | {u:8.0f}  {pa:.4f}   {r:.3f}  {_sig}")

# ── [통계검증] 부정 리뷰 helpfulness 코호트 간 Kruskal-Wallis ──
_neg_samps = {c: df_valid[(df_valid['cohort']==c)&(df_valid['rating']<=2)]['helpfulTrueCount'].dropna().values
              for c in COHORT_ORDER}
_neg_valid  = {c: s for c, s in _neg_samps.items() if len(s)>=5}
if len(_neg_valid) >= 2:
    _H_neg, _p_neg = stats.kruskal(*_neg_valid.values())
    print(f"\\n[부정 리뷰 helpfulness 코호트 간 Kruskal-Wallis]")
    print(f"    H={_H_neg:.3f}, p={_p_neg:.4f} → {'유의 ★' if _p_neg<0.05 else '비유의'}")
    for c, s in _neg_valid.items():
        print(f"    {c}: n={len(s)}, 평균={s.mean():.2f}, 중앙값={np.median(s):.1f}")

print("\\n[해석] r = rank-biserial 효과 크기 (0.1=소, 0.3=중, 0.5=대)")
print("       ★★★ p<0.001, ★★ p<0.01, ★ p<0.05")
'''

T3_BERTOPIC_STAT = '''\
# ── [통계검증] 3-4. BERTopic 토픽 분포 × 코호트 Chi-square ───────
from scipy.stats import chi2_contingency, fisher_exact
import numpy as np

print("=" * 70)
print("BERTopic 토픽 분포 × 코호트 통계 검정")
print("=" * 70)
print("(df_tot, bert_topic_t3, ti3, topic_model_t3 는 3-4 BERTopic 셀에서 정의됨)")

_df_tc = df_tot[df_tot['cohort'].isin(COHORT_ORDER)].copy()

# ── (1) 전체 토픽 × 코호트 Chi-square ────────────────────────
_ct_tc = pd.crosstab(_df_tc['cohort'], _df_tc['bert_topic_t3']).reindex(COHORT_ORDER, fill_value=0)
_chi2, _p, _dof, _exp = chi2_contingency(_ct_tc)
_n = _ct_tc.sum().sum()
_v = (_chi2 / (_n * (min(_ct_tc.shape)-1))) ** 0.5
print(f"\\n[1] 전체 토픽 × 코호트 (아웃라이어 포함)")
print(f"    χ²={_chi2:.3f}, df={_dof}, p={_p:.4f}, Cramér's V={_v:.3f}")
print(f"    → {'코호트별 토픽 분포 차이 유의 ★' if _p<0.05 else '차이 없음'}")
print(f"    최소 기대빈도={_exp.min():.1f} {'⚠ <5 → 해석 주의' if _exp.min()<5 else '(OK)'}")

# ── (2) 배송/포장 토픽 얼리어답터 vs 후기다수 — Fisher's Exact ─
_pkg_kw = {'박스', '포장', '뽁뽁이', '상자', '에어', '충격'}
_pkg_tid = None
for _tid in [row['Topic'] for _, row in ti3[ti3['Topic']!=-1].iterrows()]:
    _tw = {w for w, _ in topic_model_t3.get_topic(_tid)}
    if _tw & _pkg_kw:
        _pkg_tid = _tid
        break

_early_col = COHORT_ORDER[0]
_late_col  = COHORT_ORDER[2]

if _pkg_tid is not None:
    print(f"\\n[2] 배송/포장 토픽(T{_pkg_tid}) 얼리어답터 vs 후기다수 — Fisher's Exact")
    _df_tc['_is_pkg'] = (_df_tc['bert_topic_t3'] == _pkg_tid).astype(int)
    _ea = _df_tc[_df_tc['cohort']==_early_col]
    _la = _df_tc[_df_tc['cohort']==_late_col]
    _ea_y, _ea_n = _ea['_is_pkg'].sum(), len(_ea)
    _la_y, _la_n = _la['_is_pkg'].sum(), len(_la)
    _ct2 = np.array([[_ea_y, _ea_n-_ea_y], [_la_y, _la_n-_la_y]])
    _or, _pfe = fisher_exact(_ct2, alternative='greater')
    print(f"    얼리어답터: {_ea_y}/{_ea_n} ({_ea_y/_ea_n*100:.1f}%)")
    print(f"    후기다수  : {_la_y}/{_la_n} ({_la_y/_la_n*100:.1f}%)")
    print(f"    Fisher's Exact: OR={_or:.3f}, p={_pfe:.4f} → {'D+0~30 배송불만 유의하게 집중 ★' if _pfe<0.05 else '비유의'}")
    print("\\n    코호트별 배송/포장 토픽 비율:")
    for c in COHORT_ORDER:
        sub = _df_tc[_df_tc['cohort']==c]
        _cy, _cn = sub['_is_pkg'].sum(), len(sub)
        print(f"    {c}: {_cy}/{_cn} ({_cy/_cn*100:.1f}%)")
else:
    print("\\n[2] 배송/포장 토픽 자동 식별 실패 — topics_over_time 시각화로 확인 권장")
    print("    hint: valid_topics 목록에서 'T1'의 키워드 확인")

# ── (3) 아웃라이어(-1) 비율 코호트 간 차이 ────────────────────
print("\\n[3] BERTopic 아웃라이어(-1) 비율 코호트 간 Chi-square")
_ct_out = pd.crosstab(_df_tc['cohort'], (_df_tc['bert_topic_t3']==-1).astype(int)).reindex(COHORT_ORDER, fill_value=0)
_chi2o, _po, _dofo, _expo = chi2_contingency(_ct_out)
_no = _ct_out.sum().sum()
_vo = (_chi2o / (_no * (min(_ct_out.shape)-1))) ** 0.5
print(f"    χ²={_chi2o:.3f}, df={_dofo}, p={_po:.4f}, Cramér's V={_vo:.3f}")
print(f"    → {'코호트별 아웃라이어 비율 차이 유의 ★' if _po<0.05 else '차이 없음'}")
for c in COHORT_ORDER:
    sub = _df_tc[_df_tc['cohort']==c]
    _oc = (sub['bert_topic_t3']==-1).sum(); _oN = len(sub)
    print(f"    {c}: {_oc}/{_oN} ({_oc/_oN*100:.1f}%)")
'''

# ─────────────────────────────────────────────────────────────
#  NOTEBOOK 수정 함수
# ─────────────────────────────────────────────────────────────

def insert_cells(cells, insert_after_idx, new_cells):
    """insert_after_idx 뒤에 new_cells 삽입. 고차원 → 저차원 순서로 호출해야 함."""
    return cells[:insert_after_idx+1] + new_cells + cells[insert_after_idx+1:]


def patch_topic2():
    with open('topic2_analysis.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    cells = nb['cells']
    n_before = len(cells)

    # 뒤에서 앞 순서로 삽입 (인덱스 밀림 방지)

    # 6) After cell 37 (Poisson) → statsmodels p-values
    cells = insert_cells(cells, 37, [
        make_md("### 2-14-[통계] Poisson 회귀 계수 유의성 (statsmodels GLM)"),
        make_code(T2_POISSON_STAT),
    ])

    # 5) After cell 33 (complaint) → complaint chi-square
    cells = insert_cells(cells, 33, [
        make_md("### 2-12-[통계] 불만 카테고리 분포 검정"),
        make_code(T2_COMPLAINT_STAT),
    ])

    # 4) After cell 31 (BERTopic group viz) → BERTopic chi-square
    cells = insert_cells(cells, 31, [
        make_md("### 2-11-[통계] BERTopic 토픽 분포 × 도메인 그룹 Chi-square"),
        make_code(T2_BERTOPIC_STAT),
    ])

    # 3) After cell 15 (survey) → survey KW
    cells = insert_cells(cells, 15, [
        make_md("### 2-5-[통계] Survey 만족도 차원별 Kruskal-Wallis"),
        make_code(T2_SURVEY_STAT),
    ])

    # 2) After cell 13 (brand) → brand chi-square
    cells = insert_cells(cells, 13, [
        make_md("### 2-4-[통계] 브랜드 선호도 독립성 검정 — Chi-square"),
        make_code(T2_BRAND_STAT),
    ])

    # 1) After cell 11 (behavior) → behavior stats
    cells = insert_cells(cells, 11, [
        make_md("### 2-3-[통계] 리뷰 행태 지표별 그룹 차이 검정"),
        make_code(T2_BEHAVIOR_STAT),
    ])

    nb['cells'] = cells
    with open('topic2_analysis.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"topic2_analysis.ipynb: {n_before} → {len(cells)} cells (+{len(cells)-n_before})")


def patch_topic3():
    with open('topic3_analysis.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    cells = nb['cells']
    n_before = len(cells)

    # 2) After cell 17 (topics_over_time) → BERTopic×코호트 chi-square
    cells = insert_cells(cells, 17, [
        make_md("### 3-4-[통계] BERTopic 토픽 분포 × 코호트 Chi-square"),
        make_code(T3_BERTOPIC_STAT),
    ])

    # 1) Modify cell 14 (helpfulness) — 기존 코드 뒤에 사후검정 코드 추가
    src_14 = ''.join(cells[14]['source'])
    cells[14]['source'] = lines(src_14+ T3_HELP_POSTHOC)

    nb['cells'] = cells
    with open('topic3_analysis.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"topic3_analysis.ipynb: {n_before} → {len(cells)} cells (+{len(cells)-n_before})")


if __name__ == '__main__':
    patch_topic2()
    patch_topic3()
    print("완료!")
