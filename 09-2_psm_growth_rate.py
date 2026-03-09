"""
===================================================================
09-2_psm_growth_rate.py
視聴施設 vs 未視聴施設: 傾向スコアマッチング(PSM)による伸長率比較 [ver2]
===================================================================
ver1 との違い:
  - 複数医師が所属する施設も含む
  - 分析単位: 施設 (facility_id) レベル
  - 複数施設所属医師は平均納入額最大の施設に割り当て
  - 共変量: 施設属性 + 施設医師数 (個人属性を除く)
===================================================================
"""

import os
import warnings
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.spatial import KDTree

warnings.filterwarnings("ignore")

for _font in ["Yu Gothic", "MS Gothic", "Meiryo", "Hiragino Sans", "IPAexGothic"]:
    try:
        matplotlib.rcParams["font.family"] = _font
        break
    except Exception:
        pass
matplotlib.rcParams["axes.unicode_minus"] = False

# ===================================================================
# パラメータ設定
# ===================================================================
ENT_PRODUCT_CODE = "00001"
ACTIVITY_CHANNEL_FILTER = "Web講演会"
CONTENT_TYPES = ["webiner", "e_contents", "Web講演会"]  # MR活動量計算で除外するデジタル種別

# ===================================================================
# 共変量設定 (ver2: 施設レベル分析)
# -------------------------------------------------------------------
# カテゴリ共変量 (one-hot 化)
COV_CAT_COLS = [
    "UHP区分名称",      # 施設: UHPセグメント (U/H/P/雑)
    "baseline_cat",    # 施設: ベースライン売上カテゴリ
]
# 連続共変量 (z-score 標準化)
# mr_pre / fac_avg_age はデータ確認後に動的追加
COV_CONT_COLS = ["n_docs"]  # 施設あたり医師数

# 医師クインタイル → 施設平均スコア共変量
# Z=0, L=1, M=2, H=3, VH=4 に変換して施設内医師の平均を連続共変量として使用
# カラムが存在しない場合は自動スキップ
QUINTILE_MAP = {"Z": 0, "L": 1, "M": 2, "H": 3, "VH": 4}
DOCTOR_QUINTILE_COLS = [
    "MR_VISIT_F2F_ER_QUINTILE_FINAL",      # MR面談関与度
    "OWNED_MEDIA_ER_QUINTILE_FINAL",       # オウンドメディア関与度
    "MR_VISIT_REMOTE_ER_QUINTILE_FINAL",   # MRリモート訪問関与度
]
_QUINTILE_DISP = {
    "MR_VISIT_F2F_ER_QUINTILE_FINAL":    "MR面談関与度",
    "OWNED_MEDIA_ER_QUINTILE_FINAL":     "オウンドメディア関与度",
    "MR_VISIT_REMOTE_ER_QUINTILE_FINAL": "MRリモート訪問関与度",
}
# ===================================================================

FILE_RW_LIST           = "rw_list.csv"
FILE_SALES             = "sales.csv"
FILE_DIGITAL           = "デジタル視聴データ.csv"
FILE_ACTIVITY          = "活動データ.csv"
FILE_DOCTOR_ATTR       = "doctor_attribute.csv"
FILE_FACILITY_MASTER   = "facility_attribute_修正.csv"
FILE_FAC_DOCTOR_LIST   = "施設医師リスト.csv"

INCLUDE_ONLY_RW     = False# True: RW医師のみ
INCLUDE_ONLY_NON_RW = False  # True: 非RW医師のみ (INCLUDE_ONLY_RW=Falseのとき有効)
EXCLUDE_ZERO_SALES_FACILITIES = False  # True: 全期間納入が0の施設を解析対象から除外
FILTER_SINGLE_FAC_DOCTOR = False  # True: 1施設1医師の施設のみを対象（複数医師施設を除外）
UHP_RANK = {"U": 0, "H": 1, "P": 2, "雑": 3}  # U>H>P>雑 (規模大→小)

# 出力ファイル名サフィックス
if INCLUDE_ONLY_RW:
    _pop_sfx = "_rw"
elif INCLUDE_ONLY_NON_RW:
    _pop_sfx = "_nonrw"
else:
    _pop_sfx = "_all"
if EXCLUDE_ZERO_SALES_FACILITIES:
    _zero_sfx = "_nozero"
else:
    _zero_sfx = ""
if FILTER_SINGLE_FAC_DOCTOR:
    _single_sfx = "_single"
else:
    _single_sfx = ""
_suffix = _pop_sfx + _zero_sfx + _single_sfx

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "本番データ")
_required  = [FILE_SALES, FILE_DIGITAL, FILE_ACTIVITY, FILE_RW_LIST]
if not all(os.path.exists(os.path.join(DATA_DIR, f)) for f in _required):
    _alt = os.path.join(SCRIPT_DIR, "data")
    if all(os.path.exists(os.path.join(_alt, f)) for f in _required):
        DATA_DIR = _alt

START_DATE          = "2023-04-01"
N_MONTHS            = 33
WASHOUT_MONTHS      = 2
LAST_ELIGIBLE_MONTH = 29

PRE_END    = 11   # month_index 0-11 (12 months)
POST_START = 12   # month_index 12-32 (21 months)
BASELINE_START_MONTH_IDX = -12  # baseline_cat 用: 解析開始前12ヶ月 (2022/4 = month_index -12)

CALIPER_MULTIPLIER = 0.2
RANDOM_SEED        = 42
N_BINS_CONTINUOUS  = 4   # 連続変数の自動ビン数（FIXED_BINS 未指定列に適用）

# 医師歴・年齢は 10 年刻みの固定区分（SUBGROUP_SPECS の is_continuous=True 列にも適用）
FIXED_BINS: dict = {
    "医師歴": {"bins": [0, 9, 19, 29, 39, np.inf], "labels": ["1~9年",  "10~19年", "20~29年", "30~39年", "40年以上"]},
    "年齢":   {"bins": [19, 29, 39, 49, 59, np.inf], "labels": ["20~29歳", "30~39歳", "40~49歳", "50~59歳", "60歳以上"]},
    "fac_avg_age": {"bins": [19, 29, 39, 49, 59, np.inf], "labels": ["20代", "30代", "40代", "50代", "60代以上"]},
}

# ===================================================================
# サブグループ分析設定 (ver2: 施設レベルのみ)
#   (表示名, unit_df の列名, is_continuous, 単位文字列)
#   列が存在しない場合は自動スキップ
# ===================================================================
SUBGROUP_SPECS = [
    ("ベースライン納入額",  "baseline_cat",  False, ""),
    ("UHP区分",            "UHP区分名称",    False, ""),
    ("施設医師数区分",     "n_docs_cat",     False, ""),
]

# ===================================================================
# ユーティリティ関数
# ===================================================================

def _fixed_cut(series, bins, labels):
    """固定ビンで pd.cut し (CategoricalSeries, levels_list) を返す。
    データに存在するラベルのみ levels_list に含める。
    """
    result = pd.cut(series, bins=bins, labels=labels)
    present = [l for l in labels if (result == l).any()]
    return result, present


def _infer_unit(col_name, unit_override=""):
    if unit_override:
        return unit_override
    if col_name in ("年齢", "卒業時年齢") or "歳" in col_name:
        return "歳"
    if col_name in ("医師歴",) or "歴" in col_name:
        return "年"
    if "床" in col_name:
        return "床"
    return ""


# カテゴリ固定表示順序
_QUINTILE_ORDER   = ["Z", "L", "M", "H", "VH", "不明"]
_BASELINE_ORDER   = ["0以下", "低", "中", "高"]
_UHP_ORDER        = ["U", "H", "P", "雑", "不明"]  # U>H>P>雑 (規模大→小)
_N_DOCS_CAT_ORDER = ["1名", "2〜3名", "4名以上"]
_COVERAGE_ORDER   = ["低Coverage", "中Coverage", "高Coverage"]


def _sort_levels(levels):
    """カテゴリレベルを固定順序で並べる。
    既知のカテゴリセット (Quintile/baseline/UHP/医師数/Coverage) は固定順、
    それ以外は文字列ソート。
    """
    str_levels = [str(v) for v in levels]
    non_missing = [v for v in str_levels if v not in ("nan", "None", "不明")]

    # QUINTILE (H/L/M/VH/Z)
    if non_missing and all(v in ("H", "L", "M", "VH", "Z") for v in non_missing):
        return [v for v in _QUINTILE_ORDER if v in str_levels]

    # ベースライン納入額カテゴリ
    if non_missing and all(v in set(_BASELINE_ORDER) for v in non_missing):
        return [v for v in _BASELINE_ORDER if v in str_levels]

    # UHP区分名称 (U/H/P/雑)
    if non_missing and all(v in {"U", "H", "P", "雑"} for v in non_missing):
        return [v for v in _UHP_ORDER if v in str_levels]

    # 施設医師数カテゴリ
    if non_missing and all(v in set(_N_DOCS_CAT_ORDER) for v in non_missing):
        return [v for v in _N_DOCS_CAT_ORDER if v in str_levels]

    # Coverageカテゴリ
    if non_missing and all(v in set(_COVERAGE_ORDER) for v in non_missing):
        return [v for v in _COVERAGE_ORDER if v in str_levels]

    return sorted(levels, key=str)  # 元の型を保持（int の 0/1 フラグも崩さない）


def _auto_range_labels(series, q=4, col_name="", unit_override=""):
    """連続変数を q 分位カテゴリ化し '最小~最大単位' 形式ラベルを返す。"""
    unit = _infer_unit(col_name, unit_override)
    s = series.dropna()
    if len(s) == 0:
        return pd.Series([pd.NA] * len(series), index=series.index), []

    actual_q = min(q, s.nunique())
    if actual_q == 1:
        label = f"{int(round(s.iloc[0]))}{unit}"
        return pd.Categorical(series.where(series.isna(), label),
                              categories=[label]), [label]
    try:
        _, bins = pd.qcut(s, q=actual_q, retbins=True, duplicates="drop")
        labels = [f"{int(round(bins[i]))}~{int(round(bins[i+1]))}{unit}"
                  for i in range(len(bins) - 1)]
        bins_cut = bins.copy()
        bins_cut[0] -= 0.001
        result = pd.cut(series, bins=bins_cut, labels=labels)
        return result, list(labels)
    except Exception:
        med = s.median()
        lo = f"{int(round(s.min()))}~{int(round(med))}{unit}"
        hi = f"{int(round(med))+1}~{int(round(s.max()))}{unit}"
        result = pd.cut(series, bins=[s.min() - 0.001, med, s.max()],
                        labels=[lo, hi])
        return result, [lo, hi]


def _baseline_4cat(series):
    """ベースライン納入額を 0以下/低/中/高 の4カテゴリに分類。"""
    positive = series[series > 0]
    if len(positive) == 0:
        cat = pd.Categorical(["0以下"] * len(series), categories=["0以下"])
        return pd.Series(cat, index=series.index), ["0以下"]
    q33 = positive.quantile(1 / 3)
    q67 = positive.quantile(2 / 3)
    if q33 == q67:
        bins, levels = [-np.inf, 0, q67, np.inf], ["0以下", "低", "高"]
    else:
        bins, levels = [-np.inf, 0, q33, q67, np.inf], ["0以下", "低", "中", "高"]
    result = pd.cut(series, bins=bins, labels=levels, include_lowest=True)
    return result, levels


def logit_ps(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def psm_1to1(treated_df, control_df, caliper):
    """1:1 最近傍マッチング（キャリパー付き、非復元）"""
    ctrl_arr = control_df["logit_ps"].values.reshape(-1, 1)
    tree = KDTree(ctrl_arr)
    np.random.seed(RANDOM_SEED)
    order = np.random.permutation(len(treated_df))
    used_ctrl = set()
    pairs = []
    for i in order:
        row   = treated_df.iloc[i]
        t_lps = row["logit_ps"]
        dists, idxs = tree.query([[t_lps]], k=len(control_df))
        for dist, ci in zip(dists[0], idxs[0]):
            if caliper is not None and dist > caliper:
                break
            c_doc = control_df.iloc[ci]["doctor_id"]
            if c_doc not in used_ctrl:
                pairs.append((row["doctor_id"], c_doc))
                used_ctrl.add(c_doc)
                break
    return pairs


def run_subgroup_psm(sg_data, caliper, global_caliper):
    """サブグループ内で 1:1 PSM を実施し ATT を返す。
    caliper 内でマッチ不成立なら caliper 緩和 → None（最近傍）の順でリトライ。
    """
    sg_t = sg_data[sg_data["treated"] == 1][["doctor_id", "logit_ps"]].reset_index(drop=True)
    sg_c = sg_data[sg_data["treated"] == 0][["doctor_id", "logit_ps"]].reset_index(drop=True)

    if len(sg_t) < 3 or len(sg_c) < 3:
        return None, len(sg_t), len(sg_c)

    pairs = psm_1to1(sg_t, sg_c, caliper)
    if len(pairs) < 2 and caliper is not None:
        pairs = psm_1to1(sg_t, sg_c, global_caliper)  # 緩和
    if len(pairs) < 2:
        pairs = psm_1to1(sg_t, sg_c, None)            # 最近傍のみ

    t_ids = [p[0] for p in pairs]
    c_ids = [p[1] for p in pairs]

    def get_growth(doc_id):
        row = sg_data[sg_data["doctor_id"] == doc_id]["growth"].values
        return row[0] if len(row) > 0 else np.nan

    diffs = np.array([get_growth(t) - get_growth(c) for t, c in zip(t_ids, c_ids)])
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) < 2:
        return None, len(sg_t), len(sg_c)

    att    = diffs.mean()
    se_att = diffs.std() / np.sqrt(len(diffs))
    t_stat = att / (se_att + 1e-12)
    p_val  = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(diffs) - 1))
    return {
        "att": att, "se": se_att, "t": t_stat, "p": p_val,
        "ci_lo": att - 1.96 * se_att, "ci_hi": att + 1.96 * se_att,
        "n_matched": len(diffs),
        "n_treated_raw": len(sg_t), "n_control_raw": len(sg_c),
    }, len(sg_t), len(sg_c)


def _run_ch_psm(ch_treat_ids, ch_ctrl_ids, base_df):
    """チャネル別 treated フラグで PS推定 → 1:1マッチング → ATT を返す。
    base_df: unit_df と同じ構造（pre_mean/growth/facility_id/全共変量）
    """
    ch_all = ch_treat_ids | ch_ctrl_ids
    df = base_df[base_df["facility_id"].isin(ch_all)].copy()
    df["treated"] = df["facility_id"].isin(ch_treat_ids).astype(int)
    df["doctor_id"] = df["facility_id"]
    n_t = int(df["treated"].sum())
    n_c = int((df["treated"] == 0).sum())
    if n_t < 3 or n_c < 3:
        return None, n_t, n_c

    _drop = df.dropna(subset=[c for c in COV_CAT_COLS if c in df.columns] + ["pre_mean"]).copy()
    _dm_cols = [c for c in COV_CAT_COLS if c in _drop.columns]
    _dm = pd.get_dummies(_drop, columns=_dm_cols, drop_first=True)
    _dm["pre_mean_std"] = (_dm["pre_mean"] - _dm["pre_mean"].mean()) / (_dm["pre_mean"].std() + 1e-9)
    _cs = []
    for _cc in COV_CONT_COLS:
        _sn = _cc + "_std"
        if _cc in _dm.columns:
            _m, _s = _dm[_cc].mean(), _dm[_cc].std() + 1e-9
            _dm[_sn] = (_dm[_cc].fillna(_m) - _m) / _s
        else:
            _dm[_sn] = 0.0
        _cs.append(_sn)
    _pfx = tuple(c + "_" for c in COV_CAT_COLS)
    _xcols = [c for c in _dm.columns if c.startswith(_pfx)] + ["pre_mean_std"] + _cs
    _y = _dm["treated"].astype(float).values
    _X = _dm[_xcols].astype(float).values

    _ps = None
    for _C in [0.1, 0.05, 0.02]:
        try:
            _lr = LogisticRegression(C=_C, max_iter=2000, random_state=RANDOM_SEED,
                                     solver="lbfgs", class_weight="balanced")
            _lr.fit(_X, _y)
            _pc = _lr.predict_proba(_X)[:, 1]
            if _pc[_y == 1].min() < _pc[_y == 0].max() and _pc[_y == 0].min() < _pc[_y == 1].max():
                _ps = _pc
                break
        except Exception:
            continue
    if _ps is None:
        try:
            _lr2 = LogisticRegression(C=0.5, max_iter=1000, random_state=RANDOM_SEED)
            _lr2.fit(_dm[["pre_mean_std"]].values, _y)
            _ps = _lr2.predict_proba(_dm[["pre_mean_std"]].values)[:, 1]
        except Exception:
            _ps = np.full(len(_dm), 0.5)

    _drop = _drop.copy()
    _drop["ps"] = _ps
    _drop["logit_ps"] = logit_ps(_ps)
    _drop["doctor_id"] = _drop["facility_id"]
    _sd = _drop["logit_ps"].std()
    _cap = CALIPER_MULTIPLIER * _sd if _sd >= 1e-6 else None
    _t_df = _drop[_drop["treated"] == 1][["doctor_id", "logit_ps"]].reset_index(drop=True)
    _c_df = _drop[_drop["treated"] == 0][["doctor_id", "logit_ps"]].reset_index(drop=True)
    _pairs = psm_1to1(_t_df, _c_df, _cap)
    if _cap is not None and len(_pairs) < len(_t_df) * 0.3:
        _pairs = psm_1to1(_t_df, _c_df, _sd)
    if len(_pairs) < 2:
        _pairs = psm_1to1(_t_df, _c_df, None)
    if len(_pairs) < 2:
        return None, n_t, n_c

    _gmap = dict(zip(_drop["facility_id"], _drop["growth"]))
    _tg = np.array([_gmap.get(t, np.nan) for t, _ in _pairs])
    _cg = np.array([_gmap.get(c, np.nan) for _, c in _pairs])
    _vm = ~(np.isnan(_tg) | np.isnan(_cg))
    _d = _tg[_vm] - _cg[_vm]
    if len(_d) < 2:
        return None, n_t, n_c

    _a = _d.mean()
    _se = _d.std() / np.sqrt(len(_d))
    _tv = _a / (_se + 1e-12)
    _pv = 2 * (1 - stats.t.cdf(abs(_tv), df=len(_d) - 1))
    return {
        "att": float(_a), "se": float(_se), "t_stat": float(_tv), "p_value": float(_pv),
        "ci_95_lower": float(_a - 1.96 * _se), "ci_95_upper": float(_a + 1.96 * _se),
        "n_matched": int(len(_d)), "n_treated_raw": n_t, "n_control_raw": n_c,
        "treated_mean_growth": float(np.nanmean(_tg[_vm])),
        "control_mean_growth": float(np.nanmean(_cg[_vm])),
    }, n_t, n_c


# ===================================================================
print("=" * 70)
print(" 09-2: PSM による伸長率比較（視聴施設 vs 未視聴施設）[ver2]")
print("       分析単位: 施設 / サブグループ: " + str(len(SUBGROUP_SPECS)) + " 次元")
print("=" * 70)

# ===================================================================
# [1] データ読み込み
# ===================================================================
print("\n[1] データ読み込み")

rw_list    = pd.read_csv(os.path.join(DATA_DIR, FILE_RW_LIST))

sales_raw  = pd.read_csv(os.path.join(DATA_DIR, FILE_SALES), dtype=str)
sales_raw["実績"]  = pd.to_numeric(sales_raw["実績"], errors="coerce").fillna(0)
daily = sales_raw[sales_raw["品目コード"].str.strip() == ENT_PRODUCT_CODE].copy()
daily["日付"]  = pd.to_datetime(daily["日付"], format="mixed")
daily = daily.rename(columns={
    "日付": "delivery_date",
    "施設（本院に合算）コード": "facility_id",
    "実績": "amount",
})
daily["facility_id"] = daily["facility_id"].astype(str).str.strip()
daily["month_index"] = (
    (daily["delivery_date"].dt.year - 2023) * 12
    + daily["delivery_date"].dt.month - 4
)

digital_raw = pd.read_csv(os.path.join(DATA_DIR, FILE_DIGITAL))
digital_raw["品目コード"] = digital_raw["品目コード"].astype(str).str.strip().str.zfill(5)
digital = digital_raw[digital_raw["品目コード"] == ENT_PRODUCT_CODE].copy()
digital = digital[digital["fac_honin"].notna() & (digital["fac_honin"].astype(str).str.strip() != "")].copy()

activity_raw = pd.read_csv(os.path.join(DATA_DIR, FILE_ACTIVITY))
activity_raw["品目コード"] = activity_raw["品目コード"].astype(str).str.strip().str.zfill(5)
web_lecture = activity_raw[
    (activity_raw["品目コード"] == ENT_PRODUCT_CODE)
    & (activity_raw["活動種別"] == ACTIVITY_CHANNEL_FILTER)
].copy()
web_lecture = web_lecture[web_lecture["fac_honin"].notna() & (web_lecture["fac_honin"].astype(str).str.strip() != "")].copy()

# MR前処置期間活動量: 非デジタル活動を施設別に集計 (washout期間内の月平均)
_mr_raw = activity_raw[
    (activity_raw["品目コード"] == ENT_PRODUCT_CODE)
    & (~activity_raw["活動種別"].isin(CONTENT_TYPES))
    & activity_raw["fac_honin"].notna()
].copy()
if len(_mr_raw) > 0:
    _mr_raw["活動日_dt"] = pd.to_datetime(_mr_raw["活動日_dt"], format="mixed")
    _mr_raw["month_index"] = (
        (_mr_raw["活動日_dt"].dt.year - 2023) * 12
        + _mr_raw["活動日_dt"].dt.month - 4
    )
    mr_pre_fac = (
        _mr_raw[_mr_raw["month_index"] < WASHOUT_MONTHS]
        .groupby("fac_honin").size()
        .reset_index(name="mr_total")
    )
    mr_pre_fac["mr_pre"] = mr_pre_fac["mr_total"] / max(WASHOUT_MONTHS, 1)
    mr_pre_fac = mr_pre_fac.rename(columns={"fac_honin": "facility_id"})[["facility_id", "mr_pre"]]
else:
    mr_pre_fac = pd.DataFrame(columns=["facility_id", "mr_pre"])

common_cols = ["活動日_dt", "品目コード", "活動種別", "fac_honin", "doc"]
viewing = pd.concat([digital[common_cols], web_lecture[common_cols]], ignore_index=True)
viewing = viewing.rename(columns={
    "活動日_dt": "view_date", "fac_honin": "facility_id", "doc": "doctor_id"
})
viewing["view_date"] = pd.to_datetime(viewing["view_date"], format="mixed")
viewing["month_index"] = (
    (viewing["view_date"].dt.year - 2023) * 12
    + viewing["view_date"].dt.month - 4
)
viewing = viewing.rename(columns={"doctor_id": "doc"})

months = pd.date_range(start=START_DATE, periods=N_MONTHS, freq="MS")
print("  データ読み込み完了")

# ================================================================
# Part 2: 解析集団の絞り込み (ver2: 複数医師施設対応)
# ================================================================
print("\n[2] 除外フロー (ver2)")

fac_doc_list = pd.read_csv(os.path.join(DATA_DIR, FILE_FAC_DOCTOR_LIST))
fac_doc_list["fac_honin"] = fac_doc_list["fac_honin"].astype(str).str.strip()
fac_df       = pd.read_csv(os.path.join(DATA_DIR, FILE_FACILITY_MASTER))
fac_df["fac_honin"] = fac_df["fac_honin"].astype(str).str.strip()
doc_attr_df  = pd.read_csv(os.path.join(DATA_DIR, FILE_DOCTOR_ATTR))

_doc_to_fac   = dict(zip(fac_doc_list["doc"], fac_doc_list["fac"]))
_doc_to_honin = dict(zip(fac_doc_list["doc"], fac_doc_list["fac_honin"]))
all_docs = set(fac_doc_list["doc"])

# 主施設割り当て (最適化版): 1施設所属は直接割り当て、複数施設所属のみ売上ベース
_doc_fac_list = fac_doc_list[["doc", "fac_honin"]].drop_duplicates()
_doc_fac_count = _doc_fac_list.groupby("doc")["fac_honin"].nunique()
_single_fac_docs = set(_doc_fac_count[_doc_fac_count == 1].index)
_multi_fac_docs  = set(_doc_fac_count[_doc_fac_count >  1].index)
print(f"  1施設所属: {len(_single_fac_docs)}名, 複数施設所属: {len(_multi_fac_docs)}名")

_single_assign = (
    _doc_fac_list[_doc_fac_list["doc"].isin(_single_fac_docs)]
    .drop_duplicates("doc")
    .set_index("doc")["fac_honin"]
)

_doc_fac_list_multi = _doc_fac_list[_doc_fac_list["doc"].isin(_multi_fac_docs)]
_sales_by_fac = (
    daily.groupby("facility_id")["amount"].mean()
    .reset_index().rename(columns={"facility_id": "fac_honin", "amount": "avg_sales"})
)
_doc_fac_sales = _doc_fac_list_multi.merge(_sales_by_fac, on="fac_honin", how="left")
_doc_fac_sales["avg_sales"] = _doc_fac_sales["avg_sales"].fillna(0)

_multi_assign = (
    _doc_fac_sales.sort_values("avg_sales", ascending=False)
    .groupby("doc")["fac_honin"].first()
)

_fac_uhp = fac_df.drop_duplicates("fac_honin").set_index("fac_honin")["UHP区分名称"] \
    if "UHP区分名称" in fac_df.columns else pd.Series(dtype=str)
_zero_sum = _doc_fac_sales.groupby("doc")["avg_sales"].sum()
_zero_docs_set = set(_zero_sum[_zero_sum == 0].index)
if _zero_docs_set:
    _zero_df = _doc_fac_sales[_doc_fac_sales["doc"].isin(_zero_docs_set)].copy()
    _zero_df["_uhp_rank"] = _zero_df["fac_honin"].map(_fac_uhp).map(
        lambda x: UHP_RANK.get(str(x), 99) if pd.notna(x) else 99
    )
    _zero_best = _zero_df.sort_values("_uhp_rank").groupby("doc")["fac_honin"].first()
    _multi_assign.update(_zero_best)

_doc_primary_all = pd.concat([_single_assign, _multi_assign])
doc_primary_fac = _doc_primary_all  # doc → fac_honin

# 医師フィルタ
rw_doc_ids = set(rw_list["doc"])
if INCLUDE_ONLY_RW:
    analysis_docs_all = all_docs & rw_doc_ids
    print(f"  [Step 3] RWフィルタ適用: {len(analysis_docs_all)} 名")
elif INCLUDE_ONLY_NON_RW:
    analysis_docs_all = all_docs - rw_doc_ids
    print(f"  [Step 3] 非RWフィルタ適用: {len(analysis_docs_all)} 名")
else:
    analysis_docs_all = all_docs
    print(f"  [Step 3] スキップ (全医師): {len(analysis_docs_all)} 名")

# 施設→医師リスト (1:N)
_prim_filt = doc_primary_fac[doc_primary_fac.index.isin(analysis_docs_all)]
_prim_df = pd.DataFrame({"doc": _prim_filt.index, "fac": _prim_filt.values})
fac_to_docs = _prim_df.groupby("fac")["doc"].agg(list).to_dict()

n_docs_map = {fac: len(docs) for fac, docs in fac_to_docs.items()}

# 全期間納入0施設の除外（フラグで制御）
if EXCLUDE_ZERO_SALES_FACILITIES:
    _fac_total_sales = daily.groupby("facility_id")["amount"].sum()
    _zero_sale_facs = set(_fac_total_sales[_fac_total_sales <= 0].index)
    _no_sale_facs   = {fac for fac in fac_to_docs if fac not in _fac_total_sales.index}
    _exclude_zero   = _zero_sale_facs | _no_sale_facs
    fac_to_docs = {fac: docs for fac, docs in fac_to_docs.items()
                   if fac not in _exclude_zero}
    n_docs_map  = {fac: len(docs) for fac, docs in fac_to_docs.items()}
    print(f"  [全期間0売上除外] {len(_exclude_zero)} 施設を除外 → 残 {len(fac_to_docs)} 施設")

if FILTER_SINGLE_FAC_DOCTOR:
    fac_to_docs = {fac: docs for fac, docs in fac_to_docs.items() if len(docs) == 1}
    n_docs_map  = {fac: len(docs) for fac, docs in fac_to_docs.items()}
    print(f"  [1施設1医師フィルタ] 複数医師施設を除外 → 残 {len(fac_to_docs)} 施設")

# 施設平均年齢計算 (fac_to_docs確定後)
_fac_avg_age: dict = {}
if "年齢" in doc_attr_df.columns:
    for _fac, _docs in fac_to_docs.items():
        _ages = doc_attr_df[doc_attr_df["doc"].isin(_docs)]["年齢"].dropna()
        _fac_avg_age[_fac] = float(_ages.mean()) if len(_ages) > 0 else float("nan")
    _n_age_valid = sum(1 for v in _fac_avg_age.values() if v == v)
    print(f"  [fac_avg_age] 施設平均年齢算出: 非欠損施設 {_n_age_valid}/{len(fac_to_docs)}")

# --- DOCTOR_SEGMENT 施設割り当て（全体構成比から最大正乖離セグメント）---
_fac_doctor_segment: dict = {}
if "DOCTOR_SEGMENT" in doc_attr_df.columns:
    _seg_all = doc_attr_df[doc_attr_df["doc"].isin(analysis_docs_all)][["doc", "DOCTOR_SEGMENT"]].dropna()
    _seg_overall_prop = _seg_all["DOCTOR_SEGMENT"].value_counts(normalize=True)
    for _fac, _docs in fac_to_docs.items():
        _fac_seg_prop = _seg_all[_seg_all["doc"].isin(_docs)]["DOCTOR_SEGMENT"].value_counts(normalize=True)
        _dev = {seg: _fac_seg_prop.get(seg, 0.0) - _seg_overall_prop.get(seg, 0.0)
                for seg in _seg_overall_prop.index}
        _fac_doctor_segment[_fac] = max(_dev, key=_dev.get)
    print(f"  [DOCTOR_SEGMENT] {len(_fac_doctor_segment)} 施設割り当て完了")

# --- DIGITAL_CHANNEL_PREFERENCE 施設割り当て（全体構成比から最大正乖離側）---
_fac_digital_pref: dict = {}
if "DIGITAL_CHANNEL_PREFERENCE" in doc_attr_df.columns:
    _dcp_all = doc_attr_df[doc_attr_df["doc"].isin(analysis_docs_all)][["doc", "DIGITAL_CHANNEL_PREFERENCE"]].dropna()
    _dcp_overall_prop = _dcp_all["DIGITAL_CHANNEL_PREFERENCE"].value_counts(normalize=True)
    for _fac, _docs in fac_to_docs.items():
        _fac_dcp_prop = _dcp_all[_dcp_all["doc"].isin(_docs)]["DIGITAL_CHANNEL_PREFERENCE"].value_counts(normalize=True)
        _dev = {seg: _fac_dcp_prop.get(seg, 0.0) - _dcp_overall_prop.get(seg, 0.0)
                for seg in _dcp_overall_prop.index}
        _fac_digital_pref[_fac] = max(_dev, key=_dev.get)
    print(f"  [DIGITAL_CHANNEL_PREFERENCE] {len(_fac_digital_pref)} 施設割り当て完了")

# 医師クインタイル施設平均スコア計算 (fac_to_docs確定後)
_fac_quintile_means: dict = {}
for _qcol in DOCTOR_QUINTILE_COLS:
    if _qcol not in doc_attr_df.columns:
        print(f"  [{_qcol}] 列が doctor_attribute.csv に存在しない → スキップ")
        continue
    _qnum = doc_attr_df[["doc", _qcol]].copy()
    _n_raw_notna = _qnum[_qcol].notna().sum()
    _qnum[_qcol + "_num"] = _qnum[_qcol].map(QUINTILE_MAP)
    _n_mapped = _qnum[_qcol + "_num"].notna().sum()
    _fac_q: dict = {}
    for _fac, _docs in fac_to_docs.items():
        _vals = _qnum[_qnum["doc"].isin(_docs)][_qcol + "_num"].dropna()
        _fac_q[_fac] = float(_vals.mean()) if len(_vals) > 0 else float("nan")
    _fac_quintile_means[_qcol] = _fac_q
    _n_valid = sum(1 for v in _fac_q.values() if v == v)
    print(f"  [{_qcol}] 列あり: 医師{_n_raw_notna}名非欠損, マップ成功{_n_mapped}名"
          f" → 施設平均スコア: 非欠損施設 {_n_valid}/{len(fac_to_docs)}")

# 視聴データに主施設ID付与 (解析対象医師のみ)
viewing_all = viewing[viewing["doc"].isin(analysis_docs_all)].copy()
viewing_all["facility_id"] = viewing_all["doc"].map(doc_primary_fac)

# ウォッシュアウト除外
washout_fac_ids = set(
    viewing_all[
        (viewing_all["month_index"] < WASHOUT_MONTHS) &
        viewing_all["facility_id"].notna()
    ]["facility_id"]
)
washout_viewers = washout_fac_ids  # 表示用

# 初回視聴月（施設レベル）
_first_view = (
    viewing_all[
        (viewing_all["month_index"] >= WASHOUT_MONTHS) &
        (viewing_all["month_index"] <= LAST_ELIGIBLE_MONTH) &
        viewing_all["facility_id"].notna() &
        ~viewing_all["facility_id"].isin(washout_fac_ids)
    ]
    .groupby("facility_id")["month_index"].min()
    .rename("first_view_month")
)

treated_fac_ids = set(_first_view.index)
control_fac_ids = set(fac_to_docs.keys()) - treated_fac_ids - washout_fac_ids
analysis_fac_ids = treated_fac_ids | control_fac_ids
analysis_doc_ids = set()  # not used in ver2

print(f"  ウォッシュアウト除外（視聴あり施設）: {len(washout_fac_ids)}")
print(f"  処置群（視聴あり）: {len(treated_fac_ids)}  対照群（未視聴）: {len(control_fac_ids)}")

# ===================================================================
# [3] 前後期売上集計 (施設レベル)
# ===================================================================
print("\n[3] 前後期売上集計 (施設レベル)")

daily_target = daily[daily["facility_id"].isin(analysis_fac_ids)].copy()
monthly = daily_target.groupby(["facility_id", "month_index"])["amount"].sum().reset_index()

full_idx = pd.MultiIndex.from_product(
    [sorted(analysis_fac_ids), list(range(N_MONTHS))],
    names=["facility_id", "month_index"]
)
panel = (
    monthly.set_index(["facility_id", "month_index"])
    .reindex(full_idx, fill_value=0).reset_index()
)

pre_avg  = (panel[panel["month_index"] <= PRE_END]
            .groupby("facility_id")["amount"].mean().rename("pre_mean"))
post_avg = (panel[panel["month_index"] >= POST_START]
            .groupby("facility_id")["amount"].mean().rename("post_mean"))

unit_df = pd.DataFrame({
    "facility_id": sorted(analysis_fac_ids)
}).set_index("facility_id").join(pre_avg).join(post_avg).reset_index()
unit_df["growth"]  = unit_df["post_mean"] - unit_df["pre_mean"]
unit_df["growth_rate"] = np.where(
    unit_df["pre_mean"] > 0,
    (unit_df["post_mean"] - unit_df["pre_mean"]) / unit_df["pre_mean"] * 100,
    np.nan
)
unit_df["treated"] = unit_df["facility_id"].isin(treated_fac_ids).astype(int)
unit_df["doctor_id"] = unit_df["facility_id"]  # alias for compatibility

# ===================================================================
# [4] 属性マージ (施設レベル)
# ===================================================================
print("\n[4] 属性マージ")

# 施設属性
fac_df2 = fac_df.rename(columns={"fac_honin": "facility_id"})
fac_cols = [c for c in fac_df2.columns if c not in {"facility_id", "fac", "fac_honin_name"}]
unit_df = unit_df.merge(fac_df2[["facility_id"] + fac_cols], on="facility_id", how="left")

# 施設医師数
unit_df["n_docs"] = unit_df["facility_id"].map(n_docs_map).fillna(1).astype(int)
print(f"  n_docs: 平均={unit_df['n_docs'].mean():.1f}, 最大={unit_df['n_docs'].max()}")

# 医師クインタイル施設平均スコア (連続共変量)
for _qcol in DOCTOR_QUINTILE_COLS:
    if _qcol in _fac_quintile_means:
        _cname = _qcol + "_mean"
        unit_df[_cname] = unit_df["facility_id"].map(_fac_quintile_means[_qcol])
        _gmean = unit_df[_cname].mean()
        unit_df[_cname] = unit_df[_cname].fillna(_gmean if _gmean == _gmean else 2.0)
        if _cname not in COV_CONT_COLS:
            COV_CONT_COLS.append(_cname)
        print(f"  {_cname}: 平均={unit_df[_cname].mean():.2f}, 欠損→平均補完")

# 施設医師数区分 (n_docs_cat)
_ndocs_cat = pd.cut(
    unit_df["n_docs"],
    bins=[0, 1, 3, np.inf],
    labels=["1名", "2〜3名", "4名以上"]
)
unit_df["n_docs_cat"] = _ndocs_cat

# ベースライン納入額カテゴリ: 施設レベル集計
_bline_fac = (
    daily_target[daily_target["month_index"].isin(range(BASELINE_START_MONTH_IDX, WASHOUT_MONTHS))]
    .groupby("facility_id")["amount"].mean().reset_index()
    .rename(columns={"amount": "baseline_mean"})
)
unit_df = unit_df.merge(_bline_fac, on="facility_id", how="left")
unit_df["baseline_mean"] = unit_df["baseline_mean"].fillna(0.0)
_bc_result, _bc_levels = _baseline_4cat(unit_df["baseline_mean"])
unit_df["baseline_cat"] = _bc_result
_n_bline_months = WASHOUT_MONTHS - BASELINE_START_MONTH_IDX
print(f"  baseline_cat: 前処置期間{_n_bline_months}ヶ月平均から4カテゴリ → " + str(_bc_levels))

# MR前処置活動量をマージ
if len(mr_pre_fac) > 0:
    unit_df = unit_df.merge(mr_pre_fac, on="facility_id", how="left")
    unit_df["mr_pre"] = unit_df["mr_pre"].fillna(0)
else:
    unit_df["mr_pre"] = 0.0
print(f"  mr_pre: WO期間施設別月平均MR活動 (非ゼロ: {unit_df['mr_pre'].gt(0).sum()}件)")
if "mr_pre" not in COV_CONT_COLS:
    COV_CONT_COLS.append("mr_pre")

# 施設平均年齢をマージ
if _fac_avg_age:
    unit_df["fac_avg_age"] = unit_df["facility_id"].map(_fac_avg_age)
    _gmean_age = unit_df["fac_avg_age"].mean()
    unit_df["fac_avg_age"] = unit_df["fac_avg_age"].fillna(_gmean_age if _gmean_age == _gmean_age else 45.0)
    if "fac_avg_age" not in COV_CONT_COLS:
        COV_CONT_COLS.append("fac_avg_age")
    print(f"  fac_avg_age: 平均={unit_df['fac_avg_age'].mean():.1f}歳, 欠損→平均補完")

# 新規施設レベル変数をunit_dfにマージ
if _fac_doctor_segment:
    unit_df["fac_doctor_segment"] = unit_df["facility_id"].map(_fac_doctor_segment)
    print(f"  fac_doctor_segment: {unit_df['fac_doctor_segment'].value_counts().to_dict()}")
if _fac_digital_pref:
    unit_df["fac_digital_pref"] = unit_df["facility_id"].map(_fac_digital_pref)
    print(f"  fac_digital_pref: {unit_df['fac_digital_pref'].value_counts().to_dict()}")
# 医師クインタイル施設平均スコア → 低/中/高 3分位 → SUBGROUP_SPECS
for _qcol in DOCTOR_QUINTILE_COLS:
    _cname = _qcol + "_mean"
    _disp = _QUINTILE_DISP.get(_qcol, _qcol)
    if _cname not in unit_df.columns:
        print(f"  [{_disp}] unit_df に {_cname} 列なし → スキップ")
        continue
    _valid = unit_df[_cname].dropna()
    _n_unique = _valid.nunique()
    if len(_valid) < 3:
        print(f"  [{_disp}] 有効施設数不足 ({len(_valid)}) → スキップ")
        continue
    if _n_unique < 2:
        print(f"  [{_disp}] ユニーク値 {_n_unique}個（分散なし） → スキップ")
        continue
    _cat_col = _qcol + "_cat"
    try:
        _qcat = pd.qcut(
            unit_df[_cname].fillna(_valid.mean()),
            q=3, labels=["低", "中", "高"], duplicates="drop"
        )
        unit_df[_cat_col] = _qcat.astype(str)
        _qlevels = [l for l in ["低", "中", "高"] if (unit_df[_cat_col] == l).any()]
        if len(_qlevels) < 2:
            print(f"  [{_disp}] qcut後のカテゴリ数が{len(_qlevels)}個 → スキップ")
            continue
        SUBGROUP_SPECS.append((_disp, _cat_col, False, ""))
        print(f"  [{_disp}] {_cat_col}: {unit_df[_cat_col].value_counts().to_dict()}")
    except Exception as _qe:
        print(f"  [{_disp}] qcut失敗: {_qe}")

# SUBGROUP_SPECS を動的拡張（列が存在する場合のみ）
if "fac_avg_age" in unit_df.columns:
    SUBGROUP_SPECS.append(("施設平均年齢", "fac_avg_age", True, "歳"))
if "fac_doctor_segment" in unit_df.columns:
    SUBGROUP_SPECS.append(("医師セグメント", "fac_doctor_segment", False, ""))
if "fac_digital_pref" in unit_df.columns:
    SUBGROUP_SPECS.append(("デジタルch嗜好", "fac_digital_pref", False, ""))
print(f"  SUBGROUP_SPECS 計 {len(SUBGROUP_SPECS)} 次元")

# 解析期間（WASHOUT_MONTHS〜LAST_ELIGIBLE_MONTH）での最終Coverage（処置群の累積視聴率）
_post_view = (
    viewing_all[
        viewing_all["month_index"].between(WASHOUT_MONTHS, LAST_ELIGIBLE_MONTH) &
        viewing_all["facility_id"].isin(treated_fac_ids)
    ]
    .groupby("facility_id")["doc"].nunique()
    .rename("n_viewed_docs_post")
)
_total_docs_ser = pd.Series({fac: len(docs) for fac, docs in fac_to_docs.items()})
_coverage_ser = (
    _post_view / _total_docs_ser
).clip(0, 1).rename("final_coverage")
unit_df["final_coverage"] = unit_df["facility_id"].map(_coverage_ser).fillna(0.0)
print(f"  final_coverage: 処置群 mean={unit_df.loc[unit_df['treated']==1, 'final_coverage'].mean():.3f}")

# ===================================================================
# [5] 傾向スコア推定
# ===================================================================
print("\n[5] 傾向スコア推定（Logistic Regression + L2正則化）")

from sklearn.linear_model import LogisticRegression

ps_data = unit_df.dropna(
    subset=[c for c in COV_CAT_COLS if c in unit_df.columns] + ["pre_mean"]
).copy()

_dummy_cols = [c for c in COV_CAT_COLS if c in ps_data.columns]
ps_dummies = pd.get_dummies(
    ps_data,
    columns=_dummy_cols,
    drop_first=True,
)
ps_dummies["pre_mean_std"] = (
    (ps_dummies["pre_mean"] - ps_dummies["pre_mean"].mean())
    / (ps_dummies["pre_mean"].std() + 1e-9)
)
# COV_CONT_COLS の標準化 (NaN は平均で補完)
_cont_std_cols = []
for _cont in COV_CONT_COLS:
    _std_name = _cont + "_std"
    if _cont in ps_dummies.columns:
        _m = ps_dummies[_cont].mean()
        _s = ps_dummies[_cont].std() + 1e-9
        ps_dummies[_std_name] = (ps_dummies[_cont].fillna(_m) - _m) / _s
    else:
        ps_dummies[_std_name] = 0.0
    _cont_std_cols.append(_std_name)

_cat_prefixes = tuple(col + "_" for col in COV_CAT_COLS)
cov_cols = [c for c in ps_dummies.columns
            if c.startswith(_cat_prefixes)
            ] + ["pre_mean_std"] + _cont_std_cols

y_ps = ps_dummies["treated"].astype(float).values
X_ps = ps_dummies[cov_cols].astype(float).values

ps_estimated = False
for C_val in [0.1, 0.05, 0.02]:
    try:
        lr = LogisticRegression(C=C_val, max_iter=2000, random_state=RANDOM_SEED,
                                solver="lbfgs", class_weight="balanced")
        lr.fit(X_ps, y_ps)
        ps_candidate = lr.predict_proba(X_ps)[:, 1]
        t_mask, c_mask = y_ps == 1, y_ps == 0
        if (ps_candidate[t_mask].min() < ps_candidate[c_mask].max() and
                ps_candidate[c_mask].min() < ps_candidate[t_mask].max()):
            ps_data = ps_data.copy()
            ps_data["ps"] = ps_candidate
            ps_estimated = True
            print("  L2 LogisticRegression (C=" + str(C_val) + ") 成功")
            break
    except Exception:
        continue

if not ps_estimated:
    print("  警告: 完全分離 -> 前期売上+施設医師数のみで単変量推定")
    try:
        _fallback_cols = ["pre_mean_std"] + [c for c in _cont_std_cols if c in ps_dummies.columns]
        X_simple = ps_dummies[_fallback_cols].values
        lr_s = LogisticRegression(C=0.5, max_iter=1000, random_state=RANDOM_SEED)
        lr_s.fit(X_simple, y_ps)
        ps_data = ps_data.copy()
        ps_data["ps"] = lr_s.predict_proba(X_simple)[:, 1]
    except Exception:
        ps_data = ps_data.copy()
        ps_data["ps"] = 0.5

ps_t = ps_data[ps_data["treated"] == 1]["ps"]
ps_c = ps_data[ps_data["treated"] == 0]["ps"]
print("  処置群 PS: mean=" + str(round(ps_t.mean(), 3))
      + "  range=[" + str(round(ps_t.min(), 3)) + ", " + str(round(ps_t.max(), 3)) + "]")
print("  対照群 PS: mean=" + str(round(ps_c.mean(), 3))
      + "  range=[" + str(round(ps_c.min(), 3)) + ", " + str(round(ps_c.max(), 3)) + "]")

# ===================================================================
# [6] 全体 1:1 マッチング
# ===================================================================
print("\n[6] 全体 1:1 最近傍マッチング")

# doctor_id alias for matching functions (facility_id based in ver2)
ps_data["doctor_id"] = ps_data["facility_id"]

ps_data["logit_ps"] = logit_ps(ps_data["ps"].values)
logit_std = ps_data["logit_ps"].std()

if logit_std < 1e-6:
    caliper = None
    print("  logit(PS) SD≈0 -> キャリパーなし最近傍")
else:
    caliper = CALIPER_MULTIPLIER * logit_std
    print("  logit(PS) SD=" + str(round(logit_std, 4)) + "  キャリパー=" + str(round(caliper, 4)))

treated_for_match = ps_data[ps_data["treated"] == 1][["doctor_id", "logit_ps"]].reset_index(drop=True)
control_for_match = ps_data[ps_data["treated"] == 0][["doctor_id", "logit_ps"]].reset_index(drop=True)

matched_pairs = psm_1to1(treated_for_match, control_for_match, caliper)
if caliper is not None and len(matched_pairs) < len(treated_for_match) * 0.3:
    caliper_wide = logit_std
    print("  マッチング率低 -> キャリパー緩和(" + str(round(caliper_wide, 4)) + ")リトライ")
    matched_pairs = psm_1to1(treated_for_match, control_for_match, caliper_wide)
    if len(matched_pairs) < 5:
        matched_pairs = psm_1to1(treated_for_match, control_for_match, None)

matched_t_ids = [p[0] for p in matched_pairs]
matched_c_ids = [p[1] for p in matched_pairs]
matched_ids   = set(matched_t_ids) | set(matched_c_ids)
matched_df    = ps_data[ps_data["doctor_id"].isin(matched_ids)].copy()

print("  マッチング成立ペア数: " + str(len(matched_pairs))
      + " / " + str(len(treated_for_match)) + " 処置群施設")

# 全体 ATT
def _get_growth(doc_id, df):
    r = df[df["doctor_id"] == doc_id]["growth"].values
    return r[0] if len(r) > 0 else np.nan

mt_growth = np.array([_get_growth(d, ps_data) for d in matched_t_ids])
mc_growth = np.array([_get_growth(d, ps_data) for d in matched_c_ids])
valid_mask = ~np.isnan(mc_growth)
diffs  = mt_growth[valid_mask] - mc_growth[valid_mask]
att    = diffs.mean()
se_att = diffs.std() / np.sqrt(len(diffs))
t_stat = att / (se_att + 1e-12)
p_val  = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(diffs) - 1))
ci_lo  = att - 1.96 * se_att
ci_hi  = att + 1.96 * se_att

print("  全体 ATT=" + str(round(att, 2)) + " 円/月  p=" + str(round(p_val, 4)))

# ===================================================================
# [6b] 共変量バランス（SMD）チェック
# ===================================================================
def _smd(t_vals, c_vals):
    t = np.asarray(t_vals, dtype=float)
    c = np.asarray(c_vals, dtype=float)
    t, c = t[~np.isnan(t)], c[~np.isnan(c)]
    if len(t) < 2 or len(c) < 2:
        return np.nan
    pv = (t.var() + c.var()) / 2
    return 0.0 if pv < 1e-12 else (t.mean() - c.mean()) / np.sqrt(pv)

_pre_t  = ps_data[ps_data["treated"] == 1]
_pre_c  = ps_data[ps_data["treated"] == 0]
_post_t = ps_data[ps_data["doctor_id"].isin(matched_t_ids)]
_post_c = ps_data[ps_data["doctor_id"].isin(matched_c_ids)]

covariate_balance_smd = []

for _col in COV_CONT_COLS + ["pre_mean"]:
    if _col not in ps_data.columns:
        continue
    smd_b = _smd(_pre_t[_col],  _pre_c[_col])
    smd_a = _smd(_post_t[_col], _post_c[_col])
    covariate_balance_smd.append({
        "変数": _col,
        "SMD_before": float(smd_b) if not np.isnan(smd_b) else 0.0,
        "SMD_after":  float(smd_a) if not np.isnan(smd_a) else 0.0,
    })

for _cat in COV_CAT_COLS:
    if _cat not in ps_data.columns:
        continue
    _levels = sorted(ps_data[_cat].dropna().astype(str).unique())
    for _lv in _levels:
        _b_t = (_pre_t[_cat].astype(str)  == _lv).astype(float)
        _b_c = (_pre_c[_cat].astype(str)  == _lv).astype(float)
        _a_t = (_post_t[_cat].astype(str) == _lv).astype(float)
        _a_c = (_post_c[_cat].astype(str) == _lv).astype(float)
        smd_b = _smd(_b_t, _b_c)
        smd_a = _smd(_a_t, _a_c)
        covariate_balance_smd.append({
            "変数": f"{_cat}={_lv}",
            "SMD_before": float(smd_b) if not np.isnan(smd_b) else 0.0,
            "SMD_after":  float(smd_a) if not np.isnan(smd_a) else 0.0,
        })

print(f"\n  共変量バランス SMD ({len(covariate_balance_smd)}項目):")
for r in covariate_balance_smd:
    flag = " OK" if abs(r["SMD_after"]) < 0.1 else " !"
    print(f"    {r['変数']}: before={r['SMD_before']:.3f} after={r['SMD_after']:.3f}{flag}")

# ===================================================================
# [7] サブグループ分析（SUBGROUP_SPECS に基づいて全次元を処理）
# ===================================================================
print("\n[7] サブグループ分析（" + str(len(SUBGROUP_SPECS)) + " 次元）")

global_caliper = logit_std if logit_std >= 1e-6 else None
all_sg_results = []

for (disp_name, col, is_continuous, unit) in SUBGROUP_SPECS:
    # 連続変数は自動カテゴリ化
    cat_col = col
    if is_continuous:
        cat_col = col + "_cat"
        if col in ps_data.columns:
            if col in FIXED_BINS:
                _fb = FIXED_BINS[col]
                result_cat, cat_levels = _fixed_cut(ps_data[col], _fb["bins"], _fb["labels"])
            else:
                result_cat, cat_levels = _auto_range_labels(
                    ps_data[col], q=N_BINS_CONTINUOUS, col_name=col, unit_override=unit
                )
            ps_data[cat_col] = result_cat
        elif cat_col in ps_data.columns:
            cat_levels = [str(v) for v in ps_data[cat_col].dropna().unique()]
        else:
            print("  [" + disp_name + "] 列 '" + col + "' が存在しないためスキップ")
            continue
    else:
        if col not in ps_data.columns:
            print("  [" + disp_name + "] 列 '" + col + "' が存在しないためスキップ")
            continue
        _col_s = ps_data[cat_col].dropna()
        if hasattr(_col_s, "cat"):
            # Categorical列: cat.categories から存在するものを取り出し固定順を適用
            _present = set(_col_s.astype(str))
            _raw = [str(v) for v in _col_s.cat.categories
                    if str(v) in _present and str(v) not in ("nan", "None", "不明")]
        else:
            _raw = [str(v) for v in _col_s.unique()
                    if str(v) not in ("nan", "None", "不明")]
        cat_levels = _sort_levels(_raw)

    if not cat_levels:
        continue

    print("  [" + disp_name + "] " + cat_col + ": " + str(len(cat_levels)) + "カテゴリ")
    dim_results = []

    for level in cat_levels:
        sg_data = ps_data[ps_data[cat_col].astype(str) == str(level)].copy()
        result, n_t, n_c = run_subgroup_psm(sg_data, caliper, global_caliper)
        row = {
            "dimension": disp_name,
            "col": cat_col,
            "level": str(level),
            "n_treated_raw": n_t,
            "n_control_raw": n_c,
        }
        if result:
            row.update(result)
            sig = ("**" if result["p"] < 0.01 else
                   ("*" if result["p"] < 0.05 else ("†" if result["p"] < 0.1 else "")))
            print("    " + str(level) + ": ATT=" + str(round(result["att"], 2))
                  + " N処置=" + str(n_t) + " N対照=" + str(n_c)
                  + " マッチ=" + str(result["n_matched"]) + " p=" + str(round(result["p"], 4))
                  + " " + sig)
        else:
            row["att"] = None
            print("    " + str(level) + ": N処置=" + str(n_t) + " N対照=" + str(n_c)
                  + " -> マッチング不成立")
        dim_results.append(row)
    all_sg_results.extend(dim_results)

sg_df = pd.DataFrame(all_sg_results)

# ===================================================================
# [7b] Coverage 別サブグループ分析（処置群のみ）
# ===================================================================
print("\n[7b] Coverage別サブグループ分析（処置群のみ）")

_coverage_corr_r = None  # JSON出力用に外部スコープで保持

_cov_treated = unit_df.loc[unit_df["treated"] == 1, "final_coverage"]
if len(_cov_treated) > 0 and _cov_treated.max() > 0:
    try:
        _full_mask_t = (unit_df["treated"] == 1) & (unit_df["final_coverage"] >= 1.0)
        _part_mask_t = (unit_df["treated"] == 1) & (unit_df["final_coverage"] < 1.0) & unit_df["final_coverage"].notna()
        if _full_mask_t.sum() > 0 and _part_mask_t.sum() >= 4:
            # coverage=1.0を「高Coverage」として確定し、残りをqcutで低・中に分割
            unit_df["coverage_cat"] = pd.NA
            unit_df.loc[_full_mask_t, "coverage_cat"] = "高Coverage"
            _part_cats = pd.qcut(
                unit_df.loc[_part_mask_t, "final_coverage"],
                q=2, labels=["低Coverage", "中Coverage"], duplicates="drop"
            )
            unit_df.loc[_part_mask_t, "coverage_cat"] = _part_cats.astype(str)
        else:
            unit_df["coverage_cat"] = pd.qcut(
                unit_df["final_coverage"].where(unit_df["treated"] == 1),
                q=3, labels=["低Coverage", "中Coverage", "高Coverage"],
                duplicates="drop"
            )
    except Exception:
        _med = _cov_treated.median()
        unit_df["coverage_cat"] = pd.cut(
            unit_df["final_coverage"].where(unit_df["treated"] == 1),
            bins=[-0.001, _med, 1.001], labels=["低Coverage", "高Coverage"]
        )

    _cov_cats = unit_df["coverage_cat"].dropna().unique()
    print(f"  Coverage分布 (処置群): {_cov_treated.describe().to_dict()}")

    # unit_df に facility_id インデックスを一時的に使うため facility_id を key にした辞書を作成
    _fac_to_growth = dict(zip(unit_df["facility_id"], unit_df["growth"]))
    _fac_to_cov_cat = dict(zip(unit_df["facility_id"], unit_df.get("coverage_cat", pd.Series(dtype=str))))

    for cat in sorted(_cov_cats, key=str):
        # 処置群のうちこのカテゴリに属する施設ID
        _cat_fac_ids = set(
            unit_df.loc[unit_df["coverage_cat"].astype(str) == str(cat), "facility_id"]
        )
        # matched_pairs（タプルリスト）からこのカテゴリの処置施設に対応するペアを抽出
        _cat_pairs = [(t, c) for t, c in matched_pairs if t in _cat_fac_ids]
        if len(_cat_pairs) < 3:
            print(f"  {cat}: サンプル不足 (N={len(_cat_pairs)})")
            continue
        _t_gr = np.array([_fac_to_growth.get(t, np.nan) for t, c in _cat_pairs])
        _c_gr = np.array([_fac_to_growth.get(c, np.nan) for t, c in _cat_pairs])
        _valid = ~(np.isnan(_t_gr) | np.isnan(_c_gr))
        _t_gr = _t_gr[_valid]
        _c_gr = _c_gr[_valid]
        if len(_t_gr) == 0 or len(_c_gr) == 0:
            continue
        _att = float(np.mean(_t_gr) - np.mean(_c_gr))
        _, _p = stats.ttest_rel(_t_gr, _c_gr)
        print(f"  {cat}: ATT={_att:.2f} N処置={len(_t_gr)} N対照={len(_c_gr)} p={_p:.4f}")

    # Coverage と伸長率（前期比%）の相関（処置群のみ）
    _cov_growth = unit_df[unit_df["treated"] == 1][["final_coverage", "growth_rate"]].dropna()
    if len(_cov_growth) > 3:
        try:
            _r, _p_corr = stats.pearsonr(_cov_growth["final_coverage"], _cov_growth["growth_rate"])
            _coverage_corr_r = float(_r)
            print(f"\n  Coverage × 伸長率 相関係数: r={_r:.3f} (p={_p_corr:.4f})")
        except Exception:
            pass
else:
    print("  処置群のCoverageデータなし、スキップ")

# ===================================================================
# [8] 可視化
# ===================================================================
print("\n[8] 可視化")

# ---- (a)(b)(c) 傾向スコア分布 / 伸長率分布 / 平均伸長率棒グラフ ----
fig_main, axes = plt.subplots(1, 3, figsize=(20, 5))
fig_main.suptitle("09-2: PSM 伸長率比較 [ver2] - 傾向スコア分布・伸長率分布・平均伸長率比較",
                  fontsize=12, fontweight="bold")

ax = axes[0]
bins_ps = np.linspace(0, 1, 25)
ax.hist(ps_data[ps_data["treated"] == 1]["ps"], bins=bins_ps, alpha=0.45,
        color="#1565C0", label="視聴施設（全体）")
ax.hist(ps_data[ps_data["treated"] == 0]["ps"], bins=bins_ps, alpha=0.45,
        color="#FF8F00", label="未視聴施設（全体）")
ax.hist(matched_df[matched_df["treated"] == 1]["ps"], bins=bins_ps,
        alpha=0.9, histtype="step", linewidth=2, color="#1565C0", label="視聴施設（マッチ後）")
ax.hist(matched_df[matched_df["treated"] == 0]["ps"], bins=bins_ps,
        alpha=0.9, histtype="step", linewidth=2, color="#FF8F00", label="未視聴施設（マッチ後）")
ax.set_xlabel("傾向スコア")
ax.set_ylabel("施設数")
ax.set_title("(a) 傾向スコア分布（マッチング前後）")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

ax = axes[1]
mt_v = mt_growth[valid_mask]
mc_v = mc_growth[valid_mask]
ax.hist(mt_v, bins=20, alpha=0.6, color="#1565C0",
        label="視聴施設 (N=" + str(valid_mask.sum()) + ")")
ax.hist(mc_v, bins=20, alpha=0.6, color="#FF8F00",
        label="未視聴施設 (N=" + str(valid_mask.sum()) + ")")
ax.axvline(mt_v.mean(), color="#1565C0", linewidth=2, linestyle="--",
           label="視聴施設 平均=" + str(round(mt_v.mean(), 1)))
ax.axvline(mc_v.mean(), color="#FF8F00", linewidth=2, linestyle="--",
           label="未視聴施設 平均=" + str(round(mc_v.mean(), 1)))
ax.set_xlabel("伸長率（後期月平均 - 前期月平均, 円）")
ax.set_ylabel("施設数")
ax.set_title("(b) マッチング後 伸長率分布\nATT=" + str(round(att, 2))
             + " 円/月, p=" + str(round(p_val, 4)))
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

# ---- (c) 平均伸長率 棒グラフ (左=未視聴施設, 右=視聴施設) ----
ax = axes[2]
n_t_bar = valid_mask.sum()
means_bar = [mc_v.mean(), mt_v.mean()]
ses_bar   = [mc_v.std() / np.sqrt(len(mc_v)), mt_v.std() / np.sqrt(len(mt_v))]
ax.bar([0, 1], means_bar,
       yerr=[1.96 * s for s in ses_bar],
       color=["#FF8F00", "#1565C0"], alpha=0.75, capsize=6,
       error_kw={"linewidth": 1.5, "ecolor": "black"})
ax.set_xticks([0, 1])
ax.set_xticklabels(["未視聴施設\n(N=" + str(n_t_bar) + ")",
                    "視聴施設\n(N=" + str(n_t_bar) + ")"])
ax.set_ylabel("平均伸長率（円/月）")
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax.grid(True, alpha=0.3, axis="y")

pstar_bar = ("***" if p_val < 0.001 else ("**" if p_val < 0.01
             else ("*" if p_val < 0.05 else "n.s.")))
ax.set_title("(c) 平均伸長率の比較（マッチ後）\nATT=" + str(round(att, 2))
             + " 円/月, p=" + str(round(p_val, 3)) + " " + pstar_bar)

# ATT ブラケット注釈
_y_max = max(m + 1.96 * s for m, s in zip(means_bar, ses_bar))
_y_top = _y_max * 1.15
ax.annotate("", xy=(0, _y_top), xytext=(1, _y_top),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
ax.text(0.5, _y_top * 1.03,
        "ATT=" + str(round(att, 2)) + "  [" + str(round(ci_lo, 1))
        + ", " + str(round(ci_hi, 1)) + "]",
        ha="center", va="bottom", fontsize=8)
ax.set_ylim(bottom=min(0, min(means_bar) - max(ses_bar) * 2),
            top=_y_top * 1.20)

plt.tight_layout()
out_main = os.path.join(SCRIPT_DIR, f"psm_growth_rate_v2{_suffix}.png")
plt.savefig(out_main, dpi=150, bbox_inches="tight")
plt.close(fig_main)

# ---- (b) サブグループ Forest Plot ----
valid_sg_df = sg_df[sg_df["att"].notna()].copy()
n_rows      = len(valid_sg_df) + 1  # +1 for overall

fig_h = max(6, n_rows * 0.55 + 2)
fig2, ax2 = plt.subplots(figsize=(14, fig_h))
fig2.suptitle("09-2: サブグループ別 ATT Forest Plot（PSM, 後期-前期差）[ver2]",
              fontsize=12, fontweight="bold")

color_map = {}
palette   = ["#1565C0", "#E53935", "#43A047", "#FB8C00",
             "#8E24AA", "#00897B", "#F06292", "#795548", "#0288D1"]
dim_names = sg_df["dimension"].unique().tolist()
for i, d in enumerate(dim_names):
    color_map[d] = palette[i % len(palette)]

y_pos = list(range(len(valid_sg_df) + 1))
y_labels = []

# 全体 ATT (一番下)
overall_row_y = 0
ax2.errorbar(att, overall_row_y,
             xerr=[[att - ci_lo], [ci_hi - att]],
             fmt="D", color="black", markersize=9, capsize=5, linewidth=2,
             label="全体 (N=" + str(valid_mask.sum()) + ")")
pstar_all = ("***" if p_val < 0.001 else ("**" if p_val < 0.01
             else ("*" if p_val < 0.05 else "")))
ax2.text(ci_hi + 0.3, overall_row_y,
         "p=" + str(round(p_val, 3)) + pstar_all, va="center", fontsize=8)
y_labels.append("全体")

# サブグループ（下から上へ配置）
prev_dim = None
for i, row in enumerate(valid_sg_df.itertuples(), start=1):
    dim  = row.dimension
    col  = color_map.get(dim, "gray")
    y    = i
    att_ = row.att
    ci_l = row.ci_lo
    ci_h = row.ci_hi
    p_   = row.p
    n_m  = int(row.n_matched)

    # 次元が変わるときに区切り線
    if dim != prev_dim and prev_dim is not None:
        ax2.axhline(y - 0.5, color="lightgray", linewidth=0.7, linestyle="--")

    ax2.errorbar(att_, y, xerr=[[att_ - ci_l], [ci_h - att_]],
                 fmt="o", color=col, markersize=7, capsize=4, linewidth=1.5)
    pstar = ("***" if p_ < 0.001 else ("**" if p_ < 0.01
             else ("*" if p_ < 0.05 else ("†" if p_ < 0.1 else ""))))
    ax2.text(max(ci_h, att_ + 0.1) + 0.3, y,
             "p=" + str(round(p_, 3)) + pstar + "  N=" + str(n_m),
             va="center", fontsize=7.5)
    y_labels.append("[" + dim + "] " + str(row.level))
    prev_dim = dim

ax2.axvline(0, color="gray", linestyle="--", linewidth=1)
ax2.set_yticks(range(len(y_labels)))
ax2.set_yticklabels(y_labels, fontsize=8.5)
ax2.set_xlabel("ATT（円/月、後期-前期差の視聴施設-対照施設差）", fontsize=10)
ax2.set_title("サブグループ別 ATT（全 " + str(len(valid_sg_df)) + " カテゴリ）", fontsize=10)
ax2.grid(True, alpha=0.3, axis="x")

# 凡例（次元の色）
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker="o", color=color_map.get(d, "gray"),
                           label=d, markersize=6, linestyle="None")
                   for d in dim_names]
legend_elements.append(Line2D([0], [0], marker="D", color="black",
                               label="全体", markersize=6, linestyle="None"))
ax2.legend(handles=legend_elements, fontsize=8, loc="lower right")

plt.tight_layout()
out_forest = os.path.join(SCRIPT_DIR, f"psm_subgroup_forest_v2{_suffix}.png")
plt.savefig(out_forest, dpi=150, bbox_inches="tight")
plt.close(fig2)

print("  psm_growth_rate_v2.png を保存")
print("  psm_subgroup_forest_v2.png を保存")

# ===================================================================
# [8c] Coverage 用量反応可視化
# ===================================================================
print("\n[8c] Coverage 用量反応可視化")

if "final_coverage" in unit_df.columns:
    try:
        fig_dose, axes_dose = plt.subplots(1, 2, figsize=(14, 5))
        fig_dose.suptitle(
            "09-2: Coverage（施設視聴率）と売上伸長率の関係 [ver2]\n"
            "左: 用量反応散布図  右: Coverage群別 平均伸長率比較",
            fontsize=11, fontweight="bold"
        )

        # --- 左: Coverage vs 伸長率（前期比%）散布図 + 回帰直線 ---
        ax_s = axes_dose[0]
        _cov_data = unit_df[unit_df["treated"] == 1][["final_coverage", "growth_rate"]].dropna()
        _x = _cov_data["final_coverage"].values
        _y = _cov_data["growth_rate"].values

        ax_s.scatter(_x, _y, alpha=0.5, color="#1565C0", s=40, label="処置施設")
        if len(_x) > 3:
            from scipy.stats import linregress as _linreg
            _slope, _intercept, _r_val, _p_reg, _ = _linreg(_x, _y)
            _xl = np.linspace(_x.min(), _x.max(), 100)
            ax_s.plot(_xl, _slope * _xl + _intercept, color="red", linewidth=2,
                      label=f"回帰直線 (r={_r_val:.3f}, p={_p_reg:.3f})")
        ax_s.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax_s.set_xlabel("最終視聴率 Coverage（0→1）")
        ax_s.set_ylabel("売上伸長率（前期比, %）")
        ax_s.set_title(f"(a) 用量反応: Coverage × 売上伸長率\n処置施設のみ (N={len(_x)})")
        ax_s.legend(fontsize=9)
        ax_s.grid(True, alpha=0.3)

        # --- 右: 対照群(マッチ後) + Coverage群別 処置群 平均伸長率（前期比%）---
        ax_b = axes_dose[1]

        # 対照群のgrowth_rateをunit_dfから取得
        _fac_to_gr = dict(zip(unit_df["facility_id"], unit_df["growth_rate"]))
        mc_gr = np.array([_fac_to_gr.get(c, np.nan) for c in matched_c_ids])
        _ctrl_gr = mc_gr[valid_mask]
        _ctrl_gr = _ctrl_gr[~np.isnan(_ctrl_gr)]
        _bar_labels = ["対照群\n(マッチ後)"]
        _bar_means  = [float(np.mean(_ctrl_gr))]
        _bar_ses    = [float(np.std(_ctrl_gr) / np.sqrt(len(_ctrl_gr))) if len(_ctrl_gr) > 1 else 0.0]
        _bar_ns     = [int(len(_ctrl_gr))]
        _bar_colors = ["#FF8F00"]

        _cov_order_lbl = [lbl for lbl in ["低Coverage", "中Coverage", "高Coverage"]
                          if "coverage_cat" in unit_df.columns
                          and lbl in unit_df["coverage_cat"].astype(str).unique()]
        _cov_palette = {"低Coverage": "#90CAF9", "中Coverage": "#42A5F5", "高Coverage": "#1565C0"}

        for _cl in _cov_order_lbl:
            _g_vals = unit_df.loc[unit_df["coverage_cat"].astype(str) == _cl, "growth_rate"].dropna().values
            if len(_g_vals) < 2:
                continue
            _bar_labels.append(f"処置群\n{_cl}")
            _bar_means.append(float(np.mean(_g_vals)))
            _bar_ses.append(float(np.std(_g_vals) / np.sqrt(len(_g_vals))))
            _bar_ns.append(int(len(_g_vals)))
            _bar_colors.append(_cov_palette.get(_cl, "#1565C0"))

        _x_pos = np.arange(len(_bar_labels))
        ax_b.bar(_x_pos, _bar_means, yerr=_bar_ses, capsize=5,
                 color=_bar_colors, alpha=0.85, width=0.6)
        _max_abs = max(abs(m) for m in _bar_means) if _bar_means else 1.0
        for xi, (m, se, n) in enumerate(zip(_bar_means, _bar_ses, _bar_ns)):
            ax_b.text(xi, m + se + _max_abs * 0.04, f"N={n}", ha="center", fontsize=9)
        ax_b.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax_b.set_xticks(_x_pos)
        ax_b.set_xticklabels(_bar_labels, fontsize=9)
        ax_b.set_ylabel("平均売上伸長率（前期比, %）")
        ax_b.set_title("(b) Coverage群別 平均伸長率\n（対照群との比較）")
        ax_b.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        _dose_path = os.path.join(SCRIPT_DIR, f"coverage_dose_response_v2{_suffix}.png")
        fig_dose.savefig(_dose_path, dpi=150, bbox_inches="tight")
        plt.close(fig_dose)
        print(f"  coverage_dose_response_v2.png を保存")
    except Exception as e:
        print(f"  Coverage用量反応可視化失敗: {e}")
else:
    print("  Coverageデータなし、スキップ")

# ===================================================================
# [9] JSON 出力
# ===================================================================
results_dir = os.path.join(SCRIPT_DIR, "results")
os.makedirs(results_dir, exist_ok=True)

def _sg_to_dict(row):
    d = {
        "dimension": row.get("dimension", ""),
        "level":     str(row.get("level", "")),
        "n_treated_raw": int(row.get("n_treated_raw", 0)),
        "n_control_raw": int(row.get("n_control_raw", 0)),
        "n_matched": int(row.get("n_matched", 0)) if pd.notna(row.get("n_matched")) else 0,
        "att": float(row["att"]) if row.get("att") is not None and pd.notna(row.get("att")) else None,
        "se":  float(row["se"])  if row.get("se")  is not None and pd.notna(row.get("se"))  else None,
        "p_value": float(row["p"]) if row.get("p") is not None and pd.notna(row.get("p")) else None,
        "ci_95_lower": float(row["ci_lo"]) if row.get("ci_lo") is not None and pd.notna(row.get("ci_lo")) else None,
        "ci_95_upper": float(row["ci_hi"]) if row.get("ci_hi") is not None and pd.notna(row.get("ci_hi")) else None,
    }
    return d

results_json = {
    "analysis_settings": {
        "version": "ver2 (facility-level, includes multi-doctor facilities)",
        "pre_period":  "month 0-" + str(PRE_END) + " (" + str(PRE_END + 1) + "ヶ月)",
        "post_period": "month " + str(POST_START) + "-32 (" + str(33 - POST_START) + "ヶ月)",
        "outcome": "後期月平均 - 前期月平均（円）",
        "caliper": float(caliper) if caliper is not None else None,
        "matching": "1:1 nearest-neighbor within caliper (facility level)",
        "n_subgroup_dimensions": len(SUBGROUP_SPECS),
    },
    "sample_sizes": {
        "washout_excluded": int(len(washout_viewers)),
        "treated_raw": int(len(treated_fac_ids)),
        "control_raw": int(len(control_fac_ids)),
        "matched_pairs": int(len(matched_pairs)),
    },
    "overall_att": {
        "att": float(att), "se": float(se_att), "t_stat": float(t_stat),
        "p_value": float(p_val), "ci_95_lower": float(ci_lo), "ci_95_upper": float(ci_hi),
    },
    "subgroup_att": [_sg_to_dict(r) for r in all_sg_results],
    "covariate_balance_smd": covariate_balance_smd,
    "baseline_cat_levels": _bc_levels,
    "interpretation": {
        "note": "PSMはobservable confoundersのみ調整。未観測交絡は残る。",
        "att_definition": "視聴施設の（後期-前期）成長率 と 対照施設の（後期-前期）成長率 の差",
        "unit": "施設レベル (facility_id)。複数医師施設含む。主施設=平均納入額最大施設。",
    },
    "coverage_stats": {
        "treated_mean_coverage": (
            float(unit_df.loc[unit_df["treated"] == 1, "final_coverage"].mean())
            if "final_coverage" in unit_df.columns else None
        ),
        "coverage_growth_corr": _coverage_corr_r,
    },
    "psm_covariates": cov_cols,
}

json_path = os.path.join(results_dir, f"psm_growth_rate_v2{_suffix}.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results_json, f, ensure_ascii=False, indent=2)
print("  結果JSON: " + json_path)

# ===================================================================
# [10] チャネル別 PSM分析
# ===================================================================
print("\n[10] チャネル別 PSM分析")

CHANNEL_DISPLAY = {
    "webiner":    "ウェビナー",
    "e_contents": "eコンテンツ",
    "Web講演会":  "Web講演会",
}

channel_results  = {}
channel_coverage = {}

for _ch in CONTENT_TYPES:
    _ch_view = viewing_all[viewing_all["活動種別"] == _ch]
    _ch_first = (
        _ch_view[
            (_ch_view["month_index"] >= WASHOUT_MONTHS) &
            (_ch_view["month_index"] <= LAST_ELIGIBLE_MONTH) &
            _ch_view["facility_id"].notna() &
            ~_ch_view["facility_id"].isin(washout_fac_ids)
        ]
        .groupby("facility_id")["month_index"].min()
    )
    _ch_treat = set(_ch_first.index)
    _ch_ctrl = set(fac_to_docs.keys()) - _ch_treat - washout_fac_ids
    _ch_disp = CHANNEL_DISPLAY.get(_ch, _ch)
    print(f"  [{_ch_disp}] 処置群: {len(_ch_treat)} 施設, 対照群: {len(_ch_ctrl)} 施設")

    _res, _nt, _nc = _run_ch_psm(_ch_treat, _ch_ctrl, unit_df)
    channel_results[_ch] = _res
    if _res:
        _sig = ("**" if _res["p_value"] < 0.01 else
                ("*" if _res["p_value"] < 0.05 else ("†" if _res["p_value"] < 0.1 else "")))
        print(f"    ATT={_res['att']:.2f} 円/月  N={_res['n_matched']}ペア"
              f"  p={_res['p_value']:.4f} {_sig}"
              f"  95%CI[{_res['ci_95_lower']:.2f}, {_res['ci_95_upper']:.2f}]")
    else:
        print(f"    マッチング不成立 (処置群={_nt}施設, 対照群={_nc}施設)")

    # チャネル別 Coverage（処置群施設の医師視聴率）
    _ch_post_view = (
        _ch_view[
            _ch_view["month_index"].between(WASHOUT_MONTHS, LAST_ELIGIBLE_MONTH)
            & _ch_view["facility_id"].isin(_ch_treat)
        ]
        .groupby("facility_id")["doc"].nunique()
    )
    _ch_total_docs = pd.Series(
        {fac: len(fac_to_docs[fac]) for fac in _ch_treat if fac in fac_to_docs}
    )
    _ch_cov_ser = (_ch_post_view / _ch_total_docs).clip(0, 1)
    _ch_cov_mean = float(_ch_cov_ser.mean()) if len(_ch_cov_ser) > 0 else 0.0
    channel_coverage[_ch] = _ch_cov_mean
    print(f"    Coverage={_ch_cov_mean:.1%}")

# ---- チャネル別 ATT 比較棒グラフ ----
_ch_valid = [(CHANNEL_DISPLAY.get(ch, ch), r) for ch, r in channel_results.items() if r is not None]
if _ch_valid:
    _all_labels = ["全体(Overall)"] + [c for c, _ in _ch_valid]
    _all_atts   = [att]             + [r["att"]           for _, r in _ch_valid]
    _all_cilo   = [ci_lo]           + [r["ci_95_lower"]   for _, r in _ch_valid]
    _all_cihi   = [ci_hi]           + [r["ci_95_upper"]   for _, r in _ch_valid]
    _all_ns     = [len(matched_pairs)] + [r["n_matched"]  for _, r in _ch_valid]
    _all_ps     = [p_val]           + [r["p_value"]       for _, r in _ch_valid]

    fig_ch, ax_ch = plt.subplots(figsize=(10, 5))
    fig_ch.suptitle("09-2: チャネル別 ATT 比較（PSM, 後期-前期差）[ver2]",
                    fontsize=12, fontweight="bold")
    _xs = np.arange(len(_all_labels))
    _colors = ["black"] + ["#1565C0", "#43A047", "#FB8C00"][:len(_ch_valid)]
    _errs_lo = [a - lo for a, lo in zip(_all_atts, _all_cilo)]
    _errs_hi = [hi - a for a, hi in zip(_all_atts, _all_cihi)]
    ax_ch.bar(_xs, _all_atts, yerr=[_errs_lo, _errs_hi], color=_colors,
              alpha=0.75, capsize=7, error_kw={"linewidth": 1.5, "ecolor": "gray"}, width=0.55)
    ax_ch.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    _scale = max(abs(a) for a in _all_atts) if _all_atts else 1.0
    for xi, (a, p_, n, eh) in enumerate(zip(_all_atts, _all_ps, _all_ns, _errs_hi)):
        _s = ("***" if p_ < 0.001 else ("**" if p_ < 0.01
              else ("*" if p_ < 0.05 else ("†" if p_ < 0.1 else "n.s."))))
        ax_ch.text(xi, a + eh + _scale * 0.04, f"{_s}\nN={n}", ha="center", fontsize=9)
    ax_ch.set_xticks(_xs)
    ax_ch.set_xticklabels(_all_labels, fontsize=10)
    ax_ch.set_ylabel("ATT（円/月）")
    ax_ch.set_title("チャネル別 ATT と 95%CI（全体+チャネル別）", fontsize=10)
    ax_ch.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out_ch = os.path.join(SCRIPT_DIR, f"psm_channel_att_v2{_suffix}.png")
    plt.savefig(out_ch, dpi=150, bbox_inches="tight")
    plt.close(fig_ch)
    print(f"  psm_channel_att_v2.png を保存")

# ---- チャネル別 Coverage + 平均伸長率 比較グラフ ----
_overall_cov = (
    float(unit_df.loc[unit_df["treated"] == 1, "final_coverage"].mean())
    if "final_coverage" in unit_df.columns else 0.0
)
_overall_t_growth = float(np.nanmean(mt_growth[valid_mask]))
_overall_c_growth = float(np.nanmean(mc_growth[valid_mask]))

_cg_ch_labels  = ["全体"] + [CHANNEL_DISPLAY.get(ch, ch) for ch in CONTENT_TYPES]
_cg_coverages  = [_overall_cov] + [channel_coverage.get(ch, 0.0) for ch in CONTENT_TYPES]
_cg_t_growths  = [_overall_t_growth] + [
    channel_results[ch]["treated_mean_growth"] if channel_results.get(ch) else float("nan")
    for ch in CONTENT_TYPES
]
_cg_c_growths  = [_overall_c_growth] + [
    channel_results[ch]["control_mean_growth"] if channel_results.get(ch) else float("nan")
    for ch in CONTENT_TYPES
]
_cg_colors = ["#555555", "#1565C0", "#43A047", "#FB8C00"]

fig_cg, (ax_cov, ax_gr) = plt.subplots(1, 2, figsize=(14, 5))
fig_cg.suptitle("09-2: チャネル別 Coverage・平均伸長率比較 [ver2]",
                fontsize=12, fontweight="bold")

# (a) Coverage
_xs = np.arange(len(_cg_ch_labels))
ax_cov.bar(_xs, [v * 100 for v in _cg_coverages],
           color=_cg_colors[:len(_cg_ch_labels)], alpha=0.75, width=0.55)
ax_cov.set_xticks(_xs)
ax_cov.set_xticklabels(_cg_ch_labels, fontsize=10)
ax_cov.set_ylabel("平均 Coverage（%）")
ax_cov.set_title("(a) チャネル別 平均Coverage（処置群）", fontsize=10)
ax_cov.grid(True, alpha=0.3, axis="y")
for xi, v in enumerate(_cg_coverages):
    ax_cov.text(xi, v * 100 + 0.5, f"{v:.1%}", ha="center", fontsize=9)

# (b) 平均伸長率（視聴施設 vs マッチング後対照群）
_w = 0.35
ax_gr.bar(_xs - _w / 2, _cg_c_growths, width=_w,
          color="#FF8F00", alpha=0.75, label="未視聴施設（マッチ後対照）")
ax_gr.bar(_xs + _w / 2, _cg_t_growths, width=_w,
          color="#1565C0", alpha=0.75, label="視聴施設（処置群）")
ax_gr.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax_gr.set_xticks(_xs)
ax_gr.set_xticklabels(_cg_ch_labels, fontsize=10)
ax_gr.set_ylabel("平均伸長率（円/月）")
ax_gr.set_title("(b) チャネル別 平均伸長率（視聴 vs 未視聴, マッチ後）", fontsize=10)
ax_gr.legend(fontsize=9)
ax_gr.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
_out_cg = os.path.join(SCRIPT_DIR, f"psm_channel_coverage_growth_v2{_suffix}.png")
plt.savefig(_out_cg, dpi=150, bbox_inches="tight")
plt.close(fig_cg)
print(f"  psm_channel_coverage_growth_v2.png を保存")

# JSON にチャネル別結果を追記
results_json["channel_att"] = {
    ch: ({"att": r["att"], "se": r["se"], "p_value": r["p_value"],
          "ci_95_lower": r["ci_95_lower"], "ci_95_upper": r["ci_95_upper"],
          "n_matched": r["n_matched"], "n_treated_raw": r["n_treated_raw"],
          "n_control_raw": r["n_control_raw"],
          "treated_mean_growth": r["treated_mean_growth"],
          "control_mean_growth": r["control_mean_growth"]}
         if r is not None else None)
    for ch, r in channel_results.items()
}
results_json["channel_coverage"] = {
    ch: float(cov) for ch, cov in channel_coverage.items()
}
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results_json, f, ensure_ascii=False, indent=2)
print("  channel_att / channel_coverage を JSON に追記")

# ===================================================================
print("\n" + "=" * 70)
print(" 分析完了 [ver2: 施設レベル PSM]")
print("=" * 70)
print("\n【全体 ATT】 " + str(round(att, 2)) + " 円/月"
      + "  (p=" + str(round(p_val, 4)) + ", 95%CI ["
      + str(round(ci_lo, 2)) + ", " + str(round(ci_hi, 2)) + "])")
print("\n【サブグループ別（有意なもの）】")
for r in all_sg_results:
    p_ = r.get("p")
    if p_ is not None and p_ < 0.1:
        sig = ("**" if p_ < 0.01 else ("*" if p_ < 0.05 else "†"))
        print("  [" + r["dimension"] + "] " + str(r["level"])
              + ": ATT=" + str(round(r["att"], 2))
              + " p=" + str(round(p_, 4)) + " " + sig)
print("\n【注意】視聴は施設の自発行動 -> 未観測交絡が残るため因果解釈は慎重に。")
