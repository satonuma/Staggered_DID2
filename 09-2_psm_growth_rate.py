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
    "UHP区分名",        # 施設: UHPセグメント
    "施設区分名",        # 施設: 施設タイプ
    "baseline_cat",    # 施設: ベースライン売上カテゴリ
]
# 連続共変量 (z-score 標準化)
COV_CONT_COLS = ["n_docs", "n_pre_viewed_docs"]  # 施設あたり医師数 / ウォッシュアウト前視聴医師数
# ===================================================================

FILE_RW_LIST           = "rw_list.csv"
FILE_SALES             = "sales.csv"
FILE_DIGITAL           = "デジタル視聴データ.csv"
FILE_ACTIVITY          = "活動データ.csv"
FILE_DOCTOR_ATTR       = "doctor_attribute.csv"
FILE_FACILITY_MASTER   = "facility_attribute_修正.csv"
FILE_FAC_DOCTOR_LIST   = "施設医師リスト.csv"

INCLUDE_ONLY_RW = False
EXCLUDE_ZERO_SALES_FACILITIES = False  # True: 全期間納入が0の施設を解析対象から除外
UHP_RANK = {"UHP-A": 0, "UHP-B": 1, "UHP-C": 2}

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
}

# ===================================================================
# サブグループ分析設定 (ver2: 施設レベルのみ)
#   (表示名, unit_df の列名, is_continuous, 単位文字列)
#   列が存在しない場合は自動スキップ
# ===================================================================
SUBGROUP_SPECS = [
    ("ベースライン納入額",  "baseline_cat",  False, ""),
    ("UHP区分",            "UHP区分名",      False, ""),
    ("施設区分",           "施設区分名",     False, ""),
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


# QUINTILE系カテゴリ (H/L/M/VH/Z) の表示順
_QUINTILE_ORDER = ["不明", "Z", "L", "M", "H", "VH"]


def _sort_levels(levels):
    """カテゴリレベルを適切な順に並べる。
    QUINTILE値 (H/L/M/VH/Z) のみで構成される場合は 不明,Z,L,M,H,VH 順、
    それ以外は文字列ソート。
    """
    str_levels = [str(v) for v in levels]
    non_missing = [v for v in str_levels if v not in ("nan", "None", "不明")]
    if non_missing and all(v in ("H", "L", "M", "VH", "Z") for v in non_missing):
        return [v for v in _QUINTILE_ORDER if v in str_levels]
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
sales_raw["日付"]  = pd.to_datetime(sales_raw["日付"], format="mixed")
daily = sales_raw[sales_raw["品目コード"].str.strip() == ENT_PRODUCT_CODE].copy()
daily = daily.rename(columns={
    "日付": "delivery_date",
    "施設（本院に合算）コード": "facility_id",
    "実績": "amount",
})
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
fac_df       = pd.read_csv(os.path.join(DATA_DIR, FILE_FACILITY_MASTER))
doc_attr_df  = pd.read_csv(os.path.join(DATA_DIR, FILE_DOCTOR_ATTR))

_doc_to_fac   = dict(zip(fac_doc_list["doc"], fac_doc_list["fac"]))
_doc_to_honin = dict(zip(fac_doc_list["doc"], fac_doc_list["fac_honin"]))
all_docs = set(fac_doc_list["doc"])

# 主施設割り当て: 平均納入額最大の施設
_doc_fac_list = fac_doc_list[["doc", "fac_honin"]].drop_duplicates()
_sales_by_fac = (
    daily.groupby("facility_id")["amount"].mean()
    .reset_index().rename(columns={"facility_id": "fac_honin", "amount": "avg_sales"})
)
_doc_fac_sales = _doc_fac_list.merge(_sales_by_fac, on="fac_honin", how="left")
_doc_fac_sales["avg_sales"] = _doc_fac_sales["avg_sales"].fillna(0)

_doc_primary_all = (
    _doc_fac_sales.sort_values("avg_sales", ascending=False)
    .groupby("doc")["fac_honin"].first()
)

# 全施設0 → UHP最上位
_fac_uhp = fac_df.drop_duplicates("fac_honin").set_index("fac_honin")["UHP区分名"] \
    if "UHP区分名" in fac_df.columns else pd.Series(dtype=str)
_zero_sum = _doc_fac_sales.groupby("doc")["avg_sales"].sum()
_zero_docs_set = set(_zero_sum[_zero_sum == 0].index)
for _doc in _zero_docs_set:
    _doc_facs = _doc_fac_sales[_doc_fac_sales["doc"] == _doc]["fac_honin"].tolist()
    if _doc_facs:
        _ranked = sorted(_doc_facs, key=lambda f: UHP_RANK.get(str(_fac_uhp.get(f, "")), 99))
        _doc_primary_all[_doc] = _ranked[0]

doc_primary_fac = _doc_primary_all  # doc → fac_honin

# RWフィルタ
rw_doc_ids = set(rw_list["doc"])
if INCLUDE_ONLY_RW:
    analysis_docs_all = all_docs & rw_doc_ids
else:
    analysis_docs_all = all_docs

# 施設→医師リスト (1:N)
fac_to_docs: dict = {}
for doc in analysis_docs_all:
    if doc in doc_primary_fac.index:
        fac = doc_primary_fac[doc]
        fac_to_docs.setdefault(fac, []).append(doc)

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

# 視聴データに主施設ID付与
viewing_all = viewing.copy()
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

# ウォッシュアウト期間の視聴医師数（施設別）を共変量として追加
_pre_viewed = (
    viewing_all[
        (viewing_all["month_index"] < WASHOUT_MONTHS) &
        viewing_all["facility_id"].isin(analysis_fac_ids)
    ]
    .groupby("facility_id")["doc"].nunique()
)
unit_df["n_pre_viewed_docs"] = unit_df["facility_id"].map(_pre_viewed).fillna(0).astype(int)
print(f"  n_pre_viewed_docs: WO前視聴医師数 (非ゼロ施設: {unit_df['n_pre_viewed_docs'].gt(0).sum()}件)")

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
            # Categorical列: pd.cut で作成された category 順を保持
            _present = set(_col_s.astype(str))
            cat_levels = [str(v) for v in _col_s.cat.categories
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

    # Coverage と伸長率の相関（処置群のみ）
    _cov_growth = unit_df[unit_df["treated"] == 1][["final_coverage", "growth"]].dropna()
    if len(_cov_growth) > 3:
        try:
            _r, _p_corr = stats.pearsonr(_cov_growth["final_coverage"], _cov_growth["growth"])
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
out_main = os.path.join(SCRIPT_DIR, "psm_growth_rate_v2.png")
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
out_forest = os.path.join(SCRIPT_DIR, "psm_subgroup_forest_v2.png")
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

        # --- 左: Coverage vs 伸長率 散布図 + 回帰直線 ---
        ax_s = axes_dose[0]
        _cov_data = unit_df[unit_df["treated"] == 1][["final_coverage", "growth"]].dropna()
        _x = _cov_data["final_coverage"].values
        _y = _cov_data["growth"].values

        ax_s.scatter(_x, _y, alpha=0.5, color="#1565C0", s=40, label="処置施設")
        if len(_x) > 3:
            from scipy.stats import linregress as _linreg
            _slope, _intercept, _r_val, _p_reg, _ = _linreg(_x, _y)
            _xl = np.linspace(_x.min(), _x.max(), 100)
            ax_s.plot(_xl, _slope * _xl + _intercept, color="red", linewidth=2,
                      label=f"回帰直線 (r={_r_val:.3f}, p={_p_reg:.3f})")
        ax_s.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax_s.set_xlabel("最終視聴率 Coverage（0→1）")
        ax_s.set_ylabel("売上伸長率（後期月平均 − 前期月平均, 円）")
        ax_s.set_title(f"(a) 用量反応: Coverage × 売上伸長率\n処置施設のみ (N={len(_x)})")
        ax_s.legend(fontsize=9)
        ax_s.grid(True, alpha=0.3)

        # --- 右: 対照群(マッチ後) + Coverage群別 処置群 平均伸長率 ---
        ax_b = axes_dose[1]

        _ctrl_g = mc_growth[valid_mask]
        _ctrl_g = _ctrl_g[~np.isnan(_ctrl_g)]
        _bar_labels = ["対照群\n(マッチ後)"]
        _bar_means  = [float(np.mean(_ctrl_g))]
        _bar_ses    = [float(np.std(_ctrl_g) / np.sqrt(len(_ctrl_g))) if len(_ctrl_g) > 1 else 0.0]
        _bar_ns     = [int(len(_ctrl_g))]
        _bar_colors = ["#FF8F00"]

        _cov_order_lbl = [lbl for lbl in ["低Coverage", "中Coverage", "高Coverage"]
                          if "coverage_cat" in unit_df.columns
                          and lbl in unit_df["coverage_cat"].astype(str).unique()]
        _cov_palette = {"低Coverage": "#90CAF9", "中Coverage": "#42A5F5", "高Coverage": "#1565C0"}

        for _cl in _cov_order_lbl:
            _g_vals = unit_df.loc[unit_df["coverage_cat"].astype(str) == _cl, "growth"].dropna().values
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
        ax_b.set_ylabel("平均売上伸長率（後期月平均 − 前期月平均, 円）")
        ax_b.set_title("(b) Coverage群別 平均伸長率\n（対照群との比較）")
        ax_b.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        _dose_path = os.path.join(SCRIPT_DIR, "coverage_dose_response_v2.png")
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

json_path = os.path.join(results_dir, "psm_growth_rate_v2.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results_json, f, ensure_ascii=False, indent=2)
print("  結果JSON: " + json_path)

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
