#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05-2_intensive_extensive_margin.py [ver2: 複数医師施設対応]
施設レベルで視聴回数別限界効果を分析。

【分析目的】
視聴回数別の限界効果を推定し、配信成功率（視聴確率）を考慮した
期待効果を計算。最適な配信戦略（既存施設 vs 新規施設）を提示。

【重要な追加分析】
1. 視聴回数別の限界効果（1回目、2回目、3回目...）
2. 視聴確率（継続率）の推定
3. 期待効果 = 視聴確率 × 限界効果
4. 最適配信戦略の閾値算出

【意思決定の問い】
同じ予算で、既存施設への追加配信 vs 新規施設への初回配信、
どちらが効果的か？
→ 「N回視聴済みの施設には配信せず、新規施設を優先すべき」のNを算出

【内生性の注意】
視聴は施設内医師の自発的行動であり、結果は相関関係として解釈すべき。

【ver1との違い】
- 複数医師施設も含む（FILTER_SINGLE_FAC_DOCTOR=False）
- 分析単位: 施設 (facility_id) レベル
- 除外フロー: ver2ロジック（主施設割り当て）
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

warnings.filterwarnings("ignore")

for _font in ["Yu Gothic", "MS Gothic", "Meiryo", "Hiragino Sans", "IPAexGothic"]:
    try:
        matplotlib.rcParams["font.family"] = _font
        break
    except Exception:
        pass
matplotlib.rcParams["axes.unicode_minus"] = False

# ================================================================
# 設定
# ================================================================
ENT_PRODUCT_CODE = "00001"
CONTENT_TYPES = ["webiner", "e_contents", "Web講演会"]
ACTIVITY_CHANNEL_FILTER = "Web講演会"

FILE_RW_LIST = "rw_list.csv"
FILE_SALES = "sales.csv"
FILE_DIGITAL = "デジタル視聴データ.csv"
FILE_ACTIVITY = "活動データ.csv"
FILE_FACILITY_MASTER = "facility_attribute_修正.csv"
FILE_DOCTOR_ATTR = "doctor_attribute.csv"
FILE_FAC_DOCTOR_LIST = "施設医師リスト.csv"

# ver2: 複数医師施設を含める
FILTER_SINGLE_FAC_DOCTOR = False
INCLUDE_ONLY_RW     = False   # True: RW医師のみ
INCLUDE_ONLY_NON_RW = False  # True: 非RW医師のみ (INCLUDE_ONLY_RW=Falseのとき有効)
EXCLUDE_ZERO_SALES_FACILITIES = False  # True: 全期間納入が0の施設を解析対象から除外

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
DATA_DIR = os.path.join(SCRIPT_DIR, "本番データ")
_required = [FILE_SALES, FILE_DIGITAL, FILE_ACTIVITY, FILE_RW_LIST]
if not all(os.path.exists(os.path.join(DATA_DIR, f)) for f in _required):
    _alt = os.path.join(SCRIPT_DIR, "data")
    if all(os.path.exists(os.path.join(_alt, f)) for f in _required):
        DATA_DIR = _alt

START_DATE = "2023-04-01"
N_MONTHS = 33
WASHOUT_MONTHS = 2
LAST_ELIGIBLE_MONTH = 29

# 配信コスト仮定（08と同じ）
COST_PER_DISTRIBUTION = 0.5  # 万円

# 視聴回数ビン定義 (var, cumul_lo, cumul_hi, 表示ラベル)
# cumulative_views==N-1 のとき「N回目の視聴」に相当
# 1〜9回目は個別、10回目以降は5刻みグループ
VIEW_BINS = [
    ("view_1",      0,   0,   "1回目"),
    ("view_2",      1,   1,   "2回目"),
    ("view_3",      2,   2,   "3回目"),
    ("view_4",      3,   3,   "4回目"),
    ("view_5",      4,   4,   "5回目"),
    ("view_6",      5,   5,   "6回目"),
    ("view_7",      6,   6,   "7回目"),
    ("view_8",      7,   7,   "8回目"),
    ("view_9",      8,   8,   "9回目"),
    ("view_10_14",  9,  13,   "10〜14回"),
    ("view_15_19", 14,  18,   "15〜19回"),
    ("view_20_24", 19,  23,   "20〜24回"),
    ("view_25_29", 24,  28,   "25〜29回"),
    ("view_30plus", 29, 9999, "30回以上"),
]
VIEW_VARS   = [b[0] for b in VIEW_BINS]
VIEW_LABELS = [b[3] for b in VIEW_BINS]
# 各ビンの期待効果計算に使う継続率レンジ (None = initial_viewing_rate を使用)
_BIN_CONT_RANGES = {
    "view_1":      None,
    "view_2":      range(1, 2),
    "view_3":      range(2, 3),
    "view_4":      range(3, 4),
    "view_5":      range(4, 5),
    "view_6":      range(5, 6),
    "view_7":      range(6, 7),
    "view_8":      range(7, 8),
    "view_9":      range(8, 9),
    "view_10_14":  range(9, 14),
    "view_15_19":  range(14, 19),
    "view_20_24":  range(19, 24),
    "view_25_29":  range(24, 29),
    "view_30plus": range(29, 35),
}

CHANNEL_DISPLAY = {
    "webiner":    "ウェビナー",
    "e_contents": "eコンテンツ",
    "Web講演会":  "Web講演会",
}

print("=" * 70)
print(" 施設視聴パターン分析（ver2）: 視聴回数別限界効果 + 期待値")
print("=" * 70)

# ================================================================
# データ読み込み
# ================================================================
print("\n[データ読み込み]")

rw_list = pd.read_csv(os.path.join(DATA_DIR, FILE_RW_LIST))

sales_raw = pd.read_csv(os.path.join(DATA_DIR, FILE_SALES), dtype=str)
sales_raw["実績"] = pd.to_numeric(sales_raw["実績"], errors="coerce").fillna(0)
daily = sales_raw[sales_raw["品目コード"].str.strip() == ENT_PRODUCT_CODE].copy()
daily["日付"] = pd.to_datetime(daily["日付"], format="mixed")
daily = daily.rename(columns={
    "日付": "delivery_date",
    "施設（本院に合算）コード": "facility_id",
    "実績": "amount",
})
daily["facility_id"] = daily["facility_id"].astype(str).str.strip()

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

common_cols = ["活動日_dt", "品目コード", "活動種別", "活動種別コード", "fac_honin", "doc"]
viewing = pd.concat([digital[common_cols], web_lecture[common_cols]], ignore_index=True)
viewing = viewing.rename(columns={
    "活動日_dt": "view_date",
    "fac_honin": "facility_id",
    "doc": "doctor_id",
    "活動種別": "channel_category",
})
viewing["view_date"] = pd.to_datetime(viewing["view_date"], format="mixed")

months = pd.date_range(start=START_DATE, periods=N_MONTHS, freq="MS")

print(f"  売上データ(ENT品目): {len(daily):,} 行")
print(f"  RW医師リスト(全体): {len(rw_list)} 行")
print(f"  視聴データ結合: {len(viewing):,} 行")

# ================================================================
# 除外フロー（ver2: 複数医師施設対応）
# ================================================================
print("\n[除外フロー (ver2)]")

# 施設医師リスト: 全医師の施設対応マスター
fac_doc_list = pd.read_csv(os.path.join(DATA_DIR, FILE_FAC_DOCTOR_LIST))
fac_doc_list["fac_honin"] = fac_doc_list["fac_honin"].astype(str).str.strip()

# [Step 1] facility_attribute_修正.csv 読み込み (施設属性用)
fac_df = pd.read_csv(os.path.join(DATA_DIR, FILE_FACILITY_MASTER))
fac_df["fac_honin"] = fac_df["fac_honin"].astype(str).str.strip()

# [Step 2] doctor_attribute.csv 読み込み (医師属性用)
doc_attr_df = pd.read_csv(os.path.join(DATA_DIR, FILE_DOCTOR_ATTR))

_doc_to_fac   = dict(zip(fac_doc_list["doc"], fac_doc_list["fac"]))
_doc_to_honin = dict(zip(fac_doc_list["doc"], fac_doc_list["fac_honin"]))
all_docs = set(fac_doc_list["doc"])

# --- 主施設割り当て (最適化版): 1施設所属は直接割り当て、複数施設所属のみ売上ベース ---
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
    .reset_index()
    .rename(columns={"facility_id": "fac_honin", "amount": "avg_sales"})
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
doc_primary = _doc_primary_all  # doc → fac_honin（主施設）

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

# --- 施設→医師リスト (主施設ベース, 1:N) ---
_prim_filt = doc_primary[doc_primary.index.isin(analysis_docs_all)]
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

print(f"\n  主施設割り当て完了:")
print(f"    解析対象医師数: {len(analysis_docs_all)}")
print(f"    解析対象施設数: {len(fac_to_docs)}")
_multi = sum(1 for docs in fac_to_docs.values() if len(docs) > 1)
print(f"    複数医師施設: {_multi} 施設")

# --- 視聴データに主施設IDを付与 (解析対象医師のみ) ---
viewing_all = viewing[viewing["doctor_id"].isin(analysis_docs_all)].copy()
viewing_all["facility_id"] = viewing_all["doctor_id"].map(doc_primary.to_dict())

# month_index を付与
viewing_all["month_index"] = (
    (viewing_all["view_date"].dt.year - 2023) * 12
    + viewing_all["view_date"].dt.month - 4
)

# ウォッシュアウト除外: 施設内いずれかの医師が washout 期間(month 0,1)に視聴
washout_fac_ids = set(
    viewing_all[
        (viewing_all["month_index"] < WASHOUT_MONTHS) &
        viewing_all["facility_id"].notna()
    ]["facility_id"]
)

# 初回視聴月: 施設内最初の視聴月（washout除外後、LAST_ELIGIBLE_MONTH 以内）
_first_view = (
    viewing_all[
        (viewing_all["month_index"] >= WASHOUT_MONTHS) &
        (viewing_all["month_index"] <= LAST_ELIGIBLE_MONTH) &
        viewing_all["facility_id"].notna() &
        ~viewing_all["facility_id"].isin(washout_fac_ids)
    ]
    .groupby("facility_id")["month_index"].min()
    .rename("cohort_month")
)

first_fac_view = _first_view  # fac_id → cohort_month

treated_fac_ids = set(_first_view.index)
control_fac_ids = set(fac_to_docs.keys()) - treated_fac_ids - washout_fac_ids
analysis_fac_ids = treated_fac_ids | control_fac_ids

print(f"\n  処置群 (視聴あり): {len(treated_fac_ids)} 施設")
print(f"  対照群 (視聴なし): {len(control_fac_ids)} 施設")
print(f"  ウォッシュアウト除外: {len(washout_fac_ids)} 施設")
print(f"  解析対象合計: {len(analysis_fac_ids)} 施設")

# ================================================================
# 施設×月次パネルデータ構築（累積視聴回数を追跡）
# ================================================================
print("\n[施設×月次パネルデータ構築]")

# 施設×月の視聴回数（施設内全医師の視聴合計）
fac_viewing_monthly = (
    viewing_all[viewing_all["facility_id"].isin(analysis_fac_ids)]
    .groupby(["facility_id", "month_index"])
    .size().reset_index(name="view_count")
)

# 全施設×全月の完全インデックスを作成し current_views を reindex
full_idx = pd.MultiIndex.from_product(
    [sorted(analysis_fac_ids), range(N_MONTHS)],
    names=["facility_id", "month_index"]
)
current_s = (fac_viewing_monthly
    .set_index(["facility_id", "month_index"])["view_count"]
    .reindex(full_idx, fill_value=0))
facility_panel = current_s.reset_index().rename(columns={"view_count": "current_views"})
facility_panel = facility_panel.sort_values(["facility_id", "month_index"])

# 累積視聴回数 = 当月含まず → shift(1) + cumsum
facility_panel["cumulative_views"] = (
    facility_panel.groupby("facility_id")["current_views"]
    .transform(lambda x: x.shift(1).fillna(0).cumsum().astype(int))
)

# 月次売上（施設レベル）
daily["month_index"] = (
    (daily["delivery_date"].dt.year - 2023) * 12
    + daily["delivery_date"].dt.month - 4
)
monthly_sales_fac = (
    daily[daily["facility_id"].isin(analysis_fac_ids)]
    .groupby(["facility_id", "month_index"])["amount"].sum().reset_index()
)

# 売上をパネルにマージ
facility_panel = facility_panel.merge(
    monthly_sales_fac[["facility_id", "month_index", "amount"]],
    on=["facility_id", "month_index"], how="left"
)
facility_panel["amount"] = facility_panel["amount"].fillna(0.0)
facility_panel["date"] = facility_panel["month_index"].map(dict(enumerate(months)))
facility_panel["current_views"] = facility_panel["current_views"].astype(int)

print(f"  施設×月パネル: {len(facility_panel):,} 行")
print(f"  施設数: {facility_panel['facility_id'].nunique()}")
print(f"  期間: {len(months)} ヶ月")

# ================================================================
# 視聴回数別ダミー変数作成
# ================================================================
print("\n[視聴回数別ダミー変数作成]")

for var, lo, hi, label in VIEW_BINS:
    facility_panel[var] = (
        (facility_panel["cumulative_views"] >= lo) &
        (facility_panel["cumulative_views"] <= hi) &
        (facility_panel["current_views"] > 0)
    ).astype(int)

for var, lo, hi, label in VIEW_BINS:
    hi_str = "∞" if hi == 9999 else str(hi)
    print(f"  {label} (累積{lo}〜{hi_str}回済): {facility_panel[var].sum():,} 回")

# ================================================================
# TWFE回帰: 視聴回数別の限界効果推定
# ================================================================
print("\n[TWFE回帰: 視聴回数別限界効果]")

from linearmodels import PanelOLS

# 施設FE + 時間FE
panel_reg = facility_panel.copy()
panel_reg = panel_reg[panel_reg["amount"] > 0].copy()  # 売上ゼロを除外
panel_reg = panel_reg.set_index(["facility_id", "month_index"])

# 観測数が5以上のビンのみ使用
active_vars = [v for v in VIEW_VARS if facility_panel[v].sum() >= 5]
print(f"  有効変数 ({len(active_vars)}個): {active_vars}")

marginal_effects = {v: {"coefficient": np.nan, "se": np.nan, "p": np.nan, "sig": "n/a"}
                   for v in VIEW_VARS}
regression_success = False

try:
    if len(active_vars) < 2:
        raise ValueError(f"有効な視聴回数ダミーが不足 ({len(active_vars)}変数)")

    model = PanelOLS(
        dependent=panel_reg["amount"],
        exog=panel_reg[active_vars],
        entity_effects=True,
        time_effects=True
    )
    result = model.fit(cov_type="clustered", cluster_entity=True)

    for var in active_vars:
        coef = float(result.params[var])
        se   = float(result.std_errors[var])
        p    = float(result.pvalues[var])
        sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        marginal_effects[var] = {"coefficient": coef, "se": se, "p": p, "sig": sig}
        print(f"  {var}: {coef:.2f} (SE={se:.2f}, p={p:.4f}, {sig})")

    regression_success = True

except Exception as e:
    print(f"  回帰エラー: {e}")
    print("  NaN を設定")

# ================================================================
# 視聴確率（継続率）の推定
# ================================================================
print("\n[視聴確率（継続率）の推定]")

# N回累積視聴した施設が、次月に視聴する確率を計算
# shift(-1) で次月視聴を一括付与（iterrows ループを排除）
fp_sorted = facility_panel.sort_values(["facility_id", "month_index"]).copy()
fp_sorted["next_views"] = fp_sorted.groupby("facility_id")["current_views"].shift(-1)

continuation_rates = {}
for n in range(0, 40):
    cohort = fp_sorted[fp_sorted["cumulative_views"] == n]
    valid = cohort["next_views"].dropna()
    continuation_rate = float((valid > 0).mean()) if len(valid) > 0 else 0.0
    continuation_rates[n] = continuation_rate
    if n < 36:
        print(f"  累積{n:2d}回 → 次月視聴確率: {continuation_rate:.1%}")

# 初回視聴確率（未視聴施設が視聴を始める確率）
# 分母は「未視聴施設×月」のみ（cumulative_views == 0）
never_viewed_months = facility_panel[facility_panel["cumulative_views"] == 0]
first_view_count = (never_viewed_months["current_views"] > 0).sum()
initial_viewing_rate = first_view_count / len(never_viewed_months) if len(never_viewed_months) > 0 else 0.0

print(f"\n  初回視聴確率（未視聴→視聴）: {initial_viewing_rate:.1%}")
print(f"  計算根拠: 未視聴施設×月 {len(never_viewed_months):,}回 のうち {first_view_count}回が初回視聴")

# ================================================================
# 期待効果の計算
# ================================================================
print("\n[期待効果の計算]")

# 各ビンの期待効果 = 視聴確率 × 限界効果
expected_effects = {}
bin_probs = {}

for var, lo, hi, label in VIEW_BINS:
    me_val = marginal_effects[var]["coefficient"]
    cr_range = _BIN_CONT_RANGES[var]
    if cr_range is None:
        prob = initial_viewing_rate
    else:
        vals = [continuation_rates.get(i, 0.0) for i in cr_range]
        prob = float(np.mean(vals)) if vals else 0.0
    bin_probs[var] = prob
    if np.isnan(me_val):
        expected_effects[var] = np.nan
        print(f"  {label}: n/a (観測不足)")
    else:
        expected_effects[var] = prob * me_val
        print(f"  {label}: {expected_effects[var]:.2f}万円 = {prob:.1%} × {me_val:.2f}")

# ================================================================
# 最適配信戦略の算出
# ================================================================
print("\n[最適配信戦略]")

new_facility_expected = expected_effects.get("view_1", np.nan)

threshold_message = None
for var, lo, hi, label in VIEW_BINS[1:]:
    ee = expected_effects.get(var, np.nan)
    if not np.isnan(ee) and not np.isnan(new_facility_expected) and ee < new_facility_expected:
        threshold_message = f"既存施設が{label}相当（累積{lo}回以上）に達したら新規施設を優先すべき"
        break

if threshold_message is None:
    threshold_message = "全ての視聴回数で既存施設の期待効果が高い（常に既存施設優先）"

print(f"  {threshold_message}")
for var, lo, hi, label in VIEW_BINS:
    ee = expected_effects.get(var, np.nan)
    if not np.isnan(ee):
        print(f"  {label}: {ee:.2f}万円")

# ================================================================
# 可視化
# ================================================================
print("\n[可視化]")

# 有効ビン（NaN除外）
valid_bins = [(var, lo, hi, label) for var, lo, hi, label in VIEW_BINS
              if not np.isnan(marginal_effects[var]["coefficient"])]
vv = [b[0] for b in valid_bins]
vl = [b[3] for b in valid_bins]

fig = plt.figure(figsize=(18, 11))

# (a) 視聴回数別の限界効果（10ビン）
ax1 = fig.add_subplot(2, 3, (1, 2))  # 2列分使用
effects = [marginal_effects[v]["coefficient"] for v in vv]
errors  = [marginal_effects[v]["se"] for v in vv]
colors_me = ["#4caf50" if i == 0 else "#2196f3" for i in range(len(vv))]
ax1.bar(vl, effects, yerr=errors, color=colors_me, alpha=0.7, capsize=4)
ax1.set_ylabel("限界効果（万円）", fontsize=10)
ax1.set_title("(a) 視聴回数別の限界効果（TWFE推定）", fontsize=11, fontweight="bold")
ax1.tick_params(axis="x", labelsize=9)
ax1.grid(axis="y", alpha=0.3)
ax1.axhline(0, color="black", linewidth=1)

# (b) 配信優先順位（期待効果でソート）
ax5 = fig.add_subplot(2, 3, 3)
priority_data = [(("新規" if var == "view_1" else "既存") + label,
                  expected_effects[var])
                 for var, lo, hi, label in valid_bins
                 if not np.isnan(expected_effects[var])]
priority_data_sorted = sorted(priority_data, key=lambda x: x[1], reverse=True)
priority_labels = [p[0] for p in priority_data_sorted]
priority_values = [p[1] for p in priority_data_sorted]
priority_colors = ["#4caf50" if "新規" in lbl else "#2196f3" for lbl in priority_labels]
ax5.barh(priority_labels, priority_values, color=priority_colors, alpha=0.7)
ax5.set_xlabel("期待効果（万円）", fontsize=10)
ax5.set_title("(b) 配信優先順位（期待効果順）", fontsize=11, fontweight="bold")
ax5.grid(axis="x", alpha=0.3)

# (c) 視聴確率（継続率）: 累積0〜35回まで
ax2 = fig.add_subplot(2, 3, 4)
cont_x = list(range(0, 36))
cont_y = [continuation_rates.get(i, 0) * 100 for i in cont_x]
ax2.plot(cont_x, cont_y, marker="o", markersize=4, color="#ff9800", linewidth=1.5)
ax2.axhline(initial_viewing_rate * 100, color="red", linestyle="--",
            label=f"初回視聴率: {initial_viewing_rate:.1%}")
ax2.set_xlabel("累積視聴回数", fontsize=10)
ax2.set_ylabel("次月視聴確率（%）", fontsize=10)
ax2.set_title("(c) 視聴確率（継続率）", fontsize=11, fontweight="bold")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# (d) 期待効果の比較
ax3 = fig.add_subplot(2, 3, 5)
exp_labels_plot = [("新規\n" if var == "view_1" else "") + label
                   for var, lo, hi, label in valid_bins
                   if not np.isnan(expected_effects[var])]
exp_values_plot = [expected_effects[var] for var, lo, hi, label in valid_bins
                   if not np.isnan(expected_effects[var])]
colors_exp = ["#4caf50" if i == 0 else "#2196f3" for i in range(len(exp_values_plot))]
ax3.bar(exp_labels_plot, exp_values_plot, color=colors_exp, alpha=0.7)
ax3.set_ylabel("期待効果（万円）", fontsize=10)
ax3.set_title("(d) 期待効果 = 視聴確率 × 限界効果", fontsize=11, fontweight="bold")
ax3.tick_params(axis="x", labelsize=8)
ax3.grid(axis="y", alpha=0.3)
ax3.axhline(0, color="black", linewidth=1)

# (e) 最適配分メッセージ
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis("off")
_top3 = "\n".join(f"  {lbl}: {val:.2f}万円" for lbl, val in priority_data_sorted[:5])
message = f"""【最適配信戦略】

{threshold_message}

期待効果 上位5位:
{_top3}

注意: 視聴は施設内医師の自発的行動
結果は相関関係として解釈すべき"""
ax6.text(0.05, 0.5, message, fontsize=9, verticalalignment="center",
         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

plt.suptitle("視聴回数別限界効果 + 配信成功率を考慮した期待効果分析（施設レベル ver2）",
             fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.97])

output_png = f"intensive_extensive_margin_v2{_suffix}.png"
plt.savefig(output_png, dpi=150, bbox_inches="tight")
print(f"  可視化を保存: {output_png}")
plt.close()

# ================================================================
# 結果保存
# ================================================================
print("\n[結果保存]")

def _safe_float(v):
    return float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else None

output_json = {
    "marginal_effects": {
        v: {k2: _safe_float(vv2) if k2 != "sig" else vv2
            for k2, vv2 in d.items()}
        for v, d in marginal_effects.items()
    },
    "continuation_rates": {str(k): v for k, v in continuation_rates.items() if k < 40},
    "initial_viewing_rate": float(initial_viewing_rate),
    "expected_effects": {v: _safe_float(e) for v, e in expected_effects.items()},
    "expected_roi": {v: _safe_float(e / COST_PER_DISTRIBUTION)
                     for v, e in expected_effects.items() if not np.isnan(e)},
    "optimal_strategy": {
        "message": threshold_message,
        "new_facility_expected": _safe_float(new_facility_expected),
        "priority_ranking": [(lbl, _safe_float(val)) for lbl, val in priority_data_sorted],
    },
    "view_bins": [{"var": var, "cumul_lo": lo, "cumul_hi": hi if hi < 9999 else None,
                   "label": label} for var, lo, hi, label in VIEW_BINS],
    "cost_assumption": {"cost_per_distribution": float(COST_PER_DISTRIBUTION)},
    "analysis_info": {
        "version": "ver2",
        "unit": "facility",
        "n_analysis_facilities": len(analysis_fac_ids),
        "n_treated_facilities": len(treated_fac_ids),
        "n_control_facilities": len(control_fac_ids),
        "n_washout_facilities": len(washout_fac_ids),
        "filter_single_fac_doctor": FILTER_SINGLE_FAC_DOCTOR,
        "include_only_rw": INCLUDE_ONLY_RW,
    },
    "interpretation": {
        "warning": "視聴は施設内医師の自発的行動（内生変数）であり、因果効果ではなく相関関係",
        "recommendation": "配信成功率を考慮すると、既存施設への配信効率が高い傾向。ただし新規施設獲得も重要",
    },
    "channel_marginal_effects": {},  # チャネル別分析で追記
}

output_path = os.path.join(SCRIPT_DIR, "results", f"physician_viewing_analysis_v2{_suffix}.json")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_json, f, ensure_ascii=False, indent=2)

print(f"  結果を保存: {output_path}")

# ================================================================
# チャネル別 視聴回数別限界効果分析
# ================================================================
print("\n" + "=" * 70)
print(" チャネル別 視聴回数別限界効果分析")
print("=" * 70)

channel_me_results = {}
channel_exp_results = {}

for _ch in CONTENT_TYPES:
    _ch_disp = CHANNEL_DISPLAY.get(_ch, _ch)
    print(f"\n[チャネル: {_ch_disp}]")

    # チャネル限定の視聴データ
    _ch_view = viewing_all[viewing_all["channel_category"] == _ch]

    # 施設×月の視聴回数
    _ch_monthly = (
        _ch_view[_ch_view["facility_id"].isin(analysis_fac_ids)]
        .groupby(["facility_id", "month_index"])
        .size().reset_index(name="view_count")
    )
    _ch_s = (_ch_monthly
             .set_index(["facility_id", "month_index"])["view_count"]
             .reindex(full_idx, fill_value=0))
    _ch_panel = _ch_s.reset_index().rename(columns={"view_count": "current_views"})
    _ch_panel = _ch_panel.sort_values(["facility_id", "month_index"])
    _ch_panel["cumulative_views"] = (
        _ch_panel.groupby("facility_id")["current_views"]
        .transform(lambda x: x.shift(1).fillna(0).cumsum().astype(int))
    )
    _ch_panel = _ch_panel.merge(
        monthly_sales_fac[["facility_id", "month_index", "amount"]],
        on=["facility_id", "month_index"], how="left"
    )
    _ch_panel["amount"] = _ch_panel["amount"].fillna(0.0)
    _ch_panel["current_views"] = _ch_panel["current_views"].astype(int)

    for _var, _lo, _hi, _lbl in VIEW_BINS:
        _ch_panel[_var] = (
            (_ch_panel["cumulative_views"] >= _lo) &
            (_ch_panel["cumulative_views"] <= _hi) &
            (_ch_panel["current_views"] > 0)
        ).astype(int)

    _active = [v for v in VIEW_VARS if _ch_panel[v].sum() >= 5]
    print(f"  有効変数 ({len(_active)}個): {_active}")

    _me = {v: {"coefficient": np.nan, "se": np.nan, "p": np.nan, "sig": "n/a"}
           for v in VIEW_VARS}
    if len(_active) >= 2:
        try:
            _pr = _ch_panel[_ch_panel["amount"] > 0].copy().set_index(["facility_id", "month_index"])
            _model = PanelOLS(
                dependent=_pr["amount"],
                exog=_pr[_active],
                entity_effects=True,
                time_effects=True
            )
            _res = _model.fit(cov_type="clustered", cluster_entity=True)
            for _v in _active:
                _c = float(_res.params[_v])
                _s = float(_res.std_errors[_v])
                _p = float(_res.pvalues[_v])
                _sig = "***" if _p < 0.001 else "**" if _p < 0.01 else "*" if _p < 0.05 else "n.s."
                _me[_v] = {"coefficient": _c, "se": _s, "p": _p, "sig": _sig}
                print(f"  {_v}: {_c:.2f} (SE={_s:.2f}, p={_p:.4f}, {_sig})")
        except Exception as _e:
            print(f"  回帰エラー: {_e}")
    else:
        print("  有効変数不足のためスキップ")

    channel_me_results[_ch] = _me

    # チャネル別継続率
    _ch_fp = _ch_panel.sort_values(["facility_id", "month_index"]).copy()
    _ch_fp["next_views"] = _ch_fp.groupby("facility_id")["current_views"].shift(-1)
    _ch_cont_rates = {}
    for _n in range(0, 40):
        _cohort = _ch_fp[_ch_fp["cumulative_views"] == _n]
        _valid = _cohort["next_views"].dropna()
        _ch_cont_rates[_n] = float((_valid > 0).mean()) if len(_valid) > 0 else 0.0

    # チャネル別初回視聴率
    _ch_never = _ch_panel[_ch_panel["cumulative_views"] == 0]
    _ch_init_rate = (
        float((_ch_never["current_views"] > 0).sum() / len(_ch_never))
        if len(_ch_never) > 0 else 0.0
    )

    # チャネル別期待効果
    _ch_exp = {}
    for _var, _lo, _hi, _lbl in VIEW_BINS:
        _me_val = _me[_var]["coefficient"]
        _cr_range = _BIN_CONT_RANGES[_var]
        if _cr_range is None:
            _prob = _ch_init_rate
        else:
            _vals = [_ch_cont_rates.get(_i, 0.0) for _i in _cr_range]
            _prob = float(np.mean(_vals)) if _vals else 0.0
        _ch_exp[_var] = float(_prob * _me_val) if not np.isnan(_me_val) else np.nan

    channel_exp_results[_ch] = _ch_exp
    print(f"  初回視聴率: {_ch_init_rate:.1%}  期待効果(1回目): "
          f"{_ch_exp.get('view_1', np.nan):.2f}万円"
          if not np.isnan(_ch_exp.get("view_1", np.nan)) else
          f"  初回視聴率: {_ch_init_rate:.1%}  期待効果(1回目): n/a")

# ---- チャネル別 限界効果 比較グラフ ----
_n_ch = len(CONTENT_TYPES)
_fig_ch, _axes_ch = plt.subplots(1, _n_ch + 1, figsize=(5 * (_n_ch + 1), 5), sharey=True)
_fig_ch.suptitle("チャネル別 視聴回数別限界効果（TWFE, 施設レベル ver2）",
                 fontsize=12, fontweight="bold")

# 全体
_ax = _axes_ch[0]
_eff = [marginal_effects[v]["coefficient"] for v in VIEW_VARS]
_err = [marginal_effects[v]["se"] if not np.isnan(marginal_effects[v]["se"]) else 0.0
        for v in VIEW_VARS]
_eff_clean = [e if not np.isnan(e) else 0.0 for e in _eff]
_ax.bar(VIEW_LABELS, _eff_clean, yerr=_err, color="#555555", alpha=0.7, capsize=3)
_ax.axhline(0, color="black", linewidth=0.8)
_ax.set_title("全体", fontsize=10, fontweight="bold")
_ax.set_ylabel("限界効果（万円）", fontsize=9)
_ax.tick_params(axis="x", rotation=45, labelsize=7)
_ax.grid(axis="y", alpha=0.3)

_ch_colors = ["#1565C0", "#43A047", "#FB8C00"]
for _i, _ch in enumerate(CONTENT_TYPES):
    _ax = _axes_ch[_i + 1]
    _me_ch = channel_me_results[_ch]
    _eff_ch = [_me_ch[v]["coefficient"] for v in VIEW_VARS]
    _err_ch = [_me_ch[v]["se"] if not np.isnan(_me_ch[v]["se"]) else 0.0 for v in VIEW_VARS]
    _eff_ch_c = [e if not np.isnan(e) else 0.0 for e in _eff_ch]
    _ax.bar(VIEW_LABELS, _eff_ch_c, yerr=_err_ch,
            color=_ch_colors[_i % len(_ch_colors)], alpha=0.7, capsize=3)
    _ax.axhline(0, color="black", linewidth=0.8)
    _ax.set_title(CHANNEL_DISPLAY.get(_ch, _ch), fontsize=10, fontweight="bold")
    _ax.tick_params(axis="x", rotation=45, labelsize=7)
    _ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
_out_ch = os.path.join(SCRIPT_DIR, f"intensive_margin_channel_v2{_suffix}.png")
plt.savefig(_out_ch, dpi=150, bbox_inches="tight")
plt.close(_fig_ch)
print(f"\n  チャネル別グラフを保存: {_out_ch}")

# ---- チャネル別 期待効果 比較グラフ ----
_fig_ch_exp, _axes_ch_exp = plt.subplots(
    1, _n_ch + 1, figsize=(5 * (_n_ch + 1), 5), sharey=True
)
_fig_ch_exp.suptitle(
    "チャネル別 期待効果（視聴確率×限界効果, 施設レベル ver2）",
    fontsize=12, fontweight="bold"
)

# 全体
_ax_exp = _axes_ch_exp[0]
_ee_all = [
    expected_effects[v] if not np.isnan(expected_effects.get(v, np.nan)) else 0.0
    for v in VIEW_VARS
]
_ax_exp.bar(VIEW_LABELS, _ee_all, color="#555555", alpha=0.7)
_ax_exp.axhline(0, color="black", linewidth=0.8)
_ax_exp.set_title("全体", fontsize=10, fontweight="bold")
_ax_exp.set_ylabel("期待効果（万円）", fontsize=9)
_ax_exp.tick_params(axis="x", rotation=45, labelsize=7)
_ax_exp.grid(axis="y", alpha=0.3)

for _i, _ch in enumerate(CONTENT_TYPES):
    _ax_exp = _axes_ch_exp[_i + 1]
    _exp_ch = channel_exp_results[_ch]
    _ee_ch = [
        _exp_ch[v] if not np.isnan(_exp_ch.get(v, np.nan)) else 0.0
        for v in VIEW_VARS
    ]
    _ax_exp.bar(VIEW_LABELS, _ee_ch,
                color=_ch_colors[_i % len(_ch_colors)], alpha=0.7)
    _ax_exp.axhline(0, color="black", linewidth=0.8)
    _ax_exp.set_title(CHANNEL_DISPLAY.get(_ch, _ch), fontsize=10, fontweight="bold")
    _ax_exp.tick_params(axis="x", rotation=45, labelsize=7)
    _ax_exp.grid(axis="y", alpha=0.3)

plt.tight_layout()
_out_ch_exp = os.path.join(SCRIPT_DIR, f"intensive_margin_channel_exp_v2{_suffix}.png")
plt.savefig(_out_ch_exp, dpi=150, bbox_inches="tight")
plt.close(_fig_ch_exp)
print(f"\n  チャネル別期待効果グラフを保存: {_out_ch_exp}")

# JSON にチャネル別結果を追記
output_json["channel_marginal_effects"] = {
    _ch: {v: {k2: _safe_float(vv2) if k2 != "sig" else vv2
               for k2, vv2 in d.items()}
           for v, d in _me.items()}
    for _ch, _me in channel_me_results.items()
}
output_json["channel_expected_effects"] = {
    _ch: {v: _safe_float(ee) for v, ee in _exp.items()}
    for _ch, _exp in channel_exp_results.items()
}
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_json, f, ensure_ascii=False, indent=2)
print(f"  channel_marginal_effects / channel_expected_effects を JSON に追記: {output_path}")

print("\n" + "=" * 70)
print(" 分析完了 (ver2: 施設レベル)")
print("=" * 70)
print(f"\n【最適配信戦略】")
print(f"  {threshold_message}")
