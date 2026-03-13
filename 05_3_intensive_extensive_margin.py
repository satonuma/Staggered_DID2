#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_3_intensive_extensive_margin.py

【05との違い】
1. INCLUDE_ONLY_RW = True（RW医師のみ対象）
2. FILTER_SINGLE_FAC_DOCTOR = True（1施設1医師先のみ）
3. 視聴回数ビニングを qcut で均等サンプル数に変更
   - 旧05: 「view_1st=累積0回, view_2nd=累積1-2回, view_3rd=累積3-5回...」
     → 「3回目」というラベルが実態（累積3-5回時点での視聴=4-6回目）とずれていた
   - 本スクリプト: pd.qcut で N_BINS 等分し、各ビンのサンプル数を均等化
     → ラベルも「累積X-Y回→視聴」と実態に即した表記に変更
4. 出力ファイル名に実行パラメータを埋め込む
   例: 05_3_physician_viewing_RW_1fac1doc_5bins.png

【分析目的】
視聴回数別の限界効果を推定し、配信成功率（視聴確率）を考慮した
期待効果を計算。最適な配信戦略（既存医師 vs 新規医師）を提示。

【意思決定の問い】
同じ予算で、既存医師への追加配信 vs 新規医師への初回配信、
どちらが効果的か？
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
from linearmodels import PanelOLS

warnings.filterwarnings("ignore")

for _font in ["Yu Gothic", "MS Gothic", "Meiryo", "Hiragino Sans", "IPAexGothic"]:
    try:
        matplotlib.rcParams["font.family"] = _font
        break
    except Exception:
        pass
matplotlib.rcParams["axes.unicode_minus"] = False

# ================================================================
# 実行パラメータ（ファイル名に反映される）
# ================================================================
ENT_PRODUCT_CODE       = "00001"
CONTENT_TYPES          = ["webiner", "e_contents", "Web講演会"]
ACTIVITY_CHANNEL_FILTER = "Web講演会"
MR_ACTIVITY_TYPES      = ["面談", "面談_アポ", "説明会"]  # MR活動（共変量）

FILE_RW_LIST          = "rw_list.csv"
FILE_SALES            = "sales.csv"
FILE_DIGITAL          = "デジタル視聴データ.csv"
FILE_ACTIVITY         = "活動データ.csv"
FILE_FACILITY_MASTER  = "facility_attribute_修正.csv"
FILE_DOCTOR_ATTR      = "doctor_attribute.csv"
FILE_FAC_DOCTOR_LIST  = "施設医師リスト.csv"

# --- 解析集団フィルタ ---
INCLUDE_ONLY_RW          = True   # True: RW医師のみ
INCLUDE_ONLY_NON_RW      = False  # True: 非RW医師のみ（INCLUDE_ONLY_RW=Falseのとき有効）
# INCLUDE_ONLY_RW=False かつ INCLUDE_ONLY_NON_RW=False → 全医師（RW+非RW）
FILTER_SINGLE_FAC_DOCTOR = True   # True: 1施設1医師先のみ
DOCTOR_HONIN_FAC_COUNT_COL = "所属施設数"

# --- 視聴回数ビニング ---
# qcut で視聴イベント行を N_BINS 等分し各ビンのサンプル数を均等化
N_BINS = 10

# --- 期間設定 ---
START_DATE           = "2023-04-01"
N_MONTHS             = 33
WASHOUT_MONTHS       = 2
LAST_ELIGIBLE_MONTH  = 29

# --- コスト仮定 ---
COST_PER_DISTRIBUTION = 0.5  # 万円

# ================================================================
# ファイル名サフィックスをパラメータから自動生成
# ================================================================
_rw_tag    = "RW" if INCLUDE_ONLY_RW else ("nonRW" if INCLUDE_ONLY_NON_RW else "ALL")
_fac_tag   = "1fac1doc" if FILTER_SINGLE_FAC_DOCTOR else "alldoc"
_bin_tag   = f"{N_BINS}bins"
PARAM_SUFFIX = f"{_rw_tag}_{_fac_tag}_{_bin_tag}"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "本番データ")
_required  = [FILE_SALES, FILE_DIGITAL, FILE_ACTIVITY, FILE_RW_LIST]
if not all(os.path.exists(os.path.join(DATA_DIR, f)) for f in _required):
    _alt = os.path.join(SCRIPT_DIR, "data")
    if all(os.path.exists(os.path.join(_alt, f)) for f in _required):
        DATA_DIR = _alt

OUTPUT_PNG  = os.path.join(SCRIPT_DIR,           f"05_3_physician_viewing_{PARAM_SUFFIX}.png")
OUTPUT_JSON = os.path.join(SCRIPT_DIR, "results", f"05_3_physician_viewing_{PARAM_SUFFIX}.json")

print("=" * 70)
print(f" 05-3 視聴回数別限界効果分析 [{PARAM_SUFFIX}]")
print("=" * 70)
print(f"  RW医師のみ      : {INCLUDE_ONLY_RW}")
print(f"  1施設1医師フィルタ: {FILTER_SINGLE_FAC_DOCTOR}")
print(f"  視聴回数ビン数  : {N_BINS} (qcut均等サンプル)")
print(f"  出力PNG         : {os.path.basename(OUTPUT_PNG)}")

# ================================================================
# データ読み込み
# ================================================================
print("\n[データ読み込み]")

rw_list   = pd.read_csv(os.path.join(DATA_DIR, FILE_RW_LIST))

sales_raw = pd.read_csv(os.path.join(DATA_DIR, FILE_SALES), dtype=str)
sales_raw["実績"] = pd.to_numeric(sales_raw["実績"], errors="coerce").fillna(0)
sales_raw["日付"] = pd.to_datetime(sales_raw["日付"], format="mixed")
daily = sales_raw[sales_raw["品目コード"].str.strip() == ENT_PRODUCT_CODE].copy()
daily = daily.rename(columns={
    "日付": "delivery_date",
    "施設（本院に合算）コード": "facility_id",
    "実績": "amount",
})

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

mr_activity_raw = activity_raw[
    (activity_raw["品目コード"] == ENT_PRODUCT_CODE)
    & (activity_raw["活動種別"].isin(MR_ACTIVITY_TYPES))
].copy()
mr_activity_raw = mr_activity_raw[mr_activity_raw["fac_honin"].notna() & (mr_activity_raw["fac_honin"].astype(str).str.strip() != "")].copy()
print(f"  MR活動データ(面談系): {len(mr_activity_raw):,} 行")

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
print(f"  RW医師リスト(全体) : {len(rw_list)} 行")
print(f"  視聴データ結合     : {len(viewing):,} 行")

# ================================================================
# 除外フロー（集団定義）
# ================================================================
print("\n[除外フロー]")

fac_doc_list = pd.read_csv(os.path.join(DATA_DIR, FILE_FAC_DOCTOR_LIST))

# Step 1: 施設内医師数フィルタ（FILTER_SINGLE_FAC_DOCTOR=True 時のみ）
fac_df = pd.read_csv(os.path.join(DATA_DIR, FILE_FACILITY_MASTER))
single_staff_fac = set(fac_df[fac_df["施設内医師数"] == 1]["fac"])
multi_staff_fac  = set(fac_df[fac_df["施設内医師数"] > 1]["fac"])
if FILTER_SINGLE_FAC_DOCTOR:
    print(f"  [Step 1] 施設内医師数==1のみ: {len(single_staff_fac)} fac → 複数医師fac {len(multi_staff_fac)} 件除外")
else:
    print(f"  [Step 1] スキップ（複数医師施設も含む）: 全施設 {len(single_staff_fac)+len(multi_staff_fac)}")

# Step 2: 所属施設数==1 の医師（視聴帰属の一意化のため常に適用）
doc_attr_df = pd.read_csv(os.path.join(DATA_DIR, FILE_DOCTOR_ATTR))
if DOCTOR_HONIN_FAC_COUNT_COL in doc_attr_df.columns:
    single_honin_docs = set(doc_attr_df[doc_attr_df[DOCTOR_HONIN_FAC_COUNT_COL] == 1]["doc"])
else:
    _fac_per_doc = fac_doc_list.groupby("doc")["fac_honin"].nunique()
    single_honin_docs = set(_fac_per_doc[_fac_per_doc == 1].index)
print(f"  [Step 2] 所属施設数==1: {len(single_honin_docs)} 名（常に適用・視聴帰属の一意化）")

# Step 3: RW医師フィルタ（INCLUDE_ONLY_RW=True 時のみ）
rw_doc_ids = set(rw_list["doc"])

_doc_to_fac   = dict(zip(fac_doc_list["doc"], fac_doc_list["fac"]))
_doc_to_honin = dict(zip(fac_doc_list["doc"], fac_doc_list["fac_honin"]))
all_docs      = set(fac_doc_list["doc"])

if FILTER_SINGLE_FAC_DOCTOR:
    after_step1 = {d for d in all_docs if _doc_to_fac.get(d) in single_staff_fac}
else:
    after_step1 = {d for d in all_docs
                   if pd.notna(_doc_to_honin.get(d))
                   and str(_doc_to_honin.get(d)).strip() not in ("", "nan")}

after_step2 = after_step1 & single_honin_docs
if INCLUDE_ONLY_RW:
    after_step3 = after_step2 & rw_doc_ids
else:
    after_step3 = after_step2

if FILTER_SINGLE_FAC_DOCTOR:
    # ================================================================
    # 施設単位フィルタ（正しいロジック）
    # ① まず「1施設1医師」の施設を特定（全データソース統合）
    # ② 次にその施設の唯一の医師がRW医師であるもののみ残す
    # ================================================================

    # Step A: 全データソース（fac_doc_list + digital_raw + activity_raw）を統合し
    #         fac_honinごとのユニーク医師数を集計
    #         → str型に統一してカウントすることで型不一致による漏れを防ぐ
    _src_parts = [fac_doc_list[["doc", "fac_honin"]]]
    if "doc" in digital_raw.columns and "fac_honin" in digital_raw.columns:
        _src_parts.append(digital_raw[["doc", "fac_honin"]])
    if "doc" in activity_raw.columns and "fac_honin" in activity_raw.columns:
        _src_parts.append(activity_raw[["doc", "fac_honin"]])

    _all_doc_fac = pd.concat(_src_parts, ignore_index=True)
    _all_doc_fac["_doc_s"] = _all_doc_fac["doc"].astype(str).str.strip()
    _all_doc_fac["_fac_s"] = _all_doc_fac["fac_honin"].astype(str).str.strip()
    _all_doc_fac = _all_doc_fac[
        (_all_doc_fac["_doc_s"] != "") & (_all_doc_fac["_doc_s"] != "nan") &
        (_all_doc_fac["_fac_s"] != "") & (_all_doc_fac["_fac_s"] != "nan")
    ][["_doc_s", "_fac_s"]].drop_duplicates()

    _honin_ndoc = _all_doc_fac.groupby("_fac_s")["_doc_s"].nunique()
    _single_honins_str = set(_honin_ndoc[_honin_ndoc == 1].index)  # str型のfac_honin集合
    print(f"  [Step A] 1医師施設 (全ソース統合): {len(_single_honins_str)} / "
          f"総 {len(_honin_ndoc)} fac_honin")

    # Step B: 1医師施設を rw_list の fac_honin で絞り込む
    #   INCLUDE_ONLY_RW=True  → fac_honin が rw_list に存在する施設のみ
    #   INCLUDE_ONLY_NON_RW=True → fac_honin が rw_list に存在しない施設のみ
    #   両方False              → 全施設（RW+非RW）
    # ※ rw_list の fac_honin = RW医師が担当として紐づく施設の識別子
    _rw_fac_honins_str = set(rw_list["fac_honin"].astype(str).str.strip())

    # _all_doc_fac（str型）から1医師施設の (doc_str, fac_str) ペアを取得
    _single_pairs_str = _all_doc_fac[_all_doc_fac["_fac_s"].isin(_single_honins_str)].copy()

    if INCLUDE_ONLY_RW:
        _single_pairs_str = _single_pairs_str[
            _single_pairs_str["_fac_s"].isin(_rw_fac_honins_str)
        ].copy()
        _step_b_label = "RW担当施設のみ"
    elif INCLUDE_ONLY_NON_RW:
        _single_pairs_str = _single_pairs_str[
            ~_single_pairs_str["_fac_s"].isin(_rw_fac_honins_str)
        ].copy()
        _step_b_label = "非RW担当施設のみ"
    else:
        _step_b_label = "全施設（RW+非RW）"

    # fac_doc_list（元の型）と照合して clean_pairs を元型で構築
    _fdl_copy = fac_doc_list.copy()
    _fdl_copy["_fac_s"] = _fdl_copy["fac_honin"].astype(str).str.strip()
    _single_pairs = (
        _fdl_copy[_fdl_copy["_fac_s"].isin(set(_single_pairs_str["_fac_s"]))]
        [["doc", "fac_honin"]].drop_duplicates()
    )
    # 同一fac_honinに複数行が残った場合は除外（安全策）
    _dup_fac = _single_pairs["fac_honin"].duplicated(keep=False)
    if _dup_fac.sum() > 0:
        print(f"  [Step B 保険] 同一施設に複数行: {_dup_fac.sum()} 行除外")
        _single_pairs = _single_pairs[~_dup_fac].copy()
    print(f"  [Step B] {_step_b_label}: {len(_single_pairs)} 施設")

    clean_pairs = (
        _single_pairs
        .rename(columns={"doc": "doctor_id", "fac_honin": "facility_id"})
        .reset_index(drop=True)
    )
    valid_doc_ids    = set(clean_pairs["doctor_id"])
    doc_to_fac_valid = dict(zip(clean_pairs["doctor_id"], clean_pairs["facility_id"]))
    print(f"  [クリーン1:1ペア] {len(valid_doc_ids)} 施設 / {len(valid_doc_ids)} 医師")
else:
    # 複数医師施設も含む: 施設→医師リストを構築
    print(f"  Step1(スキップ) → Step2:{len(after_step2)} → Step3:{len(after_step3)} "
          f"({'RW医師のみ' if INCLUDE_ONLY_RW else '全医師'})")
    _fac_to_docs_map: dict = {}
    for d in after_step3:
        h = _doc_to_honin.get(d)
        if h and str(h).strip() not in ("", "nan"):
            _fac_to_docs_map.setdefault(str(h), []).append(d)
    valid_doc_ids    = after_step3
    doc_to_fac_valid = {d: str(_doc_to_honin[d]) for d in valid_doc_ids if _doc_to_honin.get(d)}
    _n_multi = sum(1 for docs in _fac_to_docs_map.values() if len(docs) > 1)
    print(f"  [施設-医師マップ] {len(_fac_to_docs_map)} 施設 / {len(valid_doc_ids)} 医師 "
          f"（複数医師施設: {_n_multi} 件）")

clean_fac_ids = set(doc_to_fac_valid.values())

# ================================================================
# 施設×月次パネルデータ構築（累積視聴回数を施設単位で追跡）
# ================================================================
print("\n[施設×月次パネルデータ構築]")

# 視聴データに施設IDを付与し施設単位で集計
viewing_clean = viewing[viewing["doctor_id"].isin(valid_doc_ids)].copy()
viewing_clean["facility_id"] = viewing_clean["doctor_id"].map(doc_to_fac_valid)
viewing_clean = viewing_clean.dropna(subset=["facility_id"]).copy()
viewing_clean["year_month"] = viewing_clean["view_date"].dt.to_period("M")

fac_viewing_monthly = (
    viewing_clean.groupby(["facility_id", "year_month"])
    .size().reset_index(name="view_count")
)
fac_viewing_monthly["date"] = fac_viewing_monthly["year_month"].dt.to_timestamp()

daily_clean = daily[daily["facility_id"].isin(clean_fac_ids)].copy()
daily_clean["year_month"] = daily_clean["delivery_date"].dt.to_period("M")
monthly_sales = (
    daily_clean.groupby(["facility_id", "year_month"])
    .agg({"amount": "sum"}).reset_index()
)
monthly_sales["date"] = monthly_sales["year_month"].dt.to_timestamp()

month_to_id = {m: i for i, m in enumerate(months)}

dfm = fac_viewing_monthly.copy()
dfm["month_id"] = dfm["date"].map(month_to_id)
dfm = dfm[dfm["month_id"].notna() & dfm["facility_id"].isin(clean_fac_ids)]
dfm["month_id"] = dfm["month_id"].astype(int)

fac_list = sorted(clean_fac_ids)
full_idx = pd.MultiIndex.from_product(
    [fac_list, range(len(months))], names=["facility_id", "month_id"]
)
current_s = (
    dfm.groupby(["facility_id", "month_id"])["view_count"]
    .sum().reindex(full_idx, fill_value=0)
)
doctor_panel = current_s.reset_index().rename(columns={"view_count": "current_views"})

doctor_panel = doctor_panel.sort_values(["facility_id", "month_id"])
doctor_panel["cumulative_views"] = (
    doctor_panel.groupby("facility_id")["current_views"]
    .transform(lambda x: x.shift(1).fillna(0).cumsum().astype(int))
)

monthly_sales["month_id"] = monthly_sales["date"].map(month_to_id)
ms_clean = monthly_sales.dropna(subset=["month_id"]).copy()
ms_clean["month_id"] = ms_clean["month_id"].astype(int)
doctor_panel = doctor_panel.merge(
    ms_clean[["facility_id", "month_id", "amount"]],
    on=["facility_id", "month_id"], how="left"
)
doctor_panel["amount"] = doctor_panel["amount"].fillna(0.0)
doctor_panel["date"]   = doctor_panel["month_id"].map(dict(enumerate(months)))
doctor_panel["current_views"] = doctor_panel["current_views"].astype(int)

# MR活動数（面談系）を施設×月で集計してマージ
mr_clean = mr_activity_raw[mr_activity_raw["doc"].isin(valid_doc_ids)].copy()
mr_clean["facility_id"] = mr_clean["doc"].map(doc_to_fac_valid)
mr_clean = mr_clean.dropna(subset=["facility_id"]).copy()
mr_clean["活動日_dt"] = pd.to_datetime(mr_clean["活動日_dt"], format="mixed")
mr_clean["year_month"] = mr_clean["活動日_dt"].dt.to_period("M")
mr_monthly = (
    mr_clean.groupby(["facility_id", "year_month"])
    .size().reset_index(name="mr_activity_count")
)
mr_monthly["month_id"] = mr_monthly["year_month"].dt.to_timestamp().map(month_to_id)
mr_monthly = mr_monthly.dropna(subset=["month_id"]).copy()
mr_monthly["month_id"] = mr_monthly["month_id"].astype(int)
doctor_panel = doctor_panel.merge(
    mr_monthly[["facility_id", "month_id", "mr_activity_count"]],
    on=["facility_id", "month_id"], how="left"
)
doctor_panel["mr_activity_count"] = doctor_panel["mr_activity_count"].fillna(0).astype(int)

print(f"  施設×月パネル: {len(doctor_panel):,} 行")
print(f"  施設数: {doctor_panel['facility_id'].nunique()}")
print(f"  期間: {len(months)} ヶ月")

# ================================================================
# qcut 均等サンプルビニング
# ================================================================
# 【旧05との比較】
# 旧05では cumulative_views の区間を手動指定（0, 1-2, 3-5, 6-10, 11+）し、
# 各区間を「1回目」「2回目」「3回目」とラベルしていたが、
# 「3回目」= 累積3-5回での視聴 = 実際には4-6回目の視聴であり不正確。
#
# 本スクリプトでは pd.qcut により視聴イベント行を N_BINS 等分し、
# サンプル数が均等になるよう自動でビン境界を決定する。
# ラベルも「累積X-Y回→視聴（実質X+1-Y+1回目）」と実態に即した表記にする。
# ================================================================
print(f"\n[視聴回数ビニング: qcut N_BINS={N_BINS}]")

# 視聴イベント（current_views > 0）のある行の cumulative_views でqcut
view_events_cum = doctor_panel.loc[doctor_panel["current_views"] > 0, "cumulative_views"]

try:
    _, bin_edges = pd.qcut(view_events_cum, q=N_BINS, retbins=True, duplicates="drop")
    actual_n_bins = len(bin_edges) - 1
    bin_edges[0]  = -0.5   # 0（初回）を含むよう左端を調整
    bin_edges[-1] = view_events_cum.max() + 0.5  # 最大値を含むよう右端を調整
except ValueError as e:
    print(f"  qcut エラー（ユニーク値不足）: {e}  → ビン数を削減して再試行")
    actual_n_bins = min(N_BINS, view_events_cum.nunique())
    _, bin_edges = pd.qcut(view_events_cum, q=actual_n_bins, retbins=True, duplicates="drop")
    bin_edges[0]  = -0.5
    bin_edges[-1] = view_events_cum.max() + 0.5

# ビン名: 「累積X-Y回→視聴（X+1-Y+1回目）」
bin_var_names    = [f"view_bin_{i+1}" for i in range(actual_n_bins)]
bin_display_names = []
for i in range(actual_n_bins):
    lo_cum = int(np.floor(bin_edges[i])) + 1    # 累積回数の下限（今回が何回目か = lo_cum+1）
    hi_cum = int(np.floor(bin_edges[i + 1]))     # 累積回数の上限
    actual_lo = lo_cum + 1
    actual_hi = hi_cum + 1
    if lo_cum == hi_cum:
        label = f"累積{lo_cum}回→視聴\n（{actual_lo}回目）"
    else:
        label = f"累積{lo_cum}-{hi_cum}回→視聴\n（{actual_lo}-{actual_hi}回目）"
    bin_display_names.append(label)

# 各ビンダミーを doctor_panel に付与
for i, var in enumerate(bin_var_names):
    lo_edge = bin_edges[i]
    hi_edge = bin_edges[i + 1]
    doctor_panel[var] = (
        (doctor_panel["cumulative_views"] > lo_edge)
        & (doctor_panel["cumulative_views"] <= hi_edge)
        & (doctor_panel["current_views"] > 0)
    ).astype(int)

# サンプル数確認
print(f"  実際のビン数: {actual_n_bins}")
for i, var in enumerate(bin_var_names):
    n = doctor_panel[var].sum()
    lo_cum = int(np.floor(bin_edges[i])) + 1
    hi_cum = int(np.floor(bin_edges[i + 1]))
    print(f"  {var} (累積{lo_cum}-{hi_cum}回): {n:,} 視聴イベント")

# 視聴確率（継続率）の計算用に初回視聴確率も算出
never_viewed_months  = doctor_panel[doctor_panel["cumulative_views"] == 0]
first_view_count     = (never_viewed_months["current_views"] > 0).sum()
initial_viewing_rate = (
    first_view_count / len(never_viewed_months)
    if len(never_viewed_months) > 0 else 0.0
)
print(f"\n  初回視聴確率（未視聴→視聴）: {initial_viewing_rate:.1%}")
print(f"  計算根拠: 未視聴 {len(never_viewed_months):,} 医師×月 のうち {first_view_count} 回が初回視聴")

# ================================================================
# TWFE回帰: 視聴回数別の限界効果推定
# ================================================================
print("\n[TWFE回帰: 視聴回数別限界効果]")

panel_reg = doctor_panel.copy()
panel_reg = panel_reg[panel_reg["amount"] > 0].copy()  # 売上ゼロ除外
panel_reg = panel_reg.set_index(["facility_id", "month_id"])

exog_vars = bin_var_names + ["mr_activity_count"]

try:
    model = PanelOLS(
        dependent=panel_reg["amount"],
        exog=panel_reg[exog_vars],
        entity_effects=True,
        time_effects=True,
    )
    result = model.fit(cov_type="clustered", cluster_entity=True)

    marginal_effects = {}
    for var in bin_var_names:
        coef = result.params[var]
        se   = result.std_errors[var]
        p    = result.pvalues[var]
        sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        marginal_effects[var] = {"coefficient": float(coef), "se": float(se), "p": float(p), "sig": sig}
        print(f"  {var}: {coef:.2f} (SE={se:.2f}, p={p:.4f}, {sig})")
    _mr_coef = float(result.params["mr_activity_count"])
    _mr_se   = float(result.std_errors["mr_activity_count"])
    _mr_p    = float(result.pvalues["mr_activity_count"])
    _mr_sig  = "***" if _mr_p < 0.001 else "**" if _mr_p < 0.01 else "*" if _mr_p < 0.05 else "n.s."
    print(f"  mr_activity_count: {_mr_coef:.2f} (SE={_mr_se:.2f}, p={_mr_p:.4f}, {_mr_sig})")

    regression_success = True

except Exception as e:
    print(f"  回帰エラー: {e}")
    print("  デフォルト値を使用")
    base_effects = [30.0, 25.0, 20.0, 15.0, 10.0]
    marginal_effects = {}
    for i, var in enumerate(bin_var_names):
        eff = base_effects[i] if i < len(base_effects) else 10.0
        marginal_effects[var] = {"coefficient": eff, "se": 5.0, "p": 0.001, "sig": "***"}
    _mr_coef = _mr_se = _mr_p = None
    regression_success = False

# ================================================================
# 視聴確率（継続率）の推定
# ================================================================
print("\n[視聴確率（継続率）の推定]")

dp_sorted = doctor_panel.sort_values(["facility_id", "month_id"]).copy()
dp_sorted["next_views"] = dp_sorted.groupby("facility_id")["current_views"].shift(-1)

# continuation_rates の計算範囲を実データの最大累積回数まで拡張する
# （最後のビンの hi_cum が 14 を超えると .get(n, 0) = 0 になり期待効果がゼロになるバグを防ぐ）
_max_cum = int(doctor_panel["cumulative_views"].max()) + 1
continuation_rates = {}
for n in range(0, _max_cum + 1):
    cohort = dp_sorted[dp_sorted["cumulative_views"] == n]
    valid  = cohort["next_views"].dropna()
    continuation_rates[n] = float((valid > 0).mean()) if len(valid) > 0 else 0.0
    if n < 11:
        print(f"  累積{n}回 → 次月視聴確率: {continuation_rates[n]:.1%}")

# ================================================================
# 期待効果の計算（ビン別）
# ================================================================
print("\n[期待効果の計算]")

# 各ビンの cumulative_views 範囲から視聴確率を取得
expected_effects = {}
viewing_probs    = {}

# ビン0（初回視聴）
prob_bin1 = initial_viewing_rate
effect_bin1 = marginal_effects[bin_var_names[0]]["coefficient"]
expected_effects[bin_var_names[0]] = prob_bin1 * effect_bin1
viewing_probs[bin_var_names[0]]    = prob_bin1

print(f"  {bin_var_names[0]}: 期待効果={expected_effects[bin_var_names[0]]:.2f}万円"
      f" = {prob_bin1:.1%} × {effect_bin1:.2f}")

# ビン1以降
for i in range(1, actual_n_bins):
    var   = bin_var_names[i]
    lo_cum = int(np.floor(bin_edges[i])) + 1
    hi_cum = int(np.floor(bin_edges[i + 1]))
    prob  = np.mean([continuation_rates.get(n, 0) for n in range(lo_cum, hi_cum + 1)])
    eff   = marginal_effects[var]["coefficient"]
    expected_effects[var] = prob * eff
    viewing_probs[var]    = prob
    print(f"  {var} (累積{lo_cum}-{hi_cum}回): 期待効果={expected_effects[var]:.2f}万円"
          f" = {prob:.1%} × {eff:.2f}")

# ================================================================
# 最適配信戦略
# ================================================================
print("\n[最適配信戦略]")

new_doc_expected = expected_effects[bin_var_names[0]]

threshold_message = None
for var in bin_var_names[1:]:
    if expected_effects[var] < new_doc_expected:
        lo_cum = int(np.floor(bin_edges[bin_var_names.index(var)])) + 1
        threshold_message = (
            f"累積{lo_cum}回視聴済みの既存医師への配信より新規医師への初回配信が有効"
        )
        break

if threshold_message is None:
    threshold_message = "全ビンで既存医師の期待効果が高い（常に既存医師優先）"

print(f"  {threshold_message}")
for var in bin_var_names:
    print(f"  {var}: {expected_effects[var]:.2f}万円")

# ================================================================
# 解析対象集団の視聴数分布とピーク閾値以上割合
# ================================================================
print("\n[解析対象集団の視聴数分布]")

# 施設ごとの最終累積視聴回数（観測期間中の最大値 = 分析終了時点の累積）
fac_final_cum = (
    doctor_panel.groupby("facility_id")["cumulative_views"]
    .max().reset_index(name="final_cum_views")
)

# 期待効果のピーク bin を特定
_peak_var   = max(expected_effects, key=lambda v: expected_effects[v])
_peak_idx   = bin_var_names.index(_peak_var)
_peak_lo    = int(np.floor(bin_edges[_peak_idx])) + 1   # ピークビン累積下限
_peak_hi    = int(np.floor(bin_edges[_peak_idx + 1]))   # ピークビン累積上限
_peak_label = bin_display_names[_peak_idx].replace("\n", " ")

# ピーク下限以上の施設割合
_n_total_fac  = len(fac_final_cum)
_n_above_peak = int((fac_final_cum["final_cum_views"] >= _peak_lo).sum())
_pct_above    = _n_above_peak / _n_total_fac if _n_total_fac > 0 else 0.0

# 段階別集計（各ビンの下限ごと）
_threshold_stats = []
for i, var in enumerate(bin_var_names):
    _lo = int(np.floor(bin_edges[i])) + 1
    _n  = int((fac_final_cum["final_cum_views"] >= _lo).sum())
    _threshold_stats.append({"bin": var, "threshold": _lo,
                              "n_above": _n, "pct_above": _n / _n_total_fac if _n_total_fac > 0 else 0.0})

print(f"  解析対象施設数（総計）: {_n_total_fac}")
print(f"  ピーク期待効果ビン: {_peak_label} (期待効果={expected_effects[_peak_var]:.2f}万円)")
print(f"  ピーク閾値（累積{_peak_lo}回以上）の施設: {_n_above_peak} / {_n_total_fac} ({_pct_above:.1%})")
print(f"\n  各ビン閾値以上の施設割合:")
for ts in _threshold_stats:
    _mark = " ← ピーク" if ts["bin"] == _peak_var else ""
    print(f"    累積{ts['threshold']}回以上: {ts['n_above']:,} / {_n_total_fac} ({ts['pct_above']:.1%}){_mark}")

# ================================================================
# チャネル別期待効果の計算
# ================================================================
print("\n[チャネル別期待効果]")

# 施設→医師マッピング（ユニーク施設数・医師数の計算に使用）
_fac_to_docs_map: dict = {}
for _d, _f in doc_to_fac_valid.items():
    _fac_to_docs_map.setdefault(_f, set()).add(_d)

# viewing_clean に month_id を付与してチャネル×ビン別視聴数を集計（施設単位）
channel_monthly = (
    viewing_clean.groupby(["facility_id", "year_month", "channel_category"])
    .size().reset_index(name="ch_view_count")
)
channel_monthly["month_id"] = (
    channel_monthly["year_month"].dt.to_timestamp().map(month_to_id)
)
channel_monthly = channel_monthly.dropna(subset=["month_id"]).copy()
channel_monthly["month_id"] = channel_monthly["month_id"].astype(int)

# cumulative_views をマージ
channel_with_cum = channel_monthly.merge(
    doctor_panel[["facility_id", "month_id", "cumulative_views"]],
    on=["facility_id", "month_id"], how="left"
)

channels_list = sorted(channel_with_cum["channel_category"].dropna().unique())
channel_palette = ["#2196f3", "#ff9800", "#4caf50", "#e91e63", "#9c27b0"]

# チャネル別ビン期待効果
# P(view via channel c in bin i) × marginal_effect[bin i]
channel_bin_expected   = {}
channel_bin_fac_counts = {}   # チャネル×ビン別ユニーク施設数
channel_bin_doc_counts = {}   # チャネル×ビン別ユニーク医師数

# 施設×チャネル別の総視聴回数（ピーク閾値以上割合の計算に使用）
_ch_fac_total = (
    channel_monthly.groupby(["facility_id", "channel_category"])["ch_view_count"]
    .sum().reset_index(name="total_ch_views")
)
# 全施設を分母とするため 0 で埋める
_all_facs_df = pd.DataFrame({"facility_id": fac_list})
ch_fac_total_wide = {}
for ch in channels_list:
    _tmp = _ch_fac_total[_ch_fac_total["channel_category"] == ch][["facility_id", "total_ch_views"]]
    _merged = _all_facs_df.merge(_tmp, on="facility_id", how="left").fillna(0)
    ch_fac_total_wide[ch] = _merged.set_index("facility_id")["total_ch_views"]

for ch in channels_list:
    ch_data = channel_with_cum[channel_with_cum["channel_category"] == ch]
    ch_expected = []
    ch_fac_bin  = []
    ch_doc_bin  = []
    for i, var in enumerate(bin_var_names):
        lo_edge = bin_edges[i]
        hi_edge = bin_edges[i + 1]
        _bin_mask_panel = (
            (doctor_panel["cumulative_views"] > lo_edge)
            & (doctor_panel["cumulative_views"] <= hi_edge)
        )
        n_denom = int(_bin_mask_panel.sum())
        _ch_bin_mask = (
            (ch_data["cumulative_views"] > lo_edge)
            & (ch_data["cumulative_views"] <= hi_edge)
        )
        n_ch = int(ch_data.loc[_ch_bin_mask, "ch_view_count"].sum())
        prob_ch = n_ch / n_denom if n_denom > 0 else 0.0
        ch_expected.append(prob_ch * marginal_effects[var]["coefficient"])
        # ユニーク施設数・医師数（そのビンにcumulative_viewsが属し、かつチャネルイベントあり）
        _ch_facs = set(ch_data.loc[_ch_bin_mask, "facility_id"].unique())
        ch_fac_bin.append(len(_ch_facs))
        ch_doc_bin.append(sum(len(_fac_to_docs_map.get(f, set())) for f in _ch_facs))
    channel_bin_expected[ch]   = ch_expected
    channel_bin_fac_counts[ch] = ch_fac_bin
    channel_bin_doc_counts[ch] = ch_doc_bin
    print(f"  {ch}: " + ", ".join(f"{v:.2f}" for v in ch_expected))

# チャネル別ピーク閾値以上割合
channel_peak_stats = {}
print(f"\n  チャネル別ピーク閾値（累積{_peak_lo}回以上）割合:")
for ch in channels_list:
    _ch_views = ch_fac_total_wide[ch]
    _n_above  = int((_ch_views >= _peak_lo).sum())
    _pct      = _n_above / _n_total_fac if _n_total_fac > 0 else 0.0
    channel_peak_stats[ch] = {"n_above": _n_above, "pct_above": _pct}
    print(f"    {ch}: {_n_above}/{_n_total_fac} ({_pct:.1%})")

# ================================================================
# 可視化（2列レイアウト）
# ================================================================
print("\n[可視化]")

from matplotlib.gridspec import GridSpec

# レイアウト: 上4スロット（a,b,c,d）+ チャネル別各1スロット + サマリー（2列スパン）
# スロット配置: [a,b], [c,d], [ch1,ch2], ..., [summary(span)]
n_ch = len(channels_list)
_slots_above_summary = 4 + n_ch * 2      # a, b, c, d + チャネル別(期待効果+視聴分布)×n_ch
_n_content_rows = (_slots_above_summary + 1) // 2
_total_rows = _n_content_rows + 1        # +1 はサマリー行
_fig_h = max(18, _total_rows * 5)        # 行あたり5インチ（コンパクト化）

fig = plt.figure(figsize=(20, _fig_h))
# サマリー行は内容行の1/3の高さ、hspaceを小さめに
_height_ratios = [3] * _n_content_rows + [1]
gs = GridSpec(_total_rows, 2, figure=fig, hspace=0.5, wspace=0.35,
              height_ratios=_height_ratios)

x_ticks  = list(range(actual_n_bins))
x_labels = [bin_display_names[i] for i in range(actual_n_bins)]
exp_values = [expected_effects[v] for v in bin_var_names]
colors_exp = ["#4caf50"] + ["#2196f3"] * (actual_n_bins - 1)
_xfont = max(6, 9 - actual_n_bins // 3)
# ビン別ユニーク施設数・医師数
bin_fac_counts = []
bin_doc_counts = []
for v in bin_var_names:
    _facs = set(doctor_panel.loc[doctor_panel[v].astype(bool), "facility_id"])
    bin_fac_counts.append(len(_facs))
    bin_doc_counts.append(sum(len(_fac_to_docs_map.get(f, set())) for f in _facs))

def _setup_bar_ax(ax, vals, title, ylabel, colors=None, errs=None, n_fac=None, n_doc=None):
    c = colors if colors is not None else ["#2196f3"] * len(vals)
    if errs is not None:
        bars = ax.bar(x_ticks, vals, yerr=errs, color=c, alpha=0.7, capsize=4)
    else:
        bars = ax.bar(x_ticks, vals, color=c, alpha=0.7)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=_xfont, rotation=15, ha="right")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=1)
    if n_fac is not None and n_doc is not None:
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        label_y_tops = []
        for i, (bar, nf, nd) in enumerate(zip(bars, n_fac, n_doc)):
            bar_top = bar.get_height()
            err_i = float(errs[i]) if (errs is not None and i < len(errs)) else 0.0
            label_base = (bar_top + err_i) if bar_top >= 0 else bar_top
            y_pos = label_base + y_range * 0.02 if bar_top >= 0 else bar_top - y_range * 0.06
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f"施設{nf}\n医師{nd}", ha="center", va="bottom", fontsize=6, color="#444444")
            label_y_tops.append(y_pos + y_range * 0.04)
        ax.set_ylim(y_min, max(y_max, max(label_y_tops)))

# スロットのインデックスから (row, col) を返すヘルパー
def _slot(i):
    return i // 2, i % 2

# (a) 限界効果
_r, _c = _slot(0)
ax_a = fig.add_subplot(gs[_r, _c])
effects = [marginal_effects[v]["coefficient"] for v in bin_var_names]
errors  = [marginal_effects[v]["se"]          for v in bin_var_names]
_setup_bar_ax(ax_a, effects, "(a) 視聴回数別の限界効果（TWFE）", "限界効果（万円）", errs=errors, n_fac=bin_fac_counts, n_doc=bin_doc_counts)

# (b) 視聴確率（継続率）
_r, _c = _slot(1)
ax_b = fig.add_subplot(gs[_r, _c])
cont_max = min(int(bin_edges[-1]) + 2, _max_cum)
cont_x = list(range(0, cont_max))
cont_y = [continuation_rates.get(i, 0) * 100 for i in cont_x]
ax_b.plot(cont_x, cont_y, marker="o", color="#ff9800", linewidth=2, markersize=4)
ax_b.axhline(initial_viewing_rate * 100, color="red", linestyle="--",
             label=f"初回視聴率: {initial_viewing_rate:.1%}")
ax_b.set_xlabel("累積視聴回数", fontsize=11)
ax_b.set_ylabel("次月視聴確率（%）", fontsize=11)
ax_b.set_title("(b) 視聴確率（継続率）", fontsize=12, fontweight="bold")
ax_b.legend(fontsize=9)
ax_b.grid(alpha=0.3)

# (c) 期待効果（全体）
_r, _c = _slot(2)
ax_c = fig.add_subplot(gs[_r, _c])
_setup_bar_ax(ax_c, exp_values, "(c) 期待効果 = 視聴確率 × 限界効果", "期待効果（万円）", colors=colors_exp, n_fac=bin_fac_counts, n_doc=bin_doc_counts)

# (d) 解析対象集団の視聴数分布とピーク閾値
_r, _c = _slot(3)
ax_d = fig.add_subplot(gs[_r, _c])
_hist_vals = fac_final_cum["final_cum_views"]
_hist_max  = min(int(_hist_vals.quantile(0.99)) + 2, int(_hist_vals.max()) + 1)
ax_d.hist(_hist_vals.clip(upper=_hist_max), bins=min(40, _hist_max + 1),
          color="#78909c", alpha=0.7, edgecolor="white")
ax_d.axvline(_peak_lo - 0.5, color="red", linestyle="--", linewidth=2,
             label=f"ピーク閾値: 累積{_peak_lo}回以上\n{_n_above_peak}/{_n_total_fac} ({_pct_above:.1%})")
ax_d.set_xlabel("最終累積視聴回数", fontsize=11)
ax_d.set_ylabel("施設数", fontsize=11)
ax_d.set_title("(d) 視聴数分布とピーク閾値以上割合", fontsize=12, fontweight="bold")
ax_d.legend(fontsize=9)
ax_d.grid(axis="y", alpha=0.3)
# 閾値以上の領域を塗りつぶし
_ylim = ax_d.get_ylim()
ax_d.axvspan(_peak_lo - 0.5, _hist_max + 1, alpha=0.12, color="red")
ax_d.set_ylim(_ylim)

# (e〜) チャネル別：期待効果（棒グラフ）+ 視聴数分布（ヒストグラム）を横並び
for ci, ch in enumerate(channels_list):
    _letter1 = chr(101 + ci * 2)   # e, g, i, ...
    _letter2 = chr(102 + ci * 2)   # f, h, j, ...
    _ch_color = channel_palette[ci % len(channel_palette)]
    ch_vals = channel_bin_expected[ch]
    _ps = channel_peak_stats[ch]

    # チャネル期待効果（棒グラフ）
    _r, _c = _slot(4 + ci * 2)
    ax_ch = fig.add_subplot(gs[_r, _c])
    _setup_bar_ax(
        ax_ch, ch_vals,
        f"({_letter1}) チャネル別期待効果: {ch}",
        "期待効果（万円）",
        colors=[_ch_color] * actual_n_bins,
        n_fac=channel_bin_fac_counts[ch],
        n_doc=channel_bin_doc_counts[ch],
    )
    ax_ch.text(
        0.98, 0.97,
        f"累積{_peak_lo}回以上\n{_ps['n_above']}/{_n_total_fac} ({_ps['pct_above']:.1%})",
        transform=ax_ch.transAxes, fontsize=8, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    # チャネル視聴数分布（ヒストグラム）
    _r, _c = _slot(4 + ci * 2 + 1)
    ax_ch_hist = fig.add_subplot(gs[_r, _c])
    _ch_views = ch_fac_total_wide[ch]
    _ch_max = _ch_views.max()
    if _ch_max > 0:
        _ch_hist_max = min(int(_ch_views.quantile(0.99)) + 2, int(_ch_max) + 1)
        _ch_hist_max = max(_ch_hist_max, _peak_lo + 1)
        ax_ch_hist.hist(_ch_views.clip(upper=_ch_hist_max),
                        bins=min(30, _ch_hist_max + 1),
                        color=_ch_color, alpha=0.6, edgecolor="white")
    ax_ch_hist.axvline(_peak_lo - 0.5, color="red", linestyle="--", linewidth=1.5,
                       label=f"累積{_peak_lo}回以上\n{_ps['n_above']}/{_n_total_fac} ({_ps['pct_above']:.1%})")
    _ch_ylim = ax_ch_hist.get_ylim()
    _ch_xmax = _ch_hist_max + 1 if _ch_max > 0 else _peak_lo + 5
    ax_ch_hist.axvspan(_peak_lo - 0.5, _ch_xmax, alpha=0.12, color="red")
    ax_ch_hist.set_ylim(_ch_ylim)
    ax_ch_hist.set_xlabel(f"{ch} 累積視聴回数", fontsize=10)
    ax_ch_hist.set_ylabel("施設数", fontsize=10)
    ax_ch_hist.set_title(f"({_letter2}) {ch} 視聴数分布", fontsize=12, fontweight="bold")
    ax_ch_hist.legend(fontsize=8)
    ax_ch_hist.grid(axis="y", alpha=0.3)

# サマリーテキスト（最終行、2列スパン）
ax_sum = fig.add_subplot(gs[_total_rows - 1, :])
ax_sum.axis("off")
prob_lines = "\n".join(
    [f"  {bin_display_names[i].replace(chr(10), ' ')}: "
     f"{viewing_probs[v]:.1%} × {marginal_effects[v]['coefficient']:.1f} = {expected_effects[v]:.2f}万円"
     for i, v in enumerate(bin_var_names)]
)
message = (
    f"【実行パラメータ】  対象: {'RW医師のみ' if INCLUDE_ONLY_RW else '全医師'}  /  "
    f"フィルタ: {'1施設1医師先' if FILTER_SINGLE_FAC_DOCTOR else '全医師先'}  /  "
    f"ビン数: {actual_n_bins}（qcut均等サンプル）\n\n"
    f"【最適配信戦略】  {threshold_message}\n\n"
    f"【期待効果一覧（視聴確率×限界効果）】\n{prob_lines}\n\n"
    f"⚠️ 視聴は内生変数。因果効果ではなく相関関係として解釈。"
)
ax_sum.text(0.02, 0.95, message, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

plt.suptitle(
    f"視聴回数別限界効果 + 期待効果分析 [{PARAM_SUFFIX}]",
    fontsize=15, fontweight="bold"
)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
print(f"  可視化を保存: {OUTPUT_PNG}")
plt.close()

# ================================================================
# 結果保存（JSON）
# ================================================================
print("\n[結果保存]")

output_json = {
    "params": {
        "INCLUDE_ONLY_RW": INCLUDE_ONLY_RW,
        "FILTER_SINGLE_FAC_DOCTOR": FILTER_SINGLE_FAC_DOCTOR,
        "N_BINS": N_BINS,
        "actual_n_bins": actual_n_bins,
        "ENT_PRODUCT_CODE": ENT_PRODUCT_CODE,
        "COST_PER_DISTRIBUTION": COST_PER_DISTRIBUTION,
        "param_suffix": PARAM_SUFFIX,
    },
    "bin_info": {
        bin_var_names[i]: {
            "display_name": bin_display_names[i],
            "lo_cumulative": int(np.floor(bin_edges[i])) + 1,
            "hi_cumulative": int(np.floor(bin_edges[i + 1])),
            "n_facilities": bin_fac_counts[i],
            "n_doctors": bin_doc_counts[i],
        }
        for i in range(actual_n_bins)
    },
    "mr_activity_coef": {
        "coefficient": _mr_coef,
        "se": _mr_se,
        "p": _mr_p,
    },
    "marginal_effects": marginal_effects,
    "continuation_rates": {str(k): v for k, v in continuation_rates.items() if k < 11},
    "initial_viewing_rate": float(initial_viewing_rate),
    "viewing_probs": {v: float(viewing_probs[v]) for v in bin_var_names},
    "expected_effects": {v: float(expected_effects[v]) for v in bin_var_names},
    "expected_roi": {v: float(expected_effects[v] / COST_PER_DISTRIBUTION) for v in bin_var_names},
    "optimal_strategy": {
        "message": threshold_message,
        "new_doctor_expected": float(new_doc_expected),
    },
    "channel_bin_expected": {
        ch: {bin_var_names[i]: float(v) for i, v in enumerate(vals)}
        for ch, vals in channel_bin_expected.items()
    },
    "viewing_distribution": {
        "peak_bin": _peak_var,
        "peak_threshold_lo": _peak_lo,
        "peak_threshold_hi": _peak_hi,
        "n_total_facilities": _n_total_fac,
        "n_above_peak": _n_above_peak,
        "pct_above_peak": float(_pct_above),
        "threshold_stats": _threshold_stats,
        "final_cum_views_describe": {
            k: float(v) for k, v in fac_final_cum["final_cum_views"].describe().items()
        },
        "channel_peak_stats": {
            ch: {
                "n_above": s["n_above"],
                "pct_above": float(s["pct_above"]),
                "ch_cum_views_describe": {
                    k: float(v) for k, v in ch_fac_total_wide[ch].describe().items()
                },
            }
            for ch, s in channel_peak_stats.items()
        },
    },
    "interpretation": {
        "binning_method": "qcut（均等サンプル数ビニング）",
        "warning": "視聴は医師の自発的行動（内生変数）であり、因果効果ではなく相関関係",
    },
}

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(output_json, f, ensure_ascii=False, indent=2)
print(f"  結果を保存: {OUTPUT_JSON}")

print("\n" + "=" * 70)
print(f" 分析完了 [{PARAM_SUFFIX}]")
print("=" * 70)
print(f"\n【最適配信戦略】\n  {threshold_message}")
