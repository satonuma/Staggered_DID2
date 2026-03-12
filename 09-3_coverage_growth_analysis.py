"""
===================================================================
09-3_coverage_growth_analysis.py
Coverage（施設内視聴率）と売上伸長の関係分析
===================================================================
【分析の目的】
施設内の何割の医師が視聴したか（Coverage）が
売上伸長率と関係するかを用量反応的に分析する。

【重要な設計方針】
1施設1医師の施設ではCoverageが必ず0% or 100% になり
連続変数としての意味がないため、複数医師施設（n_docs >= 2）のみを対象とする。

PSMとは独立した分析として、Coverage × 実績伸長の相関・回帰を実施。
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
ENT_PRODUCT_CODE       = "00001"
ACTIVITY_CHANNEL_FILTER = "Web講演会"
CONTENT_TYPES          = ["webiner", "e_contents", "Web講演会"]

FILE_RW_LIST          = "rw_list.csv"
FILE_SALES            = "sales.csv"
FILE_DIGITAL          = "デジタル視聴データ.csv"
FILE_ACTIVITY         = "活動データ.csv"
FILE_DOCTOR_ATTR      = "doctor_attribute.csv"
FILE_FACILITY_MASTER  = "facility_attribute_修正.csv"
FILE_FAC_DOCTOR_LIST  = "施設医師リスト.csv"

# 複数医師施設のみ対象（1施設1医師はCoverageがバイナリになるため除外）
MIN_DOCS_PER_FAC = 2

INCLUDE_ONLY_RW     = False
INCLUDE_ONLY_NON_RW = False

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

PRE_END    = 11   # month_index 0–11（12ヶ月）
POST_START = 12   # month_index 12–32（21ヶ月）

UHP_RANK = {"U": 0, "H": 1, "P": 2, "雑": 3}

print("=" * 70)
print(" 09-3: Coverage（施設内視聴率）× 売上伸長分析")
print(f"       対象: 複数医師施設のみ（{MIN_DOCS_PER_FAC}名以上）")
print("=" * 70)

# ===================================================================
# [1] データ読み込み
# ===================================================================
print("\n[1] データ読み込み")

rw_list   = pd.read_csv(os.path.join(DATA_DIR, FILE_RW_LIST))

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
daily["month_index"] = (
    (daily["delivery_date"].dt.year - 2023) * 12
    + daily["delivery_date"].dt.month - 4
)

digital_raw = pd.read_csv(os.path.join(DATA_DIR, FILE_DIGITAL))
digital_raw["品目コード"] = digital_raw["品目コード"].astype(str).str.strip().str.zfill(5)
digital = digital_raw[digital_raw["品目コード"] == ENT_PRODUCT_CODE].copy()
digital = digital[
    digital["fac_honin"].notna()
    & (digital["fac_honin"].astype(str).str.strip() != "")
].copy()

activity_raw = pd.read_csv(os.path.join(DATA_DIR, FILE_ACTIVITY))
activity_raw["品目コード"] = activity_raw["品目コード"].astype(str).str.strip().str.zfill(5)
web_lecture = activity_raw[
    (activity_raw["品目コード"] == ENT_PRODUCT_CODE)
    & (activity_raw["活動種別"] == ACTIVITY_CHANNEL_FILTER)
].copy()
web_lecture = web_lecture[
    web_lecture["fac_honin"].notna()
    & (web_lecture["fac_honin"].astype(str).str.strip() != "")
].copy()

common_cols = ["活動日_dt", "品目コード", "活動種別", "fac_honin", "doc"]
viewing = pd.concat([digital[common_cols], web_lecture[common_cols]], ignore_index=True)
viewing = viewing.rename(columns={
    "活動日_dt": "view_date", "fac_honin": "facility_id", "doc": "doc"
})
viewing["view_date"] = pd.to_datetime(viewing["view_date"], format="mixed")
viewing["month_index"] = (
    (viewing["view_date"].dt.year - 2023) * 12
    + viewing["view_date"].dt.month - 4
)

print("  データ読み込み完了")

# ===================================================================
# [2] 施設-医師マッピングと複数医師施設の絞り込み
# ===================================================================
print("\n[2] 施設-医師マッピング（複数医師施設フィルタ）")

fac_doc_list = pd.read_csv(os.path.join(DATA_DIR, FILE_FAC_DOCTOR_LIST))
fac_doc_list["fac_honin"] = fac_doc_list["fac_honin"].astype(str).str.strip()

fac_df = pd.read_csv(os.path.join(DATA_DIR, FILE_FACILITY_MASTER))
fac_df["fac_honin"] = fac_df["fac_honin"].astype(str).str.strip()

doc_attr_df = pd.read_csv(os.path.join(DATA_DIR, FILE_DOCTOR_ATTR))

all_docs = set(fac_doc_list["doc"])

# 医師フィルタ
rw_doc_ids = set(rw_list["doc"])
if INCLUDE_ONLY_RW:
    analysis_docs_all = all_docs & rw_doc_ids
    print(f"  RWフィルタ適用: {len(analysis_docs_all)} 名")
elif INCLUDE_ONLY_NON_RW:
    analysis_docs_all = all_docs - rw_doc_ids
    print(f"  非RWフィルタ適用: {len(analysis_docs_all)} 名")
else:
    analysis_docs_all = all_docs
    print(f"  全医師対象: {len(analysis_docs_all)} 名")

# 主施設割り当て（複数施設所属医師は平均納入額最大施設に割り当て）
_doc_fac_list  = fac_doc_list[["doc", "fac_honin"]].drop_duplicates()
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

_fac_uhp = (
    fac_df.drop_duplicates("fac_honin").set_index("fac_honin")["UHP区分名称"]
    if "UHP区分名称" in fac_df.columns else pd.Series(dtype=str)
)
_zero_sum = _doc_fac_sales.groupby("doc")["avg_sales"].sum()
_zero_docs_set = set(_zero_sum[_zero_sum == 0].index)
_multi_assign = (
    _doc_fac_sales.sort_values("avg_sales", ascending=False)
    .groupby("doc")["fac_honin"].first()
)
if _zero_docs_set:
    _zero_df = _doc_fac_sales[_doc_fac_sales["doc"].isin(_zero_docs_set)].copy()
    _zero_df["_uhp_rank"] = _zero_df["fac_honin"].map(_fac_uhp).map(
        lambda x: UHP_RANK.get(str(x), 99) if pd.notna(x) else 99
    )
    _zero_best = _zero_df.sort_values("_uhp_rank").groupby("doc")["fac_honin"].first()
    _multi_assign.update(_zero_best)

doc_primary_fac = pd.concat([_single_assign, _multi_assign])

# 解析対象医師のみに絞って施設→医師リスト構築
_prim_filt = doc_primary_fac[doc_primary_fac.index.isin(analysis_docs_all)]
_prim_df   = pd.DataFrame({"doc": _prim_filt.index, "fac": _prim_filt.values})
fac_to_docs_all = _prim_df.groupby("fac")["doc"].agg(list).to_dict()

# 複数医師施設のみに絞り込み（メイン設定）
fac_to_docs = {
    fac: docs for fac, docs in fac_to_docs_all.items()
    if len(docs) >= MIN_DOCS_PER_FAC
}
n_docs_map = {fac: len(docs) for fac, docs in fac_to_docs.items()}

n_single = sum(1 for docs in fac_to_docs_all.values() if len(docs) == 1)
n_multi  = len(fac_to_docs)
print(f"  1医師施設: {n_single} 件（除外）")
print(f"  複数医師施設（{MIN_DOCS_PER_FAC}名以上）: {n_multi} 件 → 解析対象")
print(f"  医師数分布: 最小={min(n_docs_map.values())}, 最大={max(n_docs_map.values())}, "
      f"平均={np.mean(list(n_docs_map.values())):.1f}")

# ===================================================================
# [3] 視聴データへの主施設付与 & ウォッシュアウト処理
# ===================================================================
print("\n[3] 視聴データ処理")

viewing_all = viewing[viewing["doc"].isin(analysis_docs_all)].copy()
viewing_all["facility_id"] = viewing_all["doc"].map(doc_primary_fac)

# ウォッシュアウト除外施設（解析開始前に視聴あり）
washout_fac_ids = set(
    viewing_all[
        (viewing_all["month_index"] < WASHOUT_MONTHS)
        & viewing_all["facility_id"].notna()
    ]["facility_id"]
)
print(f"  ウォッシュアウト除外施設: {len(washout_fac_ids)} 件")

# 解析期間（WASHOUT_MONTHS〜LAST_ELIGIBLE_MONTH）での視聴
viewing_analysis = viewing_all[
    viewing_all["month_index"].between(WASHOUT_MONTHS, LAST_ELIGIBLE_MONTH)
    & viewing_all["facility_id"].notna()
    & ~viewing_all["facility_id"].isin(washout_fac_ids)
    & viewing_all["facility_id"].isin(fac_to_docs)
].copy()

# 処置施設・対照施設の特定
first_view = (
    viewing_analysis
    .groupby("facility_id")["month_index"].min()
    .rename("first_view_month")
)
treated_fac_ids = set(first_view.index)
control_fac_ids = set(fac_to_docs.keys()) - treated_fac_ids - washout_fac_ids
analysis_fac_ids = treated_fac_ids | control_fac_ids

print(f"  処置群（視聴あり）: {len(treated_fac_ids)} 施設")
print(f"  対照群（未視聴）  : {len(control_fac_ids)} 施設")
print(f"  合計解析施設      : {len(analysis_fac_ids)} 施設")

# ===================================================================
# [4] Coverage（施設内視聴率）計算
# ===================================================================
print("\n[4] Coverage 計算")

# 解析期間中に視聴した医師のユニーク数 / 施設の総医師数
viewed_docs_per_fac = (
    viewing_analysis
    .groupby("facility_id")["doc"].nunique()
    .rename("n_viewed_docs")
)
total_docs_ser = pd.Series(
    {fac: len(docs) for fac, docs in fac_to_docs.items()}
)

coverage_ser = (viewed_docs_per_fac / total_docs_ser).clip(0, 1).rename("coverage")

# 対照群は Coverage = 0
coverage_all = pd.Series(0.0, index=sorted(analysis_fac_ids), name="coverage")
coverage_all.update(coverage_ser)

print(f"  処置群 Coverage: mean={coverage_all[list(treated_fac_ids)].mean():.3f}, "
      f"min={coverage_all[list(treated_fac_ids)].min():.3f}, "
      f"max={coverage_all[list(treated_fac_ids)].max():.3f}")
print(f"  処置群 Coverage=1.0（全員視聴）: "
      f"{(coverage_all[list(treated_fac_ids)] == 1.0).sum()} 施設")
print(f"  Coverage分布（処置群）:")
_cov_desc = coverage_all[list(treated_fac_ids)].describe()
for _k, _v in _cov_desc.items():
    print(f"    {_k}: {_v:.3f}")

# ===================================================================
# [5] 売上前後期集計
# ===================================================================
print("\n[5] 売上前後期集計")

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

unit_df = (
    pd.DataFrame({"facility_id": sorted(analysis_fac_ids)})
    .set_index("facility_id")
    .join(pre_avg).join(post_avg).reset_index()
)
unit_df["growth"]      = unit_df["post_mean"] - unit_df["pre_mean"]
unit_df["growth_rate"] = np.where(
    unit_df["pre_mean"] > 0,
    (unit_df["post_mean"] - unit_df["pre_mean"]) / unit_df["pre_mean"] * 100,
    np.nan,
)
unit_df["treated"]  = unit_df["facility_id"].isin(treated_fac_ids).astype(int)
unit_df["coverage"] = unit_df["facility_id"].map(coverage_all).fillna(0.0)
unit_df["n_docs"]   = unit_df["facility_id"].map(n_docs_map).fillna(MIN_DOCS_PER_FAC).astype(int)

# 施設属性マージ
fac_df2  = fac_df.rename(columns={"fac_honin": "facility_id"})
fac_cols = [c for c in fac_df2.columns if c not in {"facility_id", "fac", "fac_honin_name"}]
unit_df  = unit_df.merge(fac_df2[["facility_id"] + fac_cols], on="facility_id", how="left")

print(f"  unit_df: {len(unit_df)} 施設")
print(f"  growth_rate 有効施設（前期売上>0）: {unit_df['growth_rate'].notna().sum()}")
print(f"  処置群 growth_rate mean: "
      f"{unit_df.loc[unit_df['treated']==1, 'growth_rate'].mean():.2f}%")
print(f"  対照群 growth_rate mean: "
      f"{unit_df.loc[unit_df['treated']==0, 'growth_rate'].mean():.2f}%")

# ===================================================================
# [6] Coverage × 売上伸長率 回帰分析（処置群のみ）
# ===================================================================
print("\n[6] Coverage × 売上伸長率 回帰分析")

# 処置群のみ・前期売上あり
treated_df = unit_df[
    (unit_df["treated"] == 1)
    & unit_df["growth_rate"].notna()
].copy()

print(f"  分析対象（処置群・前期売上あり）: {len(treated_df)} 施設")

# Pearson相関
r_pearson, p_pearson = stats.pearsonr(
    treated_df["coverage"], treated_df["growth_rate"]
) if len(treated_df) > 3 else (np.nan, np.nan)
print(f"  Pearson相関 r={r_pearson:.4f}, p={p_pearson:.4f}")

# Spearman相関（外れ値に頑健）
r_spearman, p_spearman = stats.spearmanr(
    treated_df["coverage"], treated_df["growth_rate"]
) if len(treated_df) > 3 else (np.nan, np.nan)
print(f"  Spearman相関 ρ={r_spearman:.4f}, p={p_spearman:.4f}")

# OLS回帰（単変量）
reg_results = None
if len(treated_df) > 5:
    _x = sm.add_constant(treated_df["coverage"].values)
    _y = treated_df["growth_rate"].values
    try:
        reg_results = sm.OLS(_y, _x).fit(cov_type="HC3")
        _coef = reg_results.params[1]
        _se   = reg_results.bse[1]
        _pval = reg_results.pvalues[1]
        _r2   = reg_results.rsquared
        print(f"  OLS: coef={_coef:.3f}, SE={_se:.3f}, p={_pval:.4f}, R²={_r2:.4f}")
        print(f"  解釈: Coverage が 0.1 上昇すると伸長率が {_coef*0.1:.2f}% 変化")
    except Exception as e:
        print(f"  OLS 失敗: {e}")

# n_docs を共変量に加えた重回帰
reg_results_multi = None
if len(treated_df) > 5:
    _x2 = sm.add_constant(
        np.column_stack([treated_df["coverage"].values, treated_df["n_docs"].values])
    )
    _y2 = treated_df["growth_rate"].values
    try:
        reg_results_multi = sm.OLS(_y2, _x2).fit(cov_type="HC3")
        _c_cov    = reg_results_multi.params[1]
        _c_ndocs  = reg_results_multi.params[2]
        _p_cov    = reg_results_multi.pvalues[1]
        _p_ndocs  = reg_results_multi.pvalues[2]
        _r2_m     = reg_results_multi.rsquared
        print(f"  重回帰 (Coverage + n_docs):")
        print(f"    Coverage: coef={_c_cov:.3f}, p={_p_cov:.4f}")
        print(f"    n_docs  : coef={_c_ndocs:.3f}, p={_p_ndocs:.4f}")
        print(f"    R²={_r2_m:.4f}")
    except Exception as e:
        print(f"  重回帰 失敗: {e}")

# ===================================================================
# [7] Coverage カテゴリ別 売上伸長率
# ===================================================================
print("\n[7] Coverage カテゴリ別 売上伸長率")

# カテゴリ分け: Coverage=0（未視聴）/ 低Coverage / 中Coverage / 高Coverage
# Coverage > 0 の処置群をqcutで3分割
_treated_cov = unit_df.loc[unit_df["treated"] == 1, "coverage"]

try:
    _full_mask  = unit_df["coverage"] >= 1.0
    _part_mask  = (unit_df["treated"] == 1) & (unit_df["coverage"] < 1.0) & (unit_df["coverage"] > 0)

    if _full_mask.sum() > 0 and _part_mask.sum() >= 4:
        # coverage=1.0を「高Coverage」, 残りをqcutで低/中に分割
        unit_df["coverage_cat"] = np.nan
        unit_df.loc[_full_mask, "coverage_cat"] = "高Coverage"
        _part_cats = pd.qcut(
            unit_df.loc[_part_mask, "coverage"],
            q=2, labels=["低Coverage", "中Coverage"], duplicates="drop"
        )
        unit_df.loc[_part_mask, "coverage_cat"] = _part_cats.astype(str)
    else:
        unit_df["coverage_cat"] = pd.qcut(
            unit_df["coverage"].where(unit_df["treated"] == 1),
            q=3, labels=["低Coverage", "中Coverage", "高Coverage"],
            duplicates="drop"
        )

    # 対照群は「未視聴」
    unit_df.loc[unit_df["treated"] == 0, "coverage_cat"] = "未視聴"

except Exception as e:
    print(f"  カテゴリ分け失敗: {e}")
    # フォールバック: 中央値で2分割
    _med = _treated_cov.median()
    unit_df["coverage_cat"] = np.nan
    unit_df.loc[unit_df["treated"] == 0, "coverage_cat"] = "未視聴"
    unit_df.loc[(unit_df["treated"] == 1) & (unit_df["coverage"] <= _med), "coverage_cat"] = "低Coverage"
    unit_df.loc[(unit_df["treated"] == 1) & (unit_df["coverage"] > _med),  "coverage_cat"] = "高Coverage"

# カテゴリ別の統計
cat_order = ["未視聴", "低Coverage", "中Coverage", "高Coverage"]
cat_stats = []
print("  カテゴリ別統計（前期売上>0の施設）:")
for _cat in cat_order:
    _rows = unit_df[
        (unit_df["coverage_cat"].astype(str) == _cat)
        & unit_df["growth_rate"].notna()
    ]
    if len(_rows) == 0:
        continue
    _n   = len(_rows)
    _m   = _rows["growth_rate"].mean()
    _sd  = _rows["growth_rate"].std()
    _se  = _sd / np.sqrt(_n)
    _cov_m = _rows["coverage"].mean()
    cat_stats.append({
        "category": _cat,
        "n": _n,
        "mean_growth_rate": float(_m),
        "sd": float(_sd),
        "se": float(_se),
        "mean_coverage": float(_cov_m),
        "ci_lo": float(_m - 1.96 * _se),
        "ci_hi": float(_m + 1.96 * _se),
    })
    print(f"    {_cat:10s}: N={_n:4d}, 平均伸長率={_m:+.2f}%, SE={_se:.2f}, Coverage={_cov_m:.3f}")

# 各カテゴリを未視聴と比較（独立t検定）
_ctrl_gr = unit_df.loc[
    (unit_df["coverage_cat"].astype(str) == "未視聴") & unit_df["growth_rate"].notna(),
    "growth_rate"
].values

print("\n  未視聴との比較（独立t検定）:")
for _s in cat_stats:
    if _s["category"] == "未視聴":
        continue
    _tgt_gr = unit_df.loc[
        (unit_df["coverage_cat"].astype(str) == _s["category"]) & unit_df["growth_rate"].notna(),
        "growth_rate"
    ].values
    if len(_tgt_gr) < 3 or len(_ctrl_gr) < 3:
        print(f"    {_s['category']}: サンプル不足")
        continue
    _t, _p = stats.ttest_ind(_tgt_gr, _ctrl_gr, equal_var=False)
    _diff  = np.mean(_tgt_gr) - np.mean(_ctrl_gr)
    _sig   = "***" if _p < 0.001 else "**" if _p < 0.01 else "*" if _p < 0.05 else "†" if _p < 0.1 else "n.s."
    print(f"    {_s['category']:10s} vs 未視聴: diff={_diff:+.2f}%, p={_p:.4f} {_sig}")
    _s["vs_control_diff"]  = float(_diff)
    _s["vs_control_p"]     = float(_p)
    _s["vs_control_sig"]   = _sig

# ===================================================================
# [8] 可視化
# ===================================================================
print("\n[8] 可視化")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    f"09-3: Coverage（施設内視聴率）× 売上伸長率分析\n"
    f"複数医師施設のみ（{MIN_DOCS_PER_FAC}名以上, N={len(treated_fac_ids)}処置施設）",
    fontsize=13, fontweight="bold"
)

# --- (a) Coverage 分布（処置群） ---
ax = axes[0, 0]
_cov_t = unit_df.loc[unit_df["treated"] == 1, "coverage"]
ax.hist(_cov_t, bins=20, color="#1565C0", alpha=0.75, edgecolor="white")
ax.axvline(_cov_t.mean(), color="red", linestyle="--",
           label=f"平均={_cov_t.mean():.3f}")
ax.axvline(_cov_t.median(), color="orange", linestyle=":",
           label=f"中央値={_cov_t.median():.3f}")
ax.set_xlabel("Coverage（施設内視聴率）", fontsize=10)
ax.set_ylabel("施設数", fontsize=10)
ax.set_title(f"(a) Coverage分布（処置施設, N={len(_cov_t)}）", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")

# --- (b) Coverage × 売上伸長率 散布図 + 回帰直線 ---
ax = axes[0, 1]
_x  = treated_df["coverage"].values
_y  = treated_df["growth_rate"].values
ax.scatter(_x, _y, alpha=0.5, color="#1565C0", s=40, label="処置施設")
if reg_results is not None and len(_x) > 3:
    _xl = np.linspace(_x.min(), _x.max(), 100)
    _coef = reg_results.params[1]
    _icpt = reg_results.params[0]
    _r_sq = reg_results.rsquared
    _p_v  = reg_results.pvalues[1]
    ax.plot(_xl, _coef * _xl + _icpt, color="red", linewidth=2,
            label=f"OLS: slope={_coef:.2f}\nr²={_r_sq:.3f}, p={_p_v:.3f}")
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Coverage（施設内視聴率）", fontsize=10)
ax.set_ylabel("売上伸長率（前期比, %）", fontsize=10)
_r_str = f"r={r_pearson:.3f}, p={p_pearson:.3f}" if not np.isnan(r_pearson) else ""
ax.set_title(f"(b) 用量反応: Coverage × 売上伸長率\n処置施設のみ (N={len(_x)})  {_r_str}",
             fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- (c) Coverage カテゴリ別 平均伸長率 棒グラフ ---
ax = axes[1, 0]
_cat_order_present = [c for c in cat_order if any(s["category"] == c for s in cat_stats)]
_bar_x      = np.arange(len(_cat_order_present))
_bar_means  = [next(s["mean_growth_rate"] for s in cat_stats if s["category"] == c)
               for c in _cat_order_present]
_bar_ses    = [next(s["se"] for s in cat_stats if s["category"] == c)
               for c in _cat_order_present]
_bar_ns     = [next(s["n"] for s in cat_stats if s["category"] == c)
               for c in _cat_order_present]
_palette    = {"未視聴": "#FF8F00", "低Coverage": "#90CAF9",
               "中Coverage": "#42A5F5", "高Coverage": "#1565C0"}
_bar_colors = [_palette.get(c, "gray") for c in _cat_order_present]

ax.bar(_bar_x, _bar_means, yerr=[1.96 * s for s in _bar_ses],
       color=_bar_colors, alpha=0.85, capsize=6,
       error_kw={"linewidth": 1.5, "ecolor": "black"})
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
_max_abs = max(abs(m) for m in _bar_means) if _bar_means else 1.0
for xi, (m, se, n) in enumerate(zip(_bar_means, _bar_ses, _bar_ns)):
    ax.text(xi, m + 1.96 * se + _max_abs * 0.05, f"N={n}", ha="center", fontsize=9)
ax.set_xticks(_bar_x)
ax.set_xticklabels(_cat_order_present, fontsize=9)
ax.set_ylabel("平均売上伸長率（前期比, %）", fontsize=10)
ax.set_title("(c) Coverage カテゴリ別 平均伸長率\n（95%CI バー、未視聴を基準）", fontsize=10)
ax.grid(True, alpha=0.3, axis="y")

# --- (d) 施設医師数 × Coverage 散布図 ---
ax = axes[1, 1]
_t_df = unit_df[unit_df["treated"] == 1].copy()
_sc = ax.scatter(
    _t_df["n_docs"], _t_df["coverage"],
    c=_t_df["growth_rate"], cmap="RdYlGn",
    alpha=0.7, s=50, edgecolors="white", linewidth=0.5
)
_cb = plt.colorbar(_sc, ax=ax)
_cb.set_label("売上伸長率（%）", fontsize=9)
ax.set_xlabel("施設内医師数", fontsize=10)
ax.set_ylabel("Coverage（施設内視聴率）", fontsize=10)
ax.set_title("(d) 施設医師数 × Coverage\n（色: 売上伸長率）", fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_png = os.path.join(SCRIPT_DIR, "coverage_growth_multi_doc.png")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  保存: {out_png}")

# ===================================================================
# [9] 施設医師数別 Coverage × 伸長率 補足分析
# ===================================================================
print("\n[9] 施設医師数区分別の Coverage × 伸長率 相関")

unit_df["n_docs_cat"] = pd.cut(
    unit_df["n_docs"],
    bins=[1, 3, 6, np.inf],
    labels=["2〜3名", "4〜6名", "7名以上"]
)

fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
fig2.suptitle(
    "09-3: 施設医師数区分別 Coverage × 売上伸長率",
    fontsize=12, fontweight="bold"
)

_ndocs_corr_results = {}
for i, _cat in enumerate(["2〜3名", "4〜6名", "7名以上"]):
    ax = axes2[i]
    _sub = unit_df[
        (unit_df["n_docs_cat"].astype(str) == _cat)
        & (unit_df["treated"] == 1)
        & unit_df["growth_rate"].notna()
    ]
    if len(_sub) < 5:
        ax.set_title(f"{_cat}: データ不足 (N={len(_sub)})")
        _ndocs_corr_results[_cat] = {"n": len(_sub)}
        continue

    _xv = _sub["coverage"].values
    _yv = _sub["growth_rate"].values
    ax.scatter(_xv, _yv, alpha=0.6, color="#1565C0", s=50)

    _r_s, _p_s = stats.pearsonr(_xv, _yv)
    _ndocs_corr_results[_cat] = {"n": len(_sub), "r": float(_r_s), "p": float(_p_s)}

    if len(_xv) > 3:
        _slope, _icpt, _r_v, _p_reg, _ = stats.linregress(_xv, _yv)
        _xl = np.linspace(_xv.min(), _xv.max(), 100)
        ax.plot(_xl, _slope * _xl + _icpt, color="red", linewidth=2)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Coverage", fontsize=10)
    ax.set_ylabel("売上伸長率（%）", fontsize=10)
    ax.set_title(
        f"{_cat} (N={len(_sub)})\nr={_r_s:.3f}, p={_p_s:.3f}",
        fontsize=10
    )
    ax.grid(True, alpha=0.3)
    print(f"  {_cat}: N={len(_sub)}, r={_r_s:.4f}, p={_p_s:.4f}")

plt.tight_layout()
out_png2 = os.path.join(SCRIPT_DIR, "coverage_growth_by_ndocs.png")
plt.savefig(out_png2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  保存: {out_png2}")

# ===================================================================
# [10] JSON 出力
# ===================================================================
print("\n[10] 結果保存")

results_dir = os.path.join(SCRIPT_DIR, "results")
os.makedirs(results_dir, exist_ok=True)

output_json = {
    "analysis_settings": {
        "script": "09-3_coverage_growth_analysis.py",
        "target": f"複数医師施設のみ（{MIN_DOCS_PER_FAC}名以上）",
        "rationale": "1施設1医師ではCoverageが0/1バイナリになりバイアスが生じるため除外",
        "pre_period":  f"month 0-{PRE_END} ({PRE_END + 1}ヶ月)",
        "post_period": f"month {POST_START}-32 ({33 - POST_START}ヶ月)",
        "outcome": "後期月平均 - 前期月平均（円）/ 前期月平均（%）",
        "coverage_def": "解析期間中に視聴した医師数 / 施設の総医師数",
    },
    "sample_sizes": {
        "all_facilities": int(len(fac_to_docs_all)),
        "single_doc_excluded": int(n_single),
        "multi_doc_analyzed":  int(n_multi),
        "washout_excluded": int(len(washout_fac_ids)),
        "treated_fac":  int(len(treated_fac_ids)),
        "control_fac":  int(len(control_fac_ids)),
    },
    "coverage_stats": {
        "treated_mean":   float(_cov_t.mean()),
        "treated_median": float(_cov_t.median()),
        "treated_std":    float(_cov_t.std()),
        "n_full_coverage": int((_cov_t >= 1.0).sum()),
    },
    "regression": {
        "pearson_r":   float(r_pearson) if not np.isnan(r_pearson) else None,
        "pearson_p":   float(p_pearson) if not np.isnan(p_pearson) else None,
        "spearman_rho": float(r_spearman) if not np.isnan(r_spearman) else None,
        "spearman_p":   float(p_spearman) if not np.isnan(p_spearman) else None,
        "ols_coef_coverage": float(reg_results.params[1]) if reg_results else None,
        "ols_se":            float(reg_results.bse[1])    if reg_results else None,
        "ols_p":             float(reg_results.pvalues[1]) if reg_results else None,
        "ols_r2":            float(reg_results.rsquared)  if reg_results else None,
    },
    "category_stats": cat_stats,
    "by_ndocs_cat_corr": _ndocs_corr_results,
}

json_path = os.path.join(results_dir, "coverage_growth_multi_doc.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(output_json, f, ensure_ascii=False, indent=2)
print(f"  結果JSON: {json_path}")

print("\n" + "=" * 70)
print(" 分析完了")
print("=" * 70)
print(f"\n【主要結果】")
print(f"  対象: 複数医師施設 {n_multi} 件（うち処置群 {len(treated_fac_ids)} 件）")
print(f"  Coverage × 伸長率: Pearson r={r_pearson:.4f} (p={p_pearson:.4f})")
if reg_results:
    print(f"  OLS: Coverage 0.1 上昇 → 伸長率 {reg_results.params[1]*0.1:+.2f}%")
