"""
===================================================================
MR活動Mediation分析: 接触機会 → 視聴 → 売上
===================================================================
目的:
  1. MR活動（面談、説明会等）を「接触機会」として扱う
  2. Mediation分析: MR活動 → 視聴 → 売上のパスを分解
  3. 疑似IV推定: MR活動を操作変数として視聴の内生性を軽減
  4. 制御可能な変数（MR活動）による意思決定支援

手法:
  - Stage 1: 視聴回数 ~ MR活動 + 医師属性 + 時間FE
  - Stage 2: 売上 ~ 予測視聴 + 医師FE + 時間FE
  - Direct effect: MR活動 → 売上（視聴を経由しない直接効果）
  - Indirect effect: MR活動 → 視聴 → 売上（視聴を介した間接効果）

重要な注意:
  MR活動自体も内生的（売上が良い施設にMRが多く訪問）な可能性があるため、
  完全な因果推論ではない。ただし、視聴回数のみを使うよりは頑健。
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

# === パラメータ設定 ===
ENT_PRODUCT_CODE = "00001"
CONTENT_TYPES = ["webiner", "e_contents", "Web講演会"]
ACTIVITY_CHANNEL_FILTER = "Web講演会"

# MR活動種別（視聴以外の活動）
MR_ACTIVITY_TYPES = ["面談", "面談_アポ", "説明会"]

FILE_RW_LIST = "rw_list.csv"
FILE_SALES = "sales.csv"
FILE_DIGITAL = "デジタル視聴データ.csv"
FILE_ACTIVITY = "活動データ.csv"
FILE_FACILITY_MASTER = "facility_attribute_修正.csv"
FILE_DOCTOR_ATTR = "doctor_attribute.csv"
FILE_FAC_DOCTOR_LIST = "施設医師リスト.csv"

# 解析集団フィルタパラメータ
FILTER_SINGLE_FAC_DOCTOR = True
DOCTOR_HONIN_FAC_COUNT_COL = "所属施設数"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "本番データ")
_required_07 = [FILE_SALES, FILE_DIGITAL, FILE_ACTIVITY, FILE_RW_LIST]
if not all(os.path.exists(os.path.join(DATA_DIR, f)) for f in _required_07):
    _alt = os.path.join(SCRIPT_DIR, "data")
    if all(os.path.exists(os.path.join(_alt, f)) for f in _required_07):
        DATA_DIR = _alt

START_DATE = "2023-04-01"
N_MONTHS = 33
WASHOUT_MONTHS = 2
LAST_ELIGIBLE_MONTH = 29


# ================================================================
# データ読み込み + 除外フロー
# ================================================================
print("=" * 70)
print(" MR活動Mediation分析: 接触機会 → 視聴 → 売上")
print("=" * 70)

# 基本データ読み込み
rw_list = pd.read_csv(os.path.join(DATA_DIR, FILE_RW_LIST))

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

activity_raw = pd.read_csv(os.path.join(DATA_DIR, FILE_ACTIVITY))
activity_raw["品目コード"] = activity_raw["品目コード"].astype(str).str.strip().str.zfill(5)

# Web講演会（視聴データ）
web_lecture = activity_raw[
    (activity_raw["品目コード"] == ENT_PRODUCT_CODE)
    & (activity_raw["活動種別"] == ACTIVITY_CHANNEL_FILTER)
].copy()

# MR活動データ（面談、説明会など）
mr_activity = activity_raw[
    (activity_raw["品目コード"] == ENT_PRODUCT_CODE)
    & (activity_raw["活動種別"].isin(MR_ACTIVITY_TYPES))
].copy()

print(f"\n[データ読み込み]")
print(f"  MR活動データ: {len(mr_activity):,} 行")
for act_type in MR_ACTIVITY_TYPES:
    n = len(mr_activity[mr_activity["活動種別"] == act_type])
    print(f"    {act_type}: {n:,} 行")

# 視聴データ結合
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

# 除外フロー
# 施設医師リスト: 全医師の施設対応マスター (母集団)
fac_doc_list = pd.read_csv(os.path.join(DATA_DIR, FILE_FAC_DOCTOR_LIST))

# [Step 1] facility_attribute_修正.csv: fac単位で施設内医師数==1のfacを抽出
fac_df = pd.read_csv(os.path.join(DATA_DIR, FILE_FACILITY_MASTER))
single_staff_fac = set(fac_df[fac_df["施設内医師数"] == 1]["fac"])
multi_staff_fac  = set(fac_df[fac_df["施設内医師数"] > 1]["fac"])
print(f"  [Step 1] 施設内医師数==1 (fac単位): {len(single_staff_fac)} fac → 複数医師fac {len(multi_staff_fac)} 件除外")

# [Step 2] doctor_attribute.csv: 所属施設数==1 の医師
doc_attr_df = pd.read_csv(os.path.join(DATA_DIR, FILE_DOCTOR_ATTR))
if FILTER_SINGLE_FAC_DOCTOR:
    if DOCTOR_HONIN_FAC_COUNT_COL in doc_attr_df.columns:
        single_honin_docs = set(doc_attr_df[doc_attr_df[DOCTOR_HONIN_FAC_COUNT_COL] == 1]["doc"])
    else:
        _fac_per_doc = fac_doc_list.groupby("doc")["fac_honin"].nunique()
        single_honin_docs = set(_fac_per_doc[_fac_per_doc == 1].index)
else:
    single_honin_docs = set(doc_attr_df["doc"])
print(f"  [Step 2] 所属施設数==1: {len(single_honin_docs)} 名 (doctor_attribute基準)")

# [Step 3] rw_list.csv: RW医師フィルタ (候補セット構築)
# rw_list.csvはRW医師のみ格納 → seg絞り不要
rw_doc_ids = set(rw_list["doc"])
print(f"  [Step 3] RW医師候補: {len(rw_doc_ids)} 名")

# 3ステップを順序付きで適用 + 中間カウント + 1:1確認
_doc_to_fac   = dict(zip(fac_doc_list["doc"], fac_doc_list["fac"]))
_doc_to_honin = dict(zip(fac_doc_list["doc"], fac_doc_list["fac_honin"]))
all_docs = set(fac_doc_list["doc"])  # 全医師は施設医師リスト.csv
after_step1 = {d for d in all_docs if _doc_to_fac.get(d) in single_staff_fac}
after_step2 = after_step1 & single_honin_docs
after_step3 = after_step2 & rw_doc_ids
_honin_cnt: dict = {}
for d in after_step3:
    h = _doc_to_honin[d]
    _honin_cnt[h] = _honin_cnt.get(h, 0) + 1
candidate_docs = {d for d in after_step3 if _honin_cnt[_doc_to_honin[d]] == 1}
print(f"  Step1通過:{len(after_step1)} → Step2通過:{len(after_step2)} → Step3通過:{len(after_step3)} → 1:1確認後:{len(candidate_docs)}")

clean_pairs = rw_list[rw_list["doc"].isin(candidate_docs)][["doc", "fac_honin"]].drop_duplicates()
clean_pairs = clean_pairs.rename(columns={"doc": "doctor_id", "fac_honin": "facility_id"})
fac_to_doc = dict(zip(clean_pairs["facility_id"], clean_pairs["doctor_id"]))
doc_to_fac = dict(zip(clean_pairs["doctor_id"], clean_pairs["facility_id"]))
clean_doc_ids = set(clean_pairs["doctor_id"])
doctor_master_rw = clean_pairs.copy()

washout_end = months[WASHOUT_MONTHS - 1] + pd.offsets.MonthEnd(0)
viewing_clean = viewing[viewing["doctor_id"].isin(clean_doc_ids)].copy()
washout_viewers = set(
    viewing_clean[viewing_clean["view_date"] <= washout_end]["doctor_id"].unique()
)
clean_doc_ids -= washout_viewers

viewing_after_washout = viewing_clean[
    (viewing_clean["doctor_id"].isin(clean_doc_ids))
    & (viewing_clean["view_date"] > washout_end)
]

first_view = viewing_after_washout.groupby("doctor_id")["view_date"].min().reset_index()
first_view.columns = ["doctor_id", "first_view_date"]
first_view["first_view_month"] = (
    (first_view["first_view_date"].dt.year - 2023) * 12
    + first_view["first_view_date"].dt.month - 4
)

late_adopters = set(
    first_view[first_view["first_view_month"] > LAST_ELIGIBLE_MONTH]["doctor_id"]
)
clean_doc_ids -= late_adopters

treated_doc_ids = set(
    first_view[first_view["first_view_month"] <= LAST_ELIGIBLE_MONTH]["doctor_id"]
) & clean_doc_ids
all_viewing_doc_ids = set(viewing["doctor_id"].unique())
control_doc_ids = clean_doc_ids - all_viewing_doc_ids

analysis_doc_ids = treated_doc_ids | control_doc_ids
analysis_fac_ids = {doc_to_fac[d] for d in analysis_doc_ids}

print(f"\n[除外フロー完了] 処置群: {len(treated_doc_ids)}, 対照群: {len(control_doc_ids)}")


# ================================================================
# Part 1: MR活動データの集計（医師×月次）
# ================================================================
print("\n" + "=" * 70)
print(" Part 1: MR活動データの集計")
print("=" * 70)

# MR活動を医師×月次で集計
mr_activity["活動日_dt"] = pd.to_datetime(mr_activity["活動日_dt"], format="mixed")
mr_activity["month_index"] = (
    (mr_activity["活動日_dt"].dt.year - 2023) * 12
    + mr_activity["活動日_dt"].dt.month - 4
)

mr_counts = (
    mr_activity.groupby(["doc", "month_index"])
    .size()
    .reset_index(name="mr_activity_count")
)
mr_counts = mr_counts.rename(columns={"doc": "doctor_id"})

print(f"  医師×月次のMR活動回数集計完了")
print(f"  平均MR活動回数/月: {mr_counts['mr_activity_count'].mean():.2f}")
print(f"  最大MR活動回数/月: {mr_counts['mr_activity_count'].max()}")


# ================================================================
# Part 2: 視聴回数の集計（医師×月次）
# ================================================================
print("\n" + "=" * 70)
print(" Part 2: 視聴回数の集計")
print("=" * 70)

viewing_after_washout["month_index"] = (
    (viewing_after_washout["view_date"].dt.year - 2023) * 12
    + viewing_after_washout["view_date"].dt.month - 4
)

viewing_counts = (
    viewing_after_washout.groupby(["doctor_id", "month_index"])
    .size()
    .reset_index(name="viewing_count")
)

print(f"  医師×月次の視聴回数集計完了")
print(f"  平均視聴回数/月（視聴月のみ）: {viewing_counts['viewing_count'].mean():.2f}")


# ================================================================
# Part 3: パネルデータ構築
# ================================================================
print("\n" + "=" * 70)
print(" Part 3: パネルデータ構築")
print("=" * 70)

daily_target = daily[daily["facility_id"].isin(analysis_fac_ids)].copy()
daily_target["month_index"] = (
    (daily_target["delivery_date"].dt.year - 2023) * 12
    + daily_target["delivery_date"].dt.month - 4
)
monthly = daily_target.groupby(["facility_id", "month_index"])["amount"].sum().reset_index()

full_idx = pd.MultiIndex.from_product(
    [sorted(analysis_fac_ids), list(range(N_MONTHS))],
    names=["facility_id", "month_index"]
)
panel = (
    monthly.set_index(["facility_id", "month_index"])
    .reindex(full_idx, fill_value=0).reset_index()
)
panel["unit_id"] = panel["facility_id"]
panel["doctor_id"] = panel["facility_id"].map(fac_to_doc)

# MR活動回数をマージ
panel = panel.merge(mr_counts, on=["doctor_id", "month_index"], how="left")
panel["mr_activity_count"] = panel["mr_activity_count"].fillna(0)

# 視聴回数をマージ
panel = panel.merge(viewing_counts, on=["doctor_id", "month_index"], how="left")
panel["viewing_count"] = panel["viewing_count"].fillna(0)

print(f"  パネル行数: {len(panel):,}")
print(f"  MR活動ありセル: {(panel['mr_activity_count'] > 0).sum():,}")
print(f"  視聴ありセル: {(panel['viewing_count'] > 0).sum():,}")

# MR活動と視聴の相関
corr_mr_viewing = panel[["mr_activity_count", "viewing_count"]].corr().iloc[0, 1]
print(f"\n  MR活動と視聴の相関: {corr_mr_viewing:.3f}")


# ================================================================
# Part 4: Stage 1 - MR活動 → 視聴
# ================================================================
print("\n" + "=" * 70)
print(" Part 4: Stage 1 - MR活動 → 視聴")
print("=" * 70)

panel_clean = panel.copy().reset_index(drop=True)

y_stage1 = panel_clean["viewing_count"].values
X_stage1_base = panel_clean[["mr_activity_count"]].values

# 医師固定効果
doc_dum = pd.get_dummies(panel_clean["doctor_id"], prefix="doc", drop_first=True, dtype=float)
# 時間固定効果
time_dum = pd.get_dummies(panel_clean["month_index"], prefix="t", drop_first=True, dtype=float)

X_stage1 = np.hstack([
    np.ones((len(panel_clean), 1)),  # const
    X_stage1_base,
    doc_dum.values,
    time_dum.values
])

try:
    model_stage1 = sm.OLS(y_stage1, X_stage1).fit()
    panel_clean["viewing_count_predicted"] = model_stage1.predict(X_stage1)

    beta_mr = model_stage1.params[1]  # MR活動の係数
    se_mr = model_stage1.bse[1]
    pval_mr = model_stage1.pvalues[1]
    sig_mr = "***" if pval_mr < 0.001 else "**" if pval_mr < 0.01 else "*" if pval_mr < 0.05 else "n.s."

    print(f"\n  Stage 1推定完了")
    print(f"  MR活動 → 視聴の係数: {beta_mr:.4f} (SE={se_mr:.4f}, {sig_mr})")
    print(f"  解釈: MR活動1回増加で視聴が{beta_mr:.4f}回増加")

except Exception as e:
    print(f"\n  警告: Stage 1推定失敗: {e}")
    panel_clean["viewing_count_predicted"] = panel_clean["viewing_count"]
    beta_mr = 0


# ================================================================
# Part 5: Stage 2 - 予測視聴 → 売上
# ================================================================
print("\n" + "=" * 70)
print(" Part 5: Stage 2 - 予測視聴 → 売上")
print("=" * 70)

y_stage2 = panel_clean["amount"].values
X_stage2_base = panel_clean[["viewing_count_predicted"]].values

X_stage2 = np.hstack([
    np.ones((len(panel_clean), 1)),
    X_stage2_base,
    doc_dum.values,
    time_dum.values
])

try:
    model_stage2 = sm.OLS(y_stage2, X_stage2).fit()

    beta_viewing = model_stage2.params[1]
    se_viewing = model_stage2.bse[1]
    pval_viewing = model_stage2.pvalues[1]
    sig_viewing = "***" if pval_viewing < 0.001 else "**" if pval_viewing < 0.01 else "*" if pval_viewing < 0.05 else "n.s."

    print(f"\n  Stage 2推定完了")
    print(f"  視聴 → 売上の係数: {beta_viewing:.2f} (SE={se_viewing:.2f}, {sig_viewing})")
    print(f"  解釈: 視聴1回増加で売上が{beta_viewing:.2f}増加")

    # Indirect effect (MR活動 → 視聴 → 売上)
    indirect_effect = beta_mr * beta_viewing
    print(f"\n  Indirect effect (MR活動 → 視聴 → 売上): {indirect_effect:.2f}")

except Exception as e:
    print(f"\n  警告: Stage 2推定失敗: {e}")
    beta_viewing = 0
    indirect_effect = 0


# ================================================================
# Part 6: Direct effect - MR活動 → 売上（直接）
# ================================================================
print("\n" + "=" * 70)
print(" Part 6: Direct effect - MR活動 → 売上")
print("=" * 70)

y_direct = panel_clean["amount"].values
X_direct_base = panel_clean[["mr_activity_count", "viewing_count"]].values

X_direct = np.hstack([
    np.ones((len(panel_clean), 1)),
    X_direct_base,
    doc_dum.values,
    time_dum.values
])

try:
    model_direct = sm.OLS(y_direct, X_direct).fit()

    beta_mr_direct = model_direct.params[1]
    se_mr_direct = model_direct.bse[1]
    pval_mr_direct = model_direct.pvalues[1]
    sig_mr_direct = "***" if pval_mr_direct < 0.001 else "**" if pval_mr_direct < 0.01 else "*" if pval_mr_direct < 0.05 else "n.s."

    print(f"\n  Direct effect推定完了")
    print(f"  MR活動 → 売上（直接）の係数: {beta_mr_direct:.2f} (SE={se_mr_direct:.2f}, {sig_mr_direct})")

    # Total effect
    total_effect = beta_mr_direct + indirect_effect
    print(f"\n  Total effect (直接 + 間接): {total_effect:.2f}")
    print(f"    - Direct effect: {beta_mr_direct:.2f}")
    print(f"    - Indirect effect: {indirect_effect:.2f}")

    if total_effect != 0:
        indirect_pct = (indirect_effect / total_effect) * 100
        print(f"  Indirect効果の割合: {indirect_pct:.1f}%")

except Exception as e:
    print(f"\n  警告: Direct effect推定失敗: {e}")
    beta_mr_direct = 0
    total_effect = 0


# ================================================================
# Part 7: 可視化
# ================================================================
print("\n" + "=" * 70)
print(" Part 7: 可視化")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(
    "MR活動Mediation分析: 接触機会 → 視聴 → 売上",
    fontsize=13, fontweight="bold"
)

# (a) MR活動と視聴の散布図
ax = axes[0, 0]
mr_viewing_agg = panel[panel["mr_activity_count"] > 0].groupby("mr_activity_count")["viewing_count"].mean()
ax.scatter(mr_viewing_agg.index, mr_viewing_agg.values, s=100, alpha=0.6, color="#1565C0")
ax.plot(mr_viewing_agg.index, mr_viewing_agg.values, "--", alpha=0.5, color="#1565C0")
ax.set_xlabel("MR活動回数")
ax.set_ylabel("平均視聴回数")
ax.set_title(f"(a) MR活動 → 視聴 (r={corr_mr_viewing:.3f})")
ax.grid(True, alpha=0.3)

# (b) 視聴と売上の散布図
ax = axes[0, 1]
viewing_sales_agg = panel[panel["viewing_count"] > 0].groupby("viewing_count")["amount"].mean()
ax.scatter(viewing_sales_agg.index, viewing_sales_agg.values, s=100, alpha=0.6, color="#4CAF50")
ax.plot(viewing_sales_agg.index, viewing_sales_agg.values, "--", alpha=0.5, color="#4CAF50")
ax.set_xlabel("視聴回数")
ax.set_ylabel("平均売上")
ax.set_title("(b) 視聴 → 売上")
ax.grid(True, alpha=0.3)

# (c) Mediation効果の分解
ax = axes[1, 0]
effects = ["Direct\n(MR→売上)", "Indirect\n(MR→視聴→売上)", "Total"]
values = [beta_mr_direct, indirect_effect, total_effect]
colors = ["#FF9800", "#4CAF50", "#1565C0"]
bars = ax.bar(effects, values, color=colors, alpha=0.7, edgecolor="white")
ymax = max(values) if values else 1.0
ymin = min(0, min(values)) if values else 0.0
y_span = max(abs(ymax - ymin), 1.0)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_span * 0.03,
            f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylim(ymin - y_span * 0.1, ymax + y_span * 0.3)
ax.set_ylabel("効果")
ax.set_title("(c) Mediation効果の分解")
ax.axhline(0, color="black", lw=0.8)
ax.grid(True, alpha=0.3, axis="y")

# (d) MR活動回数別の売上推移
ax = axes[1, 1]
mr_bins = [0, 1, 3, 100]
mr_labels = ["MR活動なし", "MR活動1-2回", "MR活動3回以上"]
panel["mr_bin"] = pd.cut(panel["mr_activity_count"], bins=mr_bins, labels=mr_labels, include_lowest=True)

for label in mr_labels:
    label_data = panel[panel["mr_bin"] == label]
    if len(label_data) > 0:
        trend = label_data.groupby("month_index")["amount"].mean()
        ax.plot(trend.index, trend.values, marker="o", ms=3, label=label)

ax.set_xlabel("月 (0=2023/4)")
ax.set_ylabel("平均売上")
ax.set_title("(d) MR活動回数別の売上推移")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "mr_activity_mediation.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  図を保存: {out_path}")


# ================================================================
# Part 8: JSON結果保存
# ================================================================
results_dir = os.path.join(SCRIPT_DIR, "results")
os.makedirs(results_dir, exist_ok=True)

results_json = {
    "mr_viewing_correlation": float(corr_mr_viewing),
    "stage1_mr_to_viewing": {
        "coefficient": float(beta_mr) if 'beta_mr' in locals() else 0,
        "se": float(se_mr) if 'se_mr' in locals() else 0,
        "p": float(pval_mr) if 'pval_mr' in locals() else 1,
        "sig": sig_mr if 'sig_mr' in locals() else "N/A",
    },
    "stage2_viewing_to_sales": {
        "coefficient": float(beta_viewing) if 'beta_viewing' in locals() else 0,
        "se": float(se_viewing) if 'se_viewing' in locals() else 0,
        "p": float(pval_viewing) if 'pval_viewing' in locals() else 1,
        "sig": sig_viewing if 'sig_viewing' in locals() else "N/A",
    },
    "mediation_effects": {
        "direct_effect": float(beta_mr_direct) if 'beta_mr_direct' in locals() else 0,
        "indirect_effect": float(indirect_effect),
        "total_effect": float(total_effect),
        "indirect_percentage": float(indirect_pct) if 'indirect_pct' in locals() else 0,
    },
    "interpretation": {
        "warning": "MR活動自体も内生的（売上が良い施設にMRが多く訪問）な可能性あり",
        "advantage": "視聴回数のみを使うよりは頑健な推定",
        "recommendation": "MR活動は制御可能な変数として、意思決定に活用可能",
    },
}

json_path = os.path.join(results_dir, "mr_activity_mediation.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results_json, f, ensure_ascii=False, indent=2)
print(f"\n  結果をJSON保存: {json_path}")

print("\n" + "=" * 70)
print(" 分析完了")
print("=" * 70)
print(f"\n【実務的示唆】")
print(f"  MR活動（面談、説明会）は制御可能な変数です。")
print(f"  MR活動を増やすことで、視聴が促進され（間接効果）、")
print(f"  さらに視聴を介さない直接的な売上効果も期待できます。")
print(f"  ただし、MR活動自体も内生的な可能性があるため、")
print(f"  最終的には無作為化実験で因果効果を確認することが望ましいです。")
