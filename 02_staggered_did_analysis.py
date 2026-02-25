"""
===================================================================
Staggered DID分析: デジタルコンテンツ視聴の効果検証
===================================================================
手順:
  1. 本番形式データ読み込み (sales, デジタル視聴, 活動, rw_list)
  2. 除外フロー:
     a. 1施設複数医師 → 除外
     b. 1医師複数施設 → 除外
     c. wash-out期間(2023/4-5)に視聴あり → 除外
     d. 初回視聴が2025/10以降 → 除外
  3. TWFE / CS(全体) / CS(チャネル別)
===================================================================
"""

import os
import warnings

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

# === データファイル・カラム設定 ===
ENT_PRODUCT_CODE = "00001"              # ENT品目コード (5桁文字列、パラメータ)
CONTENT_TYPES = ["Webinar", "e-contents", "web講演会"]  # チャネル大分類 (拡張可能)
ACTIVITY_CHANNEL_FILTER = "web講演会"   # 活動データから抽出する活動種別

# ファイル名
FILE_RW_LIST = "rw_list.csv"
FILE_SALES = "sales.csv"
FILE_DIGITAL = "デジタル視聴データ.csv"
FILE_ACTIVITY = "活動データ.csv"
FILE_FACILITY_MASTER = "facility_master.csv"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
# data/ の整合性チェック → data2/ にフォールバック
_required = [FILE_SALES, FILE_DIGITAL, FILE_ACTIVITY, FILE_RW_LIST]
_data_ok = all(os.path.exists(os.path.join(DATA_DIR, f)) for f in _required)
if not _data_ok:
    _alt = os.path.join(SCRIPT_DIR, "data2")
    if all(os.path.exists(os.path.join(_alt, f)) for f in _required):
        DATA_DIR = _alt

START_DATE = "2023-04-01"
N_MONTHS = 33
WASHOUT_MONTHS = 2
LAST_ELIGIBLE_MONTH = 29
MIN_ET, MAX_ET = -6, 18


# ================================================================
# 共通関数
# ================================================================

def compute_cs_attgt(pdata):
    doc_info = pdata.groupby("unit_id").agg(
        {"treated": "first", "cohort_month": "first"}
    )
    pivot = pdata.pivot_table(
        values="amount", index="unit_id", columns="month_index", aggfunc="mean",
    )
    ctrl_docs = doc_info[doc_info["treated"] == 0].index
    if len(ctrl_docs) == 0:
        return pd.DataFrame()
    ctrl_means = pivot.loc[ctrl_docs].mean()

    cohorts = sorted(
        doc_info.loc[doc_info["cohort_month"].notna(), "cohort_month"].unique()
    )
    all_times = sorted(pdata["month_index"].unique())

    rows = []
    for g in cohorts:
        g = int(g)
        base = g - 1
        if base not in pivot.columns:
            continue
        cdocs = doc_info[doc_info["cohort_month"] == g].index
        if len(cdocs) == 0:
            continue
        tmeans = pivot.loc[cdocs].mean()
        for t in all_times:
            if t not in pivot.columns:
                continue
            att = (tmeans[t] - tmeans[base]) - (ctrl_means[t] - ctrl_means[base])
            rows.append({
                "cohort": g, "time": t, "event_time": t - g,
                "att_gt": att, "n_cohort": len(cdocs),
            })
    return pd.DataFrame(rows)


def aggregate_dynamic(att_gt, min_e=MIN_ET, max_e=MAX_ET):
    sub = att_gt[(att_gt["event_time"] >= min_e) & (att_gt["event_time"] <= max_e)]
    if len(sub) == 0:
        return pd.DataFrame(columns=["event_time", "att"])
    dyn = (
        sub.groupby("event_time")
        .apply(lambda x: np.average(x["att_gt"], weights=x["n_cohort"]))
        .reset_index()
    )
    dyn.columns = ["event_time", "att"]
    return dyn


def aggregate_overall(att_gt):
    post = att_gt[att_gt["event_time"] >= 0]
    if len(post) == 0:
        return 0.0
    return np.average(post["att_gt"], weights=post["n_cohort"])


def run_cs_with_bootstrap(panel, n_boot=200, label=""):
    att_gt = compute_cs_attgt(panel)
    if len(att_gt) == 0:
        return None, None, None, None

    dynamic = aggregate_dynamic(att_gt)
    overall = aggregate_overall(att_gt)

    doc_data = {uid: grp for uid, grp in panel.groupby("unit_id")}
    treated_ids = panel.loc[panel["treated"] == 1, "unit_id"].unique()
    control_ids = panel.loc[panel["treated"] == 0, "unit_id"].unique()

    boot_overall = []
    boot_dynamic = {}

    tag = f"[{label}] " if label else ""
    print(f"  {tag}Bootstrap (n={n_boot}) ...", end="", flush=True)

    for b in range(n_boot):
        if (b + 1) % 50 == 0:
            print(f" {b + 1}", end="", flush=True)

        bt = np.random.choice(treated_ids, len(treated_ids), replace=True)
        bc = np.random.choice(control_ids, len(control_ids), replace=True)

        parts = []
        for i, uid in enumerate(np.concatenate([bt, bc])):
            d = doc_data[uid].copy()
            d["unit_id"] = f"B{i:04d}"
            parts.append(d)
        bpanel = pd.concat(parts, ignore_index=True)

        try:
            bgt = compute_cs_attgt(bpanel)
            if len(bgt) == 0:
                continue
            boot_overall.append(aggregate_overall(bgt))
            bdyn = aggregate_dynamic(bgt)
            for _, row in bdyn.iterrows():
                et = int(row["event_time"])
                boot_dynamic.setdefault(et, []).append(row["att"])
        except Exception:
            continue

    print(" done")

    se_overall = np.std(boot_overall) if boot_overall else 0.0
    se_dyn_map = {et: np.std(v) for et, v in boot_dynamic.items()}

    dynamic["se"] = dynamic["event_time"].map(se_dyn_map).fillna(0)
    dynamic["ci_lo"] = dynamic["att"] - 1.96 * dynamic["se"]
    dynamic["ci_hi"] = dynamic["att"] + 1.96 * dynamic["se"]

    return att_gt, dynamic, overall, se_overall


# ================================================================
# Part 1: データ読み込み (本番形式)
# ================================================================
print("=" * 70)
print(" Staggered DID分析: デジタルコンテンツ視聴の効果検証")
print("=" * 70)

# 1. RW医師リスト
rw_list = pd.read_csv(os.path.join(DATA_DIR, FILE_RW_LIST))
n_rw_all = len(rw_list)
doctor_master = rw_list[rw_list["seg"].notna() & (rw_list["seg"] != "")].copy()
doctor_master = doctor_master.rename(columns={"fac_honin": "facility_id", "doc": "doctor_id"})

# 2. 売上データ (日付・実績・品目コードが文字列)
sales_raw = pd.read_csv(os.path.join(DATA_DIR, FILE_SALES), dtype=str)
sales_raw["実績"] = pd.to_numeric(sales_raw["実績"], errors="coerce").fillna(0)
sales_raw["日付"] = pd.to_datetime(sales_raw["日付"], format="mixed")
n_sales_all = len(sales_raw)
daily = sales_raw[sales_raw["品目コード"].str.strip() == ENT_PRODUCT_CODE].copy()
daily = daily.rename(columns={
    "日付": "delivery_date",
    "施設（本院に合算）コード": "facility_id",
    "実績": "amount",
})

# 3. デジタル視聴データ
digital_raw = pd.read_csv(os.path.join(DATA_DIR, FILE_DIGITAL))
n_digital_all = len(digital_raw)
digital_raw["品目コード"] = digital_raw["品目コード"].astype(str).str.strip().str.zfill(5)
digital = digital_raw[digital_raw["品目コード"] == ENT_PRODUCT_CODE].copy()

# 4. 活動データ → web講演会のみ抽出
activity_raw = pd.read_csv(os.path.join(DATA_DIR, FILE_ACTIVITY))
n_activity_all = len(activity_raw)
activity_raw["品目コード"] = activity_raw["品目コード"].astype(str).str.strip().str.zfill(5)
web_lecture = activity_raw[
    (activity_raw["品目コード"] == ENT_PRODUCT_CODE)
    & (activity_raw["活動種別"] == ACTIVITY_CHANNEL_FILTER)
].copy()

# 5. 視聴データ結合 (デジタル + 活動web講演会)
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

print(f"\n[元データ]")
print(f"  売上データ(全品目)      : {n_sales_all:,} 行")
print(f"  売上データ(ENT品目)     : {len(daily):,} 行  (他品目 {n_sales_all - len(daily):,} 行除外)")
print(f"  RW医師リスト(全体)      : {n_rw_all} 行")
print(f"  RW医師リスト(seg非空)   : {len(doctor_master)} 行  (非RW {n_rw_all - len(doctor_master)} 行除外)")
print(f"  デジタル視聴データ      : {n_digital_all:,} 行 → ENT品目: {len(digital):,} 行")
print(f"  活動データ              : {n_activity_all:,} 行 → web講演会+ENT: {len(web_lecture):,} 行")
print(f"  視聴データ結合          : {len(viewing):,} 行")
print(f"  観測期間 : {months[0].strftime('%Y-%m')} ~ {months[-1].strftime('%Y-%m')} ({N_MONTHS}ヶ月)")

# ================================================================
# Part 2: 除外フロー
# ================================================================
print("\n" + "=" * 70)
print(" Part 2: 除外フロー")
print("=" * 70)

# [A] 施設内医師数==1 の施設に絞り込み (全医師ベース: facility_master.csv)
fac_master_df = pd.read_csv(os.path.join(DATA_DIR, FILE_FACILITY_MASTER))
single_staff_facs = set(fac_master_df[fac_master_df["施設内医師数"] == 1]["facility_id"])
multi_staff_facs  = set(fac_master_df[fac_master_df["施設内医師数"] > 1]["facility_id"])

print(f"\n  [A] 施設内医師数==1 の施設に絞り込み (全医師ベース: facility_master.csv)")
print(f"      1医師施設    : {len(single_staff_facs)} 施設")
print(f"      複数医師施設 : {len(multi_staff_facs)} 施設 -> 除外")

# [B] 複数施設所属RW医師の除外 (施設フィルタ前の全所属で確認)
facs_per_doc = doctor_master.groupby("doctor_id")["facility_id"].nunique()
multi_fac_docs  = set(facs_per_doc[facs_per_doc > 1].index)
single_fac_docs = set(facs_per_doc[facs_per_doc == 1].index)

excluded_multi_fac = doctor_master[
    doctor_master["facility_id"].isin(single_staff_facs)
    & doctor_master["doctor_id"].isin(multi_fac_docs)
]

print(f"\n  [B] 複数施設所属RW医師の除外")
print(f"      複数施設所属RW医師      : {len(multi_fac_docs)} 名")
print(f"      うち1医師施設に属する   : {len(excluded_multi_fac)} 名 -> 除外")

# クリーンな1:1ペア: 施設内1医師 かつ RW医師が1施設のみ所属
clean_pairs = doctor_master[
    doctor_master["facility_id"].isin(single_staff_facs)
    & doctor_master["doctor_id"].isin(single_fac_docs)
].copy()

print(f"      クリーンな1:1ペア: {len(clean_pairs)} 施設")

fac_to_doc = dict(zip(clean_pairs["facility_id"], clean_pairs["doctor_id"]))
doc_to_fac = dict(zip(clean_pairs["doctor_id"], clean_pairs["facility_id"]))
clean_fac_ids = set(clean_pairs["facility_id"])
clean_doc_ids = set(clean_pairs["doctor_id"])

washout_end = months[WASHOUT_MONTHS - 1] + pd.offsets.MonthEnd(0)
viewing_clean = viewing[viewing["doctor_id"].isin(clean_doc_ids)].copy()
washout_viewers = set(
    viewing_clean[viewing_clean["view_date"] <= washout_end]["doctor_id"].unique()
)
excluded_washout_facs = {doc_to_fac[d] for d in washout_viewers if d in doc_to_fac}

print(f"\n  [C] wash-out期間 (2023/4-5) 視聴者の除外")
print(f"      wash-out視聴医師 : {len(washout_viewers)} 名 -> 除外")

clean_fac_ids -= excluded_washout_facs
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
excluded_late_facs = {doc_to_fac[d] for d in late_adopters if d in doc_to_fac}

print(f"\n  [D] 遅延視聴者 (初回 >= 2025/10) の除外")
print(f"      遅延視聴医師     : {len(late_adopters)} 名 -> 除外")

clean_fac_ids -= excluded_late_facs
clean_doc_ids -= late_adopters

treated_doc_ids = set(first_view[
    first_view["first_view_month"] <= LAST_ELIGIBLE_MONTH
]["doctor_id"]) & clean_doc_ids
all_viewing_doc_ids = set(viewing["doctor_id"].unique())
control_doc_ids = clean_doc_ids - all_viewing_doc_ids

analysis_doc_ids = treated_doc_ids | control_doc_ids
analysis_fac_ids = {doc_to_fac[d] for d in analysis_doc_ids}

print(f"\n  [最終分析対象]")
print(f"      処置群 : {len(treated_doc_ids)} 施設")
print(f"      対照群 : {len(control_doc_ids)} 施設")
print(f"      合計   : {len(analysis_fac_ids)} 施設")

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

monthly = (
    daily_target.groupby(["facility_id", "month_index"])["amount"]
    .sum().reset_index()
)

full_idx = pd.MultiIndex.from_product(
    [sorted(analysis_fac_ids), list(range(N_MONTHS))],
    names=["facility_id", "month_index"]
)
panel_base = (
    monthly.set_index(["facility_id", "month_index"])
    .reindex(full_idx, fill_value=0).reset_index()
)
panel_base["unit_id"] = panel_base["facility_id"]
panel_base["doctor_id"] = panel_base["facility_id"].map(fac_to_doc)

first_view_eligible = first_view[
    first_view["doctor_id"].isin(treated_doc_ids)
][["doctor_id", "first_view_month"]].copy()
first_view_eligible["facility_id"] = first_view_eligible["doctor_id"].map(doc_to_fac)
first_view_eligible = first_view_eligible.rename(columns={"first_view_month": "cohort_month"})

panel = panel_base.merge(
    first_view_eligible[["facility_id", "cohort_month"]],
    on="facility_id", how="left",
)
panel["treated"] = panel["cohort_month"].notna().astype(int)

n_treated = panel.loc[panel["treated"] == 1, "unit_id"].nunique()
n_control = panel.loc[panel["treated"] == 0, "unit_id"].nunique()
n_total = n_treated + n_control

print(f"  パネル行数 : {len(panel):,} ({n_total}施設 x {N_MONTHS}月)")
print(f"  処置群     : {n_treated} 施設")
print(f"  対照群     : {n_control} 施設")

cohort_dist = first_view_eligible.groupby("cohort_month").size()
print(f"\n  コホート分布:")
for m, cnt in cohort_dist.items():
    ym = months[int(m)].strftime("%Y-%m")
    print(f"    month {int(m):>2} ({ym}): {cnt} 施設")

print(f"\n  チャネル別視聴者数:")
for ch in CONTENT_TYPES:
    n = viewing_after_washout[
        (viewing_after_washout["channel_category"] == ch)
        & (viewing_after_washout["doctor_id"].isin(treated_doc_ids))
    ]["doctor_id"].nunique()
    print(f"    {ch:<12}: {n} 名")

# ================================================================
# Part 4: 記述統計
# ================================================================
print("\n" + "=" * 70)
print(" Part 4: 記述統計")
print("=" * 70)

desc = panel.groupby("treated")["amount"].agg(["count", "mean", "std", "min", "max"])
desc.index = ["対照群(未視聴)", "処置群(視聴)"]
print(f"\n[月次施設別納入額]")
print(desc.round(1).to_string())

# ================================================================
# Part 5: TWFE推定
# ================================================================
print("\n" + "=" * 70)
print(" Part 5: TWFE推定")
print("=" * 70)

panel_r = panel.reset_index(drop=True)
y = panel_r["amount"].values

mask_t = panel_r["cohort_month"].notna()
panel_r["post"] = 0
panel_r.loc[mask_t, "post"] = (
    panel_r.loc[mask_t, "month_index"] >= panel_r.loc[mask_t, "cohort_month"]
).astype(int)
panel_r["did"] = panel_r["treated"] * panel_r["post"]

unit_dum = pd.get_dummies(panel_r["unit_id"], prefix="u", drop_first=True, dtype=float)
time_dum = pd.get_dummies(panel_r["month_index"], prefix="t", drop_first=True, dtype=float)

X_twfe = pd.concat(
    [pd.DataFrame({"const": 1.0, "did": panel_r["did"].values}), unit_dum, time_dum],
    axis=1,
)
model_twfe = sm.OLS(y, X_twfe).fit(
    cov_type="cluster", cov_kwds={"groups": panel_r["unit_id"].values}
)

beta = model_twfe.params["did"]
se_twfe = model_twfe.bse["did"]
pval_twfe = model_twfe.pvalues["did"]
ci_twfe = model_twfe.conf_int().loc["did"]
sig_twfe = (
    "***" if pval_twfe < 0.001 else "**" if pval_twfe < 0.01
    else "*" if pval_twfe < 0.05 else "n.s."
)

print(f"\n  DID推定量 : {beta:.2f}")
print(f"  SE        : {se_twfe:.2f}")
print(f"  p値       : {pval_twfe:.6f} {sig_twfe}")
print(f"  95%CI     : [{ci_twfe[0]:.2f}, {ci_twfe[1]:.2f}]")

# ================================================================
# Part 5b: ロバストネスチェック — MR活動共変量
# ================================================================
print("\n" + "=" * 70)
print(" Part 5b: ロバストネスチェック (MR活動共変量)")
print("=" * 70)

# 1. 活動データから非デジタル活動 (面談, 面談_アポ, 説明会, その他) を抽出
#    CONTENT_TYPES (Webinar, e-contents, web講演会) は処置変数 → 除外
mr_activity = activity_raw[
    (activity_raw["品目コード"] == ENT_PRODUCT_CODE)
    & (~activity_raw["活動種別"].isin(CONTENT_TYPES))
].copy()

# 2. 施設×月次で集計
mr_activity = mr_activity.rename(columns={"fac_honin": "mr_facility_id"})
mr_activity["活動日_dt"] = pd.to_datetime(mr_activity["活動日_dt"], format="mixed")
mr_activity["month_index"] = (
    (mr_activity["活動日_dt"].dt.year - 2023) * 12
    + mr_activity["活動日_dt"].dt.month - 4
)
mr_counts = (
    mr_activity.groupby(["mr_facility_id", "month_index"])
    .size()
    .reset_index(name="mr_activity_count")
)
mr_counts = mr_counts.rename(columns={"mr_facility_id": "facility_id"})

print(f"\n  MR活動レコード数 (非デジタル): {len(mr_activity):,}")
print(f"  活動種別内訳:")
for act_type, cnt in mr_activity["活動種別"].value_counts().items():
    print(f"    {act_type}: {cnt:,}")

# 3. パネルにマージ
panel_robust = panel_r.merge(mr_counts, on=["facility_id", "month_index"], how="left")
panel_robust["mr_activity_count"] = panel_robust["mr_activity_count"].fillna(0)

print(f"\n  パネルへのマージ完了:")
print(f"    MR活動ありセル: {(panel_robust['mr_activity_count'] > 0).sum():,} / {len(panel_robust):,}")
print(f"    MR活動 平均: {panel_robust['mr_activity_count'].mean():.2f}, "
      f"最大: {panel_robust['mr_activity_count'].max():.0f}")

# 4. TWFE with MR活動共変量
X_robust = pd.concat(
    [pd.DataFrame({
        "const": 1.0,
        "did": panel_robust["did"].values,
        "mr_activity": panel_robust["mr_activity_count"].values,
    }), unit_dum, time_dum],
    axis=1,
)
y_robust = panel_robust["amount"].values

model_robust = sm.OLS(y_robust, X_robust).fit(
    cov_type="cluster", cov_kwds={"groups": panel_robust["unit_id"].values}
)

beta_robust = model_robust.params["did"]
se_robust = model_robust.bse["did"]
pval_robust = model_robust.pvalues["did"]
ci_robust = model_robust.conf_int().loc["did"]
sig_robust = (
    "***" if pval_robust < 0.001 else "**" if pval_robust < 0.01
    else "*" if pval_robust < 0.05 else "n.s."
)

mr_coef = model_robust.params["mr_activity"]
mr_se = model_robust.bse["mr_activity"]
mr_pval = model_robust.pvalues["mr_activity"]
mr_sig = (
    "***" if mr_pval < 0.001 else "**" if mr_pval < 0.01
    else "*" if mr_pval < 0.05 else "n.s."
)

att_change_pct = abs(beta_robust - beta) / abs(beta) * 100 if beta != 0 else 0.0

print(f"\n  === ロバストネスチェック ===")
print(f"  TWFE (メイン)         : ATT={beta:.2f} (SE={se_twfe:.2f}) {sig_twfe}")
print(f"  TWFE (+MR活動共変量)  : ATT={beta_robust:.2f} (SE={se_robust:.2f}) {sig_robust}")
print(f"  MR活動の係数          : {mr_coef:.2f} (SE={mr_se:.2f}, p={mr_pval:.4f}) {mr_sig}")
print(f"  → ATT変化率: {att_change_pct:.1f}%")

# ================================================================
# Part 6: CS推定 (全体)
# ================================================================
print("\n" + "=" * 70)
print(" Part 6: CS推定 (全体)")
print("=" * 70)

att_gt_all, cs_dyn_all, cs_overall, se_overall = run_cs_with_bootstrap(
    panel_r, n_boot=200, label="全体"
)

z_all = cs_overall / se_overall if se_overall > 0 else float("inf")
p_cs = 2 * (1 - stats.norm.cdf(abs(z_all)))
sig_cs = (
    "***" if p_cs < 0.001 else "**" if p_cs < 0.01
    else "*" if p_cs < 0.05 else "n.s."
)

print(f"\n  ATT(全体) : {cs_overall:.2f}")
print(f"  SE        : {se_overall:.2f}")
print(f"  p値       : {p_cs:.6f} {sig_cs}")

print(f"\n  [動的効果]")
print(f"  {'ET':>5} {'ATT':>8} {'SE':>8} {'95%CI':>20}")
print(f"  {'-' * 45}")
for _, r in cs_dyn_all.iterrows():
    et = int(r["event_time"])
    marker = " <-" if et == 0 else ""
    print(
        f"  {'e=' + str(et):>5} {r['att']:>8.2f} {r['se']:>8.2f}"
        f" [{r['ci_lo']:>7.2f}, {r['ci_hi']:>7.2f}]{marker}"
    )

# ================================================================
# Part 7: CS推定 (チャネル別)
# ================================================================
print("\n" + "=" * 70)
print(" Part 7: CS チャネル別推定")
print("=" * 70)

control_fac_ids = {doc_to_fac[d] for d in control_doc_ids}
channel_results = {}

for ch in CONTENT_TYPES:
    print(f"\n  --- {ch} ---")
    ch_views = viewing_after_washout[
        (viewing_after_washout["channel_category"] == ch)
        & (viewing_after_washout["doctor_id"].isin(treated_doc_ids))
    ]
    if len(ch_views) == 0:
        print(f"  -> 該当なし")
        continue

    first_ch = ch_views.groupby("doctor_id")["view_date"].min().reset_index()
    first_ch.columns = ["doctor_id", "first_ch_date"]
    first_ch["cohort_month"] = (
        (first_ch["first_ch_date"].dt.year - 2023) * 12
        + first_ch["first_ch_date"].dt.month - 4
    )
    first_ch = first_ch[first_ch["cohort_month"] <= LAST_ELIGIBLE_MONTH]
    first_ch["facility_id"] = first_ch["doctor_id"].map(doc_to_fac)

    ch_treated_facs = set(first_ch["facility_id"].dropna())
    use_facs = ch_treated_facs | control_fac_ids

    ch_panel = panel_base[panel_base["facility_id"].isin(use_facs)].copy()
    ch_panel["unit_id"] = ch_panel["facility_id"]
    ch_panel = ch_panel.merge(
        first_ch[["facility_id", "cohort_month"]], on="facility_id", how="left"
    )
    ch_panel["treated"] = ch_panel["cohort_month"].notna().astype(int)

    n_ch_t = len(ch_treated_facs)
    print(f"  処置群: {n_ch_t} 施設, 対照群: {len(control_fac_ids)} 施設")

    result = run_cs_with_bootstrap(ch_panel, n_boot=100, label=ch)
    if result[0] is None:
        print(f"  -> 推定不可")
        continue

    att_gt_ch, dyn_ch, overall_ch, se_ch = result
    z_ch = overall_ch / se_ch if se_ch > 0 else float("inf")
    p_ch = 2 * (1 - stats.norm.cdf(abs(z_ch)))
    sig_ch = (
        "***" if p_ch < 0.001 else "**" if p_ch < 0.01
        else "*" if p_ch < 0.05 else "n.s."
    )

    print(f"  ATT={overall_ch:.2f}, SE={se_ch:.2f}, p={p_ch:.4f} {sig_ch}")
    channel_results[ch] = {
        "dynamic": dyn_ch, "overall": overall_ch,
        "se": se_ch, "p": p_ch, "sig": sig_ch, "n_treated": n_ch_t,
    }

# ================================================================
# Part 8: 推定結果比較
# ================================================================
print("\n" + "=" * 70)
print(" Part 8: 推定結果の比較")
print("=" * 70)

print(f"\n  {'手法':<25} {'ATT':>8} {'SE':>8} {'p値':>10} {'有意性':>6}")
print(f"  {'-' * 62}")
print(f"  {'TWFE (全体)':<25} {beta:>8.2f} {se_twfe:>8.2f} {pval_twfe:>10.6f} {sig_twfe:>6}")
print(f"  {'CS (全体)':<25} {cs_overall:>8.2f} {se_overall:>8.2f} {p_cs:>10.6f} {sig_cs:>6}")
for ch in CONTENT_TYPES:
    if ch in channel_results:
        r = channel_results[ch]
        lbl = f"CS ({ch})"
        print(f"  {lbl:<25} {r['overall']:>8.2f} {r['se']:>8.2f} {r['p']:>10.6f} {r['sig']:>6}")

print(f"\n  DGPの真の効果:")
print(f"    Webinar: 18, e-contents: 10, web講演会: 22")
print(f"    + 月次成長 1.0/月 (視聴継続中)")
print(f"    + 停止後減衰 -1.5/月 (猶予2ヶ月)")
print(f"    x 属性modifier (地域/施設タイプ/経験年数/診療科)")

# ================================================================
# Part 9: 可視化
# ================================================================
print("\n" + "=" * 70)
print(" Part 9: 可視化")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    "Staggered DID: デジタルコンテンツ視聴の効果\n"
    "(1施設1医師 AND 1医師1施設, wash-out/遅延視聴者除外)",
    fontsize=13, fontweight="bold",
)

ax = axes[0, 0]
avg = panel_r.groupby(["month_index", "treated"])["amount"].mean().unstack()
x_ticks = avg.index
ym_labels = [months[int(m)].strftime("%y/%m") if m < N_MONTHS else "" for m in x_ticks]
ax.plot(x_ticks, avg[0], "b-o", ms=3, label="対照群 (未視聴)")
ax.plot(x_ticks, avg[1], "r-s", ms=3, label="処置群 (視聴)")
ax.axvline(WASHOUT_MONTHS - 0.5, color="gray", ls=":", lw=0.8, label="wash-out")
ax.set_xlabel("月")
ax.set_ylabel("平均納入額 (月次/施設)")
ax.set_title("(a) 群別平均納入額の推移")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
tick_pos = list(range(0, N_MONTHS, 3))
ax.set_xticks(tick_pos)
ax.set_xticklabels([ym_labels[i] if i < len(ym_labels) else "" for i in tick_pos],
                    rotation=45, fontsize=7)

ax = axes[0, 1]
ax.axhline(0, color="black", lw=0.8)
ax.axvline(-0.5, color="red", ls="--", lw=0.8, alpha=0.7, label="処置開始")
ax.fill_between(cs_dyn_all["event_time"], cs_dyn_all["ci_lo"], cs_dyn_all["ci_hi"],
                alpha=0.2, color="steelblue")
ax.plot(cs_dyn_all["event_time"], cs_dyn_all["att"], "o-", color="steelblue", ms=4)
ax.set_xlabel("イベント時間 (月)")
ax.set_ylabel("ATT")
ax.set_title("(b) CS動的効果 (全体)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.axhline(0, color="black", lw=0.8)
ax.axvline(-0.5, color="red", ls="--", lw=0.8, alpha=0.7)
ch_colors = {"Webinar": "#1f77b4", "e-contents": "#ff7f0e", "web講演会": "#2ca02c"}
ch_markers = {"Webinar": "o", "e-contents": "s", "web講演会": "^"}
for ch in CONTENT_TYPES:
    if ch not in channel_results:
        continue
    dyn = channel_results[ch]["dynamic"]
    c, m = ch_colors[ch], ch_markers[ch]
    ax.fill_between(dyn["event_time"], dyn["ci_lo"], dyn["ci_hi"], alpha=0.1, color=c)
    ax.plot(dyn["event_time"], dyn["att"], f"{m}-", color=c, ms=4, label=ch)
ax.set_xlabel("イベント時間 (月)")
ax.set_ylabel("ATT")
ax.set_title("(c) CS動的効果 (チャネル別)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ch_names = [ch for ch in CONTENT_TYPES if ch in channel_results]
ch_atts = [channel_results[ch]["overall"] for ch in ch_names]
ch_ses = [channel_results[ch]["se"] for ch in ch_names]
ch_cols = [ch_colors[ch] for ch in ch_names]
x_pos = range(len(ch_names))
ax.bar(x_pos, ch_atts, color=ch_cols, alpha=0.8, edgecolor="gray")
ax.errorbar(x_pos, ch_atts, yerr=[1.96 * s for s in ch_ses],
            fmt="none", color="black", capsize=5)
ax.set_xticks(list(x_pos))
ax.set_xticklabels(ch_names)
ax.set_ylabel("ATT")
ax.set_title("(d) チャネル別ATT比較 (95%CI)")
ax.axhline(cs_overall, color="red", ls="--", lw=0.8, label=f"全体ATT={cs_overall:.1f}")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
for i, ch in enumerate(ch_names):
    sig = channel_results[ch]["sig"]
    y_offset = ch_atts[i] + 1.96 * ch_ses[i] + 0.5
    ax.text(i, y_offset, sig, ha="center", fontsize=10)

plt.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "staggered_did_results.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  図を保存: {out_path}")

# ================================================================
# 結論
# ================================================================
print("\n" + "=" * 70)
print(" 結論")
print("=" * 70)

print(f"""
  === 分析対象 ===
  処置群 : {n_treated} 施設, 対照群 : {n_control} 施設, 合計 : {n_total} 施設

  === 全体効果 ===
  TWFE推定           : {beta:.2f} (SE={se_twfe:.2f}, p={pval_twfe:.4f}) {sig_twfe}
  Callaway-Sant'Anna : {cs_overall:.2f} (SE={se_overall:.2f}, p={p_cs:.4f}) {sig_cs}

  === チャネル別効果 (CS推定) ===""")
for ch in CONTENT_TYPES:
    if ch in channel_results:
        r = channel_results[ch]
        print(f"  {ch:<12}: ATT={r['overall']:.2f} (SE={r['se']:.2f}, p={r['p']:.4f}) {r['sig']}  N={r['n_treated']}")
print()

# ================================================================
# JSON結果保存
# ================================================================
import json

results_dir = os.path.join(SCRIPT_DIR, "results")
os.makedirs(results_dir, exist_ok=True)

# 除外フロー情報
exclusion_flow = {
    "total_delivery_rows": n_sales_all,
    "ent_delivery_rows": len(daily),
    "total_rw_list": n_rw_all,
    "single_staff_facilities": len(single_staff_facs),
    "ent_rw_doctors": len(doctor_master),
    "total_viewing_rows": n_digital_all + n_activity_all,
    "viewing_after_filter": len(viewing),
    "total_doctors": len(doctor_master["doctor_id"].unique()),
    "total_facilities": len(doctor_master["facility_id"].unique()),
    "multi_staff_facilities": len(multi_staff_facs),
    "multi_fac_doctors": len(multi_fac_docs),
    "clean_pairs": len(fac_to_doc),
    "washout_excluded": len(washout_viewers),
    "late_excluded": len(late_adopters),
    "final_treated": len(treated_doc_ids),
    "final_control": len(control_doc_ids),
    "final_total": len(analysis_fac_ids),
}

# 除外された医師ID一覧
excluded_ids = {
    "washout": sorted(list(washout_viewers)),
    "multi_fac": sorted(list(multi_fac_docs)),
    "late": sorted(list(late_adopters)),
}

# TWFE結果
twfe_result = {
    "att": float(beta),
    "se": float(se_twfe),
    "p": float(pval_twfe),
    "ci_lo": float(ci_twfe[0]),
    "ci_hi": float(ci_twfe[1]),
    "sig": sig_twfe,
}

# TWFEロバストネスチェック結果
twfe_robust_result = {
    "att": float(beta_robust),
    "se": float(se_robust),
    "p": float(pval_robust),
    "ci_lo": float(ci_robust[0]),
    "ci_hi": float(ci_robust[1]),
    "sig": sig_robust,
    "mr_activity_coef": float(mr_coef),
    "mr_activity_se": float(mr_se),
    "mr_activity_p": float(mr_pval),
    "mr_activity_sig": mr_sig,
    "att_change_pct": float(att_change_pct),
    "covariates": ["mr_activity_count"],
}

# CS全体結果
cs_result = {
    "att": float(cs_overall),
    "se": float(se_overall),
    "p": float(p_cs),
    "ci_lo": float(cs_overall - 1.96 * se_overall),
    "ci_hi": float(cs_overall + 1.96 * se_overall),
    "sig": sig_cs,
    "dynamic": cs_dyn_all[["event_time", "att", "se", "ci_lo", "ci_hi"]].to_dict("records"),
}

# チャネル別結果
ch_results_json = {}
for ch in CONTENT_TYPES:
    if ch in channel_results:
        r = channel_results[ch]
        ch_results_json[ch] = {
            "att": float(r["overall"]),
            "se": float(r["se"]),
            "p": float(r["p"]),
            "ci_lo": float(r["overall"] - 1.96 * r["se"]),
            "ci_hi": float(r["overall"] + 1.96 * r["se"]),
            "sig": r["sig"],
            "n_treated": r["n_treated"],
            "dynamic": r["dynamic"][["event_time", "att", "se", "ci_lo", "ci_hi"]].to_dict("records"),
        }

# コホート分布
cohort_json = {
    months[int(m)].strftime("%Y-%m"): int(cnt)
    for m, cnt in cohort_dist.items()
}

# 記述統計
desc_json = {
    "control": {
        "count": int(desc.loc["対照群(未視聴)", "count"]),
        "mean": float(desc.loc["対照群(未視聴)", "mean"]),
        "std": float(desc.loc["対照群(未視聴)", "std"]),
    },
    "treated": {
        "count": int(desc.loc["処置群(視聴)", "count"]),
        "mean": float(desc.loc["処置群(視聴)", "mean"]),
        "std": float(desc.loc["処置群(視聴)", "std"]),
    },
}

did_results_json = {
    "exclusion_flow": exclusion_flow,
    "excluded_ids": excluded_ids,
    "twfe": twfe_result,
    "twfe_robust": twfe_robust_result,
    "cs_overall": cs_result,
    "cs_channel": ch_results_json,
    "cohort_distribution": cohort_json,
    "descriptive_stats": desc_json,
    "n_treated": n_treated,
    "n_control": n_control,
    "n_total": n_total,
}

json_path = os.path.join(results_dir, "did_results.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(did_results_json, f, ensure_ascii=False, indent=2)
print(f"  結果をJSON保存: {json_path}")
