"""
===================================================================
CATE分析: 属性別の異質的処置効果 (Heterogeneous Treatment Effects)
===================================================================
基本次元:
  - ベースライン納入額 (baseline_cat): 低 / 中 / 高

オプション属性ファイルが data/ に存在すれば自動拡張:
  - doctor_attributes.csv → experience_cat, specialty 等
  - facility_attributes.csv → region, facility_type 等

手法: CS推定量をサブグループ別に実行 + Bootstrap SE
===================================================================
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

for _font in ["Yu Gothic", "MS Gothic", "Meiryo", "Hiragino Sans", "IPAexGothic"]:
    try:
        matplotlib.rcParams["font.family"] = _font
        break
    except Exception:
        pass
matplotlib.rcParams["axes.unicode_minus"] = False

# === データファイル・カラム設定 (02と同一) ===
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

# 解析集団フィルタパラメータ
FILTER_SINGLE_FAC_DOCTOR = True   # True: 複数本院施設所属医師を除外
DOCTOR_HONIN_FAC_COUNT_COL = "所属施設数"  # doctor_attribute.csv の本院施設数カラム名
INCLUDE_ONLY_RW = False           # True: RW医師のみ (Step 3適用), False: 全医師 (Step 3スキップ)

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
MIN_ET, MAX_ET = -6, 18

# DGPの真のmodifier (サンプルデータ用。本番データでは参照されないが定義は残す)
TRUE_MODIFIERS = {
    "region":         {"都市部": 0.75, "郊外": 1.00, "地方": 1.40},
    "facility_type":  {"病院": 0.85, "クリニック": 1.20},
    "experience_cat": {"若手": 1.30, "中堅": 1.00, "ベテラン": 0.70},
    "specialty":      {"内科": 1.15, "外科": 0.85, "その他": 1.00},
}

# 基本次元 (常に使用)
CATE_DIMS = [
    ("baseline_cat", ["低", "中", "高"]),
]

# ===================================================================
# 属性ファイル設定 ― 分析するカラムをリストで指定
# ===================================================================

# --- 医師属性 (doctor_attribute.csv) ---
#   IDカラム  : doc
#   名前カラム: doc_name  (自動除外)
FILE_DOCTOR_ATTR = "doctor_attribute.csv"
DOCTOR_ATTR_ID_COL = "doc"
DOCTOR_ATTR_SELECTED: list = [      # ← 分析したいカラム名をここに列挙
    # "specialty",
    # "age",
    # "experience_years",
]

# --- 施設属性 (facility_attribute_修正.csv) ---
#   IDカラム  : fac_honin
#   名前カラム: fac_honin_name  (自動除外)
FILE_FACILITY_ATTR = "facility_attribute_修正.csv"
FACILITY_ATTR_ID_COL = "fac_honin"
FACILITY_ATTR_SELECTED: list = [    # ← 分析したいカラム名をここに列挙
    # "region",
    # "facility_type",
    # "bed_count",
]

# 連続値カラムのカテゴリ化設定
#   指定なし       → 自動で3分位（低 / 中 / 高）
#   "bins"         → pd.cut の境界値と labels を明示
#   "method":"median" → 中央値で2分割
CONTINUOUS_BINS: dict = {
    # "age":       {"bins": [0, 40, 55, 200], "labels": ["<40歳", "40-55歳", "55歳+"]},
    # "bed_count": {"method": "median"},
}


# ================================================================
# 属性ファイル読み込み・カテゴリ化ユーティリティ
# ================================================================

def _show_and_bin(df, col, continuous_bins):
    """連続値カラムの分布を表示しカテゴリ化する。新カラム名とlevelsを返す。"""
    s = df[col].dropna()
    desc = s.describe()
    print(f"      分布: min={desc['min']:.1f}  Q1={desc['25%']:.1f}  "
          f"中央値={desc['50%']:.1f}  Q3={desc['75%']:.1f}  max={desc['max']:.1f}")

    new_col = f"{col}_cat"
    cfg = continuous_bins.get(col, {})

    if "bins" in cfg:
        df[new_col] = pd.cut(df[col], bins=cfg["bins"], labels=cfg["labels"])
        levels = list(cfg["labels"])
        print(f"      → カスタムbinsでカテゴリ化: {levels}")
    elif cfg.get("method") == "median":
        med = s.median()
        labels = [f"≤{med:.0f}", f">{med:.0f}"]
        df[new_col] = pd.cut(df[col], bins=[-np.inf, med, np.inf], labels=labels)
        levels = labels
        print(f"      → 中央値({med:.1f})で2分割: {levels}")
    else:
        df[new_col] = pd.qcut(df[col], q=3, labels=["低", "中", "高"], duplicates="drop")
        levels = ["低", "中", "高"]
        print(f"      → 自動3分位でカテゴリ化: {levels}")

    return new_col, levels


def load_attr_file(filepath, id_col, id_rename, selected_cols, continuous_bins):
    """属性CSVを読み込み、選択カラムを取得・連続値をカテゴリ化して返す。

    Returns
    -------
    df_out : pd.DataFrame or None
        IDカラム(id_rename) + カテゴリ化済みカラムのDataFrame
    cate_dims : list of (col_name, levels)
    """
    if not os.path.exists(filepath):
        print(f"  ファイルなし: {os.path.basename(filepath)} → スキップ")
        return None, []

    raw = pd.read_csv(filepath)
    all_cols = [c for c in raw.columns if c not in (id_col,)]
    print(f"  {os.path.basename(filepath)}: {len(raw):,} 行")
    print(f"    利用可能カラム: {all_cols}")

    if id_col not in raw.columns:
        print(f"    IDカラム '{id_col}' が存在しません → スキップ")
        return None, []

    raw = raw.rename(columns={id_col: id_rename})

    avail = [c for c in selected_cols if c in raw.columns]
    missing = [c for c in selected_cols if c not in raw.columns]
    if missing:
        print(f"    警告: 選択カラムが見つかりません → {missing}")
    if not avail:
        print(f"    有効な選択カラムなし → スキップ")
        return None, []

    df_out = raw[[id_rename] + avail].drop_duplicates(subset=id_rename).copy()
    cate_dims = []

    for col in avail:
        series = df_out[col]
        is_numeric = pd.api.types.is_numeric_dtype(series)
        n_unique = series.nunique()

        if is_numeric and n_unique > 10:
            # 連続値 → 分布を表示してカテゴリ化
            print(f"    [{col}] 連続値 (ユニーク={n_unique})")
            new_col, levels = _show_and_bin(df_out, col, continuous_bins)
            cate_dims.append((new_col, levels))
        else:
            # カテゴリ値: そのまま使用
            levels = sorted(series.dropna().unique().tolist(), key=str)
            print(f"    [{col}] カテゴリ値: {levels}")
            cate_dims.append((col, levels))

    return df_out, cate_dims


# ================================================================
# CS推定関数
# ================================================================

def compute_cs_attgt(pdata):
    doc_info = pdata.groupby("unit_id").agg({"treated": "first", "cohort_month": "first"})
    pivot = pdata.pivot_table(values="amount", index="unit_id", columns="month_index", aggfunc="mean")
    ctrl_docs = doc_info[doc_info["treated"] == 0].index
    if len(ctrl_docs) == 0:
        return pd.DataFrame()
    ctrl_means = pivot.loc[ctrl_docs].mean()
    cohorts = sorted(doc_info.loc[doc_info["cohort_month"].notna(), "cohort_month"].unique())
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
            rows.append({"cohort": g, "time": t, "event_time": t - g, "att_gt": att, "n_cohort": len(cdocs)})
    return pd.DataFrame(rows)


def aggregate_dynamic(att_gt):
    sub = att_gt[(att_gt["event_time"] >= MIN_ET) & (att_gt["event_time"] <= MAX_ET)]
    if len(sub) == 0:
        return pd.DataFrame(columns=["event_time", "att"])
    dyn = sub.groupby("event_time").apply(lambda x: np.average(x["att_gt"], weights=x["n_cohort"])).reset_index()
    dyn.columns = ["event_time", "att"]
    return dyn


def aggregate_overall(att_gt):
    post = att_gt[att_gt["event_time"] >= 0]
    if len(post) == 0:
        return 0.0
    return np.average(post["att_gt"], weights=post["n_cohort"])


def cs_with_bootstrap(panel, n_boot=150, label=""):
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
    print(f"    {tag}Bootstrap (n={n_boot}) ...", end="", flush=True)
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
    return overall, se_overall, dynamic, boot_overall


# ================================================================
# データ読み込み + 除外フロー (02と同一ロジック)
# ================================================================
print("=" * 70)
print(" CATE分析: 属性別の異質的処置効果")
print("=" * 70)

# 1. RW医師リスト (除外フローで使用; 絞り込みは除外フローで実施)
rw_list = pd.read_csv(os.path.join(DATA_DIR, FILE_RW_LIST))

# 2. 売上データ
sales_raw = pd.read_csv(os.path.join(DATA_DIR, FILE_SALES), dtype=str)
sales_raw["実績"] = pd.to_numeric(sales_raw["実績"], errors="coerce").fillna(0)
sales_raw["日付"] = pd.to_datetime(sales_raw["日付"], format="mixed")
daily = sales_raw[sales_raw["品目コード"].str.strip() == ENT_PRODUCT_CODE].copy()
daily = daily.rename(columns={
    "日付": "delivery_date",
    "施設（本院に合算）コード": "facility_id",
    "実績": "amount",
})

# 3. デジタル視聴データ
digital_raw = pd.read_csv(os.path.join(DATA_DIR, FILE_DIGITAL))
digital_raw["品目コード"] = digital_raw["品目コード"].astype(str).str.strip().str.zfill(5)
digital = digital_raw[digital_raw["品目コード"] == ENT_PRODUCT_CODE].copy()
digital = digital[digital["fac_honin"].notna() & (digital["fac_honin"].astype(str).str.strip() != "")].copy()

# 4. 活動データ → Web講演会のみ
activity_raw = pd.read_csv(os.path.join(DATA_DIR, FILE_ACTIVITY))
activity_raw["品目コード"] = activity_raw["品目コード"].astype(str).str.strip().str.zfill(5)
web_lecture = activity_raw[
    (activity_raw["品目コード"] == ENT_PRODUCT_CODE)
    & (activity_raw["活動種別"] == ACTIVITY_CHANNEL_FILTER)
].copy()
web_lecture = web_lecture[web_lecture["fac_honin"].notna() & (web_lecture["fac_honin"].astype(str).str.strip() != "")].copy()

# 5. 視聴データ結合
common_cols = ["活動日_dt", "品目コード", "活動種別", "活動種別コード", "fac_honin", "doc"]
viewing = pd.concat([digital[common_cols], web_lecture[common_cols]], ignore_index=True)
viewing = viewing.rename(columns={
    "活動日_dt": "view_date",
    "fac_honin": "facility_id",
    "doc": "doctor_id",
    "活動種別": "channel_category",
})
viewing["view_date"] = pd.to_datetime(viewing["view_date"], format="mixed")

print(f"  売上データ: {len(sales_raw):,} 行 → ENT品目: {len(daily):,} 行")
print(f"  RW医師リスト(全体): {len(rw_list)} 行")
print(f"  視聴データ結合: {len(viewing):,} 行")

months = pd.date_range(start=START_DATE, periods=N_MONTHS, freq="MS")

# --- 除外フロー ---
print("\n[除外フロー]")

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

# [Step 3] RW医師フィルタ (INCLUDE_ONLY_RW=True の場合のみ適用)
rw_doc_ids = set(rw_list["doc"])

# 3ステップを順序付きで適用 + 中間カウント + 1:1確認
_doc_to_fac   = dict(zip(fac_doc_list["doc"], fac_doc_list["fac"]))
_doc_to_honin = dict(zip(fac_doc_list["doc"], fac_doc_list["fac_honin"]))
all_docs = set(fac_doc_list["doc"])  # 全医師は施設医師リスト.csv
after_step1 = {d for d in all_docs if _doc_to_fac.get(d) in single_staff_fac}
after_step2 = after_step1 & single_honin_docs
if INCLUDE_ONLY_RW:
    after_step3 = after_step2 & rw_doc_ids
else:
    after_step3 = after_step2  # Step 3スキップ (全医師対象)
_honin_cnt: dict = {}
for d in after_step3:
    h = _doc_to_honin[d]
    _honin_cnt[h] = _honin_cnt.get(h, 0) + 1
candidate_docs = {d for d in after_step3 if _honin_cnt[_doc_to_honin[d]] == 1}
print(f"  Step1通過:{len(after_step1)} → Step2通過:{len(after_step2)} → Step3通過:{len(after_step3)} ({'RW医師のみ' if INCLUDE_ONLY_RW else '全医師'}) → 1:1確認後:{len(candidate_docs)}")

_pair_src = rw_list if INCLUDE_ONLY_RW else fac_doc_list
clean_pairs = _pair_src[_pair_src["doc"].isin(candidate_docs)][["doc", "fac_honin"]].drop_duplicates()
clean_pairs = clean_pairs.rename(columns={"doc": "doctor_id", "fac_honin": "facility_id"})

fac_to_doc = dict(zip(clean_pairs["facility_id"], clean_pairs["doctor_id"]))
doc_to_fac = dict(zip(clean_pairs["doctor_id"], clean_pairs["facility_id"]))
clean_doc_ids = set(clean_pairs["doctor_id"])
_dm_src = rw_list if INCLUDE_ONLY_RW else fac_doc_list
doctor_master = _dm_src.rename(columns={"doc": "doctor_id", "fac_honin": "facility_id"})

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
late_adopters = set(first_view[first_view["first_view_month"] > LAST_ELIGIBLE_MONTH]["doctor_id"])
clean_doc_ids -= late_adopters

treated_doc_ids = set(first_view[
    first_view["first_view_month"] <= LAST_ELIGIBLE_MONTH
]["doctor_id"]) & clean_doc_ids
all_viewing_doc_ids = set(viewing["doctor_id"].unique())
control_doc_ids = clean_doc_ids - all_viewing_doc_ids

analysis_doc_ids = treated_doc_ids | control_doc_ids
analysis_fac_ids = {doc_to_fac[d] for d in analysis_doc_ids}

print(f"  処置群: {len(treated_doc_ids)} 施設, 対照群: {len(control_doc_ids)} 施設, 合計: {len(analysis_fac_ids)} 施設")

# ================================================================
# パネルデータ構築
# ================================================================
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
    first_view_eligible[["facility_id", "cohort_month"]], on="facility_id", how="left",
)
panel["treated"] = panel["cohort_month"].notna().astype(int)

# ================================================================
# 属性のマージ
# ================================================================
print("\n[属性のマージ]")

# ベースライン納入額 (wash-out期間の平均)
baseline = panel[panel["month_index"] < WASHOUT_MONTHS].groupby("unit_id")["amount"].mean().reset_index()
baseline.columns = ["unit_id", "baseline_amount"]
baseline["baseline_cat"] = pd.qcut(baseline["baseline_amount"], q=3, labels=["低", "中", "高"])
panel = panel.merge(baseline[["unit_id", "baseline_cat"]], on="unit_id", how="left")
print(f"  baseline_cat: wash-out期間平均から3分位 → 低/中/高")

# 属性ファイルの読み込み・カテゴリ化・マージ
_attr_configs = [
    (os.path.join(DATA_DIR, FILE_DOCTOR_ATTR),   DOCTOR_ATTR_ID_COL,   "doctor_id",   DOCTOR_ATTR_SELECTED),
    (os.path.join(DATA_DIR, FILE_FACILITY_ATTR), FACILITY_ATTR_ID_COL, "facility_id", FACILITY_ATTR_SELECTED),
]

for _filepath, _id_col, _id_rename, _selected in _attr_configs:
    _fname = os.path.basename(_filepath)
    if not _selected:
        print(f"  {_fname}: 選択カラム未設定 → スキップ")
        print(f"    (DOCTOR_ATTR_SELECTED / FACILITY_ATTR_SELECTED にカラム名を追加してください)")
        continue

    print(f"\n  {_fname} を読み込み中 ...")
    _attr_df, _new_dims = load_attr_file(_filepath, _id_col, _id_rename, _selected, CONTINUOUS_BINS)

    if _attr_df is None or not _new_dims:
        continue

    # panelへマージ (panel には doctor_id / facility_id 列が存在する)
    _dim_cols = [c for c, _ in _new_dims]
    panel = panel.merge(_attr_df[[_id_rename] + _dim_cols], on=_id_rename, how="left")

    # CATE_DIMSへ追加
    for _dim_name, _levels in _new_dims:
        CATE_DIMS.append((_dim_name, _levels))
        print(f"    CATE次元追加: {_dim_name} → {_levels}")

# DGPのmodifierが参照可能な次元があるか確認
has_true_mods = any(dim_name in TRUE_MODIFIERS for dim_name, _ in CATE_DIMS)

if has_true_mods:
    print("\n[DGPに組み込まれた処置効果の異質性 (modifier)]")
    print("  処置効果 = チャネル基本効果 x modifier")
    for dim_name, mods in TRUE_MODIFIERS.items():
        if any(dim_name == d for d, _ in CATE_DIMS):
            print(f"  {dim_name}:")
            for level, val in mods.items():
                bar = "#" * int(val * 20)
                print(f"    {level:<10}: {val:.2f}  {bar}")

# 属性分布
print("\n[属性分布]")
for attr, levels in CATE_DIMS:
    if attr not in panel.columns:
        continue
    dist = panel.drop_duplicates("unit_id").groupby(["treated", attr]).size().unstack(fill_value=0)
    print(f"\n  {attr}:")
    print(f"    {'':>8} " + " ".join(f"{c:>8}" for c in dist.columns))
    for idx in dist.index:
        lbl = "処置群" if idx == 1 else "対照群"
        print(f"    {lbl:>8} " + " ".join(f"{dist.loc[idx, c]:>8}" for c in dist.columns))


# ================================================================
# CATE推定
# ================================================================
print("\n" + "=" * 70)
print(" CATE推定: サブグループ別CS")
print("=" * 70)

N_BOOT = 150
cate_results = {}

for dim_name, levels in CATE_DIMS:
    print(f"\n{'='*50}")
    print(f"  次元: {dim_name}")
    print(f"{'='*50}")

    dim_results = {}
    for level in levels:
        unit_info = panel.drop_duplicates("unit_id")
        treated_units = unit_info[(unit_info["treated"] == 1) & (unit_info[dim_name] == level)]["unit_id"].unique()
        control_units = unit_info[unit_info["treated"] == 0]["unit_id"].unique()
        n_t = len(treated_units)
        print(f"\n  {level}: 処置群={n_t}, 対照群={len(control_units)}")

        if n_t < 2:
            print(f"    -> スキップ (N<2)")
            dim_results[level] = {"att": np.nan, "se": np.nan, "ci_lo": np.nan,
                                   "ci_hi": np.nan, "n": n_t, "boot_samples": [], "dynamic": None}
            continue

        sub_panel = panel[panel["unit_id"].isin(set(treated_units) | set(control_units))].copy()
        result = cs_with_bootstrap(sub_panel, n_boot=N_BOOT, label=level)
        overall, se, dynamic, boot_samples = result

        if overall is None:
            dim_results[level] = {"att": np.nan, "se": np.nan, "ci_lo": np.nan,
                                   "ci_hi": np.nan, "n": n_t, "boot_samples": [], "dynamic": None}
            continue

        ci_lo = overall - 1.96 * se
        ci_hi = overall + 1.96 * se
        z = overall / se if se > 0 else float("inf")
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"    ATT={overall:.1f}, SE={se:.1f}, 95%CI=[{ci_lo:.1f}, {ci_hi:.1f}] {sig}")

        dim_results[level] = {"att": overall, "se": se, "ci_lo": ci_lo, "ci_hi": ci_hi,
                               "n": n_t, "boot_samples": boot_samples, "dynamic": dynamic}

    cate_results[dim_name] = dim_results


# ================================================================
# サブグループ間の差の検定
# ================================================================
print("\n" + "=" * 70)
print(" サブグループ間の差の検定")
print("=" * 70)

diff_results = {}
for dim_name, levels in CATE_DIMS:
    print(f"\n  --- {dim_name} ---")
    valid_levels = [l for l in levels if not np.isnan(cate_results[dim_name][l]["att"])]
    if len(valid_levels) < 2:
        continue
    dim_diffs = {}
    for i, l1 in enumerate(valid_levels):
        for l2 in valid_levels[i+1:]:
            r1, r2 = cate_results[dim_name][l1], cate_results[dim_name][l2]
            diff = r1["att"] - r2["att"]
            b1, b2 = np.array(r1["boot_samples"]), np.array(r2["boot_samples"])
            n_common = min(len(b1), len(b2))
            if n_common > 0:
                boot_diff = b1[:n_common] - b2[:n_common]
                se_diff = np.std(boot_diff)
                z_diff = diff / se_diff if se_diff > 0 else float("inf")
                p_diff = 2 * (1 - stats.norm.cdf(abs(z_diff)))
                sig_diff = "***" if p_diff < 0.001 else "**" if p_diff < 0.01 else "*" if p_diff < 0.05 else "n.s."
            else:
                se_diff, p_diff, sig_diff = np.nan, np.nan, "N/A"
            print(f"    {l1} - {l2}: diff={diff:.1f}, SE={se_diff:.1f}, p={p_diff:.4f} {sig_diff}")
            dim_diffs[f"{l1} - {l2}"] = {"diff": diff, "se": se_diff, "p": p_diff, "sig": sig_diff}
    diff_results[dim_name] = dim_diffs


# ================================================================
# 推定結果サマリー
# ================================================================
print("\n" + "=" * 70)
print(" 推定結果サマリー")
print("=" * 70)

print(f"\n  {'次元':<16} {'レベル':<10} {'N':>4} {'ATT':>8} {'SE':>8} {'95%CI':>20}" +
      (f" {'DGP':>6}" if has_true_mods else ""))
print(f"  {'-' * (78 if has_true_mods else 70)}")

for dim_name, levels in CATE_DIMS:
    for level in levels:
        r = cate_results[dim_name][level]
        if np.isnan(r["att"]):
            att_s, se_s, ci_s = "   N/A", "   N/A", "        N/A         "
        else:
            att_s = f"{r['att']:>8.1f}"
            se_s = f"{r['se']:>8.1f}"
            ci_s = f"[{r['ci_lo']:>7.1f}, {r['ci_hi']:>7.1f}]"

        line = f"  {dim_name:<16} {level:<10} {r['n']:>4} {att_s} {se_s} {ci_s}"
        if has_true_mods:
            true_mod = TRUE_MODIFIERS.get(dim_name, {}).get(level, None)
            mod_s = f"{true_mod:>6.2f}" if true_mod is not None else "   N/A"
            line += f" {mod_s}"

        print(line)
    print()


# ================================================================
# 主な知見
# ================================================================
print("=" * 70)
print(" 主な知見")
print("=" * 70)

for dim_name, levels in CATE_DIMS:
    valid = {l: cate_results[dim_name][l] for l in levels
             if not np.isnan(cate_results[dim_name][l]["att"])}
    if len(valid) < 2:
        continue

    max_l = max(valid, key=lambda l: valid[l]["att"])
    min_l = min(valid, key=lambda l: valid[l]["att"])
    diff_val = valid[max_l]["att"] - valid[min_l]["att"]

    p_str = ""
    if dim_name in diff_results:
        for key, d in diff_results[dim_name].items():
            if (max_l in key and min_l in key):
                p_str = f" (p={d['p']:.3f} {d['sig']})"
                break

    dgp_check = ""
    if dim_name in TRUE_MODIFIERS:
        true_max = max(TRUE_MODIFIERS[dim_name], key=TRUE_MODIFIERS[dim_name].get)
        true_min = min(TRUE_MODIFIERS[dim_name], key=TRUE_MODIFIERS[dim_name].get)
        if max_l == true_max:
            dgp_check = " [DGP整合]"
        else:
            dgp_check = f" [DGP: {true_max}が最大のはず]"

    print(f"\n  [{dim_name}]")
    print(f"    効果が最大: {max_l} (ATT={valid[max_l]['att']:.1f}, N={valid[max_l]['n']})")
    print(f"    効果が最小: {min_l} (ATT={valid[min_l]['att']:.1f}, N={valid[min_l]['n']})")
    print(f"    差: {diff_val:.1f}{p_str}{dgp_check}")


# ================================================================
# 可視化 1: Forest plot
# ================================================================
print("\n" + "=" * 70)
print(" 可視化")
print("=" * 70)

colors_cycle = ["#4C72B0", "#C44E52", "#55A868", "#8172B2", "#CCB974", "#64B5CD"]

n_dims = len(CATE_DIMS)
n_plots = n_dims + (1 if has_true_mods else 0)
n_cols = min(3, max(1, n_plots))
n_rows = max(1, (n_plots + n_cols - 1) // n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
fig.suptitle("CATE分析: 属性別の異質的処置効果\n(Callaway-Sant'Anna, サブグループ別推定)",
             fontsize=13, fontweight="bold")
axes_flat = axes.flatten()

for idx, (dim_name, levels) in enumerate(CATE_DIMS):
    ax = axes_flat[idx]
    y_positions, y_labels, atts, ci_los, ci_his, bar_colors = [], [], [], [], [], []

    for i, level in enumerate(reversed(levels)):
        r = cate_results[dim_name][level]
        if np.isnan(r["att"]):
            continue
        y_positions.append(i)
        y_labels.append(f"{level} (N={r['n']})")
        atts.append(r["att"])
        ci_los.append(r["ci_lo"])
        ci_his.append(r["ci_hi"])
        bar_colors.append(colors_cycle[i % len(colors_cycle)])

    if not atts:
        ax.set_title(dim_name)
        continue

    ax.axvline(0, color="gray", lw=0.8, ls=":")
    for i, (yp, att, cl, ch, col) in enumerate(zip(y_positions, atts, ci_los, ci_his, bar_colors)):
        ax.errorbar(att, yp, xerr=[[att - cl], [ch - att]], fmt="o", color=col,
                     capsize=5, ms=8, capthick=1.5, elinewidth=1.5)
        ax.text(ch + 0.5, yp, f"{att:.1f}", va="center", fontsize=9)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("ATT")
    ax.set_title(dim_name)
    ax.grid(True, alpha=0.3, axis="x")

# ATT vs DGP modifier (only if applicable)
if has_true_mods:
    ax = axes_flat[n_dims]
    est_atts, true_mods, labels = [], [], []
    for dim_name, levels in CATE_DIMS:
        if dim_name not in TRUE_MODIFIERS:
            continue
        for level in levels:
            r = cate_results[dim_name][level]
            if np.isnan(r["att"]) or level not in TRUE_MODIFIERS[dim_name]:
                continue
            est_atts.append(r["att"])
            true_mods.append(TRUE_MODIFIERS[dim_name][level])
            labels.append(f"{dim_name}:{level}")

    if len(est_atts) > 1:
        ax.scatter(true_mods, est_atts, s=60, c="#4C72B0", zorder=5)
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, (true_mods[i], est_atts[i]),
                         textcoords="offset points", xytext=(5, 5), fontsize=7)
        z = np.polyfit(true_mods, est_atts, 1)
        x_line = np.linspace(min(true_mods) - 0.05, max(true_mods) + 0.05, 50)
        ax.plot(x_line, np.polyval(z, x_line), "r--", lw=1, alpha=0.7, label=f"slope={z[0]:.1f}")
        corr = np.corrcoef(true_mods, est_atts)[0, 1]
        ax.set_title(f"推定ATT vs DGP modifier (r={corr:.2f})")
        ax.legend(fontsize=9)

    ax.set_xlabel("DGP modifier")
    ax.set_ylabel("推定ATT")
    ax.grid(True, alpha=0.3)

# 未使用のaxesを非表示
for idx in range(n_plots, len(axes_flat)):
    axes_flat[idx].set_visible(False)

plt.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "cate_results.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  図を保存: {out_path}")

# 可視化 2: 動的効果 (動的効果データのある次元のみ)
plot_dims = [(d, l) for d, l in CATE_DIMS if any(
    cate_results[d][lv]["dynamic"] is not None for lv in l
)]

if plot_dims:
    n_dyn = len(plot_dims)
    n_dyn_cols = min(2, n_dyn)
    n_dyn_rows = max(1, (n_dyn + n_dyn_cols - 1) // n_dyn_cols)
    fig2, axes2 = plt.subplots(n_dyn_rows, n_dyn_cols, figsize=(8 * n_dyn_cols, 5 * n_dyn_rows), squeeze=False)
    fig2.suptitle("CATE動的効果: サブグループ別イベントスタディ", fontsize=13, fontweight="bold")
    markers = ["o", "s", "^", "D", "v", "p"]

    for idx, (dim_name, levels) in enumerate(plot_dims):
        ax = axes2[idx // n_dyn_cols, idx % n_dyn_cols]
        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(-0.5, color="red", ls="--", lw=0.8, alpha=0.5)
        for j, level in enumerate(levels):
            r = cate_results[dim_name][level]
            if r["dynamic"] is None:
                continue
            dyn = r["dynamic"]
            col = colors_cycle[j % len(colors_cycle)]
            ax.fill_between(dyn["event_time"], dyn["ci_lo"], dyn["ci_hi"], alpha=0.08, color=col)
            ax.plot(dyn["event_time"], dyn["att"], f"{markers[j % len(markers)]}-",
                    color=col, ms=3, label=f"{level} (N={r['n']})", lw=1)
        ax.set_xlabel("イベント時間 (月)")
        ax.set_ylabel("ATT")
        ax.set_title(dim_name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 未使用のaxesを非表示
    for idx in range(n_dyn, n_dyn_rows * n_dyn_cols):
        axes2[idx // n_dyn_cols, idx % n_dyn_cols].set_visible(False)

    plt.tight_layout()
    out_path2 = os.path.join(SCRIPT_DIR, "cate_dynamic_effects.png")
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  図を保存: {out_path2}")


# ================================================================
# 結論
# ================================================================
print("\n" + "=" * 70)
print(" 結論")
print("=" * 70)

print("""
  === 方法論 ===
  - Callaway-Sant'Anna (2021) をサブグループ別に適用
  - 処置群を属性レベルで分割し、共通の対照群と比較
  - Bootstrap (N=150) による標準誤差推定
  - サブグループ間差はBootstrap分布の差で検定""")

if has_true_mods:
    print("  === DGPの真の処置効果異質性 ===")
    for dim_name, mods in TRUE_MODIFIERS.items():
        if any(dim_name == d for d, _ in CATE_DIMS):
            vals = ", ".join(f"{k}={v}" for k, v in mods.items())
            print(f"    {dim_name}: {vals}")

print("\n  === 推定されたCATEの要約 ===")
for dim_name, levels in CATE_DIMS:
    valid = {l: cate_results[dim_name][l] for l in levels
             if not np.isnan(cate_results[dim_name][l]["att"])}
    if len(valid) < 2:
        continue
    max_l = max(valid, key=lambda l: valid[l]["att"])
    min_l = min(valid, key=lambda l: valid[l]["att"])
    print(f"    {dim_name}: {max_l}({valid[max_l]['att']:.1f}) > {min_l}({valid[min_l]['att']:.1f})")

print(f"""
  === 分析次元 ===
  使用次元: {', '.join(d for d, _ in CATE_DIMS)}
  (オプション属性ファイルが data/ に存在すれば自動拡張)

  === 注意点 ===
  - サブグループのNが小さいため検出力は限定的
  - 各サブグループのATTは共通の対照群を使用
  - baseline_catは処置前アウトカムからの導出 (内生性なし)
""")

# ================================================================
# JSON結果保存
# ================================================================
import json

results_dir = os.path.join(SCRIPT_DIR, "results")
os.makedirs(results_dir, exist_ok=True)

# CATE推定結果
cate_json = {}
for dim_name, levels in CATE_DIMS:
    dim_json = {}
    for level in levels:
        r = cate_results[dim_name][level]
        dim_json[level] = {
            "att": float(r["att"]) if not np.isnan(r["att"]) else None,
            "se": float(r["se"]) if not np.isnan(r["se"]) else None,
            "ci_lo": float(r["ci_lo"]) if not np.isnan(r["ci_lo"]) else None,
            "ci_hi": float(r["ci_hi"]) if not np.isnan(r["ci_hi"]) else None,
            "n": int(r["n"]),
        }
        if r["dynamic"] is not None:
            dim_json[level]["dynamic"] = r["dynamic"][
                ["event_time", "att", "se", "ci_lo", "ci_hi"]
            ].to_dict("records")
    cate_json[dim_name] = dim_json

# サブグループ間差の検定結果
diff_json = {}
for dim_name, diffs in diff_results.items():
    dim_diff = {}
    for key, d in diffs.items():
        dim_diff[key] = {
            "diff": float(d["diff"]),
            "se": float(d["se"]) if not np.isnan(d["se"]) else None,
            "p": float(d["p"]) if not np.isnan(d["p"]) else None,
            "sig": d["sig"],
        }
    diff_json[dim_name] = dim_diff

# 属性分布テーブル
attr_dist_json = {}
for attr, _ in CATE_DIMS:
    if attr not in panel.columns:
        continue
    dist = panel.drop_duplicates("unit_id").groupby(["treated", attr]).size().unstack(fill_value=0)
    attr_dist_json[attr] = {
        "treated": {str(c): int(dist.loc[1, c]) for c in dist.columns} if 1 in dist.index else {},
        "control": {str(c): int(dist.loc[0, c]) for c in dist.columns} if 0 in dist.index else {},
    }

# TRUE_MODIFIERS (使用された次元のみ)
true_mods_json = {}
if has_true_mods:
    for dim_name in TRUE_MODIFIERS:
        if any(dim_name == d for d, _ in CATE_DIMS):
            true_mods_json[dim_name] = TRUE_MODIFIERS[dim_name]

cate_results_json = {
    "cate": cate_json,
    "diff_tests": diff_json,
    "attr_distribution": attr_dist_json,
    "true_modifiers": true_mods_json,
    "dimensions": [{"name": d, "levels": l} for d, l in CATE_DIMS],
}

json_path = os.path.join(results_dir, "cate_results.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(cate_results_json, f, ensure_ascii=False, indent=2)
print(f"  結果をJSON保存: {json_path}")
