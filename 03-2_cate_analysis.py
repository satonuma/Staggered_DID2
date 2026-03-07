"""
CATE分析 ver2: 属性別の異質的処置効果（複数医師施設対応）
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
FILTER_SINGLE_FAC_DOCTOR = False  # ver2: 複数医師施設を含める
INCLUDE_ONLY_RW     = False        # True: RW医師のみ (Step 3適用)
INCLUDE_ONLY_NON_RW = False       # True: 非RW医師のみ (INCLUDE_ONLY_RW=Falseのとき有効)
EXCLUDE_ZERO_SALES_FACILITIES = False  # True: 全期間納入が0の施設を解析対象から除外
UHP_RANK = {"U": 0, "H": 1, "P": 2, "雑": 3}  # 数値小さいほど上位 (U>H>P>雑)

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
_suffix = _pop_sfx + _zero_sfx

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
BASELINE_START_MONTH_IDX = -12  # 解析開始前のベースライン期間開始 (2022/4 = month_index -12)
MIN_ET, MAX_ET = -6, 18

# 基本次元 (ver2: 施設属性のみ)
CATE_DIMS = [
    ("baseline_cat", ["低", "中", "高"]),  # ベースライン（施設売上）
]

# ===================================================================
# 属性ファイル設定 ― 分析するカラムをリストで指定
# ===================================================================

# --- 医師属性 (doctor_attribute.csv) ---
DOCTOR_ATTR_ID_COL = "doc"
DOCTOR_ATTR_SELECTED: list = []  # ver2: 施設属性のみ

# --- 施設属性 (facility_attribute_修正.csv) ---
FILE_FACILITY_ATTR = "facility_attribute_修正.csv"
FACILITY_ATTR_ID_COL = "fac_honin"
FACILITY_ATTR_SELECTED: list = [    # ← 分析したいカラム名をここに列挙
    "UHP区分名",
    "許可病床数_合計",
]

# 連続値カラムのカテゴリ化設定
CONTINUOUS_BINS: dict = {
    "許可病床数_合計": {"bins": [-1, 19, 199, np.inf], "labels": ["20床未満", "20~199床", "200床以上"]},
}

# 自動分位数（CONTINUOUS_BINS 未指定の連続変数に適用）
N_AUTO_BINS = 4   # 分位数


# ================================================================
# 属性ファイル読み込み・カテゴリ化ユーティリティ
# ================================================================

def _infer_unit(col_name):
    """カラム名から値の単位文字列を推定"""
    if col_name in ("年齢", "卒業時年齢") or "歳" in col_name:
        return "歳"
    if col_name in ("医師歴",) or ("歴" in col_name and "年" not in col_name):
        return "年"
    if "床" in col_name:
        return "床"
    return ""


# QUINTILE系カテゴリ (H/L/M/VH/Z) の表示順
# カテゴリ固定表示順序
_QUINTILE_ORDER = ["Z", "L", "M", "H", "VH", "不明"]
_BASELINE_ORDER = ["0以下", "低", "中", "高"]
_UHP_ORDER      = ["U", "H", "P", "雑", "不明"]  # U>H>P>雑 (規模大→小)


def _sort_levels(levels):
    """カテゴリレベルを固定順序で並べる。
    既知のカテゴリセット (Quintile/baseline/UHP) は固定順、
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

    # UHP区分名 (U/H/P/雑)
    if non_missing and all(v in {"U", "H", "P", "雑"} for v in non_missing):
        return [v for v in _UHP_ORDER if v in str_levels]

    return sorted(levels, key=str)  # 元の型を保持（int の 0/1 フラグも崩さない）


def _auto_range_labels(series, q=4, col_name=""):
    """連続変数をN数ベースのq分位でカテゴリ化し '最小~最大 単位' 形式ラベルを生成。
    不明以外で最大 q カテゴリを作成する。
    Returns (Categorical Series, levels_list)
    """
    unit = _infer_unit(col_name)
    s = series.dropna()
    if len(s) == 0:
        return pd.Categorical([pd.NA] * len(series)), []

    n_unique = s.nunique()
    actual_q = min(q, n_unique)

    if actual_q == 1:
        label = f"{int(round(s.iloc[0]))}{unit}"
        result = pd.Categorical(
            series.where(series.isna(), label), categories=[label]
        )
        return result, [label]

    try:
        _, bins = pd.qcut(s, q=actual_q, retbins=True, duplicates="drop")
        labels = []
        for i in range(len(bins) - 1):
            lo = int(round(bins[i]))
            hi = int(round(bins[i + 1]))
            labels.append(f"{lo}~{hi}{unit}")
        bins_cut = bins.copy()
        bins_cut[0] = bins_cut[0] - 0.001
        result = pd.cut(series, bins=bins_cut, labels=labels)
        return result, list(labels)
    except Exception:
        med = s.median()
        lo_label = f"{int(round(s.min()))}~{int(round(med))}{unit}"
        hi_label = f"{int(round(med))+1}~{int(round(s.max()))}{unit}"
        result = pd.cut(series, bins=[s.min() - 0.001, med, s.max()],
                        labels=[lo_label, hi_label])
        return result, [lo_label, hi_label]


def _baseline_4cat(series):
    """ベースライン納入額を '0以下' / '低' / '中' / '高' の4カテゴリに分類。
    0以下: <= 0 (ゼロ・マイナス)
    低/中/高: 正の値の3等分位
    """
    positive = series[series > 0]
    if len(positive) == 0:
        cat = pd.Categorical(["0以下"] * len(series), categories=["0以下"])
        return pd.Series(cat, index=series.index), ["0以下"]

    q33 = positive.quantile(1 / 3)
    q67 = positive.quantile(2 / 3)
    if q33 == q67:
        bins   = [-np.inf, 0, q67, np.inf]
        levels = ["0以下", "低", "高"]
    else:
        bins   = [-np.inf, 0, q33, q67, np.inf]
        levels = ["0以下", "低", "中", "高"]

    result = pd.cut(series, bins=bins, labels=levels, include_lowest=True)
    return result, levels


def _safe_qcut(series, q=3, labels=("低", "中", "高")):
    """重複binエッジ（ゼロ多数など）があっても動作するqcut。
    実際に作れるbin数に合わせてlabelsを自動調整する。
    Returns (Categorical Series, levels_list)
    """
    default_labels = list(labels)
    try:
        result = pd.qcut(series, q=q, labels=default_labels, duplicates="raise")
        return result, default_labels
    except ValueError:
        pass

    nonzero = series[series > 0]
    if len(nonzero) == 0:
        lvl = [default_labels[0]]
        cat = pd.Categorical([default_labels[0]] * len(series), categories=lvl)
        return pd.Series(cat, index=series.index), lvl

    med_nonzero = nonzero.median()
    max_val = series.max()

    if med_nonzero >= max_val:
        lvl = [default_labels[0], default_labels[-1]]
        result = pd.cut(series, bins=[-np.inf, 0, np.inf], labels=lvl)
        return result, lvl

    try:
        bins3 = [-np.inf, 0, med_nonzero, np.inf]
        lvl3 = default_labels[:3]
        result = pd.cut(series, bins=bins3, labels=lvl3)
        return result, lvl3
    except ValueError:
        lvl2 = [default_labels[0], default_labels[-1]]
        result = pd.cut(series, bins=[-np.inf, 0, np.inf], labels=lvl2)
        return result, lvl2


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
        result, levels = _auto_range_labels(df[col], q=N_AUTO_BINS, col_name=col)
        df[new_col] = result
        print(f"      → 自動{N_AUTO_BINS}分位でカテゴリ化: {levels}")

    n_null = df[new_col].isna().sum()
    if n_null > 0:
        df[new_col] = df[new_col].cat.add_categories("不明").fillna("不明")
        levels = list(levels) + ["不明"]
        print(f"      欠損/範囲外 {n_null} 件 → '不明' カテゴリに追加")

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

        force_continuous = is_numeric and (col in continuous_bins)
        if is_numeric and (n_unique > 10 or force_continuous):
            print(f"    [{col}] 連続値 (ユニーク={n_unique})")
            new_col, levels = _show_and_bin(df_out, col, continuous_bins)
            cate_dims.append((new_col, levels))
        else:
            n_null = series.isna().sum()
            if n_null > 0:
                df_out[col] = series.astype(object).fillna("不明")
                series = df_out[col]
                print(f"      欠損値 {n_null} 件 → '不明' カテゴリに追加")
            levels = _sort_levels(series.unique().tolist())
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
    # ── 1. pivot を1回だけ計算し numpy 配列に変換 ──────────────────────
    doc_info = panel.groupby("unit_id").agg(
        {"treated": "first", "cohort_month": "first"}
    )
    pivot = panel.pivot_table(
        values="amount", index="unit_id", columns="month_index", aggfunc="mean"
    )
    pivot_np   = pivot.to_numpy(dtype=float)
    time_cols  = np.asarray(pivot.columns)
    unit_order = pivot.index

    ctrl_mask = (doc_info.loc[unit_order, "treated"] == 0).values
    ctrl_np   = pivot_np[ctrl_mask]
    n_ctrl    = len(ctrl_np)
    if n_ctrl == 0:
        return None, None, None, None

    cohorts = sorted(doc_info["cohort_month"].dropna().unique())
    coh_data = {}
    for g in cohorts:
        g    = int(g)
        base = g - 1
        if base not in time_cols:
            continue
        base_col = int(np.where(time_cols == base)[0][0])
        cmask    = (doc_info.loc[unit_order, "cohort_month"] == g).values
        treat_np = pivot_np[cmask]
        if len(treat_np) == 0:
            continue
        coh_data[g] = {"base": base_col, "T": treat_np, "n": len(treat_np)}

    if not coh_data:
        return None, None, None, None

    # ── 2. 点推定 (ベクトル化) ─────────────────────────────────────────
    rows = []
    for g, cd in coh_data.items():
        b   = cd["base"]
        dT  = cd["T"]  - cd["T"][:, b:b+1]
        dC  = ctrl_np  - ctrl_np[:, b:b+1]
        att = np.nanmean(dT, axis=0) - np.nanmean(dC, axis=0)
        for ti, t in enumerate(time_cols):
            rows.append({"cohort": g, "time": int(t), "event_time": int(t) - g,
                         "att_gt": att[ti], "n_cohort": cd["n"]})
    att_gt = pd.DataFrame(rows)
    if len(att_gt) == 0:
        return None, None, None, None

    dynamic = aggregate_dynamic(att_gt)
    overall  = aggregate_overall(att_gt)

    # ── 3. ベクトル化ブートストラップ (全 n_boot 回を一括処理) ──────────
    tag = f"[{label}] " if label else ""
    print(f"    {tag}Bootstrap (n={n_boot}, vectorised) ...", end="", flush=True)

    boot_per_gt = {}
    for g, cd in coh_data.items():
        b    = cd["base"]
        n_t  = cd["n"]
        dT   = cd["T"]  - cd["T"][:, b:b+1]
        dC   = ctrl_np  - ctrl_np[:, b:b+1]
        idx_T = np.random.randint(0, n_t,    (n_boot, n_t))
        idx_C = np.random.randint(0, n_ctrl, (n_boot, n_ctrl))
        n_cols = len(time_cols)
        bT = np.empty((n_boot, n_cols))
        bC = np.empty((n_boot, n_cols))
        for col in range(n_cols):
            bT[:, col] = np.nanmean(dT[:, col][idx_T], axis=1)
            bC[:, col] = np.nanmean(dC[:, col][idx_C], axis=1)
        boot_att = bT - bC
        for ti in range(len(time_cols)):
            boot_per_gt[(g, ti)] = boot_att[:, ti]

    post_keys = [(g, ti) for g in coh_data
                 for ti, t in enumerate(time_cols) if int(t) - g >= 0]
    if post_keys:
        w    = np.array([coh_data[g]["n"] for g, ti in post_keys], float)
        w   /= w.sum()
        mat  = np.stack([boot_per_gt[k] for k in post_keys], axis=1)
        boot_overall_arr  = (mat * w).sum(1)
        se_overall        = float(np.std(boot_overall_arr))
        boot_overall_list = boot_overall_arr.tolist()
    else:
        se_overall        = 0.0
        boot_overall_list = []

    se_dyn_map = {}
    for et in dynamic["event_time"].values:
        keys = [(g, ti) for g in coh_data
                for ti, t in enumerate(time_cols) if int(t) - g == et]
        if not keys:
            se_dyn_map[int(et)] = 0.0
            continue
        w   = np.array([coh_data[g]["n"] for g, ti in keys], float)
        w  /= w.sum()
        mat = np.stack([boot_per_gt[k] for k in keys], axis=1)
        se_dyn_map[int(et)] = float(np.std((mat * w).sum(1)))

    print(" done")
    dynamic["se"]    = dynamic["event_time"].map(se_dyn_map).fillna(0)
    dynamic["ci_lo"] = dynamic["att"] - 1.96 * dynamic["se"]
    dynamic["ci_hi"] = dynamic["att"] + 1.96 * dynamic["se"]
    return overall, se_overall, dynamic, boot_overall_list


# ================================================================
# Part 1: データ読み込み (本番形式)
# ================================================================
print("=" * 70)
print(" CATE分析 ver2: 属性別の異質的処置効果（複数医師施設対応）")
print("=" * 70)

# 1. RW医師リスト (除外フローで使用; 絞り込みはPart 2で実施)
rw_list = pd.read_csv(os.path.join(DATA_DIR, FILE_RW_LIST))
n_rw_all = len(rw_list)

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
digital = digital[digital["fac_honin"].notna() & (digital["fac_honin"].astype(str).str.strip() != "")].copy()

# 4. 活動データ → Web講演会のみ抽出
activity_raw = pd.read_csv(os.path.join(DATA_DIR, FILE_ACTIVITY))
n_activity_all = len(activity_raw)
activity_raw["品目コード"] = activity_raw["品目コード"].astype(str).str.strip().str.zfill(5)
web_lecture = activity_raw[
    (activity_raw["品目コード"] == ENT_PRODUCT_CODE)
    & (activity_raw["活動種別"] == ACTIVITY_CHANNEL_FILTER)
].copy()
web_lecture = web_lecture[web_lecture["fac_honin"].notna() & (web_lecture["fac_honin"].astype(str).str.strip() != "")].copy()

# 5. 視聴データ結合 (デジタル + 活動Web講演会)
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
print(f"  デジタル視聴データ      : {n_digital_all:,} 行 → ENT品目: {len(digital):,} 行")
print(f"  活動データ              : {n_activity_all:,} 行 → Web講演会+ENT: {len(web_lecture):,} 行")
print(f"  視聴データ結合          : {len(viewing):,} 行")
print(f"  観測期間 : {months[0].strftime('%Y-%m')} ~ {months[-1].strftime('%Y-%m')} ({N_MONTHS}ヶ月)")

# ================================================================
# Part 2: 解析集団の絞り込み (ver2: 複数医師施設対応)
# ================================================================
print("\n" + "=" * 70)
print(" Part 2: 解析集団の絞り込み (ver2)")
print("=" * 70)

# 施設医師リスト: 全医師の施設マッピング
fac_doc_list = pd.read_csv(os.path.join(DATA_DIR, FILE_FAC_DOCTOR_LIST))

# [Step 1] facility_attribute_修正.csv 読み込み (施設属性用)
fac_df = pd.read_csv(os.path.join(DATA_DIR, FILE_FACILITY_MASTER))

# [Step 2] doctor_attribute.csv 読み込み (医師属性用)
doc_attr_df = pd.read_csv(os.path.join(DATA_DIR, FILE_DOCTOR_ATTR))

_doc_to_fac   = dict(zip(fac_doc_list["doc"], fac_doc_list["fac"]))
_doc_to_honin = dict(zip(fac_doc_list["doc"], fac_doc_list["fac_honin"]))
all_docs = set(fac_doc_list["doc"])

# --- 主施設割り当て: 医師ごとの平均納入額が最大の fac_honin を主施設とする ---
_doc_fac_list = fac_doc_list[["doc", "fac_honin"]].drop_duplicates()
_sales_by_fac = (
    daily.groupby("facility_id")["amount"].mean()
    .reset_index()
    .rename(columns={"facility_id": "fac_honin", "amount": "avg_sales"})
)
_doc_fac_sales = _doc_fac_list.merge(_sales_by_fac, on="fac_honin", how="left")
_doc_fac_sales["avg_sales"] = _doc_fac_sales["avg_sales"].fillna(0)

# 各医師: 最大平均売上の fac_honin を主施設に
_doc_primary_all = (
    _doc_fac_sales.sort_values("avg_sales", ascending=False)
    .groupby("doc")["fac_honin"].first()
)

# 全施設0の場合: UHP区分最上位の施設を主施設に
_fac_uhp = fac_df.drop_duplicates("fac_honin").set_index("fac_honin")["UHP区分名"] \
    if "UHP区分名" in fac_df.columns else pd.Series(dtype=str)
_zero_sum_docs = (
    _doc_fac_sales.groupby("doc")["avg_sales"].sum()
)
_zero_docs_set = set(_zero_sum_docs[_zero_sum_docs == 0].index)
for _doc in _zero_docs_set:
    _doc_facs = _doc_fac_sales[_doc_fac_sales["doc"] == _doc]["fac_honin"].tolist()
    if not _doc_facs:
        continue
    _ranked = sorted(
        _doc_facs,
        key=lambda f: UHP_RANK.get(str(_fac_uhp.get(f, "")), 99)
    )
    _doc_primary_all[_doc] = _ranked[0]

doc_primary = _doc_primary_all  # doc → fac_honin (主施設)

# Step 3: 医師フィルタ
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
fac_to_docs: dict = {}
for doc in analysis_docs_all:
    if doc in doc_primary.index:
        fac = doc_primary[doc]
        fac_to_docs.setdefault(fac, []).append(doc)

# --- 施設医師数 ---
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

print(f"\n  主施設割り当て完了:")
print(f"    解析対象医師数: {len(analysis_docs_all)}")
print(f"    解析対象施設数: {len(fac_to_docs)}")
_multi = sum(1 for docs in fac_to_docs.values() if len(docs) > 1)
print(f"    複数医師施設: {_multi} 施設")

# --- 視聴データに主施設IDを付与 ---
viewing_all = viewing.copy()
viewing_all["facility_id"] = viewing_all["doctor_id"].map(doc_primary)

# ウォッシュアウト除外: 施設内いずれかの医師が washout 期間(month 0,1)に視聴
washout_fac_ids = set(
    viewing_all[
        (viewing_all["view_date"] < months[WASHOUT_MONTHS]) &
        viewing_all["facility_id"].notna()
    ]["facility_id"]
)

# month_index を付与
viewing_all["month_index"] = (
    (viewing_all["view_date"].dt.year - 2023) * 12
    + viewing_all["view_date"].dt.month - 4
)

# 初回視聴月: 施設内最初の視聴月 (washout除外後, LAST_ELIGIBLE_MONTH 以内)
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

treated_fac_ids = set(_first_view.index)
control_fac_ids = set(fac_to_docs.keys()) - treated_fac_ids - washout_fac_ids
analysis_fac_ids = treated_fac_ids | control_fac_ids

print(f"\n  処置群 (視聴あり): {len(treated_fac_ids)} 施設")
print(f"  対照群 (視聴なし): {len(control_fac_ids)} 施設")
print(f"  ウォッシュアウト除外: {len(washout_fac_ids)} 施設")
print(f"  解析対象合計: {len(analysis_fac_ids)} 施設")

# コホート分布
cohort_dist = _first_view.value_counts().sort_index()
print(f"\n  コホート分布:")
for m, cnt in cohort_dist.items():
    ym = months[int(m)].strftime("%Y-%m")
    print(f"    month {int(m):>2} ({ym}): {cnt} 施設")

print(f"\n  チャネル別視聴者数 (施設):")
for ch in CONTENT_TYPES:
    n = viewing_all[
        (viewing_all["channel_category"] == ch)
        & (viewing_all["facility_id"].isin(treated_fac_ids))
    ]["facility_id"].nunique()
    print(f"    {ch:<12}: {n} 施設")

# ================================================================
# Part 3: パネルデータ構築
# ================================================================
print("\n" + "=" * 70)
print(" Part 3: パネルデータ構築")
print("=" * 70)

daily["month_index"] = (
    (daily["delivery_date"].dt.year - 2023) * 12
    + daily["delivery_date"].dt.month - 4
)

full_idx = pd.MultiIndex.from_product(
    [sorted(analysis_fac_ids), list(range(N_MONTHS))],
    names=["facility_id", "month_index"]
)
monthly = (
    daily[daily["facility_id"].isin(analysis_fac_ids)]
    .groupby(["facility_id", "month_index"])["amount"].sum().reset_index()
)
panel_base = (
    monthly.set_index(["facility_id", "month_index"])
    .reindex(full_idx, fill_value=0).reset_index()
)
panel_base["unit_id"] = panel_base["facility_id"]

# 処置・コホート変数を施設レベルで付与
_cohort_map = _first_view.to_dict()  # fac_honin → cohort_month
panel_base["cohort_month"] = panel_base["facility_id"].map(_cohort_map)
panel_base["treated"] = panel_base["cohort_month"].notna().astype(int)

panel = panel_base.copy()

n_treated = panel.loc[panel["treated"] == 1, "unit_id"].nunique()
n_control = panel.loc[panel["treated"] == 0, "unit_id"].nunique()
n_total = n_treated + n_control

print(f"  パネル行数 : {len(panel):,} ({n_total}施設 x {N_MONTHS}月)")
print(f"  処置群     : {n_treated} 施設")
print(f"  対照群     : {n_control} 施設")

# ================================================================
# 属性のマージ
# ================================================================
print("\n[属性のマージ]")

# ベースライン納入額 (解析開始前 BASELINE_START_MONTH_IDX 〜 WASHOUT_MONTHS-1 の平均: 例 2022/4-2023/5)
_bline_raw = (
    daily[
        daily["facility_id"].isin(analysis_fac_ids) &
        daily["month_index"].isin(range(BASELINE_START_MONTH_IDX, WASHOUT_MONTHS))
    ]
    .groupby("facility_id")["amount"].mean().reset_index()
)
_bline_raw.columns = ["unit_id", "baseline_amount"]
_all_units = pd.DataFrame({"unit_id": sorted(analysis_fac_ids)})
baseline = _all_units.merge(_bline_raw, on="unit_id", how="left").fillna({"baseline_amount": 0.0})
_bc_result, _bc_levels = _baseline_4cat(baseline["baseline_amount"])
baseline["baseline_cat"] = _bc_result
CATE_DIMS[0] = ("baseline_cat", _bc_levels)  # 実際のカテゴリ数に合わせてlevelsを更新
panel = panel.merge(baseline[["unit_id", "baseline_cat"]], on="unit_id", how="left")
_n_bline_months = WASHOUT_MONTHS - BASELINE_START_MONTH_IDX
print(f"  baseline_cat: 前処置期間{_n_bline_months}ヶ月平均から4カテゴリ (0以下/低/中/高) → {_bc_levels}")

# 施設属性: n_docs を fac_df にマージ
panel["n_docs"] = panel["facility_id"].map(n_docs_map).fillna(1).astype(int)

# 属性ファイルの読み込み・カテゴリ化・マージ
# ver2: 医師属性 (DOCTOR_ATTR_SELECTED = []) はスキップ、施設属性のみ
_attr_configs = [
    (os.path.join(DATA_DIR, FILE_FACILITY_ATTR), FACILITY_ATTR_ID_COL, "facility_id", FACILITY_ATTR_SELECTED),
]

for _filepath, _id_col, _id_rename, _selected in _attr_configs:
    _fname = os.path.basename(_filepath)
    if not _selected:
        print(f"  {_fname}: 選択カラム未設定 → スキップ")
        continue

    print(f"\n  {_fname} を読み込み中 ...")
    _attr_df, _new_dims = load_attr_file(_filepath, _id_col, _id_rename, _selected, CONTINUOUS_BINS)

    if _attr_df is None or not _new_dims:
        continue

    # panelへマージ (panel には facility_id 列が存在する)
    _dim_cols = [c for c, _ in _new_dims]
    panel = panel.merge(_attr_df[[_id_rename] + _dim_cols], on=_id_rename, how="left")

    # CATE_DIMSへ追加
    for _dim_name, _levels in _new_dims:
        CATE_DIMS.append((_dim_name, _levels))
        print(f"    CATE次元追加: {_dim_name} → {_levels}")

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
print(" CATE推定: サブグループ別CS (ver2: 施設レベル)")
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

print(f"\n  {'次元':<16} {'レベル':<10} {'N':>4} {'ATT':>8} {'SE':>8} {'95%CI':>20}")
print(f"  {'-' * 70}")

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
            if set(key.split(" - ")) == {str(max_l), str(min_l)}:
                p_str = f" (p={d['p']:.3f} {d['sig']})"
                break

    print(f"\n  [{dim_name}]")
    print(f"    効果が最大: {max_l} (ATT={valid[max_l]['att']:.1f}, N={valid[max_l]['n']})")
    print(f"    効果が最小: {min_l} (ATT={valid[min_l]['att']:.1f}, N={valid[min_l]['n']})")
    print(f"    差: {diff_val:.1f}{p_str}")


# ================================================================
# 可視化 1: Forest plot
# ================================================================
print("\n" + "=" * 70)
print(" 可視化")
print("=" * 70)

colors_cycle = ["#4C72B0", "#C44E52", "#55A868", "#8172B2", "#CCB974", "#64B5CD"]

n_dims = len(CATE_DIMS)
n_cols = min(3, max(1, n_dims))
n_rows = max(1, (n_dims + n_cols - 1) // n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
fig.suptitle("CATE分析 ver2: 施設属性別の異質的処置効果\n(Callaway-Sant'Anna, サブグループ別推定)",
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

# 未使用のaxesを非表示
for idx in range(n_dims, len(axes_flat)):
    axes_flat[idx].set_visible(False)

plt.tight_layout()
out_path = os.path.join(SCRIPT_DIR, f"cate_results_v2{_suffix}.png")
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
    fig2.suptitle("CATE動的効果 ver2: 施設属性別サブグループ別イベントスタディ", fontsize=13, fontweight="bold")
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
    out_path2 = os.path.join(SCRIPT_DIR, f"cate_dynamic_effects_v2{_suffix}.png")
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
  === 方法論 (ver2: 施設レベル) ===
  - Callaway-Sant'Anna (2021) をサブグループ別に適用
  - 分析単位: 施設 (facility_id) ← ver1 の医師個人から変更
  - 複数医師所属施設を含む (FILTER_SINGLE_FAC_DOCTOR=False)
  - 処置: 施設内いずれかの医師が初回視聴した月 (バイナリ)
  - 処置群を施設属性レベルで分割し、共通の対照群と比較
  - Bootstrap (N=150) による標準誤差推定
  - サブグループ間差はBootstrap分布の差で検定""")

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

  === 注意点 ===
  - サブグループのNが小さいため検出力は限定的
  - 各サブグループのATTは共通の対照群を使用
  - baseline_catは処置前アウトカムからの導出 (内生性なし)
  - 施設レベル分析のため ver1 (医師個人) と結果が異なる場合あり
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
        dim_json[str(level)] = {
            "att": float(r["att"]) if not np.isnan(r["att"]) else None,
            "se": float(r["se"]) if not np.isnan(r["se"]) else None,
            "ci_lo": float(r["ci_lo"]) if not np.isnan(r["ci_lo"]) else None,
            "ci_hi": float(r["ci_hi"]) if not np.isnan(r["ci_hi"]) else None,
            "n": int(r["n"]),
        }
        if r["dynamic"] is not None:
            dim_json[str(level)]["dynamic"] = r["dynamic"][
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

cate_results_json = {
    "cate": cate_json,
    "diff_tests": diff_json,
    "attr_distribution": attr_dist_json,
    "dimensions": [{"name": d, "levels": [str(lv) for lv in l]} for d, l in CATE_DIMS],
    "version": "v2",
    "note": "施設レベル分析 (複数医師施設対応)",
}

json_path = os.path.join(results_dir, f"cate_results_v2{_suffix}.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(cate_results_json, f, ensure_ascii=False, indent=2)
print(f"  結果をJSON保存: {json_path}")
