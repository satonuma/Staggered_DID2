"""
===================================================================
09_psm_growth_rate.py
視聴群 vs 未視聴群: 傾向スコアマッチング(PSM)による伸長率比較
===================================================================
目的:
  ウォッシュアウト期間（2023/4-5）に視聴していた医師を除外した母集団で、
  解析期間中に視聴を開始した「視聴群」と全期間未視聴の「未視聴群」を
  1:1 傾向スコアマッチング（最近傍、キャリパー付き）で比較。
  アウトカム: 後期月平均 - 前期月平均（万円差）= 伸長率

サブグループ設定 (SUBGROUP_SPECS):
  (表示名, unit_df内の列名, 連続変数か, 単位文字列)
  - 列が存在しない場合は自動スキップ
  - 連続変数 (is_continuous=True) は自動で N_BINS_CONTINUOUS 分位カテゴリ化
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

FILE_RW_LIST           = "rw_list.csv"
FILE_SALES             = "sales.csv"
FILE_DIGITAL           = "デジタル視聴データ.csv"
FILE_ACTIVITY          = "活動データ.csv"
FILE_DOCTOR_ATTR       = "doctor_attribute.csv"
FILE_FACILITY_MASTER   = "facility_attribute_修正.csv"
FILE_FAC_DOCTOR_LIST   = "施設医師リスト.csv"

FILTER_SINGLE_FAC_DOCTOR   = True
DOCTOR_HONIN_FAC_COUNT_COL = "所属施設数"
INCLUDE_ONLY_RW            = False

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

CALIPER_MULTIPLIER = 0.2
RANDOM_SEED        = 42
N_BINS_CONTINUOUS  = 4   # 連続変数の自動ビン数

# ===================================================================
# サブグループ分析設定
#   (表示名, unit_df の列名, is_continuous, 単位文字列)
#   列が存在しない場合は自動スキップ
# ===================================================================
SUBGROUP_SPECS = [
    ("医師歴区分",               "experience_cat",                 False, ""),
    ("ベースライン納入額",       "baseline_cat",                   False, ""),
    ("年齢",                     "年齢",                           True,  "歳"),
    ("デジタルチャネル選好",     "DIGITAL_CHANNEL_PREFERENCE",     False, ""),
    ("医師セグメント",           "DOCTOR_SEGEMNT",                 False, ""),
    ("経営層フラグ",             "経営層_統合_flg",                False, ""),
    ("オウンドメディアER五分位", "OWNED_MEDIA_ER_QUINTILE_FINAL",  False, ""),
    ("WebセミナーER五分位",      "WEB_SEMINAR_ER_QUINTILE_FINAL",  False, ""),
    ("非デジタルチャネル選好",   "NON_DIGITAL_CHANNEL_PREFERENCE", False, ""),
]

# ===================================================================
# ユーティリティ関数
# ===================================================================

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
print(" 09: PSM による伸長率比較（視聴群 vs 未視聴群）")
print("     サブグループ: " + str(len(SUBGROUP_SPECS)) + " 次元")
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

common_cols = ["活動日_dt", "品目コード", "活動種別", "fac_honin", "doc"]
viewing = pd.concat([digital[common_cols], web_lecture[common_cols]], ignore_index=True)
viewing = viewing.rename(columns={
    "活動日_dt": "view_date", "fac_honin": "facility_id", "doc": "doctor_id"
})
viewing["view_date"] = pd.to_datetime(viewing["view_date"], format="mixed")

months = pd.date_range(start=START_DATE, periods=N_MONTHS, freq="MS")
print("  データ読み込み完了")

# ===================================================================
# [2] 除外フロー（他スクリプトと同一）
# ===================================================================
print("\n[2] 除外フロー")

fac_doc_list = pd.read_csv(os.path.join(DATA_DIR, FILE_FAC_DOCTOR_LIST))
fac_df       = pd.read_csv(os.path.join(DATA_DIR, FILE_FACILITY_MASTER))
doc_attr_df  = pd.read_csv(os.path.join(DATA_DIR, FILE_DOCTOR_ATTR))

single_staff_fac = set(fac_df[fac_df["施設内医師数"] == 1]["fac"])

if FILTER_SINGLE_FAC_DOCTOR and DOCTOR_HONIN_FAC_COUNT_COL in doc_attr_df.columns:
    single_honin_docs = set(doc_attr_df[doc_attr_df[DOCTOR_HONIN_FAC_COUNT_COL] == 1]["doc"])
else:
    single_honin_docs = set(doc_attr_df["doc"])

_doc_to_fac   = dict(zip(fac_doc_list["doc"], fac_doc_list["fac"]))
_doc_to_honin = dict(zip(fac_doc_list["doc"], fac_doc_list["fac_honin"]))
all_docs = set(fac_doc_list["doc"])

after_step1 = {d for d in all_docs if _doc_to_fac.get(d) in single_staff_fac}
after_step2 = after_step1 & single_honin_docs
after_step3 = after_step2 if not INCLUDE_ONLY_RW else after_step2 & set(rw_list["doc"])

_honin_cnt: dict = {}
for d in after_step3:
    h = _doc_to_honin[d]
    _honin_cnt[h] = _honin_cnt.get(h, 0) + 1
candidate_docs = {d for d in after_step3 if _honin_cnt[_doc_to_honin[d]] == 1}

_pair_src   = rw_list if INCLUDE_ONLY_RW else fac_doc_list
clean_pairs = (_pair_src[_pair_src["doc"].isin(candidate_docs)][["doc", "fac_honin"]]
               .drop_duplicates())
clean_pairs = clean_pairs[
    clean_pairs["fac_honin"].notna()
    & (~clean_pairs["fac_honin"].astype(str).str.strip().isin(["", "nan"]))
].copy()
clean_pairs = clean_pairs.rename(columns={"doc": "doctor_id", "fac_honin": "facility_id"})

fac_to_doc  = dict(zip(clean_pairs["facility_id"], clean_pairs["doctor_id"]))
doc_to_fac  = dict(zip(clean_pairs["doctor_id"],   clean_pairs["facility_id"]))
clean_doc_ids = set(clean_pairs["doctor_id"])

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

clean_doc_ids -= set(first_view[first_view["first_view_month"] > LAST_ELIGIBLE_MONTH]["doctor_id"])

all_viewing_doc_ids = set(viewing["doctor_id"].unique())
treated_doc_ids = (
    set(first_view[first_view["first_view_month"] <= LAST_ELIGIBLE_MONTH]["doctor_id"])
    & clean_doc_ids
)
control_doc_ids  = clean_doc_ids - all_viewing_doc_ids
analysis_doc_ids = treated_doc_ids | control_doc_ids
analysis_fac_ids = {doc_to_fac[d] for d in analysis_doc_ids}

print("  ウォッシュアウト除外（継続視聴者）: " + str(len(washout_viewers)))
print("  処置群（視聴あり）: " + str(len(treated_doc_ids))
      + "  対照群（未視聴）: " + str(len(control_doc_ids)))

# ===================================================================
# [3] パネルデータ + 前期・後期集計
# ===================================================================
print("\n[3] 前後期売上集計")

daily_target = daily[daily["facility_id"].isin(analysis_fac_ids)].copy()
monthly = daily_target.groupby(["facility_id", "month_index"])["amount"].sum().reset_index()

full_idx = pd.MultiIndex.from_product(
    [sorted(analysis_fac_ids), list(range(N_MONTHS))],
    names=["facility_id", "month_index"],
)
panel = (monthly.set_index(["facility_id", "month_index"])
         .reindex(full_idx, fill_value=0).reset_index())
panel["doctor_id"] = panel["facility_id"].map(fac_to_doc)
panel["treated"]   = panel["doctor_id"].isin(treated_doc_ids).astype(int)

pre_avg  = (panel[panel["month_index"] <= PRE_END]
            .groupby("doctor_id")["amount"].mean().rename("pre_mean"))
post_avg = (panel[panel["month_index"] >= POST_START]
            .groupby("doctor_id")["amount"].mean().rename("post_mean"))

unit_df = (clean_pairs[clean_pairs["doctor_id"].isin(analysis_doc_ids)]
           .set_index("doctor_id").join(pre_avg).join(post_avg).reset_index())
unit_df["growth"]  = unit_df["post_mean"] - unit_df["pre_mean"]
unit_df["treated"] = unit_df["doctor_id"].isin(treated_doc_ids).astype(int)

# ===================================================================
# [4] 属性マージ（doctor_attribute.csv の全カラムを取り込む）
# ===================================================================
print("\n[4] 属性マージ")

doc_attr_df2 = doc_attr_df.rename(columns={"doc": "doctor_id"})

def _exp_cat(y):
    if y <= 10:   return "若手"
    elif y <= 20: return "中堅"
    return "ベテラン"

if "医師歴" in doc_attr_df2.columns:
    doc_attr_df2["experience_cat"] = doc_attr_df2["医師歴"].apply(_exp_cat)

# 名前カラム以外を全てマージ（IDカラムを除く）
exclude_cols = {"doctor_id", "doc_name", "DCF医師コード"}
attr_cols_to_merge = [c for c in doc_attr_df2.columns
                      if c not in exclude_cols and c != "doctor_id"]
unit_df = unit_df.merge(
    doc_attr_df2[["doctor_id"] + attr_cols_to_merge],
    on="doctor_id", how="left"
)

# 施設属性マージ
fac_df2 = fac_df.rename(columns={"fac_honin": "facility_id"})
fac_cols = [c for c in fac_df2.columns if c not in {"facility_id", "fac", "fac_honin_name"}]
unit_df = unit_df.merge(fac_df2[["facility_id"] + fac_cols], on="facility_id", how="left")

# ベースライン納入額カテゴリ（0以下/低/中/高）を追加
_bc_result, _bc_levels = _baseline_4cat(unit_df["pre_mean"])
unit_df["baseline_cat"] = _bc_result
print("  baseline_cat: " + str(_bc_levels))
print("  doctor_attribute 読み込み済みカラム: " + str(attr_cols_to_merge))

# ===================================================================
# [5] 傾向スコア推定
# ===================================================================
print("\n[5] 傾向スコア推定（Logistic Regression + L2正則化）")

from sklearn.linear_model import LogisticRegression

ps_data = unit_df.dropna(
    subset=["experience_cat", "DOCTOR_SEGEMNT", "DIGITAL_CHANNEL_PREFERENCE",
            "施設区分名", "UHP区分名", "pre_mean"]
).copy()

ps_dummies = pd.get_dummies(
    ps_data,
    columns=["experience_cat", "DOCTOR_SEGEMNT", "DIGITAL_CHANNEL_PREFERENCE",
             "施設区分名", "UHP区分名"],
    drop_first=True,
)
ps_dummies["pre_mean_std"] = (
    (ps_dummies["pre_mean"] - ps_dummies["pre_mean"].mean())
    / (ps_dummies["pre_mean"].std() + 1e-9)
)
if "医師歴" in ps_dummies.columns:
    ps_dummies["exp_years_std"] = (
        (ps_dummies["医師歴"] - ps_dummies["医師歴"].mean())
        / (ps_dummies["医師歴"].std() + 1e-9)
    )
else:
    ps_dummies["exp_years_std"] = 0.0

cov_cols = [c for c in ps_dummies.columns
            if c.startswith(("experience_cat_", "DOCTOR_SEGEMNT_",
                             "DIGITAL_CHANNEL_PREFERENCE_",
                             "施設区分名_", "UHP区分名_"))
            ] + ["pre_mean_std", "exp_years_std"]

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
    print("  警告: 完全分離 -> 前期売上+医師歴のみで単変量推定")
    try:
        X_simple = ps_dummies[["pre_mean_std", "exp_years_std"]].values
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
      + " / " + str(len(treated_for_match)) + " 処置群")

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

print("  全体 ATT=" + str(round(att, 2)) + " 万円/月  p=" + str(round(p_val, 4)))

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
        cat_levels = [str(v) for v in ps_data[cat_col].dropna().unique()
                      if str(v) not in ("nan", "None", "不明")]

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
# [8] 可視化
# ===================================================================
print("\n[8] 可視化")

# ---- (a) 傾向スコア分布 ----
fig_main, axes = plt.subplots(1, 2, figsize=(14, 5))
fig_main.suptitle("09: PSM 伸長率比較 - 傾向スコア分布とマッチング後伸長率",
                  fontsize=12, fontweight="bold")

ax = axes[0]
bins_ps = np.linspace(0, 1, 25)
ax.hist(ps_data[ps_data["treated"] == 1]["ps"], bins=bins_ps, alpha=0.45,
        color="#1565C0", label="視聴群（全体）")
ax.hist(ps_data[ps_data["treated"] == 0]["ps"], bins=bins_ps, alpha=0.45,
        color="#FF8F00", label="未視聴群（全体）")
ax.hist(matched_df[matched_df["treated"] == 1]["ps"], bins=bins_ps,
        alpha=0.9, histtype="step", linewidth=2, color="#1565C0", label="視聴群（マッチ後）")
ax.hist(matched_df[matched_df["treated"] == 0]["ps"], bins=bins_ps,
        alpha=0.9, histtype="step", linewidth=2, color="#FF8F00", label="未視聴群（マッチ後）")
ax.set_xlabel("傾向スコア")
ax.set_ylabel("医師数")
ax.set_title("(a) 傾向スコア分布（マッチング前後）")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

ax = axes[1]
mt_v = mt_growth[valid_mask]
mc_v = mc_growth[valid_mask]
ax.hist(mt_v, bins=20, alpha=0.6, color="#1565C0",
        label="視聴群 (N=" + str(valid_mask.sum()) + ")")
ax.hist(mc_v, bins=20, alpha=0.6, color="#FF8F00",
        label="未視聴群 (N=" + str(valid_mask.sum()) + ")")
ax.axvline(mt_v.mean(), color="#1565C0", linewidth=2, linestyle="--",
           label="視聴群 平均=" + str(round(mt_v.mean(), 1)))
ax.axvline(mc_v.mean(), color="#FF8F00", linewidth=2, linestyle="--",
           label="未視聴群 平均=" + str(round(mc_v.mean(), 1)))
ax.set_xlabel("伸長率（後期月平均 - 前期月平均, 万円）")
ax.set_ylabel("医師数")
ax.set_title("(b) マッチング後 伸長率分布\nATT=" + str(round(att, 2))
             + " 万円/月, p=" + str(round(p_val, 4)))
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
out_main = os.path.join(SCRIPT_DIR, "psm_growth_rate.png")
plt.savefig(out_main, dpi=150, bbox_inches="tight")
plt.close(fig_main)

# ---- (b) サブグループ Forest Plot ----
valid_sg_df = sg_df[sg_df["att"].notna()].copy()
n_rows      = len(valid_sg_df) + 1  # +1 for overall

fig_h = max(6, n_rows * 0.55 + 2)
fig2, ax2 = plt.subplots(figsize=(14, fig_h))
fig2.suptitle("09: サブグループ別 ATT Forest Plot（PSM, 後期-前期差）",
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
ax2.set_xlabel("ATT（万円/月、後期-前期差の視聴群-対照群差）", fontsize=10)
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
out_forest = os.path.join(SCRIPT_DIR, "psm_subgroup_forest.png")
plt.savefig(out_forest, dpi=150, bbox_inches="tight")
plt.close(fig2)

print("  psm_growth_rate.png を保存")
print("  psm_subgroup_forest.png を保存")

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
        "pre_period":  "month 0-" + str(PRE_END) + " (" + str(PRE_END + 1) + "ヶ月)",
        "post_period": "month " + str(POST_START) + "-32 (" + str(33 - POST_START) + "ヶ月)",
        "outcome": "後期月平均 - 前期月平均（万円）",
        "caliper": float(caliper) if caliper is not None else None,
        "matching": "1:1 nearest-neighbor within caliper",
        "n_subgroup_dimensions": len(SUBGROUP_SPECS),
    },
    "sample_sizes": {
        "washout_excluded": int(len(washout_viewers)),
        "treated_raw": int(len(treated_doc_ids)),
        "control_raw": int(len(control_doc_ids)),
        "matched_pairs": int(len(matched_pairs)),
    },
    "overall_att": {
        "att": float(att), "se": float(se_att), "t_stat": float(t_stat),
        "p_value": float(p_val), "ci_95_lower": float(ci_lo), "ci_95_upper": float(ci_hi),
    },
    "subgroup_att": [_sg_to_dict(r) for r in all_sg_results],
    "baseline_cat_levels": _bc_levels,
    "interpretation": {
        "note": "PSMはobservable confoundersのみ調整。未観測交絡は残る。",
        "att_definition": "視聴群の（後期-前期）成長率 と 対照群の（後期-前期）成長率 の差",
    },
}

json_path = os.path.join(results_dir, "psm_growth_rate.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results_json, f, ensure_ascii=False, indent=2)
print("  結果JSON: " + json_path)

# ===================================================================
print("\n" + "=" * 70)
print(" 分析完了")
print("=" * 70)
print("\n【全体 ATT】 " + str(round(att, 2)) + " 万円/月"
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
print("\n【注意】視聴は医師の自発行動 -> 未観測交絡が残るため因果解釈は慎重に。")
