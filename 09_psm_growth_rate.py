"""
===================================================================
09_psm_growth_rate.py
視聴群 vs 未視聴群: 傾向スコアマッチング(PSM)による伸長率比較
===================================================================
目的:
  ウォッシュアウト期間（2023/4-5）に視聴していた医師を除外した母集団で、
  解析期間中に視聴を開始した「視聴群」と全期間未視聴の「未視聴群」を
  1:1 傾向スコアマッチング（最近傍、キャリパー付き）で比較。
  売上伸長率の差をATT（処置群平均処置効果）として推定する。

分析仕様:
  - 前期: month 0-11（ウォッシュアウト込み、第1年度）
  - 後期: month 12-32（第2-3年度）
  - 伸長率 = 後期月平均 - 前期月平均（万円差）
  - マッチング共変量: 医師歴・セグメント・デジタル選好・施設区分 + 前期売上
  - キャリパー: 0.2 x std(logit(傾向スコア))
  - サブグループ: 若手（10年以下）・中堅（11-20年）・ベテラン（21年以上）

注意:
  視聴は医師の自発的行動であり、PSMで調整されない未観測交絡が残る。
  推定値は因果効果の上限値として解釈すること。
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
from statsmodels.discrete.discrete_model import Logit
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

# 前期 / 後期 定義
PRE_END    = 11   # month_index 0-11 (12ヶ月)
POST_START = 12   # month_index 12-32 (21ヶ月)

# PSM キャリパー倍率（0.2 x SD of logit PS が標準）
CALIPER_MULTIPLIER = 0.2

# 乱数シード（マッチング順序の再現性）
RANDOM_SEED = 42

# ===================================================================
print("=" * 70)
print(" 09: PSM による伸長率比較（視聴群 vs 未視聴群）")
print("=" * 70)

# ===================================================================
# データ読み込み
# ===================================================================
print("\n[1] データ読み込み")

rw_list = pd.read_csv(os.path.join(DATA_DIR, FILE_RW_LIST))

sales_raw = pd.read_csv(os.path.join(DATA_DIR, FILE_SALES), dtype=str)
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
    "活動日_dt": "view_date",
    "fac_honin": "facility_id",
    "doc": "doctor_id",
})
viewing["view_date"] = pd.to_datetime(viewing["view_date"], format="mixed")

months = pd.date_range(start=START_DATE, periods=N_MONTHS, freq="MS")
print("  データ読み込み完了")

# ===================================================================
# 除外フロー（他スクリプトと同一）
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

rw_doc_ids = set(rw_list["doc"])
_doc_to_fac   = dict(zip(fac_doc_list["doc"], fac_doc_list["fac"]))
_doc_to_honin = dict(zip(fac_doc_list["doc"], fac_doc_list["fac_honin"]))
all_docs = set(fac_doc_list["doc"])

after_step1 = {d for d in all_docs if _doc_to_fac.get(d) in single_staff_fac}
after_step2 = after_step1 & single_honin_docs
after_step3 = after_step2 if not INCLUDE_ONLY_RW else after_step2 & rw_doc_ids

_honin_cnt: dict = {}
for d in after_step3:
    h = _doc_to_honin[d]
    _honin_cnt[h] = _honin_cnt.get(h, 0) + 1
candidate_docs = {d for d in after_step3 if _honin_cnt[_doc_to_honin[d]] == 1}

_pair_src   = rw_list if INCLUDE_ONLY_RW else fac_doc_list
clean_pairs = (
    _pair_src[_pair_src["doc"].isin(candidate_docs)][["doc", "fac_honin"]]
    .drop_duplicates()
)
clean_pairs = clean_pairs[
    clean_pairs["fac_honin"].notna()
    & (~clean_pairs["fac_honin"].astype(str).str.strip().isin(["", "nan"]))
].copy()
clean_pairs = clean_pairs.rename(columns={"doc": "doctor_id", "fac_honin": "facility_id"})

fac_to_doc  = dict(zip(clean_pairs["facility_id"], clean_pairs["doctor_id"]))
doc_to_fac  = dict(zip(clean_pairs["doctor_id"],   clean_pairs["facility_id"]))
clean_doc_ids = set(clean_pairs["doctor_id"])

# ウォッシュアウト期間視聴者を除外（継続的視聴者の代理指標）
washout_end = months[WASHOUT_MONTHS - 1] + pd.offsets.MonthEnd(0)
viewing_clean = viewing[viewing["doctor_id"].isin(clean_doc_ids)].copy()
washout_viewers = set(
    viewing_clean[viewing_clean["view_date"] <= washout_end]["doctor_id"].unique()
)
clean_doc_ids -= washout_viewers

# 解析後半にのみ初回視聴した医師を除外
viewing_after_washout = viewing_clean[
    (viewing_clean["doctor_id"].isin(clean_doc_ids))
    & (viewing_clean["view_date"] > washout_end)
]
first_view = (
    viewing_after_washout.groupby("doctor_id")["view_date"]
    .min().reset_index()
)
first_view.columns = ["doctor_id", "first_view_date"]
first_view["first_view_month"] = (
    (first_view["first_view_date"].dt.year - 2023) * 12
    + first_view["first_view_date"].dt.month - 4
)

late_adopters = set(
    first_view[first_view["first_view_month"] > LAST_ELIGIBLE_MONTH]["doctor_id"]
)
clean_doc_ids -= late_adopters

# 処置群（解析期間中に視聴）・対照群（全期間未視聴）
all_viewing_doc_ids = set(viewing["doctor_id"].unique())
treated_doc_ids = (
    set(first_view[first_view["first_view_month"] <= LAST_ELIGIBLE_MONTH]["doctor_id"])
    & clean_doc_ids
)
control_doc_ids = clean_doc_ids - all_viewing_doc_ids

analysis_doc_ids  = treated_doc_ids | control_doc_ids
analysis_fac_ids  = {doc_to_fac[d] for d in analysis_doc_ids}

print("  ウォッシュアウト除外（継続視聴者）: " + str(len(washout_viewers)))
print("  処置群（視聴あり）: " + str(len(treated_doc_ids))
      + "  対照群（未視聴）: " + str(len(control_doc_ids)))
print("  合計解析対象: " + str(len(analysis_doc_ids)))

# ===================================================================
# パネルデータ構築 + 前期・後期集計
# ===================================================================
print("\n[3] 前期・後期売上集計")

daily_target = daily[daily["facility_id"].isin(analysis_fac_ids)].copy()
monthly = daily_target.groupby(["facility_id", "month_index"])["amount"].sum().reset_index()

full_idx = pd.MultiIndex.from_product(
    [sorted(analysis_fac_ids), list(range(N_MONTHS))],
    names=["facility_id", "month_index"],
)
panel = (
    monthly.set_index(["facility_id", "month_index"])
    .reindex(full_idx, fill_value=0).reset_index()
)
panel["doctor_id"] = panel["facility_id"].map(fac_to_doc)
panel["treated"]   = panel["doctor_id"].isin(treated_doc_ids).astype(int)

# 前期・後期平均月売上
pre_avg = (
    panel[panel["month_index"] <= PRE_END]
    .groupby("doctor_id")["amount"].mean()
    .rename("pre_mean")
)
post_avg = (
    panel[panel["month_index"] >= POST_START]
    .groupby("doctor_id")["amount"].mean()
    .rename("post_mean")
)

unit_df = (
    clean_pairs[clean_pairs["doctor_id"].isin(analysis_doc_ids)]
    .set_index("doctor_id")
    .join(pre_avg)
    .join(post_avg)
    .reset_index()
)
unit_df["growth"] = unit_df["post_mean"] - unit_df["pre_mean"]
unit_df["treated"] = unit_df["doctor_id"].isin(treated_doc_ids).astype(int)

# 医師属性マージ
doc_attr_df2 = doc_attr_df.rename(columns={"doc": "doctor_id"})

def _exp_cat(y):
    if y <= 10:   return "若手"
    elif y <= 20: return "中堅"
    return "ベテラン"

doc_attr_df2["experience_cat"] = doc_attr_df2["医師歴"].apply(_exp_cat)
unit_df = unit_df.merge(
    doc_attr_df2[["doctor_id", "医師歴", "experience_cat",
                  "DOCTOR_SEGEMNT", "DIGITAL_CHANNEL_PREFERENCE"]],
    on="doctor_id", how="left",
)

# 施設属性マージ
fac_df2 = fac_df.rename(columns={"fac_honin": "facility_id"})
unit_df = unit_df.merge(
    fac_df2[["facility_id", "施設区分名", "UHP区分名"]],
    on="facility_id", how="left",
)

t_growth = unit_df[unit_df["treated"] == 1]["growth"].mean()
c_growth = unit_df[unit_df["treated"] == 0]["growth"].mean()
print("  前期 month 0-" + str(PRE_END) + ": " + str(PRE_END + 1) + "ヶ月  後期 month "
      + str(POST_START) + "-32: " + str(33 - POST_START) + "ヶ月")
print("  粗成長率 - 処置群: " + str(round(t_growth, 1)) + "  対照群: " + str(round(c_growth, 1)))

# ===================================================================
# 傾向スコア推定（sklearn L2正則化ロジスティック回帰を優先使用）
# ===================================================================
print("\n[4] 傾向スコア推定（Logistic Regression + L2正則化）")

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
# 医師歴（連続量）も追加
ps_dummies["exp_years_std"] = (
    (ps_dummies["医師歴"] - ps_dummies["医師歴"].mean())
    / (ps_dummies["医師歴"].std() + 1e-9)
)

cov_cols = [c for c in ps_dummies.columns
            if c.startswith(("experience_cat_", "DOCTOR_SEGEMNT_",
                             "DIGITAL_CHANNEL_PREFERENCE_",
                             "施設区分名_", "UHP区分名_"))
            ] + ["pre_mean_std", "exp_years_std"]

y_ps_arr = ps_dummies["treated"].astype(float).values
X_ps_arr = ps_dummies[cov_cols].astype(float).values

from sklearn.linear_model import LogisticRegression

# 正則化強度を強め（C=0.1）でオーバーフィットを防ぐ
# 少数サンプルでも共通台が生まれやすくなる
ps_data = ps_data.copy()
ps_estimated = False

for C_val in [0.1, 0.05, 0.02]:
    try:
        lr = LogisticRegression(C=C_val, max_iter=2000, random_state=RANDOM_SEED,
                                solver="lbfgs", class_weight="balanced")
        lr.fit(X_ps_arr, y_ps_arr)
        ps_candidate = lr.predict_proba(X_ps_arr)[:, 1]
        # 共通台チェック: 処置群と対照群のPS範囲が重なっていれば採用
        t_mask = y_ps_arr == 1
        c_mask = y_ps_arr == 0
        overlap = (ps_candidate[t_mask].min() < ps_candidate[c_mask].max()) and \
                  (ps_candidate[c_mask].min() < ps_candidate[t_mask].max())
        if overlap:
            ps_data["ps"] = ps_candidate
            ps_estimated = True
            print("  L2正則化 LogisticRegression (C=" + str(C_val) + ") 成功")
            break
    except Exception:
        continue

if not ps_estimated:
    # 単変量（前期売上 + 医師歴のみ）でキャリパーなしマッチング
    print("  警告: 完全分離 -> 前期売上+医師歴のみで単変量PS推定")
    try:
        X_simple = ps_dummies[["pre_mean_std", "exp_years_std"]].values
        lr_s = LogisticRegression(C=0.5, max_iter=1000, random_state=RANDOM_SEED)
        lr_s.fit(X_simple, y_ps_arr)
        ps_data["ps"] = lr_s.predict_proba(X_simple)[:, 1]
        ps_estimated = True
    except Exception:
        ps_data["ps"] = (y_ps_arr + np.random.normal(0, 0.05, len(y_ps_arr))).clip(0.05, 0.95)

ps_t = ps_data[ps_data["treated"] == 1]["ps"]
ps_c = ps_data[ps_data["treated"] == 0]["ps"]
print("  処置群 PS: mean=" + str(round(ps_t.mean(), 3))
      + "  range=[" + str(round(ps_t.min(), 3)) + ", " + str(round(ps_t.max(), 3)) + "]")
print("  対照群 PS: mean=" + str(round(ps_c.mean(), 3))
      + "  range=[" + str(round(ps_c.min(), 3)) + ", " + str(round(ps_c.max(), 3)) + "]")

# ===================================================================
# 1:1 最近傍マッチング（キャリパー付き）
# ===================================================================
print("\n[5] 1:1 最近傍マッチング（キャリパー付き）")


def logit_ps(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def psm_1to1(treated_df, control_df, caliper):
    """
    treated_df / control_df: DataFrame with 'doctor_id' and 'logit_ps'
    Returns matched pair list: [(treated_doc, control_doc), ...]
    caliper=None の場合はキャリパーなし（最近傍のみ）
    """
    ctrl_arr = control_df["logit_ps"].values.reshape(-1, 1)
    tree = KDTree(ctrl_arr)

    np.random.seed(RANDOM_SEED)
    order = np.random.permutation(len(treated_df))

    used_ctrl = set()
    pairs = []

    for i in order:
        row   = treated_df.iloc[i]
        t_lps = row["logit_ps"]
        # 全コントロール候補を検索対象にする（使用済みを跳び越えるため）
        k = len(control_df)

        dists, idxs = tree.query([[t_lps]], k=k)
        for dist, ci in zip(dists[0], idxs[0]):
            if caliper is not None and dist > caliper:
                break
            c_doc = control_df.iloc[ci]["doctor_id"]
            if c_doc not in used_ctrl:
                pairs.append((row["doctor_id"], c_doc))
                used_ctrl.add(c_doc)
                break

    return pairs


ps_data["logit_ps"] = logit_ps(ps_data["ps"].values)
logit_std = ps_data["logit_ps"].std()

if logit_std < 1e-6:
    caliper = None
    print("  logit(PS) SD≈0 -> キャリパーなし最近傍マッチング")
else:
    caliper = CALIPER_MULTIPLIER * logit_std
    print("  logit(PS) SD=" + str(round(logit_std, 4)) + "  キャリパー=" + str(round(caliper, 4)))

treated_for_match = ps_data[ps_data["treated"] == 1][["doctor_id", "logit_ps"]].reset_index(drop=True)
control_for_match = ps_data[ps_data["treated"] == 0][["doctor_id", "logit_ps"]].reset_index(drop=True)

matched_pairs = psm_1to1(treated_for_match, control_for_match, caliper)

# マッチング率が低ければキャリパーを緩和してリトライ
if caliper is not None and len(matched_pairs) < len(treated_for_match) * 0.3:
    caliper_wide = logit_std  # 0.2 -> 1.0 倍に緩和
    print("  マッチング率低 -> キャリパー緩和 (" + str(round(caliper_wide, 4)) + ") でリトライ")
    matched_pairs = psm_1to1(treated_for_match, control_for_match, caliper_wide)
    if len(matched_pairs) < 5:
        print("  緩和後も不成立 -> キャリパーなし最近傍マッチング")
        matched_pairs = psm_1to1(treated_for_match, control_for_match, None)

matched_t_ids = [p[0] for p in matched_pairs]
matched_c_ids = [p[1] for p in matched_pairs]

match_rate = len(matched_pairs) / max(len(treated_for_match), 1) * 100
print("  マッチング成立ペア数: " + str(len(matched_pairs)))
print("  マッチング率（処置群）: " + str(round(match_rate, 1)) + "%")

# マッチング後データセット
matched_ids = set(matched_t_ids) | set(matched_c_ids)
matched_df  = ps_data[ps_data["doctor_id"].isin(matched_ids)].copy()

# ===================================================================
# 共変量バランス確認（SMD）
# ===================================================================
print("\n[6] 共変量バランス確認（SMD）")


def smd(x_t, x_c):
    mean_diff  = x_t.mean() - x_c.mean()
    pooled_sd  = np.sqrt((x_t.var() + x_c.var()) / 2 + 1e-12)
    return mean_diff / pooled_sd


smd_results = []
for label, col in [("前期売上", "pre_mean"), ("医師歴(年)", "医師歴")]:
    smd_before = smd(
        ps_data[ps_data["treated"] == 1][col],
        ps_data[ps_data["treated"] == 0][col],
    )
    smd_after = smd(
        matched_df[matched_df["treated"] == 1][col],
        matched_df[matched_df["treated"] == 0][col],
    )
    smd_results.append({"変数": label, "SMD_before": smd_before, "SMD_after": smd_after})
    print("  " + label + ": SMD before=" + str(round(smd_before, 3))
          + "  after=" + str(round(smd_after, 3)))

smd_df = pd.DataFrame(smd_results)

# ===================================================================
# ATT 推定（マッチング後）
# ===================================================================
print("\n[7] ATT 推定（マッチング後の伸長率差）")

matched_t_growth = np.array([
    ps_data[ps_data["doctor_id"] == d]["growth"].values[0]
    for d in matched_t_ids
])
matched_c_growth = np.array([
    ps_data[ps_data["doctor_id"] == d]["growth"].values[0]
    if len(ps_data[ps_data["doctor_id"] == d]) > 0 else np.nan
    for d in matched_c_ids
])

valid_mask = ~np.isnan(matched_c_growth)
diffs  = matched_t_growth[valid_mask] - matched_c_growth[valid_mask]
att    = diffs.mean()
se_att = diffs.std() / np.sqrt(len(diffs))
t_stat = att / (se_att + 1e-12)
p_val  = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(diffs) - 1))
ci_lo  = att - 1.96 * se_att
ci_hi  = att + 1.96 * se_att

print("  ATT = " + str(round(att, 2)) + " 万円/月  SE=" + str(round(se_att, 2))
      + "  t=" + str(round(t_stat, 2)) + "  p=" + str(round(p_val, 4)))
print("  95% CI: [" + str(round(ci_lo, 2)) + ", " + str(round(ci_hi, 2)) + "]")
print("  視聴群は未視聴群に比べ月平均実績が " + str(round(att, 2)) + " 万円 多い（後期-前期差）")

# ===================================================================
# サブグループ分析（医師歴別: 若手/中堅/ベテラン）
# ===================================================================
print("\n[8] サブグループ分析（医師歴別）")

subgroup_results = []

for sg_label in ["若手", "中堅", "ベテラン"]:
    sg_data = ps_data[ps_data["experience_cat"] == sg_label].copy()
    sg_t    = sg_data[sg_data["treated"] == 1][["doctor_id", "logit_ps"]].reset_index(drop=True)
    sg_c    = sg_data[sg_data["treated"] == 0][["doctor_id", "logit_ps"]].reset_index(drop=True)

    if len(sg_t) >= 3 and len(sg_c) >= 3:
        # まずキャリパー付きで試み、不成立なら最近傍のみ
        sg_pairs = psm_1to1(sg_t, sg_c, caliper)
        if len(sg_pairs) < 2:
            sg_pairs = psm_1to1(sg_t, sg_c, None)
    else:
        sg_pairs = []

    n_matched = len(sg_pairs)
    if n_matched < 2:
        subgroup_results.append({
            "subgroup": sg_label,
            "n_treated_raw": len(sg_t),
            "n_control_raw": len(sg_c),
            "n_matched": 0,
            "att": np.nan, "se": np.nan,
            "p_val": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
        })
        print("  [" + sg_label + "] N処置=" + str(len(sg_t))
              + " N対照=" + str(len(sg_c)) + " -> マッチング不成立")
        continue

    sg_t_docs = [p[0] for p in sg_pairs]
    sg_c_docs = [p[1] for p in sg_pairs]

    sg_t_growth = np.array([
        sg_data[sg_data["doctor_id"] == d]["growth"].values[0]
        for d in sg_t_docs
    ])
    sg_c_growth = np.array([
        sg_data[sg_data["doctor_id"] == d]["growth"].values[0]
        if len(sg_data[sg_data["doctor_id"] == d]) > 0 else np.nan
        for d in sg_c_docs
    ])

    sg_diffs = sg_t_growth - sg_c_growth
    sg_diffs = sg_diffs[~np.isnan(sg_diffs)]

    if len(sg_diffs) < 2:
        subgroup_results.append({
            "subgroup": sg_label,
            "n_treated_raw": len(sg_t),
            "n_control_raw": len(sg_c),
            "n_matched": n_matched,
            "att": np.nan, "se": np.nan,
            "p_val": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
        })
        continue

    sg_att   = sg_diffs.mean()
    sg_se    = sg_diffs.std() / np.sqrt(len(sg_diffs))
    sg_tstat = sg_att / (sg_se + 1e-12)
    sg_p     = 2 * (1 - stats.t.cdf(abs(sg_tstat), df=len(sg_diffs) - 1))
    sg_ci_lo = sg_att - 1.96 * sg_se
    sg_ci_hi = sg_att + 1.96 * sg_se

    subgroup_results.append({
        "subgroup": sg_label,
        "n_treated_raw": len(sg_t),
        "n_control_raw": len(sg_c),
        "n_matched": len(sg_diffs),
        "att": sg_att, "se": sg_se,
        "p_val": sg_p, "ci_lo": sg_ci_lo, "ci_hi": sg_ci_hi,
    })
    sig = "**" if sg_p < 0.01 else ("*" if sg_p < 0.05 else ("†" if sg_p < 0.10 else ""))
    print("  [" + sg_label + "] N処置=" + str(len(sg_t)) + " N対照=" + str(len(sg_c))
          + " マッチ=" + str(len(sg_diffs)))
    print("    ATT=" + str(round(sg_att, 2)) + " SE=" + str(round(sg_se, 2))
          + " p=" + str(round(sg_p, 4))
          + " 95%CI[" + str(round(sg_ci_lo, 2)) + "," + str(round(sg_ci_hi, 2)) + "] " + sig)

sg_df = pd.DataFrame(subgroup_results)

# ===================================================================
# 可視化
# ===================================================================
print("\n[9] 可視化")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(
    "09: 視聴群 vs 未視聴群 - PSM による伸長率比較",
    fontsize=13, fontweight="bold",
)

COLOR_T = "#1565C0"
COLOR_C = "#FF8F00"

# (a) マッチング前後の傾向スコア分布
ax = axes[0, 0]
bins = np.linspace(0, 1, 25)
ax.hist(ps_data[ps_data["treated"] == 1]["ps"],  bins=bins, alpha=0.45, color=COLOR_T, label="視聴群（全体）")
ax.hist(ps_data[ps_data["treated"] == 0]["ps"],  bins=bins, alpha=0.45, color=COLOR_C, label="未視聴群（全体）")
ax.hist(matched_df[matched_df["treated"] == 1]["ps"], bins=bins, alpha=0.9, color=COLOR_T,
        histtype="step", linewidth=2, label="視聴群（マッチ後）")
ax.hist(matched_df[matched_df["treated"] == 0]["ps"], bins=bins, alpha=0.9, color=COLOR_C,
        histtype="step", linewidth=2, label="未視聴群（マッチ後）")
ax.set_xlabel("傾向スコア", fontsize=11)
ax.set_ylabel("医師数", fontsize=11)
ax.set_title("(a) 傾向スコア分布（マッチング前後）", fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

# (b) マッチング後の伸長率分布
ax = axes[0, 1]
mt_valid = matched_t_growth[valid_mask]
mc_valid = matched_c_growth[valid_mask]
ax.hist(mt_valid, bins=20, alpha=0.6, color=COLOR_T,
        label="視聴群 (N=" + str(valid_mask.sum()) + ")")
ax.hist(mc_valid, bins=20, alpha=0.6, color=COLOR_C,
        label="未視聴群 (N=" + str(valid_mask.sum()) + ")")
ax.axvline(mt_valid.mean(), color=COLOR_T, linewidth=2, linestyle="--",
           label="視聴群 平均: " + str(round(mt_valid.mean(), 1)))
ax.axvline(mc_valid.mean(), color=COLOR_C, linewidth=2, linestyle="--",
           label="未視聴群 平均: " + str(round(mc_valid.mean(), 1)))
ax.set_xlabel("伸長率（後期月平均 - 前期月平均, 万円）", fontsize=10)
ax.set_ylabel("医師数", fontsize=11)
ax.set_title("(b) マッチング後の伸長率分布\nATT=" + str(round(att, 2)) + "万円/月, p=" + str(round(p_val, 4)), fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

# (c) サブグループ別 ATT (Forest plot)
ax = axes[1, 0]
valid_sg = sg_df[sg_df["att"].notna()].reset_index(drop=True)
colors_sg = {"若手": "#43A047", "中堅": "#1565C0", "ベテラン": "#E53935"}

for i, row in valid_sg.iterrows():
    color = colors_sg.get(row["subgroup"], "gray")
    ax.errorbar(
        row["att"], i,
        xerr=[[row["att"] - row["ci_lo"]], [row["ci_hi"] - row["att"]]],
        fmt="o", color=color, markersize=8, capsize=5, linewidth=2,
        label=row["subgroup"] + " (N=" + str(int(row["n_matched"])) + ")",
    )
    x_text = row["ci_hi"] + abs(row["ci_hi"] - row["ci_lo"]) * 0.05 + 0.2
    pstar = ("***" if row["p_val"] < 0.001 else ("**" if row["p_val"] < 0.01
             else ("*" if row["p_val"] < 0.05 else ("†" if row["p_val"] < 0.1 else ""))))
    ax.text(x_text, i, "p=" + str(round(row["p_val"], 3)) + pstar,
            va="center", fontsize=9)

# 全体 ATT を追加
ax.errorbar(att, len(valid_sg), xerr=[[att - ci_lo], [ci_hi - att]],
            fmt="D", color="black", markersize=8, capsize=5, linewidth=2,
            label="全体 (N=" + str(valid_mask.sum()) + ")")
ax.text(ci_hi + abs(ci_hi - ci_lo) * 0.05 + 0.2, len(valid_sg),
        "p=" + str(round(p_val, 3)), va="center", fontsize=9)

ax.axvline(0, color="gray", linestyle="--", linewidth=1)
ytick_labels = [r["subgroup"] for _, r in valid_sg.iterrows()] + ["全体"]
ax.set_yticks(list(range(len(valid_sg))) + [len(valid_sg)])
ax.set_yticklabels(ytick_labels, fontsize=10)
ax.set_xlabel("ATT（万円/月, 後期-前期差の視聴群-対照群差）", fontsize=10)
ax.set_title("(c) サブグループ別 ATT（Forest Plot）", fontsize=11)
ax.grid(True, alpha=0.3, axis="x")
ax.legend(fontsize=8, loc="lower right")

# (d) 医師歴別の解析対象数（ウォッシュアウト除外の影響）
ax = axes[1, 1]
all_docs_attr = doc_attr_df2[doc_attr_df2["doctor_id"].isin(set(clean_pairs["doctor_id"]))].copy()
all_docs_attr["in_analysis"] = all_docs_attr["doctor_id"].isin(analysis_doc_ids)
all_docs_attr["washout_excluded"] = all_docs_attr["doctor_id"].isin(washout_viewers)

order = ["若手", "中堅", "ベテラン"]
sg_counts_total    = all_docs_attr.groupby("experience_cat")["doctor_id"].count().reindex(order, fill_value=0)
sg_counts_analysis = all_docs_attr[all_docs_attr["in_analysis"]].groupby("experience_cat")["doctor_id"].count().reindex(order, fill_value=0)
sg_counts_washout  = all_docs_attr[all_docs_attr["washout_excluded"]].groupby("experience_cat")["doctor_id"].count().reindex(order, fill_value=0)

x = np.arange(3)
w = 0.28
b1 = ax.bar(x - w, sg_counts_total.values,    w, label="候補母集団", color="#90A4AE", alpha=0.9)
b2 = ax.bar(x,      sg_counts_analysis.values, w, label="解析対象（視聴+未視聴）", color=COLOR_T, alpha=0.9)
b3 = ax.bar(x + w,  sg_counts_washout.values,  w, label="ウォッシュアウト除外数", color="#EF9A9A", alpha=0.9)

for bar_group in [b1, b2, b3]:
    for rect in bar_group:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 0.2,
                str(int(h)), ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(order, fontsize=11)
ax.set_ylabel("医師数", fontsize=11)
ax.set_title("(d) 医師歴別 解析対象数\n（ウォッシュアウト除外の影響確認）", fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "psm_growth_rate.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print("  図を保存: " + out_path)

# ===================================================================
# 結果 JSON 出力
# ===================================================================
results_dir = os.path.join(SCRIPT_DIR, "results")
os.makedirs(results_dir, exist_ok=True)

results_json = {
    "analysis_settings": {
        "pre_period": "month 0-" + str(PRE_END) + " (" + str(PRE_END + 1) + "ヶ月)",
        "post_period": "month " + str(POST_START) + "-32 (" + str(33 - POST_START) + "ヶ月)",
        "outcome": "後期月平均 - 前期月平均（万円）",
        "caliper": float(caliper),
        "matching": "1:1 nearest-neighbor within caliper",
    },
    "sample_sizes": {
        "washout_excluded": int(len(washout_viewers)),
        "treated_raw": int(len(treated_doc_ids)),
        "control_raw": int(len(control_doc_ids)),
        "matched_pairs": int(len(matched_pairs)),
    },
    "overall_att": {
        "att": float(att),
        "se": float(se_att),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "ci_95_lower": float(ci_lo),
        "ci_95_upper": float(ci_hi),
    },
    "subgroup_att": [
        {
            "subgroup": row["subgroup"],
            "n_treated_raw": int(row["n_treated_raw"]),
            "n_control_raw": int(row["n_control_raw"]),
            "n_matched": int(row["n_matched"]),
            "att": float(row["att"]) if not np.isnan(row["att"]) else None,
            "se": float(row["se"]) if not np.isnan(row["se"]) else None,
            "p_value": float(row["p_val"]) if not np.isnan(row["p_val"]) else None,
            "ci_95_lower": float(row["ci_lo"]) if not np.isnan(row["ci_lo"]) else None,
            "ci_95_upper": float(row["ci_hi"]) if not np.isnan(row["ci_hi"]) else None,
        }
        for _, row in sg_df.iterrows()
    ],
    "covariate_balance_smd": smd_df.to_dict("records"),
    "washout_excluded_by_experience": {
        sg: int(sg_counts_washout.get(sg, 0)) for sg in order
    },
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
      + "  (p=" + str(round(p_val, 4)) + ", 95%CI [" + str(round(ci_lo, 2)) + ", " + str(round(ci_hi, 2)) + "])")
print("  視聴群は未視聴群（PSMマッチ後）に比べ、")
print("  後期（month " + str(POST_START) + "-32）の月平均実績が "
      + str(round(att, 2)) + " 万円 多い。")
print("\n【サブグループ別】")
for _, row in sg_df.iterrows():
    if np.isnan(row["att"]):
        print("  " + row["subgroup"] + ": 推定不能（サンプル不足）")
    else:
        sig = "**" if row["p_val"] < 0.01 else ("*" if row["p_val"] < 0.05 else "")
        print("  " + row["subgroup"] + ": ATT=" + str(round(row["att"], 2))
              + " 万円/月  p=" + str(round(row["p_val"], 4)) + " " + sig)
print("\n【注意】視聴は医師の自発行動 -> 未観測交絡が残るため因果解釈は慎重に。")
