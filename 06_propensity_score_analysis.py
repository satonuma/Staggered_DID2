"""
===================================================================
視聴傾向スコア分析: セッションベース + 選択バイアス調整
===================================================================
目的:
  1. セッションベースの視聴パターン分類（時間的パターンを考慮）
  2. 医師・施設属性による視聴傾向スコア算出
  3. IPW（Inverse Probability Weighting）による選択バイアス調整
  4. 傾向スコア調整後のIntensive/Extensive Margin効果

手法:
  - 視聴間隔が30日以上 → 別セッション
  - パターン分類: 短期集中型、長期継続型、定期視聴型、断続視聴型
  - Propensity Score: Logit(視聴有無 ~ 属性)
  - IPW推定で観測可能な選択バイアスを軽減

重要な注意:
  視聴回数は内生変数（医師の自発的行動）であり、
  本分析でも未観測の交絡（医師の関心度など）は完全には除去できない。
  結果は「上限値」として解釈すべき。
===================================================================
"""

import os
import warnings
import json
from datetime import timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
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

FILE_RW_LIST = "rw_list.csv"
FILE_SALES = "sales.csv"
FILE_DIGITAL = "デジタル視聴データ.csv"
FILE_ACTIVITY = "活動データ.csv"
FILE_DOCTOR_MASTER = "doctor_attribute.csv"
FILE_DOCTOR_ATTR = "doctor_attribute.csv"
FILE_FACILITY_MASTER = "facility_attribute_修正.csv"
FILE_FAC_DOCTOR_LIST = "施設医師リスト.csv"

# 解析集団フィルタパラメータ
FILTER_SINGLE_FAC_DOCTOR = True
DOCTOR_HONIN_FAC_COUNT_COL = "所属施設数"
INCLUDE_ONLY_RW = True            # True: RW医師のみ (Step 3適用), False: 全医師 (Step 3スキップ)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "本番データ")
_required_06 = [FILE_SALES, FILE_DIGITAL, FILE_ACTIVITY, FILE_RW_LIST]
if not all(os.path.exists(os.path.join(DATA_DIR, f)) for f in _required_06):
    _alt = os.path.join(SCRIPT_DIR, "data")
    if all(os.path.exists(os.path.join(_alt, f)) for f in _required_06):
        DATA_DIR = _alt

START_DATE = "2023-04-01"
N_MONTHS = 33
WASHOUT_MONTHS = 2
LAST_ELIGIBLE_MONTH = 29

# セッション定義: 視聴間隔が30日以上空いたら別セッション
SESSION_GAP_DAYS = 30


# ================================================================
# セッション分類関数
# ================================================================

def classify_viewing_sessions(viewing_dates):
    """
    視聴日のリストをセッションに分割し、パターンを分類

    Parameters:
    - viewing_dates: list of datetime

    Returns:
    - pattern: str (短期集中型 / 長期継続型 / 定期視聴型 / 断続視聴型)
    - n_sessions: int
    - total_span_days: int
    """
    if len(viewing_dates) == 0:
        return "未視聴", 0, 0

    if len(viewing_dates) == 1:
        return "単発視聴", 1, 0

    # ソート
    dates = sorted(viewing_dates)

    # セッション分割
    sessions = []
    current_session = [dates[0]]

    for i in range(1, len(dates)):
        days_gap = (dates[i] - dates[i-1]).days

        if days_gap > SESSION_GAP_DAYS:
            sessions.append(current_session)
            current_session = [dates[i]]
        else:
            current_session.append(dates[i])

    sessions.append(current_session)

    n_sessions = len(sessions)
    total_span_days = (dates[-1] - dates[0]).days

    # パターン分類
    if n_sessions == 1:
        # 単一セッション
        session_duration = (sessions[0][-1] - sessions[0][0]).days
        if session_duration <= SESSION_GAP_DAYS:
            pattern = "短期集中型"  # 1ヶ月以内に集中視聴
        else:
            pattern = "長期継続型"  # 同一セッションが長期継続
    else:
        # 複数セッション
        session_gaps = []
        for i in range(1, len(sessions)):
            gap = (sessions[i][0] - sessions[i-1][-1]).days
            session_gaps.append(gap)

        avg_gap = np.mean(session_gaps)

        if avg_gap <= 60:
            pattern = "定期視聴型"  # 平均2ヶ月以内の間隔で定期視聴
        else:
            pattern = "断続視聴型"  # 長期間空けて断続的に視聴

    return pattern, n_sessions, total_span_days


# ================================================================
# データ読み込み + 除外フロー
# ================================================================
print("=" * 70)
print(" 視聴傾向スコア分析: セッションベース + 選択バイアス調整")
print("=" * 70)

# 1. 基本データ読み込み（除外フローで絞り込み）
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
web_lecture = activity_raw[
    (activity_raw["品目コード"] == ENT_PRODUCT_CODE)
    & (activity_raw["活動種別"] == ACTIVITY_CHANNEL_FILTER)
].copy()

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

print(f"\n[データ読み込み完了]")

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

print(f"[除外フロー完了] 処置群: {len(treated_doc_ids)}, 対照群: {len(control_doc_ids)}")


# ================================================================
# Part 1: セッションベースの視聴パターン分類
# ================================================================
print("\n" + "=" * 70)
print(" Part 1: セッションベースの視聴パターン分類")
print("=" * 70)

session_results = []

for doc_id in analysis_doc_ids:
    doc_views = viewing_after_washout[
        viewing_after_washout["doctor_id"] == doc_id
    ]["view_date"].tolist()

    pattern, n_sessions, span_days = classify_viewing_sessions(doc_views)

    session_results.append({
        "doctor_id": doc_id,
        "facility_id": doc_to_fac.get(doc_id),
        "viewing_pattern": pattern,
        "n_sessions": n_sessions,
        "total_views": len(doc_views),
        "span_days": span_days,
        "has_viewed": 1 if len(doc_views) > 0 else 0,
    })

session_df = pd.DataFrame(session_results)

pattern_dist = session_df["viewing_pattern"].value_counts()
print(f"\n[視聴パターン分布]")
print(f"  セッション定義: 視聴間隔>{SESSION_GAP_DAYS}日で別セッション")
print(f"\n  分布:")
for pattern, count in pattern_dist.items():
    pct = count / len(session_df) * 100
    print(f"    {pattern}: {count:>4}医師 ({pct:>5.1f}%)")


# ================================================================
# Part 2: 医師・施設属性のマージ
# ================================================================
print("\n" + "=" * 70)
print(" Part 2: 医師・施設属性のマージ")
print("=" * 70)

# 医師属性 (doctor_attribute.csv)
doctor_attrs = pd.read_csv(os.path.join(DATA_DIR, FILE_DOCTOR_MASTER))
doctor_attrs = doctor_attrs.rename(columns={"doc": "doctor_id"})

def _exp_cat(y):
    if y <= 10: return "若手"
    elif y <= 20: return "中堅"
    return "ベテラン"

doctor_attrs["experience_cat"] = doctor_attrs["医師歴"].apply(_exp_cat)
session_df = session_df.merge(
    doctor_attrs[["doctor_id", "experience_cat", "DOCTOR_SEGEMNT", "DIGITAL_CHANNEL_PREFERENCE"]],
    on="doctor_id", how="left"
)

# 施設属性 (facility_attribute_修正.csv)
facility_attrs = pd.read_csv(os.path.join(DATA_DIR, FILE_FACILITY_MASTER))
facility_attrs = facility_attrs.rename(columns={"fac_honin": "facility_id"})
session_df = session_df.merge(
    facility_attrs[["facility_id", "施設区分名", "UHP区分名"]],
    on="facility_id", how="left"
)

print(f"  属性マージ完了")
print(f"  欠損なし医師数: {session_df.dropna().shape[0]} / {len(session_df)}")


# ================================================================
# Part 3: 視聴傾向スコア（Propensity Score）の推定
# ================================================================
print("\n" + "=" * 70)
print(" Part 3: 視聴傾向スコア推定")
print("=" * 70)

# ダミー変数化
ps_data = session_df.dropna().copy()
ps_data_dummies = pd.get_dummies(
    ps_data,
    columns=["experience_cat", "DOCTOR_SEGEMNT", "DIGITAL_CHANNEL_PREFERENCE", "施設区分名", "UHP区分名"],
    drop_first=True
)

# Logitモデル
y_ps = ps_data_dummies["has_viewed"].astype(float)
X_ps_cols = [c for c in ps_data_dummies.columns
             if c.startswith(("experience_cat_", "DOCTOR_SEGEMNT_", "DIGITAL_CHANNEL_PREFERENCE_",
                              "施設区分名_", "UHP区分名_"))]
X_ps = ps_data_dummies[X_ps_cols].astype(float)
X_ps = sm.add_constant(X_ps)

try:
    ps_model = Logit(y_ps, X_ps).fit(disp=False)
    ps_data_dummies["propensity_score"] = ps_model.predict(X_ps)

    # 元のps_dataにマージ
    ps_data = ps_data.merge(
        ps_data_dummies[["doctor_id", "propensity_score"]],
        on="doctor_id", how="left"
    )

    print(f"\n  Logitモデル推定完了")
    print(f"  Pseudo R2: {ps_model.prsquared:.4f}")  # ² → 2

    # 視聴群と未視聴群の傾向スコア分布
    ps_treated = ps_data[ps_data["has_viewed"] == 1]["propensity_score"]
    ps_control = ps_data[ps_data["has_viewed"] == 0]["propensity_score"]

    print(f"\n  視聴群の傾向スコア: 平均={ps_treated.mean():.3f}, 中央値={ps_treated.median():.3f}")
    print(f"  未視聴群の傾向スコア: 平均={ps_control.mean():.3f}, 中央値={ps_control.median():.3f}")

    # Common Support チェック
    overlap = (ps_treated.min() < ps_control.max()) and (ps_control.min() < ps_treated.max())
    print(f"  Common Support: {'あり' if overlap else 'なし（要注意）'}")

except Exception as e:
    print(f"\n  警告: Logitモデル推定失敗: {e}")
    ps_data["propensity_score"] = 0.5
    ps_treated = pd.Series([0.5])
    ps_control = pd.Series([0.5])

# session_dfにマージ
session_df = session_df.merge(
    ps_data[["doctor_id", "propensity_score"]],
    on="doctor_id", how="left"
)
session_df["propensity_score"] = session_df["propensity_score"].fillna(0.5)


# ================================================================
# Part 4: IPW推定
# ================================================================
print("\n" + "=" * 70)
print(" Part 4: IPW（Inverse Probability Weighting）推定")
print("=" * 70)

# パネルデータ構築（05と同様）
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

# 視聴パターン情報をマージ
panel = panel.merge(
    session_df[["doctor_id", "viewing_pattern", "has_viewed", "propensity_score"]],
    on="doctor_id", how="left"
)

# IPW重み
panel["ipw_weight"] = 1.0
panel.loc[panel["has_viewed"] == 1, "ipw_weight"] = 1 / panel.loc[panel["has_viewed"] == 1, "propensity_score"]
panel.loc[panel["has_viewed"] == 0, "ipw_weight"] = 1 / (1 - panel.loc[panel["has_viewed"] == 0, "propensity_score"])

# 極端な重みの調整（trimming）
panel["ipw_weight"] = panel["ipw_weight"].clip(upper=panel["ipw_weight"].quantile(0.99))

# 視聴パターン別の平均実績（IPW調整後）
pattern_means_ipw = []
for pattern in session_df["viewing_pattern"].unique():
    pattern_panel = panel[panel["viewing_pattern"] == pattern]
    if len(pattern_panel) > 0:
        weighted_mean = np.average(
            pattern_panel["amount"],
            weights=pattern_panel["ipw_weight"]
        )
        pattern_means_ipw.append({
            "pattern": pattern,
            "mean_ipw": weighted_mean,
            "n": len(pattern_panel) / N_MONTHS  # 施設数
        })

pattern_means_ipw_df = pd.DataFrame(pattern_means_ipw)

print(f"\n  IPW調整後の視聴パターン別平均実績:")
for _, row in pattern_means_ipw_df.iterrows():
    print(f"    {row['pattern']}: {row['mean_ipw']:.1f} (N={row['n']:.0f})")


# ================================================================
# Part 5: 可視化
# ================================================================
print("\n" + "=" * 70)
print(" Part 5: 可視化")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(
    "視聴傾向スコア分析: セッションベース + 選択バイアス調整",
    fontsize=13, fontweight="bold"
)

# (a) 視聴パターン分布
ax = axes[0, 0]
pattern_counts = session_df["viewing_pattern"].value_counts()
colors_map = {
    "短期集中型": "#FF6F00", "長期継続型": "#4CAF50",
    "定期視聴型": "#1565C0", "断続視聴型": "#9C27B0",
    "未視聴": "#757575", "単発視聴": "#FFC107"
}
bars = ax.bar(
    range(len(pattern_counts)),
    pattern_counts.values,
    color=[colors_map.get(p, "gray") for p in pattern_counts.index],
    alpha=0.8, edgecolor="white"
)
for bar, val in zip(bars, pattern_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            str(val), ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticks(range(len(pattern_counts)))
ax.set_xticklabels(pattern_counts.index, rotation=45, ha="right")
ax.set_ylabel("医師数")
ax.set_title("(a) 視聴パターン分布（セッションベース）")
ax.grid(True, alpha=0.3, axis="y")

# (b) 傾向スコア分布
ax = axes[0, 1]
if "propensity_score" in ps_data.columns:
    ax.hist(ps_treated, bins=20, alpha=0.6, color="#4CAF50", label="視聴群", edgecolor="white")
    ax.hist(ps_control, bins=20, alpha=0.6, color="#1565C0", label="未視聴群", edgecolor="white")
    ax.set_xlabel("視聴傾向スコア")
    ax.set_ylabel("医師数")
    ax.set_title("(b) 視聴傾向スコア分布")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

# (c) IPW調整前後の比較
ax = axes[1, 0]
# 調整前
pattern_means_raw = session_df.merge(
    panel.groupby("doctor_id")["amount"].mean().reset_index(),
    on="doctor_id", how="left"
).groupby("viewing_pattern")["amount"].mean()

x_pos = np.arange(len(pattern_means_raw))
width = 0.35
ax.bar(x_pos - width/2, pattern_means_raw.values, width,
       label="調整前", alpha=0.7, color="#FF9800")
ax.bar(x_pos + width/2, pattern_means_ipw_df.set_index("pattern").loc[pattern_means_raw.index, "mean_ipw"].values, width,
       label="IPW調整後", alpha=0.7, color="#4CAF50")
ax.set_xticks(x_pos)
ax.set_xticklabels(pattern_means_raw.index, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("平均納入額")
ax.set_title("(c) IPW調整前後の比較")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")

# (d) 視聴パターン別の時系列（IPW調整）
ax = axes[1, 1]
for pattern in ["短期集中型", "長期継続型", "定期視聴型"]:
    pattern_panel = panel[panel["viewing_pattern"] == pattern]
    if len(pattern_panel) > 0:
        weighted_trend = []
        for m in range(N_MONTHS):
            m_data = pattern_panel[pattern_panel["month_index"] == m]
            if len(m_data) > 0:
                weighted_trend.append(
                    np.average(m_data["amount"], weights=m_data["ipw_weight"])
                )
            else:
                weighted_trend.append(np.nan)
        ax.plot(range(N_MONTHS), weighted_trend, marker="o", ms=3,
                label=pattern, color=colors_map.get(pattern, "gray"))
ax.set_xlabel("月 (0=2023/4)")
ax.set_ylabel("平均納入額（IPW調整）")
ax.set_title("(d) 視聴パターン別 実績推移（IPW調整）")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "propensity_score_analysis.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  図を保存: {out_path}")


# ================================================================
# Part 6: JSON結果保存
# ================================================================
results_dir = os.path.join(SCRIPT_DIR, "results")
os.makedirs(results_dir, exist_ok=True)

results_json = {
    "session_classification": {
        "gap_threshold_days": SESSION_GAP_DAYS,
        "pattern_distribution": {
            pattern: int(count) for pattern, count in pattern_dist.items()
        },
    },
    "propensity_score_model": {
        "pseudo_r2": float(ps_model.prsquared) if 'ps_model' in locals() else None,
        "treated_mean": float(ps_treated.mean()) if len(ps_treated) > 0 else None,
        "control_mean": float(ps_control.mean()) if len(ps_control) > 0 else None,
    },
    "ipw_adjusted_means": pattern_means_ipw_df.to_dict("records"),
    "interpretation": {
        "warning": "視聴回数は内生変数（医師の自発的行動）であり、未観測の交絡は残る",
        "recommendation": "結果は相関関係として解釈し、因果効果の上限値と考えるべき",
    },
}

json_path = os.path.join(results_dir, "propensity_score_analysis.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results_json, f, ensure_ascii=False, indent=2)
print(f"\n  結果をJSON保存: {json_path}")

print("\n" + "=" * 70)
print(" 分析完了")
print("=" * 70)
print(f"\n【重要な注意事項】")
print(f"  本分析は観測可能な属性で選択バイアスを調整していますが、")
print(f"  未観測の交絡（医師の関心度、患者層の違いなど）は完全には除去できません。")
print(f"  結果は「視聴と売上の関連性」であり、「視聴の因果効果」ではありません。")
print(f"  推定値は真の効果の上限値として解釈すべきです。")
