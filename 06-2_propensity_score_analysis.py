"""
===================================================================
視聴傾向スコア分析: セッションベース + 選択バイアス調整 (ver2)
===================================================================
目的:
  1. セッションベースの視聴パターン分類（時間的パターンを考慮）
  2. 施設属性による視聴傾向スコア算出
  3. IPW（Inverse Probability Weighting）による選択バイアス調整
  4. 傾向スコア調整後のIntensive/Extensive Margin効果

ver2 変更点:
  - ver1: 1施設1医師のみ対象（FILTER_SINGLE_FAC_DOCTOR=True）
  - ver2: 複数医師施設も含む全施設対象、施設レベルで解析

手法:
  - 視聴間隔が30日以上 → 別セッション
  - パターン分類: 短期集中型、長期継続型、定期視聴型、断続視聴型
  - Propensity Score: Logit(視聴有無 ~ 施設属性 + 医師数)
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

# 解析集団フィルタパラメータ (ver2)
INCLUDE_ONLY_RW = False
UHP_RANK = {"UHP-A": 0, "UHP-B": 1, "UHP-C": 2}

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
print(" 視聴傾向スコア分析: セッションベース + 選択バイアス調整 (ver2)")
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

print(f"\n[データ読み込み完了]")

# 施設医師リスト: 全医師の施設対応マスター (母集団)
fac_doc_list = pd.read_csv(os.path.join(DATA_DIR, FILE_FAC_DOCTOR_LIST))

# 施設マスタ読み込み
fac_df = pd.read_csv(os.path.join(DATA_DIR, FILE_FACILITY_MASTER))

# ================================================================
# 解析集団の絞り込み (ver2: 複数医師施設対応)
# ================================================================
_doc_to_fac   = dict(zip(fac_doc_list["doc"], fac_doc_list["fac"]))
_doc_to_honin = dict(zip(fac_doc_list["doc"], fac_doc_list["fac_honin"]))
all_docs = set(fac_doc_list["doc"])

# 主施設割り当て: 平均納入額最大の施設
_doc_fac_list = fac_doc_list[["doc", "fac_honin"]].drop_duplicates()
_sales_by_fac = (
    daily.groupby("facility_id")["amount"].mean()
    .reset_index().rename(columns={"facility_id": "fac_honin", "amount": "avg_sales"})
)
_doc_fac_sales = _doc_fac_list.merge(_sales_by_fac, on="fac_honin", how="left")
_doc_fac_sales["avg_sales"] = _doc_fac_sales["avg_sales"].fillna(0)

_doc_primary_all = (
    _doc_fac_sales.sort_values("avg_sales", ascending=False)
    .groupby("doc")["fac_honin"].first()
)

# 全施設0 → UHP最上位
_fac_uhp = fac_df.drop_duplicates("fac_honin").set_index("fac_honin")["UHP区分名"] \
    if "UHP区分名" in fac_df.columns else pd.Series(dtype=str)
_zero_sum = _doc_fac_sales.groupby("doc")["avg_sales"].sum()
for _doc in set(_zero_sum[_zero_sum == 0].index):
    _doc_facs = _doc_fac_sales[_doc_fac_sales["doc"] == _doc]["fac_honin"].tolist()
    if _doc_facs:
        _ranked = sorted(_doc_facs, key=lambda f: UHP_RANK.get(str(_fac_uhp.get(f, "")), 99))
        _doc_primary_all[_doc] = _ranked[0]

doc_primary_fac = _doc_primary_all

# RWフィルタ
rw_doc_ids = set(rw_list["doc"])
analysis_docs_all = all_docs & rw_doc_ids if INCLUDE_ONLY_RW else all_docs

# 施設→医師リスト (1:N)
fac_to_docs: dict = {}
for doc in analysis_docs_all:
    if doc in doc_primary_fac.index:
        fac = doc_primary_fac[doc]
        fac_to_docs.setdefault(fac, []).append(doc)

n_docs_map = {fac: len(docs) for fac, docs in fac_to_docs.items()}

# 視聴データに主施設ID付与
viewing_fac = viewing.copy()
viewing_fac["facility_id"] = viewing_fac["doctor_id"].map(doc_primary_fac)

# ウォッシュアウト除外
washout_fac_ids = set(
    viewing_fac[
        (viewing_fac["view_date"] < months[WASHOUT_MONTHS])
        & viewing_fac["facility_id"].notna()
    ]["facility_id"]
)

_first_view = (
    viewing_fac[
        (viewing_fac["view_date"] >= months[WASHOUT_MONTHS])
        & viewing_fac["facility_id"].notna()
        & ~viewing_fac["facility_id"].isin(washout_fac_ids)
    ]
    .copy()
)
_first_view["month_index"] = (
    (_first_view["view_date"].dt.year - 2023) * 12
    + _first_view["view_date"].dt.month - 4
)
_first_view = _first_view[_first_view["month_index"] <= LAST_ELIGIBLE_MONTH]
_first_view_min = _first_view.groupby("facility_id")["month_index"].min()

treated_fac_ids = set(_first_view_min.index)
control_fac_ids = set(fac_to_docs.keys()) - treated_fac_ids - washout_fac_ids
analysis_fac_ids = treated_fac_ids | control_fac_ids

print(f"  処置群（視聴あり）: {len(treated_fac_ids)}  対照群（未視聴）: {len(control_fac_ids)}")
print(f"  ウォッシュアウト除外: {len(washout_fac_ids)}  解析対象: {len(analysis_fac_ids)}")

# ウォッシュアウト後の視聴データ（施設レベル）
viewing_after_washout_fac = viewing_fac[
    (viewing_fac["view_date"] >= months[WASHOUT_MONTHS])
    & viewing_fac["facility_id"].isin(analysis_fac_ids)
].copy()

# 医師レベルのウォッシュアウト後視聴データも構築（セッション分類用）
analysis_doc_ids = {doc for fac in analysis_fac_ids for doc in fac_to_docs.get(fac, [])}
viewing_after_washout = viewing_fac[
    (viewing_fac["view_date"] >= months[WASHOUT_MONTHS])
    & viewing_fac["doctor_id"].isin(analysis_doc_ids)
].copy()


# ================================================================
# Part 1: セッションベースの視聴パターン分類
# ================================================================
print("\n" + "=" * 70)
print(" Part 1: セッションベースの視聴パターン分類")
print("=" * 70)

# 事前に doctor_id → view_date リストの辞書を構築（ループ内フィルタを排除）
viewing_by_doc = (
    viewing_after_washout.sort_values("view_date")
    .groupby("doctor_id")["view_date"].apply(list)
    .to_dict()
)

session_results = []
for doc_id in analysis_doc_ids:
    doc_views = viewing_by_doc.get(doc_id, [])
    pattern, n_sessions, span_days = classify_viewing_sessions(doc_views)
    session_results.append({
        "doctor_id": doc_id,
        "facility_id": doc_primary_fac.get(doc_id),
        "viewing_pattern": pattern,
        "n_sessions": n_sessions,
        "total_views": len(doc_views),
        "span_days": span_days,
        "has_viewed": 1 if len(doc_views) > 0 else 0,
    })

session_df = pd.DataFrame(session_results)

# 施設レベルに集約
session_df["facility_id"] = session_df["doctor_id"].map(doc_primary_fac)

fac_session = session_df.groupby("facility_id").agg(
    has_viewed=("has_viewed", "max"),  # 1 if any doctor viewed
    viewing_pattern=("viewing_pattern", lambda x: x[x != "未視聴"].mode()[0] if any(x != "未視聴") else "未視聴"),
    pre_mean=("total_views", "mean"),  # facility-level average (views per doctor)
).reset_index()
fac_session["n_docs"] = fac_session["facility_id"].map(n_docs_map).fillna(1)

# 解析対象施設のみ
fac_session = fac_session[fac_session["facility_id"].isin(analysis_fac_ids)].copy()

pattern_dist = fac_session["viewing_pattern"].value_counts()
print(f"\n[視聴パターン分布（施設レベル）]")
print(f"  セッション定義: 視聴間隔>{SESSION_GAP_DAYS}日で別セッション")
print(f"\n  分布:")
for pattern, count in pattern_dist.items():
    pct = count / len(fac_session) * 100
    print(f"    {pattern}: {count:>4}施設 ({pct:>5.1f}%)")


# ================================================================
# Part 2: 施設属性のマージ
# ================================================================
print("\n" + "=" * 70)
print(" Part 2: 施設属性のマージ")
print("=" * 70)

# 施設属性マージ (ver2: 施設レベル属性のみ、医師レベル属性は不使用)
fac_attrs2 = fac_df.rename(columns={"fac_honin": "facility_id"})
fac_session = fac_session.merge(
    fac_attrs2[["facility_id", "施設区分名", "UHP区分名"]],
    on="facility_id", how="left"
)

print(f"  属性マージ完了")
print(f"  欠損なし施設数: {fac_session.dropna().shape[0]} / {len(fac_session)}")


# ================================================================
# Part 3: 視聴傾向スコア（Propensity Score）の推定
# ================================================================
print("\n" + "=" * 70)
print(" Part 3: 視聴傾向スコア推定")
print("=" * 70)

# PS推定用共変量 (ver2: 施設区分名, UHP区分名, n_docs)
ps_data = fac_session.dropna(subset=["施設区分名", "UHP区分名"]).copy()
ps_dummies = pd.get_dummies(
    ps_data,
    columns=["施設区分名", "UHP区分名"],
    drop_first=True
)
_n_docs_std = (ps_dummies["n_docs"] - ps_dummies["n_docs"].mean()) / (ps_dummies["n_docs"].std() + 1e-9)
ps_dummies["n_docs_std"] = _n_docs_std

X_ps_cols = [c for c in ps_dummies.columns
             if c.startswith(("施設区分名_", "UHP区分名_"))] + ["n_docs_std"]

# Logitモデル
y_ps = ps_dummies["has_viewed"].astype(float)
X_ps = ps_dummies[X_ps_cols].astype(float)
X_ps = sm.add_constant(X_ps)

try:
    ps_model = Logit(y_ps, X_ps).fit(disp=False)
    ps_dummies["propensity_score"] = ps_model.predict(X_ps)

    # 元のps_dataにマージ
    ps_data = ps_data.merge(
        ps_dummies[["facility_id", "propensity_score"]],
        on="facility_id", how="left"
    )

    print(f"\n  Logitモデル推定完了")
    print(f"  Pseudo R2: {ps_model.prsquared:.4f}")

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

# fac_sessionにマージ
fac_session = fac_session.merge(
    ps_data[["facility_id", "propensity_score"]],
    on="facility_id", how="left"
)
fac_session["propensity_score"] = fac_session["propensity_score"].fillna(0.5)


# ================================================================
# Part 4: IPW推定
# ================================================================
print("\n" + "=" * 70)
print(" Part 4: IPW（Inverse Probability Weighting）推定")
print("=" * 70)

# パネルデータ構築（施設レベル）
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

# 視聴パターン情報をマージ（施設レベル）
panel = panel.merge(
    fac_session[["facility_id", "viewing_pattern", "has_viewed", "propensity_score"]],
    on="facility_id", how="left"
)

# IPW重み
panel["ipw_weight"] = 1.0
panel.loc[panel["has_viewed"] == 1, "ipw_weight"] = 1 / panel.loc[panel["has_viewed"] == 1, "propensity_score"]
panel.loc[panel["has_viewed"] == 0, "ipw_weight"] = 1 / (1 - panel.loc[panel["has_viewed"] == 0, "propensity_score"])

# 極端な重みの調整（trimming）
panel["ipw_weight"] = panel["ipw_weight"].clip(upper=panel["ipw_weight"].quantile(0.99))

# 視聴パターン別の平均実績（IPW調整後）
pattern_means_ipw = []
for pattern in fac_session["viewing_pattern"].unique():
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
    "視聴傾向スコア分析: セッションベース + 選択バイアス調整 (ver2: 施設レベル)",
    fontsize=13, fontweight="bold"
)

# (a) 視聴パターン分布
ax = axes[0, 0]
pattern_counts = fac_session["viewing_pattern"].value_counts()
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
ax.set_ylabel("施設数")
ax.set_title("(a) 視聴パターン分布（セッションベース、施設レベル）")
ax.grid(True, alpha=0.3, axis="y")

# (b) 傾向スコア分布
ax = axes[0, 1]
if "propensity_score" in ps_data.columns:
    ax.hist(ps_treated, bins=20, alpha=0.6, color="#4CAF50", label="視聴群", edgecolor="white")
    ax.hist(ps_control, bins=20, alpha=0.6, color="#1565C0", label="未視聴群", edgecolor="white")
    ax.set_xlabel("視聴傾向スコア")
    ax.set_ylabel("施設数")
    ax.set_title("(b) 視聴傾向スコア分布（施設レベル）")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

# (c) IPW調整前後の比較
ax = axes[1, 0]
# 調整前
pattern_means_raw = fac_session.merge(
    panel.groupby("facility_id")["amount"].mean().reset_index(),
    on="facility_id", how="left"
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
ax.set_title("(c) IPW調整前後の比較（施設レベル）")
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
ax.set_title("(d) 視聴パターン別 実績推移（IPW調整、施設レベル）")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "propensity_score_analysis_v2.png")
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

json_path = os.path.join(results_dir, "propensity_score_analysis_v2.json")
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
