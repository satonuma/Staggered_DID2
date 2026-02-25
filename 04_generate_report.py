"""
===================================================================
HTML総合レポート生成
===================================================================
02/03の分析結果(JSON)と既存PNGを読み込み、
自己完結型HTMLレポートを生成する。

- Jinja2テンプレート (Python文字列)
- matplotlib → base64 PNG変換 (CONSORT図, 視聴パターン)
- 既存PNG → base64読み込み
- @media print CSSで印刷/PDF最適化
===================================================================
"""

import base64
import io
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
from jinja2 import Template

warnings.filterwarnings("ignore")

for _font in ["Yu Gothic", "MS Gothic", "Meiryo", "Hiragino Sans", "IPAexGothic"]:
    try:
        matplotlib.rcParams["font.family"] = _font
        break
    except Exception:
        pass
matplotlib.rcParams["axes.unicode_minus"] = False

# === データファイル・カラム設定 (02/03と同一) ===
ENT_PRODUCT_CODE = "00001"
CONTENT_TYPES = ["Webinar", "e-contents", "web講演会"]
ACTIVITY_CHANNEL_FILTER = "web講演会"

FILE_RW_LIST = "rw_list.csv"
FILE_SALES = "sales.csv"
FILE_DIGITAL = "デジタル視聴データ.csv"
FILE_ACTIVITY = "活動データ.csv"
FILE_FACILITY_MASTER = "facility_attribute.csv"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
_required = [FILE_SALES, FILE_DIGITAL, FILE_ACTIVITY, FILE_RW_LIST]
_data_ok = all(os.path.exists(os.path.join(DATA_DIR, f)) for f in _required)
if not _data_ok:
    _alt = os.path.join(SCRIPT_DIR, "data2")
    if all(os.path.exists(os.path.join(_alt, f)) for f in _required):
        DATA_DIR = _alt

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
REPORTS_DIR = os.path.join(SCRIPT_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

START_DATE = "2023-04-01"
N_MONTHS = 33
WASHOUT_MONTHS = 2
LAST_ELIGIBLE_MONTH = 29


# ================================================================
# ユーティリティ関数
# ================================================================

def fig_to_base64(fig, dpi=120):
    """matplotlib figureをbase64 PNG文字列に変換"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)
    return b64


def png_to_base64(filepath):
    """既存PNGファイルをbase64文字列に変換"""
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ================================================================
# データ読み込み (本番形式)
# ================================================================
print("=" * 60)
print(" HTMLレポート生成")
print("=" * 60)

print("\n[データ読み込み]")

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
digital = pd.read_csv(os.path.join(DATA_DIR, FILE_DIGITAL))
n_digital_all = len(digital)
digital["品目コード"] = digital["品目コード"].astype(str).str.strip().str.zfill(5)
digital = digital[digital["品目コード"] == ENT_PRODUCT_CODE].copy()

# 4. 活動データ → web講演会のみ抽出
activity = pd.read_csv(os.path.join(DATA_DIR, FILE_ACTIVITY))
n_activity_all = len(activity)
activity["品目コード"] = activity["品目コード"].astype(str).str.strip().str.zfill(5)
web_lecture = activity[
    (activity["品目コード"] == ENT_PRODUCT_CODE)
    & (activity["活動種別"] == ACTIVITY_CHANNEL_FILTER)
].copy()

# 5. 視聴データ結合 (デジタル + web講演会)
common_cols = ["活動日_dt", "品目コード", "活動種別", "活動種別コード", "fac_honin", "doc"]
viewing = pd.concat([digital[common_cols], web_lecture[common_cols]], ignore_index=True)
n_viewing_combined = len(viewing)
viewing = viewing.rename(columns={
    "活動日_dt": "view_date",
    "fac_honin": "facility_id",
    "doc": "doctor_id",
    "活動種別": "channel_category",
})
viewing["view_date"] = pd.to_datetime(viewing["view_date"], format="mixed")

daily["delivery_date"] = pd.to_datetime(daily["delivery_date"])
months = pd.date_range(start=START_DATE, periods=N_MONTHS, freq="MS")

print(f"  売上データ: {n_sales_all:,} 行 → ENT品目: {len(daily):,} 行")
print(f"  RW医師リスト: {n_rw_all} 行 → seg非空: {len(doctor_master)} 行")
print(f"  デジタル視聴: {n_digital_all:,} 行, 活動データ: {n_activity_all:,} 行")
print(f"  視聴データ結合 (ENT, web講演会抽出): {n_viewing_combined:,} 行")

# JSON結果読み込み
print("\n[JSON結果読み込み]")
with open(os.path.join(RESULTS_DIR, "did_results.json"), "r", encoding="utf-8") as f:
    did_results = json.load(f)
with open(os.path.join(RESULTS_DIR, "cate_results.json"), "r", encoding="utf-8") as f:
    cate_results = json.load(f)

# 医師視聴パターン分析結果
physician_viewing_path = os.path.join(RESULTS_DIR, "physician_viewing_analysis.json")
propensity_score_path = os.path.join(RESULTS_DIR, "propensity_score_analysis.json")
mr_mediation_path = os.path.join(RESULTS_DIR, "mr_activity_mediation.json")
mr_digital_balance_path = os.path.join(RESULTS_DIR, "mr_digital_balance.json")

physician_viewing_results = None
propensity_score_results = None
mr_mediation_results = None
mr_digital_balance_results = None

loaded_files = []
if os.path.exists(physician_viewing_path):
    with open(physician_viewing_path, "r", encoding="utf-8") as f:
        physician_viewing_results = json.load(f)
    loaded_files.append("physician_viewing_analysis.json")

if os.path.exists(propensity_score_path):
    with open(propensity_score_path, "r", encoding="utf-8") as f:
        propensity_score_results = json.load(f)
    loaded_files.append("propensity_score_analysis.json")

if os.path.exists(mr_mediation_path):
    with open(mr_mediation_path, "r", encoding="utf-8") as f:
        mr_mediation_results = json.load(f)
    loaded_files.append("mr_activity_mediation.json")

if os.path.exists(mr_digital_balance_path):
    with open(mr_digital_balance_path, "r", encoding="utf-8") as f:
        mr_digital_balance_results = json.load(f)
    loaded_files.append("mr_digital_balance.json")

print(f"  did_results.json, cate_results.json, {', '.join(loaded_files) if loaded_files else '(医師視聴分析なし)'} 読み込み完了")


# ================================================================
# 除外フロー再実行 (カウント取得 + 対象医師特定)
# ================================================================
print("\n[除外フロー再実行]")

# [A] 施設内医師数==1 の施設に絞り込み (全医師ベース: facility_master.csv)
fac_master_df = pd.read_csv(os.path.join(DATA_DIR, FILE_FACILITY_MASTER))
single_staff_facs = set(fac_master_df[fac_master_df["施設内医師数"] == 1]["fac_honin"])

# [B] 複数施設所属RW医師の除外 (施設フィルタ前の全所属で確認)
facs_per_doc = doctor_master.groupby("doctor_id")["facility_id"].nunique()
single_fac_docs = set(facs_per_doc[facs_per_doc == 1].index)
multi_fac_docs = set(facs_per_doc[facs_per_doc > 1].index)

# クリーンな1:1ペア: 施設内1医師 かつ RW医師が1施設のみ所属
clean_pairs = doctor_master[
    doctor_master["facility_id"].isin(single_staff_facs)
    & doctor_master["doctor_id"].isin(single_fac_docs)
].copy()
fac_to_doc = dict(zip(clean_pairs["facility_id"], clean_pairs["doctor_id"]))
doc_to_fac = dict(zip(clean_pairs["doctor_id"], clean_pairs["facility_id"]))
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

print(f"  処置群: {len(treated_doc_ids)}, 対照群: {len(control_doc_ids)}, 合計: {len(analysis_fac_ids)}")


# ================================================================
# CONSORT フロー図生成
# ================================================================
print("\n[CONSORT フロー図生成]")

def create_consort_diagram(flow):
    fig, ax = plt.subplots(figsize=(10, 18))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis("off")
    fig.suptitle("解析対象集団の選定", fontsize=14, fontweight="bold", y=0.98)

    def draw_box(ax, x, y, w, h, text, color="#E8F4FD", edge="#2196F3"):
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.15", facecolor=color,
                             edgecolor=edge, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=9, fontweight="bold")

    def draw_excluded(ax, x, y, w, h, text, color="#FFEBEE", edge="#E53935"):
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.1", facecolor=color,
                             edgecolor=edge, linewidth=1.2)
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=8, color="#B71C1C")

    def draw_arrow(ax, x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="-|>", color="#455A64", lw=1.5))

    def draw_arrow_right(ax, x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="-|>", color="#E53935", lw=1.2, ls="--"))

    # Data from flow
    n_sales_all = flow.get("total_delivery_rows", 0)
    n_ent_rows = flow.get("ent_delivery_rows", 0)
    n_rw_all = flow.get("total_rw_list", 0)
    n_ent_rw = flow.get("ent_rw_doctors", 0)
    n_view_all = flow.get("total_viewing_rows", 0)
    n_view_after = flow.get("viewing_after_filter", 0)
    total_docs = flow["total_doctors"]
    total_facs = flow["total_facilities"]
    n_single = flow.get("single_staff_facilities", flow.get("single_doc_facilities", 0))
    n_multi_doc = flow.get("multi_staff_facilities", flow.get("multi_doc_facilities", 0))
    n_multi_fac = flow["multi_fac_doctors"]
    n_clean = flow["clean_pairs"]
    n_washout = flow["washout_excluded"]
    n_late = flow["late_excluded"]
    n_treated = flow["final_treated"]
    n_control = flow["final_control"]
    n_final = flow["final_total"]

    cx = 4.0  # center x for main boxes
    ex = 8.0  # x for exclusion boxes
    sp = 1.45  # vertical spacing

    # Step 0a: 全品目売上データ → ENT品目のみ
    y = 19.0
    draw_box(ax, cx, y, 4.2, 0.9,
             f"全品目売上データ\n{n_sales_all:,} 行")
    draw_arrow(ax, cx, y - 0.45, cx, y - sp + 0.45)
    y -= sp
    draw_box(ax, cx, y, 4.2, 0.9,
             f"ENT品目の売上データ\n{n_ent_rows:,} 行")
    draw_arrow_right(ax, cx + 2.1, y, ex - 1.5, y)
    draw_excluded(ax, ex, y, 2.8, 0.7,
                  f"他品目\n{n_sales_all - n_ent_rows:,} 行除外")

    # Step 0b: RW医師リスト → seg非空の医師
    draw_arrow(ax, cx, y - 0.45, cx, y - sp + 0.45)
    y -= sp
    draw_box(ax, cx, y, 4.2, 0.9,
             f"RW対象医師 (seg非空)\n{n_ent_rw} 行 / {total_facs}施設 / {total_docs}医師")
    draw_arrow_right(ax, cx + 2.1, y, ex - 1.5, y)
    draw_excluded(ax, ex, y, 2.8, 0.7,
                  f"seg空欄\n{n_rw_all - n_ent_rw} 行除外")

    # Step 0c: 視聴データ結合 (デジタル + web講演会)
    draw_arrow(ax, cx, y - 0.45, cx, y - sp + 0.45)
    y -= sp
    draw_box(ax, cx, y, 4.2, 0.9,
             f"視聴データ結合\n(デジタル+web講演会)\n{n_view_after:,} 行")
    draw_arrow_right(ax, cx + 2.1, y, ex - 1.5, y)
    draw_excluded(ax, ex, y, 2.8, 0.7,
                  f"他品目/その他活動\n{n_view_all - n_view_after:,} 行除外")

    # Step 1: 1施設複数医師除外
    draw_arrow(ax, cx, y - 0.45, cx, y - sp + 0.45)
    y -= sp
    draw_box(ax, cx, y, 4.2, 0.9,
             f"1施設1医師の施設\n{n_single}施設")
    draw_arrow_right(ax, cx + 2.1, y, ex - 1.5, y)
    draw_excluded(ax, ex, y, 2.8, 0.7,
                  f"複数医師施設\n{n_multi_doc}施設 除外")

    # Step 2: 1医師複数施設除外
    draw_arrow(ax, cx, y - 0.45, cx, y - sp + 0.45)
    y -= sp
    draw_box(ax, cx, y, 4.2, 0.9,
             f"1医師1施設ペア\n{n_clean}施設")
    draw_arrow_right(ax, cx + 2.1, y, ex - 1.5, y)
    draw_excluded(ax, ex, y, 2.8, 0.7,
                  f"複数施設所属\n{n_multi_fac}医師 除外")

    # Step 3: Wash-out除外
    draw_arrow(ax, cx, y - 0.45, cx, y - sp + 0.45)
    y -= sp
    after_washout = n_clean - n_washout
    draw_box(ax, cx, y, 4.2, 0.9,
             f"wash-out後\n{after_washout}施設")
    draw_arrow_right(ax, cx + 2.1, y, ex - 1.5, y)
    draw_excluded(ax, ex, y, 2.8, 0.7,
                  f"wash-out視聴\n{n_washout}医師 除外")

    # Step 4: 遅延除外
    draw_arrow(ax, cx, y - 0.45, cx, y - sp + 0.45)
    y -= sp
    after_late = after_washout - n_late
    draw_box(ax, cx, y, 4.2, 0.9,
             f"遅延除外後\n{after_late}施設")
    draw_arrow_right(ax, cx + 2.1, y, ex - 1.5, y)
    draw_excluded(ax, ex, y, 2.8, 0.7,
                  f"遅延視聴者\n{n_late}医師 除外")

    # Final: analysis sample
    draw_arrow(ax, cx, y - 0.45, cx, y - sp + 0.45)
    y -= sp
    draw_box(ax, cx, y, 4.2, 0.9,
             f"最終分析対象\n{n_final}施設", color="#E8F5E9", edge="#4CAF50")

    # Split to treated / control
    draw_arrow(ax, cx - 0.8, y - 0.45, cx - 1.8, y - sp + 0.45)
    draw_arrow(ax, cx + 0.8, y - 0.45, cx + 1.8, y - sp + 0.45)
    y -= sp

    draw_box(ax, cx - 2.0, y, 3.0, 0.9,
             f"処置群\n{n_treated}施設", color="#FFF3E0", edge="#FF9800")
    draw_box(ax, cx + 2.0, y, 3.0, 0.9,
             f"対照群\n{n_control}施設", color="#E3F2FD", edge="#1565C0")

    return fig


consort_fig = create_consort_diagram(did_results["exclusion_flow"])
consort_b64 = fig_to_base64(consort_fig, dpi=150)
print("  CONSORT図生成完了")


# ================================================================
# 視聴パターン可視化
# ================================================================
print("\n[視聴パターン可視化]")

def create_viewing_patterns_figure(viewing, doctor_master, months,
                                   washout_viewers, late_adopters, multi_fac_docs,
                                   treated_doc_ids, doc_to_fac):
    """除外例と解析対象の視聴パターンを可視化"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("視聴パターンの可視化: 除外例と解析対象", fontsize=13, fontweight="bold")

    # --- (a) Wash-out除外例 ---
    ax = axes[0, 0]
    wo_docs = sorted(list(washout_viewers))[:5]
    if wo_docs:
        for i, doc_id in enumerate(wo_docs):
            doc_views = viewing[viewing["doctor_id"] == doc_id].copy()
            doc_views["month_index"] = (
                (doc_views["view_date"].dt.year - 2023) * 12
                + doc_views["view_date"].dt.month - 4
            )
            view_months = sorted(doc_views["month_index"].unique())
            for vm in view_months:
                ax.barh(i, 1, left=vm - 0.5, height=0.6, color="#E53935", alpha=0.7)
            ax.text(-1.5, i, doc_id, ha="right", va="center", fontsize=7)
    ax.axvline(WASHOUT_MONTHS - 0.5, color="blue", ls="--", lw=1.5, label="wash-out境界")
    ax.set_xlabel("月 (0=2023/4)")
    ax.set_title("(a) 除外: wash-out期間視聴者", fontsize=10)
    ax.set_xlim(-2, N_MONTHS)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="x")

    # --- (b) 複数施設所属除外例 ---
    ax = axes[0, 1]
    mf_docs = sorted(list(multi_fac_docs))[:5]
    if mf_docs:
        for i, doc_id in enumerate(mf_docs):
            doc_facs = doctor_master[doctor_master["doctor_id"] == doc_id]["facility_id"].unique()
            colors = ["#1565C0", "#FF9800", "#4CAF50", "#9C27B0"]
            for j, fac_id in enumerate(doc_facs):
                ax.barh(i, 0.4, left=j, height=0.6, color=colors[j % len(colors)], alpha=0.8)
                ax.text(j + 0.2, i + 0.35, fac_id, ha="center", va="bottom", fontsize=6)
            ax.text(-0.5, i, doc_id, ha="right", va="center", fontsize=7)
    ax.set_xlabel("施設数")
    ax.set_title("(b) 除外: 複数施設所属医師", fontsize=10)
    ax.grid(True, alpha=0.2, axis="x")

    # --- (c) 遅延視聴者除外例 ---
    ax = axes[1, 0]
    late_docs = sorted(list(late_adopters))[:5]
    if late_docs:
        for i, doc_id in enumerate(late_docs):
            doc_views = viewing[viewing["doctor_id"] == doc_id].copy()
            doc_views["month_index"] = (
                (doc_views["view_date"].dt.year - 2023) * 12
                + doc_views["view_date"].dt.month - 4
            )
            view_months = sorted(doc_views["month_index"].unique())
            for vm in view_months:
                ax.barh(i, 1, left=vm - 0.5, height=0.6, color="#FF6F00", alpha=0.7)
            ax.text(-1.5, i, doc_id, ha="right", va="center", fontsize=7)
    ax.axvline(LAST_ELIGIBLE_MONTH + 0.5, color="red", ls="--", lw=1.5, label="遅延境界(2025/10)")
    ax.set_xlabel("月 (0=2023/4)")
    ax.set_title("(c) 除外: 遅延視聴者 (初回>=2025/10)", fontsize=10)
    ax.set_xlim(-2, N_MONTHS)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="x")

    # --- (d) 解析対象の視聴パターン (代表例) ---
    ax = axes[1, 1]
    treated_list = sorted(list(treated_doc_ids))
    sample_docs = treated_list[:min(15, len(treated_list))]
    ch_colors = {"Webinar": "#1f77b4", "e-contents": "#ff7f0e", "web講演会": "#2ca02c"}

    for i, doc_id in enumerate(sample_docs):
        doc_views = viewing[viewing["doctor_id"] == doc_id].copy()
        doc_views["month_index"] = (
            (doc_views["view_date"].dt.year - 2023) * 12
            + doc_views["view_date"].dt.month - 4
        )
        for ch in ch_colors:
            ch_views = doc_views[doc_views["channel_category"] == ch]
            if len(ch_views) == 0:
                continue
            view_months = sorted(ch_views["month_index"].unique())
            for vm in view_months:
                ax.barh(i, 1, left=vm - 0.5, height=0.5,
                        color=ch_colors[ch], alpha=0.7)

    ax.set_xlabel("月 (0=2023/4)")
    ax.set_title("(d) 解析対象: 処置群の視聴パターン (代表例)", fontsize=10)
    ax.set_xlim(-2, N_MONTHS)
    ax.grid(True, alpha=0.2, axis="x")

    # legend for channels
    legend_patches = [mpatches.Patch(color=c, label=ch, alpha=0.7)
                      for ch, c in ch_colors.items()]
    ax.legend(handles=legend_patches, fontsize=8, loc="upper left")

    plt.tight_layout()
    return fig


viewing_fig = create_viewing_patterns_figure(
    viewing, doctor_master, months,
    washout_viewers, late_adopters, multi_fac_docs,
    treated_doc_ids, doc_to_fac
)
viewing_b64 = fig_to_base64(viewing_fig, dpi=130)
print("  視聴パターン図生成完了")


# ================================================================
# コホート分布グラフ生成
# ================================================================
print("\n[コホート分布グラフ生成]")

def create_cohort_distribution_figure(cohort_dist):
    """コホート分布（初回視聴月ごとの施設数）をバーチャートで可視化"""
    labels = list(cohort_dist.keys())
    values = list(cohort_dist.values())

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(range(len(labels)), values, color="#1565C0", alpha=0.85, edgecolor="white")

    # 棒の上に数値ラベル
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                str(v), ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("施設数", fontsize=10)
    ax.set_title("コホート分布: 処置群の初回視聴月ごとの施設数", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(True, alpha=0.2, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig

cohort_dist_fig = create_cohort_distribution_figure(did_results["cohort_distribution"])
cohort_dist_b64 = fig_to_base64(cohort_dist_fig, dpi=130)
print("  コホート分布グラフ生成完了")


# ================================================================
# 既存PNGの読み込み
# ================================================================
print("\n[既存PNG読み込み]")

existing_pngs = {}
png_files = [
    "staggered_did_results.png",
    "cate_results.png",
    "cate_dynamic_effects.png",
    "physician_viewing_analysis.png",
    "propensity_score_analysis.png",
    "mr_activity_mediation.png",
    "mr_digital_balance.png"
]
for name in png_files:
    path = os.path.join(SCRIPT_DIR, name)
    if os.path.exists(path):
        existing_pngs[name] = png_to_base64(path)
        print(f"  {name} 読み込み完了")
    else:
        print(f"  {name} が見つかりません")
        existing_pngs[name] = ""


# ================================================================
# HTMLテンプレート
# ================================================================
HTML_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Staggered DID分析 総合レポート</title>
<style>
  :root {
    --primary: #1565C0;
    --accent: #FF6F00;
    --success: #2E7D32;
    --danger: #C62828;
    --bg: #FAFAFA;
    --card-bg: #FFFFFF;
    --border: #E0E0E0;
    --text: #212121;
    --text-secondary: #616161;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: "Yu Gothic", "Meiryo", "Hiragino Sans", sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
  }
  .container { max-width: 1100px; margin: 0 auto; padding: 20px; }
  header {
    background: linear-gradient(135deg, var(--primary), #0D47A1);
    color: white;
    padding: 30px 0;
    text-align: center;
  }
  header h1 { font-size: 1.8em; margin-bottom: 8px; }
  header p { opacity: 0.9; font-size: 0.95em; }

  nav {
    background: var(--card-bg);
    border-bottom: 1px solid var(--border);
    padding: 12px 0;
    position: sticky;
    top: 0;
    z-index: 100;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
  }
  nav .container { display: flex; flex-wrap: wrap; gap: 8px; padding: 0 20px; }
  nav a {
    text-decoration: none;
    color: var(--primary);
    padding: 4px 12px;
    border-radius: 4px;
    font-size: 0.85em;
    font-weight: bold;
    transition: background 0.2s;
  }
  nav a:hover { background: #E3F2FD; }

  section {
    background: var(--card-bg);
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    margin: 24px 0;
    padding: 28px;
  }
  h2 {
    color: var(--primary);
    font-size: 1.35em;
    border-bottom: 3px solid var(--primary);
    padding-bottom: 8px;
    margin-bottom: 20px;
  }
  h3 {
    color: #37474F;
    font-size: 1.1em;
    margin: 18px 0 10px 0;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 0.9em;
  }
  th, td {
    border: 1px solid var(--border);
    padding: 8px 12px;
    text-align: center;
  }
  th {
    background: #E3F2FD;
    font-weight: bold;
    color: #1565C0;
  }
  tr:nth-child(even) { background: #F5F5F5; }
  tr:hover { background: #E8F5E9; }

  .sig { color: var(--danger); font-weight: bold; }
  .ns { color: var(--text-secondary); }

  .img-container {
    text-align: center;
    margin: 16px 0;
  }
  .img-container img {
    max-width: 100%;
    height: auto;
    border: 1px solid var(--border);
    border-radius: 4px;
  }

  .param-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin: 12px 0;
  }
  .param-box {
    background: #F5F5F5;
    border-radius: 6px;
    padding: 14px;
    border-left: 4px solid var(--primary);
  }
  .param-box h4 {
    color: var(--primary);
    margin-bottom: 8px;
    font-size: 0.95em;
  }
  .param-box ul {
    list-style: none;
    font-size: 0.88em;
  }
  .param-box ul li { padding: 2px 0; }
  .param-box ul li::before { content: "\\25B8 "; color: var(--primary); }

  .highlight-box {
    background: #FFF8E1;
    border: 1px solid #FFE082;
    border-radius: 6px;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 0.92em;
  }

  .conclusion-box {
    background: #E8F5E9;
    border: 1px solid #A5D6A7;
    border-radius: 6px;
    padding: 18px;
    margin: 16px 0;
  }

  footer {
    text-align: center;
    padding: 20px;
    color: var(--text-secondary);
    font-size: 0.85em;
    border-top: 1px solid var(--border);
    margin-top: 32px;
  }

  @media print {
    @page { size: A4 landscape; margin: 15mm; }
    body { background: white; font-size: 10pt; }
    nav { display: none; }
    section { break-inside: avoid; box-shadow: none; border: 1px solid #ccc; }
    .img-container img { max-width: 90%; }
    header { background: white; color: black; border-bottom: 2px solid black; }
  }
</style>
</head>
<body>

<header>
  <div class="container">
    <h1>Staggered DID分析 総合レポート</h1>
    <p>デジタルコンテンツ視聴の因果効果検証 | 観測期間: 2023/4 - 2025/12</p>
  </div>
</header>

<nav>
  <div class="container">
    <a href="#sec1">1. 分析要件</a>
    <a href="#sec2">2. 解析対象集団の選定</a>
    <a href="#sec3">3. 基礎集計</a>
    <a href="#sec4">4. 視聴パターン</a>
    <a href="#sec5">5. DID推定結果</a>
    <a href="#sec6">6. CATE分析</a>
    <a href="#sec7">7. 医師視聴パターン分析</a>
    <a href="#sec8">8. 結論</a>
  </div>
</nav>

<div class="container">

<!-- ============================================================ -->
<!-- Section 1: 分析要件定義 -->
<!-- ============================================================ -->
<section id="sec1">
<h2>1. 分析要件定義</h2>

<h3>1.1 分析目的</h3>
<p>デジタルコンテンツ（Webinar, e-contents, web講演会）の視聴が、品目コード={{ ent_product_code }}<の納入額に与える因果効果を推定する。視聴開始時期が施設医師ごとに異なる「ずれた処置（staggered treatment）」に対応したDID推定量を使用する。</p>
<p style="margin-top:8px;">分析対象は <strong>品目コード={{ ent_product_code }}</strong> (ENT) の売上データに限定し、医師は <strong>RW医師</strong> のみを使用する。</p>

<h3>1.2 データ構造</h3>
<div class="param-grid">
  <div class="param-box">
    <h4>売上データ (sales.csv)</h4>
    <ul>
      <li>日別・施設別・品目別の売上実績</li>
      <li>カラム: 日付, 施設（本院に合算）コード, DCF施設コード, 品目コード, 実績</li>
      <li>日付: YYYYMMDD文字列、実績: 数値文字列</li>
      <li>全品目: {{ n_sales_all_rows }} 行 → ENT品目: {{ n_sales_rows }} 行</li>
    </ul>
  </div>
  <div class="param-box">
    <h4>デジタル視聴データ (デジタル視聴データ.csv)</h4>
    <ul>
      <li>Webinar / e-contents の視聴ログ</li>
      <li>カラム: 活動日_dt, 品目コード, 活動種別, 活動種別コード, fac_honin, doc 等</li>
      <li>{{ n_digital_all_rows }} 行</li>
    </ul>
  </div>
  <div class="param-box">
    <h4>活動データ (活動データ.csv)</h4>
    <ul>
      <li>MR入力の活動記録 (web講演会を含む)</li>
      <li>カラム: 活動日_dt, 品目コード, 活動種別コード, 活動種別, fac_honin, doc 等</li>
      <li>{{ n_activity_all_rows }} 行 → web講演会抽出後に視聴データと結合</li>
    </ul>
  </div>
  <div class="param-box">
    <h4>RW医師リスト (rw_list.csv)</h4>
    <ul>
      <li>ENT対象医師リスト (品目カラムなし)</li>
      <li>カラム: doc, doc_name, fac_honin, fac_honin_name, fac, fac_name, seg</li>
      <li>seg非空 = RW対象</li>
      <li>全体: {{ n_rw_all_rows }} 行 → seg非空: {{ n_doctor_rows }} 行</li>
    </ul>
  </div>
</div>

<div class="highlight-box">
  <strong>視聴データの結合:</strong>
  デジタル視聴データ (Webinar + e-contents) と 活動データ (web講演会のみ抽出) を結合して分析用の視聴データとする。
  結合後: {{ n_viewing_combined_rows }} 行
</div>

<h3>1.3 分析パラメータ</h3>
<div class="param-grid">
  <div class="param-box">
    <h4>期間設定</h4>
    <ul>
      <li>観測期間: 2023/4 - 2025/12 (33ヶ月)</li>
      <li>Wash-out期間: 2ヶ月 (2023/4-5)</li>
      <li>最終適格月: 2025/9 (month 29)</li>
    </ul>
  </div>
  <div class="param-box">
    <h4>除外基準</h4>
    <ul>
      <li>売上データ: ENT品目コード ({{ ent_product_code }}) 以外を除外</li>
      <li>医師リスト: seg列が空欄の医師を除外</li>
      <li>視聴データ: ENT品目 + 対象チャネル (Webinar, e-contents, web講演会) のみ</li>
      <li>1施設に複数医師が所属 → 施設除外</li>
      <li>1医師が複数施設に所属 → 医師除外</li>
      <li>wash-out期間に視聴あり → 除外</li>
      <li>初回視聴が2025/10以降 → 除外</li>
    </ul>
  </div>
</div>

<h3>1.4 推定手法</h3>
<div class="param-grid">
  <div class="param-box">
    <h4>TWFE (Two-Way Fixed Effects)</h4>
    <ul>
      <li>標準的な双方向固定効果モデル</li>
      <li>施設固定効果 + 時間固定効果</li>
      <li>クラスターロバストSE（施設レベル）</li>
    </ul>
  </div>
  <div class="param-box">
    <h4>Callaway-Sant'Anna (2021)</h4>
    <ul>
      <li>ずれた処置に頑健な推定量</li>
      <li>コホート別ATT(g,t) → 動的効果集約</li>
      <li>Bootstrap SE (N=200全体, N=100チャネル別)</li>
    </ul>
  </div>
</div>

<h3>1.5 用語解説</h3>
<table>
  <tr><th>用語</th><th>補足説明</th></tr>
  <tr><td>DID (差の差分法)</td><td>処置群と対照群の「前後差の差」で因果効果を推定する準実験的手法</td></tr>
  <tr><td>ATT (Average Treatment effect on the Treated)</td><td>処置を受けた群における平均処置効果</td></tr>
  <tr><td>TWFE (Two-Way Fixed Effects)</td><td>個体（施設）と時点の両方の固定効果を制御する回帰モデル</td></tr>
  <tr><td>Callaway-Sant'Anna (CS)</td><td>処置開始時期が異なる場合のバイアスを補正するDID推定量</td></tr>
  <tr><td>CATE (Conditional Average Treatment Effect)</td><td>属性条件付きの平均処置効果（異質的処置効果）</td></tr>
  <tr><td>コホート</td><td>同一時期に処置を受け始めた施設のグループ</td></tr>
  <tr><td>イベント時間 (event time)</td><td>処置開始からの相対的な経過月数 (e=0が処置開始月)</td></tr>
  <tr><td>Bootstrap</td><td>リサンプリングにより標準誤差・信頼区間を算出する統計手法</td></tr>
  <tr><td>95% CI (信頼区間)</td><td>真の効果がこの範囲に含まれる確率が95%である区間</td></tr>
  <tr><td>SE (標準誤差)</td><td>推定値のばらつきの大きさ、小さいほど推定が安定</td></tr>
  <tr><td>p値</td><td>帰無仮説（効果ゼロ）が正しい場合にこの推定値以上が観測される確率</td></tr>
  <tr><td>クラスターロバストSE</td><td>施設内の観測値の相関を考慮した標準誤差の補正</td></tr>
  <tr><td>解析対象集団の選定フロー</td><td>データソースから段階的な除外基準を適用して最終解析対象を確定する過程の図示</td></tr>
  <tr><td>fac_honin (施設本院コード)</td><td>本院に合算したコードで、分析の施設単位として使用</td></tr>
</table>
</section>

<!-- ============================================================ -->
<!-- Section 2: 解析対象集団の選定 -->
<!-- ============================================================ -->
<section id="sec2">
<h2>2. 解析対象集団の選定</h2>
<p>解析対象集団の選定フローを以下に示す。売上データ・RW医師リスト・視聴データの各ソースからENT品目を抽出し、施設-医師の1対1対応・wash-out・遅延視聴者の除外を経て、最終的な処置群・対照群を確定した。</p>
<div class="img-container">
  <img src="data:image/png;base64,{{ consort_b64 }}" alt="解析対象集団の選定">
</div>

<h3>除外フロー集計</h3>
<table>
  <tr>
    <th>ステップ</th>
    <th>条件</th>
    <th>除外数</th>
    <th>残余数</th>
  </tr>
  <tr>
    <td>初期</td>
    <td>全品目売上データ</td>
    <td>-</td>
    <td>{{ flow.total_delivery_rows }} 行</td>
  </tr>
  <tr>
    <td>[0a] ENT品目フィルタ</td>
    <td>品目コード = {{ ent_product_code }} のみ</td>
    <td>{{ flow.total_delivery_rows - flow.ent_delivery_rows }} 行</td>
    <td>{{ flow.ent_delivery_rows }} 行</td>
  </tr>
  <tr>
    <td>[0b] RW医師フィルタ</td>
    <td>rw_list.csv の seg 非空</td>
    <td>{{ flow.total_rw_list - flow.ent_rw_doctors }} 行</td>
    <td>{{ flow.ent_rw_doctors }} 行 ({{ flow.total_facilities }}施設 / {{ flow.total_doctors }}医師)</td>
  </tr>
  <tr>
    <td>[0c] 視聴データ結合</td>
    <td>デジタル視聴 + 活動(web講演会) ENT品目</td>
    <td>{{ flow.total_viewing_rows - flow.viewing_after_filter }} 行</td>
    <td>{{ flow.viewing_after_filter }} 行</td>
  </tr>
  <tr>
    <td>[A] 複数医師施設</td>
    <td>1施設に2名以上の医師</td>
    <td>{{ flow.multi_staff_facilities if flow.multi_staff_facilities is defined else flow.get('multi_doc_facilities', 0) }}施設</td>
    <td>{{ flow.single_staff_facilities if flow.single_staff_facilities is defined else flow.get('single_doc_facilities', 0) }}施設</td>
  </tr>
  <tr>
    <td>[B] 複数施設所属</td>
    <td>1医師が2施設以上に所属</td>
    <td>{{ flow.multi_fac_doctors }}医師</td>
    <td>{{ flow.clean_pairs }}施設</td>
  </tr>
  <tr>
    <td>[C] Wash-out視聴</td>
    <td>2023/4-5に視聴あり</td>
    <td>{{ flow.washout_excluded }}医師</td>
    <td>{{ flow.clean_pairs - flow.washout_excluded }}施設</td>
  </tr>
  <tr>
    <td>[D] 遅延視聴者</td>
    <td>初回視聴 >= 2025/10</td>
    <td>{{ flow.late_excluded }}医師</td>
    <td>{{ flow.clean_pairs - flow.washout_excluded - flow.late_excluded }}施設</td>
  </tr>
  <tr style="font-weight: bold; background: #E8F5E9;">
    <td>最終</td>
    <td>分析対象</td>
    <td>-</td>
    <td>{{ flow.final_total }}施設 (処置{{ flow.final_treated }} + 対照{{ flow.final_control }})</td>
  </tr>
</table>
</section>

<!-- ============================================================ -->
<!-- Section 3: 基礎集計 -->
<!-- ============================================================ -->
<section id="sec3">
<h2>3. 基礎集計</h2>

<h3>3.1 処置群・対照群の記述統計</h3>
<table>
  <tr>
    <th>群</th>
    <th>施設数</th>
    <th>月次売上額 平均</th>
    <th>月次売上額 SD</th>
    <th>パネル行数</th>
  </tr>
  <tr>
    <td>対照群 (未視聴)</td>
    <td>{{ did.n_control }}</td>
    <td>{{ "%.1f"|format(desc.control.mean) }}</td>
    <td>{{ "%.1f"|format(desc.control.std) }}</td>
    <td>{{ desc.control.count }}</td>
  </tr>
  <tr>
    <td>処置群 (視聴)</td>
    <td>{{ did.n_treated }}</td>
    <td>{{ "%.1f"|format(desc.treated.mean) }}</td>
    <td>{{ "%.1f"|format(desc.treated.std) }}</td>
    <td>{{ desc.treated.count }}</td>
  </tr>
</table>

<h3>3.2 コホート分布</h3>
<p>処置群の初回視聴月ごとの施設数分布:</p>
<div class="img-container">
  <img src="data:image/png;base64,{{ cohort_dist_b64 }}" alt="コホート分布">
</div>

{% if attr_dist %}
<h3>3.3 属性分布 (処置群 / 対照群)</h3>
{% for attr_name, dist in attr_dist.items() %}
<h4 style="margin-top:12px; color:#37474F;">{{ attr_name }}</h4>
<table>
  <tr>
    <th>群</th>
    {% for level in dist.treated.keys() %}
    <th>{{ level }}</th>
    {% endfor %}
  </tr>
  <tr>
    <td>処置群</td>
    {% for level, n in dist.treated.items() %}
    <td>{{ n }}</td>
    {% endfor %}
  </tr>
  <tr>
    <td>対照群</td>
    {% for level, n in dist.control.items() %}
    <td>{{ n }}</td>
    {% endfor %}
  </tr>
</table>
{% endfor %}
{% endif %}
</section>

<!-- ============================================================ -->
<!-- Section 4: 視聴パターン -->
<!-- ============================================================ -->
<section id="sec4">
<h2>4. 視聴パターンの可視化</h2>
<p>除外された医師と解析対象の医師の視聴パターンを比較する。</p>

<div class="highlight-box">
  <strong>凡例:</strong>
  (a) wash-out期間（2023/4-5）に視聴がある医師 → 除外 /
  (b) 複数施設に所属する医師 → 除外 /
  (c) 初回視聴が2025/10以降の医師 → 除外 /
  (d) 解析対象（処置群）の代表的視聴パターン
</div>

<div class="img-container">
  <img src="data:image/png;base64,{{ viewing_b64 }}" alt="視聴パターン可視化">
</div>
</section>

<!-- ============================================================ -->
<!-- Section 5: DID推定結果 -->
<!-- ============================================================ -->
<section id="sec5">
<h2>5. DID推定結果</h2>

<h3>5.1 全体推定結果の比較</h3>
<table>
  <tr>
    <th>手法</th>
    <th>ATT <small>(処置群の平均処置効果)</small></th>
    <th>SE <small>(標準誤差)</small></th>
    <th>p値</th>
    <th>95% CI <small>(信頼区間)</small></th>
    <th>有意性</th>
  </tr>
  <tr>
    <td>TWFE (全体)</td>
    <td>{{ "%.2f"|format(twfe.att) }}</td>
    <td>{{ "%.2f"|format(twfe.se) }}</td>
    <td>{{ "%.6f"|format(twfe.p) }}</td>
    <td>[{{ "%.2f"|format(twfe.ci_lo) }}, {{ "%.2f"|format(twfe.ci_hi) }}]</td>
    <td class="{{ 'sig' if twfe.sig != 'n.s.' else 'ns' }}">{{ twfe.sig }}</td>
  </tr>
  <tr>
    <td>CS (全体)</td>
    <td>{{ "%.2f"|format(cs.att) }}</td>
    <td>{{ "%.2f"|format(cs.se) }}</td>
    <td>{{ "%.6f"|format(cs.p) }}</td>
    <td>[{{ "%.2f"|format(cs.ci_lo) }}, {{ "%.2f"|format(cs.ci_hi) }}]</td>
    <td class="{{ 'sig' if cs.sig != 'n.s.' else 'ns' }}">{{ cs.sig }}</td>
  </tr>
  {% for ch_name, ch in channels.items() %}
  <tr>
    <td>CS ({{ ch_name }})</td>
    <td>{{ "%.2f"|format(ch.att) }}</td>
    <td>{{ "%.2f"|format(ch.se) }}</td>
    <td>{{ "%.6f"|format(ch.p) }}</td>
    <td>[{{ "%.2f"|format(ch.ci_lo) }}, {{ "%.2f"|format(ch.ci_hi) }}]</td>
    <td class="{{ 'sig' if ch.sig != 'n.s.' else 'ns' }}">{{ ch.sig }}</td>
  </tr>
  {% endfor %}
</table>

<h3>5.2 CS動的効果 (全体)</h3>
<table>
  <tr>
    <th>イベント時間</th>
    <th>ATT</th>
    <th>SE</th>
    <th>95% CI下限</th>
    <th>95% CI上限</th>
  </tr>
  {% for row in cs.dynamic %}
  <tr{% if row.event_time == 0 %} style="background:#FFF3E0; font-weight:bold;"{% endif %}>
    <td>e={{ row.event_time }}</td>
    <td>{{ "%.2f"|format(row.att) }}</td>
    <td>{{ "%.2f"|format(row.se) }}</td>
    <td>{{ "%.2f"|format(row.ci_lo) }}</td>
    <td>{{ "%.2f"|format(row.ci_hi) }}</td>
  </tr>
  {% endfor %}
</table>

<h3>5.3 可視化</h3>
{% if png_did %}
<div class="img-container">
  <img src="data:image/png;base64,{{ png_did }}" alt="Staggered DID Results">
</div>
{% else %}
<p>staggered_did_results.png が見つかりません。</p>
{% endif %}

{% if twfe_robust %}
<h3>5.4 ロバストネスチェック: MR活動共変量</h3>
<p>MR活動（面談、面談_アポ、説明会等）の月次実施回数を共変量として追加した場合の推定結果。
デジタル視聴の効果推定がMR活動の時変交絡に頑健かどうかを検証する。</p>

<table>
  <tr>
    <th>モデル</th>
    <th>ATT</th>
    <th>SE</th>
    <th>p値</th>
    <th>95% CI</th>
    <th>有意性</th>
  </tr>
  <tr>
    <td>TWFE (メイン)</td>
    <td>{{ "%.2f"|format(twfe.att) }}</td>
    <td>{{ "%.2f"|format(twfe.se) }}</td>
    <td>{{ "%.6f"|format(twfe.p) }}</td>
    <td>[{{ "%.2f"|format(twfe.ci_lo) }}, {{ "%.2f"|format(twfe.ci_hi) }}]</td>
    <td class="{{ 'sig' if twfe.sig != 'n.s.' else 'ns' }}">{{ twfe.sig }}</td>
  </tr>
  <tr>
    <td>TWFE (+MR活動共変量)</td>
    <td>{{ "%.2f"|format(twfe_robust.att) }}</td>
    <td>{{ "%.2f"|format(twfe_robust.se) }}</td>
    <td>{{ "%.6f"|format(twfe_robust.p) }}</td>
    <td>[{{ "%.2f"|format(twfe_robust.ci_lo) }}, {{ "%.2f"|format(twfe_robust.ci_hi) }}]</td>
    <td class="{{ 'sig' if twfe_robust.sig != 'n.s.' else 'ns' }}">{{ twfe_robust.sig }}</td>
  </tr>
</table>

<table>
  <tr>
    <th>共変量</th>
    <th>係数</th>
    <th>SE</th>
    <th>p値</th>
    <th>有意性</th>
  </tr>
  <tr>
    <td>MR活動回数</td>
    <td>{{ "%.2f"|format(twfe_robust.mr_activity_coef) }}</td>
    <td>{{ "%.2f"|format(twfe_robust.mr_activity_se) }}</td>
    <td>{{ "%.4f"|format(twfe_robust.mr_activity_p) }}</td>
    <td class="{{ 'sig' if twfe_robust.mr_activity_sig != 'n.s.' else 'ns' }}">{{ twfe_robust.mr_activity_sig }}</td>
  </tr>
</table>

<p style="margin-top:8px;">ATT変化率: <strong>{{ "%.1f"|format(twfe_robust.att_change_pct) }}%</strong></p>

<div class="highlight-box">
  <strong>解釈:</strong>
  MR活動（面談等）を共変量として追加してもATTの変化が小さい場合（目安: 変化率10%未満）、
  MR活動による時変交絡の影響は限定的であり、メインTWFE推定の信頼性が支持される。
</div>
{% endif %}
</section>

<!-- ============================================================ -->
<!-- Section 6: CATE分析 -->
<!-- ============================================================ -->
<section id="sec6">
<h2>6. CATE分析 (条件付き平均処置効果)</h2>

<h3>6.1 サブグループ別ATT推定値</h3>
{% for dim in cate_dims %}
<h4 style="margin-top:14px; color:#37474F;">{{ dim.name }}</h4>
<table>
  <tr>
    <th>レベル</th>
    <th>N</th>
    <th>ATT</th>
    <th>SE</th>
    <th>95% CI</th>
    {% if true_mods and dim.name in true_mods %}
    <th>DGP modifier</th>
    {% endif %}
  </tr>
  {% for level in dim.levels %}
  {% set r = cate[dim.name][level] %}
  <tr>
    <td>{{ level }}</td>
    <td>{{ r.n }}</td>
    {% if r.att is not none %}
    <td>{{ "%.1f"|format(r.att) }}</td>
    <td>{{ "%.1f"|format(r.se) }}</td>
    <td>[{{ "%.1f"|format(r.ci_lo) }}, {{ "%.1f"|format(r.ci_hi) }}]</td>
    {% else %}
    <td>N/A</td><td>N/A</td><td>N/A</td>
    {% endif %}
    {% if true_mods and dim.name in true_mods %}
    <td>{{ true_mods.get(dim.name, {}).get(level, "N/A") }}</td>
    {% endif %}
  </tr>
  {% endfor %}
</table>
{% endfor %}

{% if diff_tests %}
<h3>6.2 サブグループ間差の検定</h3>
{% for dim_name, diffs in diff_tests.items() %}
<h4 style="margin-top:12px; color:#37474F;">{{ dim_name }}</h4>
<table>
  <tr>
    <th>比較</th>
    <th>差</th>
    <th>SE</th>
    <th>p値</th>
    <th>有意性</th>
  </tr>
  {% for key, d in diffs.items() %}
  <tr>
    <td>{{ key }}</td>
    <td>{{ "%.1f"|format(d.diff) }}</td>
    <td>{{ "%.1f"|format(d.se) if d.se is not none else "N/A" }}</td>
    <td>{{ "%.4f"|format(d.p) if d.p is not none else "N/A" }}</td>
    <td class="{{ 'sig' if d.sig not in ['n.s.', 'N/A'] else 'ns' }}">{{ d.sig }}</td>
  </tr>
  {% endfor %}
</table>
{% endfor %}
{% endif %}

<h3>6.3 Forest Plot</h3>
{% if png_cate %}
<div class="img-container">
  <img src="data:image/png;base64,{{ png_cate }}" alt="CATE Forest Plot">
</div>
{% endif %}

<h3>6.4 動的効果 (サブグループ別)</h3>
{% if png_cate_dyn %}
<div class="img-container">
  <img src="data:image/png;base64,{{ png_cate_dyn }}" alt="CATE Dynamic Effects">
</div>
{% endif %}
</section>

<!-- ============================================================ -->
<!-- Section 7: 医師視聴パターン分析 -->
<!-- ============================================================ -->
<section id="sec7">
<h2>7. 医師視聴パターン分析</h2>

<div class="highlight-box" style="background:#FFF3E0; border-color:#FFB300;">
  <strong>⚠️ 重要な注意：内生性の問題</strong><br>
  視聴回数は医師の自発的行動であり、制御不可能な変数です。<br>
  - 元々関心が高い医師ほど多く視聴（選択バイアス）<br>
  - 処方意向が高い医師ほど視聴（逆因果）<br>
  - 配信はできるが視聴は強制できない<br>
  <br>
  したがって、本分析の結果は <strong>「関連性」</strong> であり <strong>「因果効果」ではありません</strong>。<br>
  推定値は真の効果の上限値として解釈すべきです。
</div>

<h3>概要</h3>
<p>以下の3つの分析アプローチで、視聴パターンと売上の関連性を多角的に検証します：</p>
<ul>
  <li><strong>7.1</strong>: Intensive vs Extensive Margin（視聴回数ベース）</li>
  <li><strong>7.2</strong>: セッションベース視聴パターン + 傾向スコア調整</li>
  <li><strong>7.3</strong>: MR活動Mediation分析（制御可能な変数）</li>
  <li><strong>7.4</strong>: 統合的解釈と実務的示唆</li>
</ul>

<hr style="margin:20px 0;">

<!-- 7.1: Intensive vs Extensive Margin -->
<h3 id="sec7-1">7.1 視聴回数別限界効果 + 配信成功率を考慮した期待効果分析</h3>

{% if pv_results %}
<p><strong>分析の目的:</strong> 視聴1回目、2回目、3回目...それぞれの限界効果を推定し、配信成功率（視聴確率）を考慮した期待効果を算出。
同じ予算で、既存医師への追加配信 vs 新規医師への初回配信、どちらが効果的かを定量的に評価。</p>

<div class="highlight-box" style="background-color:#fff3cd; border-left:4px solid #ffc107;">
  <strong>重要な発見:</strong><br>
  新規医師の視聴確率は極めて低く（約2%）、既存医師（既に視聴経験がある医師）への配信の方が
  <strong>期待効果が圧倒的に高い</strong>（最大219倍）。
</div>

<h4>7.1.1 視聴回数別の限界効果</h4>
<p>医師×月レベルのパネルデータで、視聴回数別の限界効果を推定（TWFE回帰）。</p>

<table>
  <tr>
    <th>視聴回数</th>
    <th>限界効果（万円）</th>
    <th>SE</th>
    <th>p値</th>
    <th>有意性</th>
  </tr>
  <tr>
    <td>1回目</td>
    <td>{{ "%.2f"|format(pv_results.marginal_effects.view_1st.coefficient) }}</td>
    <td>{{ "%.2f"|format(pv_results.marginal_effects.view_1st.se) }}</td>
    <td>{{ "%.4f"|format(pv_results.marginal_effects.view_1st.p) }}</td>
    <td class="{{ 'sig' if pv_results.marginal_effects.view_1st.sig != 'n.s.' else 'ns' }}">
      {{ pv_results.marginal_effects.view_1st.sig }}
    </td>
  </tr>
  <tr>
    <td>2回目</td>
    <td>{{ "%.2f"|format(pv_results.marginal_effects.view_2nd.coefficient) }}</td>
    <td>{{ "%.2f"|format(pv_results.marginal_effects.view_2nd.se) }}</td>
    <td>{{ "%.4f"|format(pv_results.marginal_effects.view_2nd.p) }}</td>
    <td class="{{ 'sig' if pv_results.marginal_effects.view_2nd.sig != 'n.s.' else 'ns' }}">
      {{ pv_results.marginal_effects.view_2nd.sig }}
    </td>
  </tr>
  <tr>
    <td>3回目</td>
    <td>{{ "%.2f"|format(pv_results.marginal_effects.view_3rd.coefficient) }}</td>
    <td>{{ "%.2f"|format(pv_results.marginal_effects.view_3rd.se) }}</td>
    <td>{{ "%.4f"|format(pv_results.marginal_effects.view_3rd.p) }}</td>
    <td class="{{ 'sig' if pv_results.marginal_effects.view_3rd.sig != 'n.s.' else 'ns' }}">
      {{ pv_results.marginal_effects.view_3rd.sig }}
    </td>
  </tr>
  <tr>
    <td>4回目</td>
    <td>{{ "%.2f"|format(pv_results.marginal_effects.view_4th.coefficient) }}</td>
    <td>{{ "%.2f"|format(pv_results.marginal_effects.view_4th.se) }}</td>
    <td>{{ "%.4f"|format(pv_results.marginal_effects.view_4th.p) }}</td>
    <td class="{{ 'sig' if pv_results.marginal_effects.view_4th.sig != 'n.s.' else 'ns' }}">
      {{ pv_results.marginal_effects.view_4th.sig }}
    </td>
  </tr>
  <tr>
    <td>5回目以上</td>
    <td>{{ "%.2f"|format(pv_results.marginal_effects.view_5plus.coefficient) }}</td>
    <td>{{ "%.2f"|format(pv_results.marginal_effects.view_5plus.se) }}</td>
    <td>{{ "%.4f"|format(pv_results.marginal_effects.view_5plus.p) }}</td>
    <td class="{{ 'sig' if pv_results.marginal_effects.view_5plus.sig != 'n.s.' else 'ns' }}">
      {{ pv_results.marginal_effects.view_5plus.sig }}
    </td>
  </tr>
</table>

<p style="margin-top:10px; font-size:0.95em;">
  視聴回数が増えるほど限界効果が増加する傾向（<strong>逓増効果</strong>）。
  5回目以上の限界効果は1回目の約8倍。
</p>

<h4>7.1.2 視聴確率（配信成功率）</h4>
<p>配信履歴から、各段階での視聴確率を推定。</p>

<table>
  <tr>
    <th>段階</th>
    <th>視聴確率</th>
  </tr>
  <tr>
    <td>新規医師（初回視聴）</td>
    <td style="background-color:#ffebee; font-weight:bold;">{{ "%.1f"|format(pv_results.initial_viewing_rate * 100) }}%</td>
  </tr>
  <tr>
    <td>既存1回 → 2回目</td>
    <td>{{ "%.1f"|format(pv_results.continuation_rates['1'] * 100) }}%</td>
  </tr>
  <tr>
    <td>既存2回 → 3回目</td>
    <td>{{ "%.1f"|format(pv_results.continuation_rates['2'] * 100) }}%</td>
  </tr>
  <tr>
    <td>既存3回 → 4回目</td>
    <td>{{ "%.1f"|format(pv_results.continuation_rates['3'] * 100) }}%</td>
  </tr>
  <tr>
    <td>既存4回 → 5回目</td>
    <td>{{ "%.1f"|format(pv_results.continuation_rates['4'] * 100) }}%</td>
  </tr>
</table>

<div class="highlight-box" style="background-color:#ffebee; border-left:4px solid #f44336;">
  <strong>重大な発見:</strong><br>
  新規医師の視聴確率は <strong>わずか{{ "%.1f"|format(pv_results.initial_viewing_rate * 100) }}%</strong>。
  つまり、<strong>{{ "%.0f"|format((1 - pv_results.initial_viewing_rate) * 100) }}%の配信が無駄</strong>になる。<br>
  一方、既存医師の継続視聴確率は29-62%と高く、配信効率が圧倒的に良い。
</div>

<h4>7.1.3 期待効果（視聴確率 × 限界効果）</h4>
<p>配信成功率を考慮した、実質的な期待効果を算出。</p>

<table>
  <tr>
    <th>配信対象</th>
    <th>視聴確率</th>
    <th>限界効果</th>
    <th>期待効果</th>
    <th>対新規比</th>
  </tr>
  <tr style="background-color:#ffebee;">
    <td><strong>新規医師 1回目</strong></td>
    <td>{{ "%.1f"|format(pv_results.initial_viewing_rate * 100) }}%</td>
    <td>{{ "%.2f"|format(pv_results.marginal_effects.view_1st.coefficient) }}万円</td>
    <td style="font-weight:bold;">{{ "%.2f"|format(pv_results.expected_effects['1st']) }}万円</td>
    <td>1.0倍</td>
  </tr>
  <tr style="background-color:#e8f5e9;">
    <td><strong>既存医師 2回目</strong></td>
    <td>35.5%</td>
    <td>{{ "%.2f"|format(pv_results.marginal_effects.view_2nd.coefficient) }}万円</td>
    <td style="font-weight:bold;">{{ "%.2f"|format(pv_results.expected_effects['2nd']) }}万円</td>
    <td>{{ "%.0f"|format(pv_results.expected_effects['2nd'] / pv_results.expected_effects['1st']) }}倍</td>
  </tr>
  <tr style="background-color:#e8f5e9;">
    <td><strong>既存医師 3回目</strong></td>
    <td>47.4%</td>
    <td>{{ "%.2f"|format(pv_results.marginal_effects.view_3rd.coefficient) }}万円</td>
    <td style="font-weight:bold;">{{ "%.2f"|format(pv_results.expected_effects['3rd']) }}万円</td>
    <td>{{ "%.0f"|format(pv_results.expected_effects['3rd'] / pv_results.expected_effects['1st']) }}倍</td>
  </tr>
  <tr style="background-color:#e8f5e9;">
    <td><strong>既存医師 4回目</strong></td>
    <td>56.6%</td>
    <td>{{ "%.2f"|format(pv_results.marginal_effects.view_4th.coefficient) }}万円</td>
    <td style="font-weight:bold;">{{ "%.2f"|format(pv_results.expected_effects['4th']) }}万円</td>
    <td>{{ "%.0f"|format(pv_results.expected_effects['4th'] / pv_results.expected_effects['1st']) }}倍</td>
  </tr>
  <tr style="background-color:#e8f5e9;">
    <td><strong>既存医師 5回以上</strong></td>
    <td>50.5%</td>
    <td>{{ "%.2f"|format(pv_results.marginal_effects.view_5plus.coefficient) }}万円</td>
    <td style="font-weight:bold;">{{ "%.2f"|format(pv_results.expected_effects['5plus']) }}万円</td>
    <td>{{ "%.0f"|format(pv_results.expected_effects['5plus'] / pv_results.expected_effects['1st']) }}倍</td>
  </tr>
</table>

<div class="conclusion-box" style="background-color:#e8f5e9; border-left:4px solid #4caf50;">
<h4>💡 最適配信戦略</h4>
<p style="font-size:1.1em; font-weight:bold;">{{ pv_results.optimal_strategy.message }}</p>

<table style="margin-top:15px; border:none;">
  <tr>
    <td style="border:none; padding:10px; vertical-align:top; width:50%;">
      <strong>📊 期待効果ランキング:</strong><br>
      {% for label, value in pv_results.optimal_strategy.priority_ranking %}
      {{ loop.index }}. {{ label }}: {{ "%.2f"|format(value) }}万円<br>
      {% endfor %}
    </td>
    <td style="border:none; padding:10px; vertical-align:top; width:50%; background-color:#fff9c4;">
      <strong>✅ 実務的推奨:</strong><br>
      • 既存視聴医師への配信を最優先<br>
      • 視聴回数が多いほど効率的<br>
      • 新規医師への配信は効率が極めて低い<br>
      • 限られた予算は既存医師に集中投下
    </td>
  </tr>
</table>
</div>

<h4>7.1.4 可視化</h4>
{% if png_physician_viewing %}
<div class="img-container">
  <img src="data:image/png;base64,{{ png_physician_viewing }}" alt="Physician Viewing Analysis">
</div>
<p style="font-size:0.9em; color:#616161; margin-top:8px;">
  (a) 視聴回数別の限界効果 / (b) 視聴確率（継続率） / (c) 期待効果の比較 /
  (d) 期待ROI / (e) 配信優先順位 / (f) 最適配分メッセージ
</p>
{% else %}
<p>physician_viewing_analysis.png が見つかりません。</p>
{% endif %}

{% else %}
<p>医師視聴パターン分析結果が見つかりません。<code>05_intensive_extensive_margin.py</code>を実行してください。</p>
{% endif %}

<!-- 7.2: セッションベース視聴パターン + 傾向スコア調整 -->
<hr style="margin:30px 0;">
<h3 id="sec7-2">7.2 セッションベース視聴パターン + 傾向スコア調整</h3>

{% if pv_ps_results %}
<p>視聴回数だけでなく <strong>視聴の時間的パターン</strong> を考慮し、医師・施設属性による <strong>選択バイアスを調整</strong>。</p>

<h4>7.2.1 セッション分類の考え方</h4>
<div class="highlight-box">
  <strong>セッション定義:</strong><br>
  視聴と視聴の間隔が {{ pv_ps_results.session_classification.gap_threshold_days }} 日以上空いた場合、別セッションとみなす。<br><br>

  <strong>視聴パターン分類:</strong><br>
  - <strong>短期集中型</strong>: 短期間に集中視聴（1セッション、期間≤30日）<br>
  - <strong>長期継続型</strong>: 長期間継続視聴（1セッション、期間>30日）<br>
  - <strong>定期視聴型</strong>: 複数セッション、間隔が短い（平均≤60日）<br>
  - <strong>断続視聴型</strong>: 複数セッション、間隔が長い（平均>60日）<br>
  - <strong>単発視聴</strong>: 1回のみ視聴<br>
  - <strong>未視聴</strong>: 視聴なし
</div>

<h4>7.2.2 視聴パターン分布</h4>
<table>
  <tr>
    <th>パターン</th>
    <th>医師数</th>
    <th>割合</th>
  </tr>
  {% for pattern, count in pv_ps_results.session_classification.pattern_distribution.items() %}
  <tr>
    <td>{{ pattern }}</td>
    <td>{{ count }}</td>
    <td>{{ "%.1f"|format(count / pv_ps_total_docs * 100) }}%</td>
  </tr>
  {% endfor %}
</table>

<h4>7.2.3 傾向スコア推定</h4>
<p>視聴有無を、医師・施設属性（経験年数、診療科、地域、施設タイプ）で予測するLogitモデルを推定。</p>

<table>
  <tr>
    <th>指標</th>
    <th>値</th>
  </tr>
  <tr>
    <td>Pseudo R2</td>
    <td>{{ "%.4f"|format(pv_ps_results.propensity_score_model.pseudo_r2) }}</td>
  </tr>
  <tr>
    <td>視聴群の平均傾向スコア</td>
    <td>{{ "%.4f"|format(pv_ps_results.propensity_score_model.treated_mean) }}</td>
  </tr>
  <tr>
    <td>未視聴群の平均傾向スコア</td>
    <td>{{ "%.4f"|format(pv_ps_results.propensity_score_model.control_mean) }}</td>
  </tr>
</table>

<p style="margin-top:10px; font-size:0.95em;">
  Pseudo R2が {{ "%.2f"|format(pv_ps_results.propensity_score_model.pseudo_r2 * 100) }}% で、視聴群の平均傾向スコアが未視聴群より高い。
  これは視聴医師が特定の属性（経験豊富、都市部など）に偏っていることを示唆。
</p>

<h4>7.2.4 IPW調整後の平均売上</h4>
<p>逆確率重み付け（IPW）により、属性バイアスを調整した各パターンの平均売上を推定。</p>

<table>
  <tr>
    <th>パターン</th>
    <th>IPW調整後平均売上</th>
    <th>医師数</th>
  </tr>
  {% for item in pv_ps_results.ipw_adjusted_means %}
  <tr>
    <td>{{ item.pattern }}</td>
    <td>{{ "%.1f"|format(item.mean_ipw) }}</td>
    <td>{{ "%.0f"|format(item.n) }}</td>
  </tr>
  {% endfor %}
</table>

<h4>7.2.5 可視化</h4>
{% if png_propensity_score %}
<div class="img-container">
  <img src="data:image/png;base64,{{ png_propensity_score }}" alt="Propensity Score Analysis">
</div>
<p style="font-size:0.9em; color:#616161; margin-top:8px;">
  (a) セッション分類分布 / (b) 傾向スコア分布 / (c) IPW調整後の平均売上
</p>
{% else %}
<p>propensity_score_analysis.png が見つかりません。</p>
{% endif %}

<div class="highlight-box" style="background-color:#fff3cd; border-left:4px solid #ffc107;">
  <strong>解釈の注意:</strong><br>
  {{ pv_ps_results.interpretation.warning }}<br>
  {{ pv_ps_results.interpretation.recommendation }}
</div>

{% else %}
<p>傾向スコア分析結果が見つかりません。<code>06_propensity_score_analysis.py</code>を実行してください。</p>
{% endif %}

<!-- 7.3: MR活動Mediation分析 -->
<hr style="margin:30px 0;">
<h3 id="sec7-3">7.3 MR活動によるMediation分析</h3>

{% if mr_results %}
<p><strong>MR活動</strong>（訪問回数）は <strong>企業が制御可能な変数</strong>。MR活動が視聴を介して売上に影響する経路を検証。</p>

<h4>7.3.1 分析の枠組み</h4>
<div class="highlight-box">
  <strong>Mediation仮説:</strong><br>
  MR活動 → 視聴機会増加 → 売上向上<br><br>

  <strong>2段階推定:</strong><br>
  ① Stage 1: MR活動 → 視聴回数 (相関)<br>
  ② Stage 2: 予測視聴 → 売上 (間接効果)<br>
  ③ Direct: MR活動 → 売上 (直接効果、視聴を制御)
</div>

<h4>7.3.2 Stage 1: MR活動 → 視聴</h4>
<table>
  <tr>
    <th>指標</th>
    <th>値</th>
  </tr>
  <tr>
    <td>MR活動-視聴相関係数</td>
    <td>{{ "%.4f"|format(mr_results.mr_viewing_correlation) }}</td>
  </tr>
  <tr>
    <td>MR活動の係数</td>
    <td>{{ "%.4f"|format(mr_results.stage1_mr_to_viewing.coefficient) }}</td>
  </tr>
  <tr>
    <td>p値</td>
    <td>{{ "%.4f"|format(mr_results.stage1_mr_to_viewing.p) }}</td>
  </tr>
  <tr>
    <td>有意性</td>
    <td>{{ mr_results.stage1_mr_to_viewing.sig }}</td>
  </tr>
</table>

<p style="margin-top:10px; font-size:0.95em;">
  相関係数が {{ "%.3f"|format(mr_results.mr_viewing_correlation) }} と非常に小さく、
  <strong>MR活動だけでは視聴をほとんど説明できない</strong>。
  これは視聴が複数チャネル（ベンダーサイト、MRメール、web講演会など）から発生し、
  MR活動以外の要因が大きいことを示唆。
</p>

<h4>7.3.3 Stage 2: 視聴 → 売上</h4>
<table>
  <tr>
    <th>変数</th>
    <th>係数</th>
    <th>SE</th>
    <th>p値</th>
    <th>有意性</th>
  </tr>
  <tr>
    <td>視聴回数</td>
    <td>{{ "%.3f"|format(mr_results.stage2_viewing_to_sales.coefficient) }}</td>
    <td>{{ "%.3f"|format(mr_results.stage2_viewing_to_sales.se) }}</td>
    <td>{{ "%.6f"|format(mr_results.stage2_viewing_to_sales.p) }}</td>
    <td class="{{ 'sig' if mr_results.stage2_viewing_to_sales.sig != 'n.s.' else 'ns' }}">
      {{ mr_results.stage2_viewing_to_sales.sig }}
    </td>
  </tr>
</table>

<h4>7.3.4 Mediation効果の分解</h4>
<table>
  <tr>
    <th>効果</th>
    <th>値</th>
  </tr>
  <tr>
    <td>Direct Effect (直接効果)</td>
    <td>{{ "%.3f"|format(mr_results.mediation_effects.direct_effect) }}</td>
  </tr>
  <tr>
    <td>Indirect Effect (間接効果)</td>
    <td>{{ "%.3f"|format(mr_results.mediation_effects.indirect_effect) }}</td>
  </tr>
  <tr>
    <td>Total Effect (総効果)</td>
    <td>{{ "%.3f"|format(mr_results.mediation_effects.total_effect) }}</td>
  </tr>
  <tr>
    <td>間接効果の割合</td>
    <td>{{ "%.1f"|format(mr_results.mediation_effects.indirect_percentage) }}%</td>
  </tr>
</table>

<p style="margin-top:10px; font-size:0.95em;">
  間接効果（MR活動→視聴→売上）が総効果の約 {{ "%.0f"|format(mr_results.mediation_effects.indirect_percentage) }}% を占める。
</p>

<h4>7.3.5 可視化</h4>
{% if png_mr_mediation %}
<div class="img-container">
  <img src="data:image/png;base64,{{ png_mr_mediation }}" alt="MR Activity Mediation Analysis">
</div>
<p style="font-size:0.9em; color:#616161; margin-top:8px;">
  (a) MR活動-視聴の散布図 / (b) MR活動と視聴の時系列 / (c) 直接効果の係数
</p>
{% else %}
<p>mr_activity_mediation.png が見つかりません。</p>
{% endif %}

<div class="conclusion-box">
<h4>Mediation分析の結論</h4>
<p style="font-size:0.95em; margin-top:8px;">
  <strong>{{ mr_results.interpretation.warning }}</strong><br><br>
  {{ mr_results.interpretation.advantage }}<br><br>
  {{ mr_results.interpretation.recommendation }}
</p>
</div>

{% else %}
<p>MR活動Mediation分析結果が見つかりません。<code>07_mr_activity_mediation.py</code>を実行してください。</p>
{% endif %}

<!-- 7.4: 統合的解釈と実務的示唆 -->
<hr style="margin:30px 0;">
<h3 id="sec7-4">7.4 統合的解釈と実務的示唆</h3>

<div class="conclusion-box" style="background-color:#e8f5e9; border-left:4px solid #4caf50;">
<h4>3つの分析から得られた知見の統合</h4>

<h5 style="margin-top:15px;">1️⃣ 視聴パターンと売上の関連性 (7.1)</h5>
<ul style="margin:8px 0; padding-left:20px;">
  <li><strong>Intensive Margin（既存医師への追加視聴）</strong> の方が売上との関連性が強い</li>
  <li>視聴回数が多い医師ほど高売上の傾向（ただし因果関係ではない）</li>
</ul>

<h5 style="margin-top:15px;">2️⃣ 時間的パターンと選択バイアス (7.2)</h5>
<ul style="margin:8px 0; padding-left:20px;">
  <li>視聴パターンを <strong>セッションベース</strong> で分類すると、定期視聴型が最も高売上</li>
  <li>傾向スコアによる調整後も、この傾向は維持される</li>
  <li>ただし、視聴意欲の高い医師が元々高売上である可能性は排除できない</li>
</ul>

<h5 style="margin-top:15px;">3️⃣ 制御可能な変数としてのMR活動 (7.3)</h5>
<ul style="margin:8px 0; padding-left:20px;">
  <li>MR活動と視聴の相関は <strong>ほぼゼロ</strong>（相関係数 {{ "%.3f"|format(mr_results.mr_viewing_correlation) if mr_results else "N/A" }}）</li>
  <li>視聴は多様なチャネル（ベンダーサイト、メール、web講演会）から発生</li>
  <li>MR活動単独では視聴行動を制御できない</li>
</ul>
</div>

<div class="highlight-box" style="background-color:#fff3cd; border-left:4px solid #ffc107;">
<h4>実務的な示唆と推奨アクション</h4>

<h5 style="margin-top:15px;">✅ 推奨される戦略</h5>
<ol style="margin:8px 0; padding-left:25px;">
  <li><strong>既存視聴医師へのフォローアップ強化</strong>
    <ul style="margin:5px 0; padding-left:20px;">
      <li>定期的なリマインド配信（メール、MR経由）</li>
      <li>新コンテンツのプッシュ通知</li>
      <li>視聴履歴に基づくパーソナライズ配信</li>
    </ul>
  </li>

  <li><strong>チャネル横断的な接触機会の創出</strong>
    <ul style="margin:5px 0; padding-left:20px;">
      <li>ベンダーサイトでの露出強化</li>
      <li>web講演会との連携</li>
      <li>MR訪問時のコンテンツ紹介</li>
    </ul>
  </li>

  <li><strong>属性ベースのターゲティング精緻化</strong>
    <ul style="margin:5px 0; padding-left:20px;">
      <li>傾向スコアモデルを活用し、視聴確率の高い医師を優先</li>
      <li>経験年数、地域、施設タイプなどの属性を考慮</li>
    </ul>
  </li>
</ol>

<h5 style="margin-top:15px;">⚠️ 注意すべき点</h5>
<ul style="margin:8px 0; padding-left:20px;">
  <li><strong>視聴は結果であって原因ではない可能性</strong>: 元々興味のある医師が視聴し、その医師が処方する</li>
  <li><strong>配信数を増やすだけでは視聴増加は保証されない</strong>: 多様なチャネルからのアクセスが重要</li>
  <li><strong>MR活動だけでは視聴をコントロールできない</strong>: 統合的なマーケティング戦略が必要</li>
</ul>

<h5 style="margin-top:15px;">📊 今後の分析の方向性</h5>
<ul style="margin:8px 0; padding-left:20px;">
  <li>チャネル別の視聴効率の測定（どのチャネルが最も視聴に繋がるか）</li>
  <li>コンテンツタイプ別の効果検証（疾患情報 vs 製品情報など）</li>
  <li>視聴タイミングと処方タイミングのラグ分析</li>
  <li>RCT（ランダム化比較試験）による因果効果の厳密な検証</li>
</ul>
</div>
</section>

<!-- ============================================================ -->
<!-- Section 8: MR vs デジタルバランス分析 -->
<!-- ============================================================ -->
<section id="sec8">
<h2>8. MR vs デジタルバランス分析：最適リソース配分</h2>

{% if mr_balance_results %}
<div class="highlight-box" style="background-color:#e3f2fd; border-left:4px solid #2196f3;">
  <strong>分析目的:</strong> MR活動とデジタルチャネルの最適なバランスを定量的に評価し、
  具体的なリソース配分シナリオを提示する。
</div>

<h3>8.1 限界効果の推定</h3>
<p>TWFE回帰により、MR活動とデジタル視聴それぞれの売上への限界効果を推定。</p>

<table>
  <tr>
    <th>変数</th>
    <th>限界効果（万円/回）</th>
    <th>SE</th>
    <th>p値</th>
    <th>有意性</th>
  </tr>
  <tr>
    <td>MR活動</td>
    <td>{{ "%.2f"|format(mr_balance_results.marginal_effects.mr.coefficient) }}</td>
    <td>{{ "%.2f"|format(mr_balance_results.marginal_effects.mr.se) }}</td>
    <td>{{ "%.6f"|format(mr_balance_results.marginal_effects.mr.p) }}</td>
    <td class="{{ 'sig' if mr_balance_results.marginal_effects.mr.sig != 'n.s.' else 'ns' }}">
      {{ mr_balance_results.marginal_effects.mr.sig }}
    </td>
  </tr>
  <tr>
    <td>デジタル視聴</td>
    <td>{{ "%.2f"|format(mr_balance_results.marginal_effects.digital.coefficient) }}</td>
    <td>{{ "%.2f"|format(mr_balance_results.marginal_effects.digital.se) }}</td>
    <td>{{ "%.6f"|format(mr_balance_results.marginal_effects.digital.p) }}</td>
    <td class="{{ 'sig' if mr_balance_results.marginal_effects.digital.sig != 'n.s.' else 'ns' }}">
      {{ mr_balance_results.marginal_effects.digital.sig }}
    </td>
  </tr>
</table>

<div class="conclusion-box" style="background-color:#fff3cd; border-left:4px solid #ffc107;">
  <strong>重要な知見:</strong><br>
  デジタル視聴の限界効果（{{ "%.2f"|format(mr_balance_results.marginal_effects.digital.coefficient) }}万円）は、
  MR活動（{{ "%.2f"|format(mr_balance_results.marginal_effects.mr.coefficient) }}万円）の
  <strong>約{{ "%.1f"|format(mr_balance_results.marginal_effects.digital.coefficient / mr_balance_results.marginal_effects.mr.coefficient) }}倍</strong>。
</div>

<h3>8.2 コスト効率性の比較</h3>
<p>コスト仮定を用いて、各チャネルの費用対効果を計算。</p>

<div class="highlight-box">
  <strong>コスト仮定（万円）:</strong><br>
  - MR 1名あたり年間コスト: {{ "{:,.0f}".format(mr_balance_results.cost_assumptions.mr_fte_annual) }}万円<br>
  - MR活動1回あたりコスト: {{ "%.1f"|format(mr_balance_results.cost_assumptions.mr_per_visit) }}万円<br>
  - デジタル配信1回あたりコスト: {{ "%.1f"|format(mr_balance_results.cost_assumptions.digital_per_view) }}万円
</div>

<table>
  <tr>
    <th>指標</th>
    <th>MR活動</th>
    <th>デジタル視聴</th>
  </tr>
  <tr>
    <td>1回あたりコスト（万円）</td>
    <td>{{ "%.1f"|format(mr_balance_results.cost_assumptions.mr_per_visit) }}</td>
    <td>{{ "%.1f"|format(mr_balance_results.cost_assumptions.digital_per_view) }}</td>
  </tr>
  <tr>
    <td>1回あたり売上貢献（万円）</td>
    <td>{{ "%.2f"|format(mr_balance_results.marginal_effects.mr.coefficient) }}</td>
    <td>{{ "%.2f"|format(mr_balance_results.marginal_effects.digital.coefficient) }}</td>
  </tr>
  <tr>
    <td>費用対効果（売上/コスト）</td>
    <td>{{ "%.2f"|format(mr_balance_results.marginal_effects.mr.coefficient / mr_balance_results.cost_assumptions.mr_per_visit) }}</td>
    <td>{{ "%.2f"|format(mr_balance_results.marginal_effects.digital.coefficient / mr_balance_results.cost_assumptions.digital_per_view) }}</td>
  </tr>
</table>

<div class="conclusion-box" style="background-color:#e8f5e9; border-left:4px solid #4caf50;">
  <strong>コスト効率性:</strong><br>
  デジタルの費用対効果は、MRの
  <strong>約{{ "%.0f"|format((mr_balance_results.marginal_effects.digital.coefficient / mr_balance_results.cost_assumptions.digital_per_view) / (mr_balance_results.marginal_effects.mr.coefficient / mr_balance_results.cost_assumptions.mr_per_visit)) }}倍</strong>。
</div>

<h3>8.3 リソース配分シナリオ</h3>
<p>複数のシナリオで、コストと売上のトレードオフを検証。</p>

<table>
  <tr>
    <th>シナリオ</th>
    <th>MR FTE</th>
    <th>デジタル予算<br>（万円）</th>
    <th>総コスト<br>（万円）</th>
    <th>コスト変化</th>
    <th>売上変化</th>
    <th>ROI</th>
  </tr>
  {% for scenario in mr_balance_scenarios %}
  <tr style="{{ 'background-color:#fff9c4;' if loop.index == 1 else '' }}">
    <td><strong>{{ scenario.scenario_name }}</strong></td>
    <td>{{ "%.0f"|format(scenario.mr_fte) }}名</td>
    <td>{{ "{:,.0f}".format(scenario.digital_budget) }}</td>
    <td>{{ "{:,.0f}".format(scenario.total_cost) }}</td>
    <td class="{{ 'sig' if scenario.cost_change < 0 else '' }}">
      {{ "{:+,.0f}".format(scenario.cost_change) }}<br>
      <small>({{ "{:+.1f}".format(scenario.cost_change_pct) }}%)</small>
    </td>
    <td>
      {{ "{:+.1f}".format(scenario.sales_change_pct) }}%
    </td>
    <td>{{ "%.2f"|format(scenario.roi) }}</td>
  </tr>
  {% endfor %}
</table>

<h3>8.4 可視化</h3>
{% if png_mr_balance %}
<div class="img-container">
  <img src="data:image/png;base64,{{ png_mr_balance }}" alt="MR vs Digital Balance Analysis">
</div>
<p style="font-size:0.9em; color:#616161; margin-top:8px;">
  (a) シナリオ別コスト / (b) 売上変化率 / (c) ROI /
  (d) 効率的フロンティア / (e) 配分マップ / (f) 限界効果 / (g) コスト効率性
</p>
{% else %}
<p>mr_digital_balance.png が見つかりません。</p>
{% endif %}

<h3>8.5 実務的推奨アクション</h3>

<div class="conclusion-box" style="background-color:#e8f5e9; border-left:4px solid #4caf50;">
<h4>💡 最適シナリオの提案</h4>

{% if mr_balance_best and mr_balance_current %}

<p style="font-size:1.1em; font-weight:bold; margin-top:10px;">
  推奨: {{ mr_balance_best.scenario_name }}
</p>

<table style="margin-top:10px;">
  <tr>
    <td style="width:50%; padding:10px; vertical-align:top;">
      <strong>📋 現状</strong><br>
      MR FTE: {{ "%.0f"|format(mr_balance_results.baseline.mr_fte) }}名<br>
      デジタル予算: {{ "{:,.0f}".format(mr_balance_results.baseline.digital_budget) }}万円<br>
      総コスト: {{ "{:,.0f}".format(mr_balance_results.current_status.total_cost) }}万円<br>
      ROI: {{ "%.2f"|format(mr_balance_current.roi) }}
    </td>
    <td style="width:50%; padding:10px; vertical-align:top; background-color:#e8f5e9;">
      <strong>✅ 推奨配分</strong><br>
      MR FTE: {{ "%.0f"|format(mr_balance_best.mr_fte) }}名<br>
      デジタル予算: {{ "{:,.0f}".format(mr_balance_best.digital_budget) }}万円<br>
      総コスト: {{ "{:,.0f}".format(mr_balance_best.total_cost) }}万円<br>
      ROI: {{ "%.2f"|format(mr_balance_best.roi) }}
    </td>
  </tr>
</table>

<ul style="margin-top:15px; padding-left:20px;">
  <li><strong>コスト削減額</strong>: {{ "{:,.0f}".format(-mr_balance_best.cost_change) }}万円
      ({{ "%.0f"|format(-mr_balance_best.cost_change_pct) }}%削減)</li>
  <li><strong>売上への影響</strong>: {{ "{:+.1f}".format(mr_balance_best.sales_change_pct) }}%</li>
  <li><strong>ROI改善</strong>: {{ "%.2f"|format(mr_balance_current.roi) }} →
      {{ "%.2f"|format(mr_balance_best.roi) }}
      {% if mr_balance_current.roi != 0 %}
      ({{ "%.1f"|format((mr_balance_best.roi / mr_balance_current.roi - 1) * 100) }}%向上)
      {% endif %}</li>
</ul>
{% endif %}
</div>

<div class="highlight-box" style="background-color:#fff3cd; border-left:4px solid #ffc107;">
<h4>⚠️ 重要な注意事項</h4>
<ul style="margin:8px 0; padding-left:20px;">
  <li>{{ mr_balance_results.interpretation.warning }}</li>
  <li>{{ mr_balance_results.interpretation.recommendation }}</li>
  <li>実際のリソース配分変更は、段階的に実施し、効果を検証しながら進めることを推奨</li>
  <li>MR活動には定量化されない価値（関係構築、情報収集など）も存在する</li>
</ul>
</div>

{% else %}
<p>MR vs デジタルバランス分析結果が見つかりません。<code>08_mr_digital_balance.py</code>を実行してください。</p>
{% endif %}
</section>

<!-- ============================================================ -->
<!-- Section 9: 結論 -->
<!-- ============================================================ -->
<section id="sec9">
<h2>9. 結論・主な知見</h2>

<div class="conclusion-box">
<h3>全体効果</h3>
<ul style="margin:10px 0; padding-left:20px;">
  <li>TWFE推定 <small>(双方向固定効果)</small>: ATT <small>(処置群の平均処置効果)</small> = {{ "%.2f"|format(twfe.att) }} (SE={{ "%.2f"|format(twfe.se) }}, {{ twfe.sig }})</li>
  <li>CS推定 <small>(Callaway-Sant'Anna)</small>: ATT = {{ "%.2f"|format(cs.att) }} (SE={{ "%.2f"|format(cs.se) }}, {{ cs.sig }})</li>
</ul>

<h3>チャネル別効果 (CS推定)</h3>
<ul style="margin:10px 0; padding-left:20px;">
  {% for ch_name, ch in channels.items() %}
  <li>{{ ch_name }}: ATT = {{ "%.2f"|format(ch.att) }} (N={{ ch.n_treated }}, {{ ch.sig }})</li>
  {% endfor %}
</ul>

<h3>CATE <small>(条件付き平均処置効果 / 異質的処置効果)</small></h3>
<ul style="margin:10px 0; padding-left:20px;">
  {% for dim in cate_dims %}
  {% set valid_levels = [] %}
  {% for level in dim.levels %}
    {% if cate[dim.name][level].att is not none %}
      {% if valid_levels.append(level) %}{% endif %}
    {% endif %}
  {% endfor %}
  {% if valid_levels|length >= 2 %}
  <li><strong>{{ dim.name }}:</strong>
    {% for level in valid_levels %}
    {{ level }}({{ "%.1f"|format(cate[dim.name][level].att) }}){{ ", " if not loop.last else "" }}
    {% endfor %}
  </li>
  {% endif %}
  {% endfor %}
</ul>
</div>

<div class="highlight-box">
  <strong>注意事項:</strong><br>
  - サブグループのNが小さいため検出力は限定的<br>
  - 各サブグループのATTは共通の対照群を使用<br>
  - 本分析はシミュレーションデータに基づく方法論検証
</div>
</section>

</div>

<footer>
  <p>Staggered DID分析 総合レポート | 自動生成</p>
  <p>ブラウザの「印刷 → PDF保存」でPDF出力可能</p>
</footer>

</body>
</html>
""")


# ================================================================
# テンプレートデータ構築 & HTML生成
# ================================================================
print("\n[HTML生成]")

# チャネル結果をdot-access可能な辞書に変換
class DotDict(dict):
    __getattr__ = dict.__getitem__

# ─── mr_balance シナリオを Python 側で正規化 ───────────────────────────
# Jinja2 内で list[0] / selectattr チェーンを使わず、Python 側で安全に計算する
_mr_balance_scenarios: list = []
_mr_balance_current = None   # scenario_id=0 (現状維持)
_mr_balance_best    = None   # ROI最大シナリオ
if mr_digital_balance_results:
    _raw_sc = mr_digital_balance_results.get("scenarios", [])
    if isinstance(_raw_sc, dict):          # dict 形式の場合もリストに変換
        _raw_sc = list(_raw_sc.values())
    _mr_balance_scenarios = [DotDict(s) for s in _raw_sc if isinstance(s, dict)]
    _mr_balance_current = next(
        (s for s in _mr_balance_scenarios if s.get("scenario_id", -1) == 0),
        _mr_balance_scenarios[0] if _mr_balance_scenarios else None,
    )
    if _mr_balance_scenarios:
        _mr_balance_best = max(_mr_balance_scenarios, key=lambda s: s.get("roi", 0))

channels = {}
for ch_name, ch_data in did_results.get("cs_channel", {}).items():
    channels[ch_name] = DotDict(ch_data)

# CATE結果をdot-access可能に
cate_data = {}
for dim_name, dim_data in cate_results.get("cate", {}).items():
    cate_data[dim_name] = {}
    for level, level_data in dim_data.items():
        cate_data[dim_name][level] = DotDict(level_data)

# diff_tests
diff_tests = {}
for dim_name, diffs in cate_results.get("diff_tests", {}).items():
    diff_tests[dim_name] = {}
    for key, d in diffs.items():
        diff_tests[dim_name][key] = DotDict(d)

template_data = {
    # 基本情報
    "ent_product_code": ENT_PRODUCT_CODE,
    "n_sales_all_rows": f"{n_sales_all:,}",
    "n_sales_rows": f"{len(daily):,}",
    "n_digital_all_rows": f"{n_digital_all:,}",
    "n_activity_all_rows": f"{n_activity_all:,}",
    "n_viewing_combined_rows": f"{n_viewing_combined:,}",
    "n_rw_all_rows": f"{n_rw_all:,}",
    "n_doctor_rows": f"{len(doctor_master):,}",

    # 除外フロー
    "flow": DotDict(did_results["exclusion_flow"]),

    # 記述統計
    "did": DotDict(did_results),
    "desc": DotDict({
        "control": DotDict(did_results["descriptive_stats"]["control"]),
        "treated": DotDict(did_results["descriptive_stats"]["treated"]),
    }),

    # 属性分布
    "attr_dist": {
        k: DotDict({
            "treated": v.get("treated", {}),
            "control": v.get("control", {}),
        })
        for k, v in cate_results.get("attr_distribution", {}).items()
    },

    # TWFE/CS結果
    "twfe": DotDict(did_results["twfe"]),
    "twfe_robust": DotDict(did_results["twfe_robust"]) if "twfe_robust" in did_results else None,
    "cs": DotDict(did_results["cs_overall"]),

    # チャネル別
    "channels": channels,

    # CATE
    "cate_dims": cate_results.get("dimensions", []),
    "cate": cate_data,
    "true_mods": cate_results.get("true_modifiers", {}),
    "diff_tests": diff_tests,

    # 画像
    "consort_b64": consort_b64,
    "viewing_b64": viewing_b64,
    "cohort_dist_b64": cohort_dist_b64,
    "png_did": existing_pngs.get("staggered_did_results.png", ""),
    "png_cate": existing_pngs.get("cate_results.png", ""),
    "png_cate_dyn": existing_pngs.get("cate_dynamic_effects.png", ""),
    "png_physician_viewing": existing_pngs.get("physician_viewing_analysis.png", ""),
    "png_propensity_score": existing_pngs.get("propensity_score_analysis.png", ""),
    "png_mr_mediation": existing_pngs.get("mr_activity_mediation.png", ""),
    "png_mr_balance": existing_pngs.get("mr_digital_balance.png", ""),

    # 医師視聴パターン分析
    "pv_results": DotDict(physician_viewing_results) if physician_viewing_results else None,
    "pv_total_docs": sum(physician_viewing_results.get("viewing_pattern_distribution", {}).values()) if physician_viewing_results else 0,

    # 傾向スコア分析
    "pv_ps_results": DotDict(propensity_score_results) if propensity_score_results else None,
    "pv_ps_total_docs": sum(propensity_score_results.get("session_classification", {}).get("pattern_distribution", {}).values()) if propensity_score_results else 0,

    # MR活動Mediation分析
    "mr_results": DotDict(mr_mediation_results) if mr_mediation_results else None,

    # MR vs デジタルバランス分析
    "mr_balance_results": DotDict(mr_digital_balance_results) if mr_digital_balance_results else None,
    "mr_balance_scenarios": _mr_balance_scenarios,
    "mr_balance_current":   _mr_balance_current,
    "mr_balance_best":      _mr_balance_best,
}

try:
    html_content = HTML_TEMPLATE.render(**template_data)
except Exception as _render_err:
    import traceback
    print(f"\n[警告] テンプレートレンダリングでエラーが発生しました: {_render_err}")
    traceback.print_exc()
    # エラー内容を埋め込んだ最小HTMLを生成して処理を継続
    html_content = (
        "<!DOCTYPE html><html><head><meta charset='UTF-8'></head><body>"
        f"<h1>レポート生成エラー</h1><pre>{traceback.format_exc()}</pre>"
        "</body></html>"
    )

output_path = os.path.join(REPORTS_DIR, "analysis_report.html")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"  HTMLレポートを保存: {output_path}")
print(f"  ファイルサイズ: {os.path.getsize(output_path) / 1024:.0f} KB")
print("\n  ブラウザで開いて確認してください。")
print("  「印刷 → PDF保存」でPDF出力も可能です。")
print("\n" + "=" * 60)
print(" レポート生成完了")
print("=" * 60)
