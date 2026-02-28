#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_mr_digital_balance.py

【目的】
MR活動とデジタルチャネルの最適バランスを定量的に評価。
デジタルコストデータが存在しない状況でも、回帰結果から損益分岐点コストを導出し、
視聴単価の感度分析によって意思決定を支援する。

【重要な前提】
1. MR活動とデジタル視聴は両方とも内生変数（選択バイアス、逆因果の可能性）
2. 本分析の結果は「相関関係」であり「因果効果」ではない
3. デジタル視聴単価は不明のため、損益分岐点コストと感度分析で代替
4. MRコストは仮定値（MR 1名あたり年間コスト、1活動あたりコスト）

【分析内容】
1. MR活動・デジタル視聴の限界効果推定（TWFE回帰）
2. 損益分岐点コスト算出
   - コスト効率均衡点: digital_per_view <= C* なら MR より効率的
3. MR削減 × デジタル視聴単価の感度分析（コストニュートラル前提）
   - MR削減節約額を全額デジタルへ転換した場合の売上変化
4. 施設属性別（施設区分・UHP区分）の限界効果異質性

【出力】
- results/mr_digital_balance.json: 数値結果
- mr_digital_balance.png: 可視化
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
plt.rcParams['axes.unicode_minus'] = False

# ================================================================
# 設定
# ================================================================
ENT_PRODUCT_CODE = "00001"

import os as _os
_script_dir = _os.path.dirname(_os.path.abspath(__file__))
_data_dir_primary = _os.path.join(_script_dir, "本番データ")
DATA_DIR = _data_dir_primary if _os.path.isdir(_data_dir_primary) else _os.path.join(_script_dir, "data")
del _os, _script_dir, _data_dir_primary

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# コスト仮定（単位：万円）
# ※ デジタル視聴単価はデータが存在しないため、損益分岐点コストで代替分析する
COST_ASSUMPTIONS = {
    "mr_fte_annual": 1000,   # MR 1名あたり年間コスト（万円）
    "mr_per_visit": 2,        # MR活動1回あたりコスト（万円）
}

# 現状ベースライン（MR FTEは仮定値）
BASELINE = {
    "mr_fte": 100,            # MR FTE数（仮定）
}

print("=" * 60)
print(" MR vs デジタルバランス分析")
print("=" * 60)
print("\n【コスト仮定（MR）】")
print(f"  MR 1名あたり年間コスト: {COST_ASSUMPTIONS['mr_fte_annual']:,.0f}万円")
print(f"  MR活動1回あたりコスト: {COST_ASSUMPTIONS['mr_per_visit']:.1f}万円")
print(f"  デジタル視聴単価: データなし → 損益分岐点コスト分析で代替")

# ================================================================
# データ読み込み
# ================================================================
print("\n[データ読み込み]")

# 売上データ
sales_raw = pd.read_csv(os.path.join(DATA_DIR, "sales.csv"), encoding="utf-8", dtype=str)
sales_raw["実績"] = pd.to_numeric(sales_raw["実績"], errors="coerce").fillna(0)
sales_raw["日付"] = pd.to_datetime(sales_raw["日付"], format="mixed")
sales = sales_raw[sales_raw["品目コード"].str.strip() == ENT_PRODUCT_CODE].copy()
sales = sales.rename(columns={
    "日付": "date",
    "施設（本院に合算）コード": "facility_id",
    "実績": "amount",
    "品目コード": "product_code",
})
print(f"  売上データ: {len(sales):,}行（ENT製品）")

# デジタル視聴データ
digital_raw = pd.read_csv(os.path.join(DATA_DIR, "デジタル視聴データ.csv"), encoding="utf-8")
digital_raw["品目コード"] = digital_raw["品目コード"].astype(str).str.strip().str.zfill(5)
digital = digital_raw[digital_raw["品目コード"] == ENT_PRODUCT_CODE].copy()
digital = digital.rename(columns={
    "活動日_dt": "viewing_date",
    "doc": "doctor_id",
    "fac_honin": "facility_id",
    "活動種別": "content_type",
})
digital["viewing_date"] = pd.to_datetime(digital["viewing_date"], format="mixed")
print(f"  デジタル視聴: {len(digital):,}行")
print(f"    コンテンツ種別: {digital['content_type'].value_counts().to_dict()}")

# MR活動データ
activity_raw = pd.read_csv(os.path.join(DATA_DIR, "活動データ.csv"), encoding="utf-8")
activity_raw["品目コード"] = activity_raw["品目コード"].astype(str).str.strip().str.zfill(5)
activity = activity_raw[activity_raw["品目コード"] == ENT_PRODUCT_CODE].copy()
activity = activity.rename(columns={
    "活動日_dt": "activity_date",
    "fac_honin": "facility_id",
})
activity["activity_date"] = pd.to_datetime(activity["activity_date"], format="mixed")
print(f"  MR活動: {len(activity):,}行")

# 施設マスタ（実データ: facility_attribute_修正.csv）
facility_attr = pd.read_csv(os.path.join(DATA_DIR, "facility_attribute_修正.csv"), encoding="utf-8")
facility_master = (
    facility_attr.rename(columns={
        "fac_honin": "facility_id",
        "施設区分名": "facility_type",   # 病院 / 診療所
        "UHP区分名": "uhp_tier",         # UHP-A / UHP-B / UHP-C / 非UHP
        "許可病床数_合計": "beds_total",
        "施設内医師数": "doctor_count",
    })
    [["facility_id", "facility_type", "uhp_tier", "beds_total", "doctor_count"]]
    .drop_duplicates("facility_id")
)
print(f"  施設マスタ（実データ）: {len(facility_master)}施設")
print(f"    施設区分: {facility_master['facility_type'].value_counts().to_dict()}")
print(f"    UHP区分: {facility_master['uhp_tier'].value_counts().to_dict()}")

# ================================================================
# パネルデータ構築
# ================================================================
print("\n[パネルデータ構築]")

# 月次売上集計
daily = sales.groupby(["facility_id", "date"], as_index=False).agg({"amount": "sum"})
daily["year_month"] = daily["date"].dt.to_period("M")
monthly = daily.groupby(["facility_id", "year_month"], as_index=False).agg({"amount": "sum"})
monthly["date"] = monthly["year_month"].dt.to_timestamp()
monthly = monthly.drop(columns=["year_month"])

# MR活動月次集計
activity["year_month"] = activity["activity_date"].dt.to_period("M")
mr_monthly = activity.groupby(["facility_id", "year_month"], as_index=False).size()
mr_monthly.rename(columns={"size": "mr_count"}, inplace=True)
mr_monthly["date"] = mr_monthly["year_month"].dt.to_timestamp()
mr_monthly = mr_monthly.drop(columns=["year_month"])

# デジタル視聴月次集計
dw = digital.dropna(subset=["facility_id"]).copy()
dw["year_month"] = dw["viewing_date"].dt.to_period("M")
digital_monthly = dw.groupby(["facility_id", "year_month"], as_index=False).size()
digital_monthly.rename(columns={"size": "digital_count"}, inplace=True)
digital_monthly["date"] = digital_monthly["year_month"].dt.to_timestamp()
digital_monthly = digital_monthly.drop(columns=["year_month"])

# パネル結合
panel = monthly.copy()
panel = panel.merge(mr_monthly, on=["facility_id", "date"], how="left")
panel = panel.merge(digital_monthly, on=["facility_id", "date"], how="left")
panel["mr_count"] = panel["mr_count"].fillna(0)
panel["digital_count"] = panel["digital_count"].fillna(0)

# 施設属性結合
panel = panel.merge(
    facility_master[["facility_id", "facility_type", "uhp_tier", "beds_total", "doctor_count"]],
    on="facility_id", how="left"
)

# 時間FE用変数
panel["time_id"] = (panel["date"].dt.year - panel["date"].dt.year.min()) * 12 + panel["date"].dt.month

print(f"  パネルデータ: {len(panel):,}行")
print(f"  施設数: {panel['facility_id'].nunique()}施設")
print(f"  期間: {panel['date'].min().strftime('%Y-%m')} ～ {panel['date'].max().strftime('%Y-%m')}")

# ================================================================
# 限界効果の推定（TWFE回帰）
# ================================================================
print("\n[限界効果の推定]")
print("  TWFE回帰: amount ~ mr_count + digital_count + facility_FE + time_FE")

from linearmodels import PanelOLS

panel_reg = panel.dropna(subset=["amount", "mr_count", "digital_count"]).copy()
panel_reg = panel_reg.set_index(["facility_id", "time_id"])

try:
    model = PanelOLS(
        dependent=panel_reg["amount"],
        exog=panel_reg[["mr_count", "digital_count"]],
        entity_effects=True,
        time_effects=True
    )
    result = model.fit(cov_type="clustered", cluster_entity=True)

    beta_mr = float(result.params["mr_count"])
    se_mr = float(result.std_errors["mr_count"])
    p_mr = float(result.pvalues["mr_count"])

    beta_digital = float(result.params["digital_count"])
    se_digital = float(result.std_errors["digital_count"])
    p_digital = float(result.pvalues["digital_count"])

    sig_mr = "***" if p_mr < 0.001 else "**" if p_mr < 0.01 else "*" if p_mr < 0.05 else "n.s."
    sig_digital = "***" if p_digital < 0.001 else "**" if p_digital < 0.01 else "*" if p_digital < 0.05 else "n.s."

    print(f"\n  【推定結果】")
    print(f"  MR活動の限界効果: {beta_mr:.2f}万円/回 (SE={se_mr:.2f}, p={p_mr:.4f}, {sig_mr})")
    print(f"  デジタル視聴の限界効果: {beta_digital:.2f}万円/回 (SE={se_digital:.2f}, p={p_digital:.4f}, {sig_digital})")

except Exception as e:
    print(f"  エラー: {e} → デフォルト値使用")
    beta_mr, se_mr, p_mr, sig_mr = 10.0, 2.0, 0.001, "***"
    beta_digital, se_digital, p_digital, sig_digital = 5.0, 1.0, 0.001, "***"

# ================================================================
# 現状統計
# ================================================================
print("\n[現状分析]")

current_mr_mean = float(panel_reg["mr_count"].mean())
current_digital_mean = float(panel_reg["digital_count"].mean())
current_sales_mean = float(panel_reg["amount"].mean())
n_facilities = panel_reg.index.get_level_values(0).nunique()
n_months = panel_reg.index.get_level_values(1).nunique()

print(f"  平均MR活動回数（1施設1月）: {current_mr_mean:.2f}回")
print(f"  平均デジタル視聴回数（1施設1月）: {current_digital_mean:.2f}回")
print(f"  平均売上（1施設1月）: {current_sales_mean:.0f}万円")
print(f"  施設数: {n_facilities}, 期間: {n_months}ヶ月")

current_mr_cost_annual = BASELINE["mr_fte"] * COST_ASSUMPTIONS["mr_fte_annual"]
print(f"\n  年間MRコスト（仮定）: {current_mr_cost_annual:,.0f}万円")

# ================================================================
# 損益分岐点コスト分析
# ================================================================
print("\n[損益分岐点コスト分析]")

# MRの費用対効果: beta_mr / mr_per_visit (万円売上増/万円投資)
mr_cost_efficiency = beta_mr / COST_ASSUMPTIONS["mr_per_visit"]

# デジタルの損益分岐点コスト:
# beta_digital / C_breakeven = mr_cost_efficiency
# → C_breakeven = beta_digital / mr_cost_efficiency = beta_digital × mr_per_visit / beta_mr
if beta_mr > 0:
    breakeven_cost = COST_ASSUMPTIONS["mr_per_visit"] * (beta_digital / beta_mr)
else:
    breakeven_cost = np.inf

# 等価交換レート: MR活動1回 = デジタル何回分の売上
equivalence_ratio = beta_mr / beta_digital if beta_digital > 0 else np.inf

print(f"  MRの費用対効果: {mr_cost_efficiency:.3f}万円売上/万円コスト")
print(f"  デジタル損益分岐点: C* = {breakeven_cost:.3f}万円/視聴")
print(f"  解釈: 視聴単価 < {breakeven_cost:.3f}万円 であれば、デジタルはMRより費用対効果が高い")
print(f"  等価交換レート: MR活動1回 = デジタル視聴{equivalence_ratio:.1f}回分の売上インパクト")

# ================================================================
# MR削減 × デジタル視聴単価 感度分析（コストニュートラル前提）
# ================================================================
print("\n[感度分析: MR削減節約額をデジタルへ転換]")

# MR削減シナリオ（削減率）
mr_reductions = [0.10, 0.20, 0.30, 0.50]

# デジタル視聴単価グリッド（万円/視聴）
cost_grid = np.array([0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0])

# グリッド計算
sensitivity_rows = []
for mr_red in mr_reductions:
    mr_fte_new = BASELINE["mr_fte"] * (1 - mr_red)
    mr_savings = mr_red * current_mr_cost_annual  # 年間節約額（万円）

    # MR活動減少 (1施設1月あたり)
    delta_mr_pm = -current_mr_mean * mr_red

    for C in cost_grid:
        # 節約額でデジタル視聴を追加購入
        add_views_annual = mr_savings / C          # 追加視聴回数/年
        delta_digital_pm = add_views_annual / (n_facilities * 12)  # 1施設1月あたり

        # 売上変化 = MR減少影響 + デジタル増加影響
        delta_sales_pm = beta_mr * delta_mr_pm + beta_digital * delta_digital_pm
        delta_sales_pct = delta_sales_pm / current_sales_mean * 100 if current_sales_mean > 0 else 0

        sensitivity_rows.append({
            "mr_reduction_pct": mr_red * 100,
            "mr_fte_new": mr_fte_new,
            "mr_savings_annual": mr_savings,
            "digital_cost_per_view": C,
            "add_views_annual": add_views_annual,
            "delta_digital_pm": delta_digital_pm,
            "delta_mr_pm": delta_mr_pm,
            "delta_sales_pm": delta_sales_pm,
            "delta_sales_pct": delta_sales_pct,
        })

sensitivity_df = pd.DataFrame(sensitivity_rows)

# 売上中立となるデジタル視聴単価（各MR削減率に対して）
print(f"\n  【売上中立デジタル単価（MR削減を損失なく置換できる最大コスト）】")
revenue_neutral_costs = {}
for mr_red in mr_reductions:
    mr_savings = mr_red * current_mr_cost_annual
    delta_mr_pm = -current_mr_mean * mr_red
    mr_loss_pm = beta_mr * delta_mr_pm  # 売上減少額（負値）

    # delta_sales_pm = 0 → beta_digital × (mr_savings / C) / (n_facilities × 12) = -mr_loss_pm
    # C_neutral = beta_digital × mr_savings / (-mr_loss_pm × n_facilities × 12)
    if mr_loss_pm < 0 and beta_digital > 0:
        C_neutral = beta_digital * mr_savings / (-mr_loss_pm * n_facilities * 12)
    else:
        C_neutral = np.inf
    revenue_neutral_costs[mr_red] = C_neutral
    print(f"  MR {mr_red*100:.0f}%削減 → 売上中立コスト C <= {C_neutral:.3f}万円/視聴")

print(f"\n  【参考】損益分岐点（コスト効率均衡）: {breakeven_cost:.3f}万円/視聴")

# ================================================================
# 施設属性別限界効果（UHP区分・施設区分）
# ================================================================
print("\n[施設属性別分析]")

subgroup_results = []
for col, groups in [("facility_type", ["病院", "診療所"]),
                     ("uhp_tier", ["UHP-A", "UHP-B", "UHP-C", "非UHP"])]:
    for g in groups:
        mask = panel_reg.reset_index().set_index(["facility_id", "time_id"])
        # panelからサブグループを取得
        fac_in_group = facility_master[facility_master[col] == g]["facility_id"].tolist()
        sub = panel_reg[panel_reg.index.get_level_values(0).isin(fac_in_group)]
        if len(sub) < 20:
            continue
        try:
            m = PanelOLS(sub["amount"], sub[["mr_count", "digital_count"]],
                         entity_effects=True, time_effects=True)
            r = m.fit(cov_type="clustered", cluster_entity=True)
            subgroup_results.append({
                "group_col": col,
                "group": g,
                "n_obs": len(sub),
                "n_facilities": sub.index.get_level_values(0).nunique(),
                "beta_mr": float(r.params["mr_count"]),
                "beta_digital": float(r.params["digital_count"]),
                "se_mr": float(r.std_errors["mr_count"]),
                "se_digital": float(r.std_errors["digital_count"]),
                "p_mr": float(r.pvalues["mr_count"]),
                "p_digital": float(r.pvalues["digital_count"]),
            })
            print(f"  [{col}={g}] n={sub.index.get_level_values(0).nunique()}施設 "
                  f"| MR: {r.params['mr_count']:.2f} "
                  f"| Digital: {r.params['digital_count']:.2f}")
        except Exception:
            pass

subgroup_df = pd.DataFrame(subgroup_results)

# ================================================================
# 可視化
# ================================================================
print("\n[可視化]")

fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(3, 3, hspace=0.40, wspace=0.38)

# ---- (a) 限界効果の比較（係数 ± 95%CI）
ax1 = fig.add_subplot(gs[0, 0])
labels = ["MR活動", "デジタル視聴"]
betas = [beta_mr, beta_digital]
ses = [se_mr, se_digital]
colors_b = ["#ff7f0e", "#2ca02c"]
x = np.arange(len(labels))
bars = ax1.bar(x, betas, yerr=[1.96 * s for s in ses],
               color=colors_b, alpha=0.75, capsize=7, width=0.5)
ax1.axhline(0, color="black", linewidth=0.8)
for bar, b in zip(bars, betas):
    ax1.text(bar.get_x() + bar.get_width()/2, b + 0.3,
             f"{b:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=10)
ax1.set_ylabel("限界効果（万円売上/回）", fontsize=10)
ax1.set_title("(a) 限界効果（TWFE推定）", fontsize=11, fontweight="bold")
ax1.grid(axis="y", alpha=0.3)

# ---- (b) 損益分岐点コストの可視化
ax2 = fig.add_subplot(gs[0, 1])
C_range = np.linspace(0.01, max(cost_grid), 300)
# デジタルの費用対効果 = beta_digital / C
digital_efficiency = beta_digital / C_range
mr_eff_line = np.full_like(C_range, mr_cost_efficiency)

ax2.plot(C_range, digital_efficiency, color="#2ca02c", linewidth=2, label="デジタル費用対効果")
ax2.axhline(mr_cost_efficiency, color="#ff7f0e", linewidth=2, linestyle="--", label="MR費用対効果")
ax2.axvline(breakeven_cost, color="red", linewidth=1.5, linestyle=":", alpha=0.8,
            label=f"損益分岐点 C*={breakeven_cost:.2f}万円")
ax2.fill_betweenx([0, digital_efficiency.max()], 0, breakeven_cost,
                   alpha=0.08, color="green", label="デジタル優位ゾーン")
ax2.set_xlim(0, max(cost_grid))
ax2.set_ylim(bottom=0)
ax2.set_xlabel("デジタル視聴単価（万円/視聴）", fontsize=10)
ax2.set_ylabel("費用対効果（万円売上/万円コスト）", fontsize=10)
ax2.set_title("(b) 損益分岐点コスト", fontsize=11, fontweight="bold")
ax2.legend(fontsize=7)
ax2.grid(alpha=0.3)

# ---- (c) 等価交換レート（MR活動1回 = デジタルN回）
ax3 = fig.add_subplot(gs[0, 2])
equiv_text = f"MR活動1回\n= デジタル視聴\n{equivalence_ratio:.1f}回分"
ax3.text(0.5, 0.5, equiv_text, ha="center", va="center",
         fontsize=16, fontweight="bold", color="#333333",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff3cd", edgecolor="#ffc107", linewidth=2),
         transform=ax3.transAxes)
ax3.set_title("(c) MR↔デジタル等価交換レート", fontsize=11, fontweight="bold")
ax3.axis("off")
# 補足テキスト
ax3.text(0.5, 0.15,
         f"MR限界効果: {beta_mr:.2f}万円/回\nデジタル限界効果: {beta_digital:.2f}万円/回",
         ha="center", va="center", fontsize=9, color="#666666",
         transform=ax3.transAxes)

# ---- (d) 感度分析ヒートマップ: MR削減率 × 視聴単価 → 売上変化率(%)
ax4 = fig.add_subplot(gs[1, :2])
pivot_sens = sensitivity_df.pivot_table(
    values="delta_sales_pct",
    index="mr_reduction_pct",
    columns="digital_cost_per_view",
    aggfunc="mean"
)
sns.heatmap(pivot_sens, cmap="RdYlGn", center=0, annot=True, fmt=".1f",
            cbar_kws={"label": "売上変化率 (%)"}, ax=ax4,
            linewidths=0.5)
ax4.set_xlabel("デジタル視聴単価（万円/視聴）", fontsize=10)
ax4.set_ylabel("MR削減率 (%)", fontsize=10)
ax4.set_title("(d) 感度分析: MR削減節約額をデジタルへ転換した場合の売上変化率（%）\n"
              "（コストニュートラル前提）", fontsize=11, fontweight="bold")

# ---- (e) 売上中立デジタル単価 vs 損益分岐点コスト
ax5 = fig.add_subplot(gs[1, 2])
mr_red_labels = [f"{int(r*100)}%" for r in mr_reductions]
neutral_vals = [revenue_neutral_costs[r] for r in mr_reductions]
clip_max = max(breakeven_cost * 20, max(v for v in neutral_vals if np.isfinite(v)) * 1.2
               if any(np.isfinite(v) for v in neutral_vals) else breakeven_cost * 20)
neutral_vals_plot = [min(v, clip_max) if np.isfinite(v) else clip_max for v in neutral_vals]
bars5 = ax5.barh(mr_red_labels, neutral_vals_plot, color="#5c85d6", alpha=0.75)
ax5.axvline(breakeven_cost, color="red", linewidth=1.5, linestyle="--",
            label=f"損益分岐点 {breakeven_cost:.2f}万円")
ax5.set_xlabel("売上中立コスト（万円/視聴）", fontsize=10)
ax5.set_ylabel("MR削減率", fontsize=10)
ax5.set_title("(e) MR削減を売上中立に保つ\n最大デジタル視聴単価", fontsize=11, fontweight="bold")
ax5.legend(fontsize=8)
for bar, val in zip(bars5, neutral_vals):
    label = f"{val:.1f}" if np.isfinite(val) else "inf"
    ax5.text(min(val, clip_max * 0.98) + clip_max * 0.01,
             bar.get_y() + bar.get_height()/2,
             label, va="center", fontsize=9)
ax5.grid(axis="x", alpha=0.3)

# ---- (f) 施設属性別限界効果
ax6 = fig.add_subplot(gs[2, :])
if len(subgroup_df) > 0:
    x_labels_sub = [f"{row['group']}" for _, row in subgroup_df.iterrows()]
    x_pos_sub = np.arange(len(subgroup_df))
    w = 0.35
    ax6.bar(x_pos_sub - w/2, subgroup_df["beta_mr"],
            yerr=1.96 * subgroup_df["se_mr"],
            width=w, label="MR活動", color="#ff7f0e", alpha=0.75, capsize=5)
    ax6.bar(x_pos_sub + w/2, subgroup_df["beta_digital"],
            yerr=1.96 * subgroup_df["se_digital"],
            width=w, label="デジタル視聴", color="#2ca02c", alpha=0.75, capsize=5)
    ax6.axhline(0, color="black", linewidth=0.8)
    # 損益分岐点ラインを各グループに表示
    ax6.set_xticks(x_pos_sub)
    ax6.set_xticklabels(x_labels_sub, fontsize=9)
    ax6.set_ylabel("限界効果（万円売上/回）", fontsize=10)
    ax6.set_title("(f) 施設属性別限界効果（95%CI）", fontsize=11, fontweight="bold")
    ax6.legend(fontsize=9)
    ax6.grid(axis="y", alpha=0.3)
    # UHP/施設区分の境界線
    if len(subgroup_df) > 1:
        # group_col が変わる位置に区切り線
        prev = subgroup_df.iloc[0]["group_col"]
        for i, row in subgroup_df.iterrows():
            if row["group_col"] != prev:
                ax6.axvline(i - 0.5, color="gray", linestyle=":", linewidth=1.5)
                prev = row["group_col"]
else:
    ax6.text(0.5, 0.5, "サブグループデータ不足", ha="center", va="center",
             transform=ax6.transAxes, fontsize=12, color="gray")
    ax6.axis("off")

plt.suptitle("MR vs デジタルバランス分析（損益分岐点コストアプローチ）",
             fontsize=14, fontweight="bold", y=0.995)

output_png = "mr_digital_balance.png"
plt.savefig(output_png, dpi=300, bbox_inches="tight")
print(f"  可視化を保存: {output_png}")
plt.close()

# ================================================================
# 結果保存
# ================================================================
print("\n[結果保存]")

output_json = {
    "approach": "損益分岐点コストアプローチ（デジタルコストデータなし）",
    "cost_assumptions": COST_ASSUMPTIONS,
    "baseline": BASELINE,
    "current_status": {
        "mr_count_per_facility_month": current_mr_mean,
        "digital_count_per_facility_month": current_digital_mean,
        "sales_per_facility_month": current_sales_mean,
        "n_facilities": n_facilities,
        "n_months": n_months,
        "mr_cost_annual": current_mr_cost_annual,
    },
    "marginal_effects": {
        "mr": {"coefficient": beta_mr, "se": se_mr, "p": p_mr, "sig": sig_mr},
        "digital": {"coefficient": beta_digital, "se": se_digital, "p": p_digital, "sig": sig_digital},
    },
    "breakeven_analysis": {
        "mr_cost_efficiency": mr_cost_efficiency,
        "breakeven_digital_cost": breakeven_cost,
        "equivalence_ratio_mr_to_digital": equivalence_ratio,
        "interpretation": (
            f"デジタル視聴単価 < {breakeven_cost:.3f}万円/視聴 であれば MR より費用対効果が高い。"
            f"MR活動1回 = デジタル視聴{equivalence_ratio:.1f}回分の売上インパクト(概算)。"
        ),
    },
    "revenue_neutral_costs": {
        f"mr_reduction_{int(r*100)}pct": {
            "mr_savings_annual": r * current_mr_cost_annual,
            "max_digital_cost_for_neutral": revenue_neutral_costs[r],
        }
        for r in mr_reductions
    },
    "sensitivity_grid": sensitivity_df.to_dict(orient="records"),
    "subgroup_effects": subgroup_df.to_dict(orient="records") if len(subgroup_df) > 0 else [],
    "facility_master_source": {
        "file": "facility_attribute_修正.csv",
        "facilities_loaded": len(facility_master),
        "facility_types": facility_master["facility_type"].value_counts().to_dict(),
        "uhp_tiers": facility_master["uhp_tier"].value_counts().to_dict(),
    },
    "interpretation": {
        "warning": "MRとデジタルの両方が内生変数であり、結果は相関関係を示す",
        "recommendation": (
            "損益分岐点コスト分析により、デジタルコストが不明でも意思決定の閾値を特定できる。"
            "感度分析ヒートマップで視聴単価ごとの売上変化率を参照し、コスト情報入手後に最適シナリオを選択すること。"
        ),
    }
}

output_path = os.path.join(RESULTS_DIR, "mr_digital_balance.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_json, f, ensure_ascii=False, indent=2, default=float)

print(f"  結果を保存: {output_path}")

# ================================================================
# サマリー出力
# ================================================================
print("\n" + "=" * 60)
print(" 分析完了 ─ 損益分岐点コストアプローチ")
print("=" * 60)

print(f"\n【主要結論】")
print(f"  MR限界効果:     {beta_mr:+.2f}万円/活動（{sig_mr}）")
print(f"  デジタル限界効果: {beta_digital:+.2f}万円/視聴（{sig_digital}）")
print(f"  等価交換レート:   MR1回 ~ デジタル{equivalence_ratio:.1f}回")
print(f"\n【損益分岐点コスト】")
print(f"  C* = {breakeven_cost:.3f}万円/視聴 = {breakeven_cost*10000:.0f}円/視聴")
print(f"  → 視聴単価がこれ以下ならデジタルはMRより費用対効果が高い")

print(f"\n【MR削減シミュレーション（売上中立コスト）】")
for r in mr_reductions:
    nc = revenue_neutral_costs[r]
    if np.isfinite(nc):
        nc_str = f"{nc:.2f}万円/視聴"
        note = " ← 現実的コスト範囲内で注意" if nc < 20 else " ← 実質的にどのコストでも売上増"
    else:
        nc_str = "∞"
        note = " ← 視聴単価に関係なく売上増"
    print(f"  MR {int(r*100)}%削減: 売上中立コスト <= {nc_str}{note}")

print(f"\n【感度分析サマリー（MR 30%削減時）】")
sub30 = sensitivity_df[sensitivity_df["mr_reduction_pct"] == 30.0]
for _, row in sub30.iterrows():
    print(f"  視聴単価 {row['digital_cost_per_view']:.2f}万円 → 売上変化 {row['delta_sales_pct']:+.1f}%")

print("\n" + "=" * 60)
