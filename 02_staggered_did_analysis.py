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
from sklearn.linear_model import LogisticRegression

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
CONTENT_TYPES = ["webiner", "e_contents", "Web講演会"]  # チャネル大分類 (拡張可能)
ACTIVITY_CHANNEL_FILTER = "Web講演会"   # 活動データから抽出する活動種別

# ファイル名
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
# 本番データ/ が存在しなければ data/ にフォールバック
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


def compute_cs_attgt_dr(pdata, cov_cols):
    """Doubly Robust CS ATT(g,t)推定 (Sant'Anna & Zhao 2020 style).
    IPW(傾向スコア) + OR(アウトカム回帰) の組み合わせ.
    cov_cols: pdata内の処置前共変量列名リスト (標準化済みを推奨).
    """
    try:
        from sklearn.linear_model import LogisticRegression, Ridge
    except ImportError:
        return compute_cs_attgt(pdata)   # sklearnなければ通常CSにフォールバック

    doc_info = pdata.groupby("unit_id").agg(
        {"treated": "first", "cohort_month": "first"}
    )
    # 共変量行列: unit × cov (time-invariant, 1行/unit)
    unit_cov = (
        pdata.groupby("unit_id")[cov_cols].first()
        if cov_cols else pd.DataFrame(index=doc_info.index)
    )
    pivot = pdata.pivot_table(
        values="amount", index="unit_id", columns="month_index", aggfunc="mean",
    )
    ctrl_ids = doc_info[doc_info["treated"] == 0].index
    if len(ctrl_ids) == 0:
        return pd.DataFrame()

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

        sample_ids = list(cdocs) + list(ctrl_ids)
        D = np.array([1 if uid in set(cdocs) else 0 for uid in sample_ids])

        X_cov = (
            unit_cov.reindex(sample_ids).fillna(0).values.astype(float)
            if cov_cols else np.ones((len(sample_ids), 1))
        )

        # ---- Step 1: 傾向スコア (ロジスティック回帰) ----
        n_t, n_c = int(D.sum()), int((1 - D).sum())
        if n_t >= 2 and n_c >= 2 and X_cov.shape[1] > 0:
            try:
                ps_model = LogisticRegression(
                    C=1.0, max_iter=500, solver="lbfgs", random_state=0
                )
                ps_model.fit(X_cov, D)
                ps = ps_model.predict_proba(X_cov)[:, 1]
            except Exception:
                ps = np.full(len(sample_ids), n_t / len(sample_ids))
        else:
            ps = np.full(len(sample_ids), n_t / len(sample_ids))
        ps = np.clip(ps, 0.025, 0.975)
        ipw = ps / (1.0 - ps)          # 対照群の非正規化IPW重み

        for t in all_times:
            if t not in pivot.columns:
                continue

            # ΔY_t = Y_t - Y_{g-1}
            dY = np.array([
                (pivot.loc[uid, t] - pivot.loc[uid, base])
                if uid in pivot.index else 0.0
                for uid in sample_ids
            ])

            ctrl_mask  = D == 0
            treat_mask = D == 1

            # ---- Step 2: アウトカム回帰 (対照群でΔYをXに回帰) ----
            dY_ctrl = dY[ctrl_mask]
            try:
                or_model = Ridge(alpha=1.0)
                or_model.fit(X_cov[ctrl_mask], dY_ctrl)
                m_hat = or_model.predict(X_cov)
            except Exception:
                m_hat = np.full(
                    len(sample_ids),
                    dY_ctrl.mean() if len(dY_ctrl) > 0 else 0.0
                )

            # ---- Step 3: DR-ATT ----
            # = E_treated[ΔY - m̂(X)] - IPW_weighted_E_control[ΔY - m̂(X)]
            resid = dY - m_hat
            treated_part = resid[treat_mask].mean() if treat_mask.sum() > 0 else 0.0
            w_ctrl = ipw[ctrl_mask]
            ctrl_resid = resid[ctrl_mask]
            control_part = (
                np.sum(w_ctrl * ctrl_resid) / w_ctrl.sum()
                if w_ctrl.sum() > 0 else 0.0
            )
            rows.append({
                "cohort": g, "time": t, "event_time": t - g,
                "att_gt": treated_part - control_part, "n_cohort": n_t,
            })
    return pd.DataFrame(rows)


def run_cs_dr_with_bootstrap(panel, cov_cols, n_boot=200, label=""):
    try:
        from sklearn.linear_model import LogisticRegression, Ridge
    except ImportError:
        return run_cs_with_bootstrap(panel, n_boot=n_boot, label=label)

    # ── 1. Pre-compute once ──────────────────────────────────────────
    doc_info = panel.groupby("unit_id").agg(
        {"treated": "first", "cohort_month": "first"}
    )
    unit_cov = (
        panel.groupby("unit_id")[cov_cols].first()
        if cov_cols else pd.DataFrame(index=doc_info.index)
    )
    pivot = panel.pivot_table(
        values="amount", index="unit_id", columns="month_index", aggfunc="mean"
    )
    pivot_np    = np.nan_to_num(pivot.to_numpy(dtype=float), nan=0.0)
    time_cols   = np.asarray(pivot.columns)
    n_months    = len(time_cols)
    unit_order  = list(pivot.index)
    unit_to_idx = {uid: i for i, uid in enumerate(unit_order)}

    ctrl_ids  = doc_info[doc_info["treated"] == 0].index
    ctrl_rows = np.array([unit_to_idx[u] for u in ctrl_ids if u in unit_to_idx])
    n_ctrl    = len(ctrl_rows)
    if n_ctrl == 0:
        return None, None, None, None

    cohorts = sorted(doc_info["cohort_month"].dropna().unique())
    coh_data = {}
    for g in cohorts:
        g    = int(g)
        base = g - 1
        if base not in time_cols:
            continue
        base_col  = int(np.where(time_cols == base)[0][0])
        cdoc_ids  = list(doc_info[doc_info["cohort_month"] == g].index)
        treat_rows = np.array([unit_to_idx[u] for u in cdoc_ids if u in unit_to_idx])
        if len(treat_rows) == 0:
            continue
        n_t = len(treat_rows)
        sample_uids = [unit_order[r] for r in treat_rows] + [unit_order[r] for r in ctrl_rows]
        X_full = (
            unit_cov.reindex(sample_uids).fillna(0).values.astype(float)
            if cov_cols else np.ones((n_t + n_ctrl, 1))
        )
        coh_data[g] = {
            "base_col": base_col, "treat_rows": treat_rows,
            "n_t": n_t, "X_full": X_full,  # treat 先頭 n_t 行, ctrl 残り
        }

    if not coh_data:
        return None, None, None, None

    # ── DR-ATT helper: multi-output Ridge で全月一括計算 ─────────────
    def _dr_att_vec(pv, X_cov, n_t, n_c, base_col):
        """pv: (n_t+n_c, n_months). Returns att: (n_months,)."""
        t_mask = np.arange(n_t + n_c) < n_t
        c_mask = ~t_mask
        dY = pv - pv[:, base_col:base_col + 1]   # (n_sample, n_months)
        D  = t_mask.astype(int)

        if n_t >= 2 and n_c >= 2:
            try:
                ps = LogisticRegression(
                    C=1.0, max_iter=500, solver="lbfgs", random_state=0
                ).fit(X_cov, D).predict_proba(X_cov)[:, 1]
            except Exception:
                ps = np.full(n_t + n_c, n_t / (n_t + n_c))
        else:
            ps = np.full(n_t + n_c, n_t / (n_t + n_c))
        ps  = np.clip(ps, 0.025, 0.975)
        ipw = ps / (1.0 - ps)
        w_c = ipw[c_mask]   # (n_ctrl,)

        # multi-output Ridge: 1 fit で全 n_months 分のアウトカム回帰
        try:
            m_hat = Ridge(alpha=1.0).fit(X_cov[c_mask], dY[c_mask]).predict(X_cov)
        except Exception:
            m_hat = np.tile(dY[c_mask].mean(axis=0), (n_t + n_c, 1))

        resid = dY - m_hat                                   # (n_sample, n_months)
        tp    = resid[t_mask].mean(axis=0)                   # (n_months,)
        w_sum = w_c.sum()
        cp    = ((w_c[:, None] * resid[c_mask]).sum(0) / w_sum
                 if w_sum > 0 else np.zeros(n_months))
        return tp - cp

    # ── 2. 点推定 ────────────────────────────────────────────────────
    rows = []
    for g, cd in coh_data.items():
        pv  = pivot_np[np.concatenate([cd["treat_rows"], ctrl_rows])]
        att = _dr_att_vec(pv, cd["X_full"], cd["n_t"], n_ctrl, cd["base_col"])
        for ti, t in enumerate(time_cols):
            rows.append({"cohort": g, "time": int(t), "event_time": int(t) - g,
                         "att_gt": att[ti], "n_cohort": cd["n_t"]})

    att_gt = pd.DataFrame(rows)
    if len(att_gt) == 0:
        return None, None, None, None

    dynamic = aggregate_dynamic(att_gt)
    overall  = aggregate_overall(att_gt)

    # ── 3. Bootstrap (DataFrame再構築・pivot_table再計算を排除) ───────
    tag = f"[{label}] " if label else ""
    print(f"  {tag}Bootstrap DR (n={n_boot}) ...", end="", flush=True)

    boot_overall = []
    boot_dynamic = {}

    for b in range(n_boot):
        if (b + 1) % 50 == 0:
            print(f" {b + 1}", end="", flush=True)

        boot_att_rows = []
        for g, cd in coh_data.items():
            n_t   = cd["n_t"]
            idx_t = np.random.randint(0, n_t,    n_t)
            idx_c = np.random.randint(0, n_ctrl, n_ctrl)
            pv    = pivot_np[np.concatenate([cd["treat_rows"][idx_t], ctrl_rows[idx_c]])]
            X_b   = np.vstack([cd["X_full"][:n_t][idx_t], cd["X_full"][n_t:][idx_c]])
            att   = _dr_att_vec(pv, X_b, n_t, n_ctrl, cd["base_col"])
            for ti, t in enumerate(time_cols):
                boot_att_rows.append({"cohort": g, "time": int(t), "event_time": int(t) - g,
                                      "att_gt": att[ti], "n_cohort": n_t})

        bgt = pd.DataFrame(boot_att_rows)
        if len(bgt) == 0:
            continue
        boot_overall.append(aggregate_overall(bgt))
        bdyn = aggregate_dynamic(bgt)
        for _, row in bdyn.iterrows():
            et = int(row["event_time"])
            boot_dynamic.setdefault(et, []).append(row["att"])

    print(" done")

    se_overall = np.std(boot_overall) if boot_overall else 0.0
    se_dyn_map = {et: np.std(v) for et, v in boot_dynamic.items()}
    dynamic["se"]    = dynamic["event_time"].map(se_dyn_map).fillna(0)
    dynamic["ci_lo"] = dynamic["att"] - 1.96 * dynamic["se"]
    dynamic["ci_hi"] = dynamic["att"] + 1.96 * dynamic["se"]
    return att_gt, dynamic, overall, se_overall


def run_cs_with_bootstrap(panel, n_boot=200, label=""):
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
    print(f"  {tag}Bootstrap (n={n_boot}, vectorised) ...", end="", flush=True)

    boot_per_gt = {}
    for g, cd in coh_data.items():
        b    = cd["base"]
        n_t  = cd["n"]
        dT   = cd["T"]  - cd["T"][:, b:b+1]
        dC   = ctrl_np  - ctrl_np[:, b:b+1]
        idx_T = np.random.randint(0, n_t,    (n_boot, n_t))
        idx_C = np.random.randint(0, n_ctrl, (n_boot, n_ctrl))
        bT    = np.nanmean(dT[idx_T], axis=1)
        bC    = np.nanmean(dC[idx_C], axis=1)
        boot_att = bT - bC
        for ti in range(len(time_cols)):
            boot_per_gt[(g, ti)] = boot_att[:, ti]

    post_keys = [(g, ti) for g in coh_data
                 for ti, t in enumerate(time_cols) if int(t) - g >= 0]
    if post_keys:
        w   = np.array([coh_data[g]["n"] for g, ti in post_keys], float)
        w  /= w.sum()
        mat = np.stack([boot_per_gt[k] for k in post_keys], axis=1)
        se_overall = float(np.std((mat * w).sum(1)))
    else:
        se_overall = 0.0

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
    return att_gt, dynamic, overall, se_overall


# ================================================================
# Part 1: データ読み込み (本番形式)
# ================================================================
print("=" * 70)
print(" Staggered DID分析: デジタルコンテンツ視聴の効果検証")
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
# Part 2: 除外フロー
# ================================================================
print("\n" + "=" * 70)
print(" Part 2: 除外フロー")
print("=" * 70)

# 施設医師リスト: 全医師の施設対応マスター (母集団)
fac_doc_list = pd.read_csv(os.path.join(DATA_DIR, FILE_FAC_DOCTOR_LIST))

# [Step 1] facility_attribute_修正.csv: fac単位で施設内医師数==1のfacを抽出
fac_df = pd.read_csv(os.path.join(DATA_DIR, FILE_FACILITY_MASTER))
single_staff_fac = set(fac_df[fac_df["施設内医師数"] == 1]["fac"])
multi_staff_fac  = set(fac_df[fac_df["施設内医師数"] > 1]["fac"])

print(f"\n  [Step 1] facility_attribute_修正.csv: 施設内医師数==1 の施設 (fac単位)")
print(f"      1医師fac             : {len(single_staff_fac)} 件")
print(f"      複数医師fac           : {len(multi_staff_fac)} 件 → 除外")

# [Step 2] doctor_attribute.csv: 所属施設数==1 の医師 (honin粒度)
doc_attr_df = pd.read_csv(os.path.join(DATA_DIR, FILE_DOCTOR_ATTR))
if FILTER_SINGLE_FAC_DOCTOR:
    if DOCTOR_HONIN_FAC_COUNT_COL in doc_attr_df.columns:
        single_honin_docs = set(doc_attr_df[doc_attr_df[DOCTOR_HONIN_FAC_COUNT_COL] == 1]["doc"])
    else:
        # フォールバック: 施設医師リスト.csvからfac_honinのユニーク数で計算
        _fac_per_doc = fac_doc_list.groupby("doc")["fac_honin"].nunique()
        single_honin_docs = set(_fac_per_doc[_fac_per_doc == 1].index)
        print(f"    ({DOCTOR_HONIN_FAC_COUNT_COL}列なし → rw_list.csvから計算)")
else:
    single_honin_docs = set(doc_attr_df["doc"])
print(f"\n  [Step 2] doctor_attribute.csv: 所属施設数==1 の医師")
print(f"      1施設所属医師 : {len(single_honin_docs)} 名 (doctor_attribute基準)")

# [Step 3] RW医師フィルタ (INCLUDE_ONLY_RW=True の場合のみ適用)
rw_doc_ids = set(rw_list["doc"])

# 3ステップを順序付きで適用 + 中間カウント + 1:1確認
_doc_to_fac   = dict(zip(fac_doc_list["doc"], fac_doc_list["fac"]))
_doc_to_honin = dict(zip(fac_doc_list["doc"], fac_doc_list["fac_honin"]))
all_docs = set(fac_doc_list["doc"])  # 全医師は施設医師リスト.csv
# Step 1 適用: 施設内医師数==1 の施設に所属する医師
after_step1 = {d for d in all_docs if _doc_to_fac.get(d) in single_staff_fac}
# Step 2 適用: 所属施設数==1 の医師 (Step 1通過者から)
after_step2 = after_step1 & single_honin_docs
# Step 3 適用: INCLUDE_ONLY_RW=True の場合のみ RW医師に絞る
if INCLUDE_ONLY_RW:
    after_step3 = after_step2 & rw_doc_ids
else:
    after_step3 = after_step2  # Step 3スキップ (全医師対象)
non_rw_excluded = len(after_step2) - len(after_step3)
multi_fac_docs = all_docs - single_honin_docs  # 可視化用 (全体)
# 同一fac_honinに複数の候補医師がいる場合は除外
_honin_cnt: dict = {}
for d in after_step3:
    h = _doc_to_honin[d]
    _honin_cnt[h] = _honin_cnt.get(h, 0) + 1
candidate_docs = {d for d in after_step3 if _honin_cnt[_doc_to_honin[d]] == 1}

print(f"\n  [順序付き適用結果]")
print(f"      全医師 (doctor_attr): {len(all_docs)} 名")
print(f"      Step 1 通過        : {len(after_step1)} 名 (施設内医師数==1の施設所属)")
print(f"      Step 2 通過        : {len(after_step2)} 名 (所属施設数==1)")
print(f"      Step 3 通過        : {len(after_step3)} 名 ({'RW医師のみ' if INCLUDE_ONLY_RW else '全医師 (Step 3スキップ)'})")
print(f"      1:1ペア確認後      : {len(candidate_docs)} 名")

_pair_src = rw_list if INCLUDE_ONLY_RW else fac_doc_list
clean_pairs = _pair_src[_pair_src["doc"].isin(candidate_docs)][["doc", "fac_honin"]].drop_duplicates()
clean_pairs = clean_pairs[clean_pairs["fac_honin"].notna() & (clean_pairs["fac_honin"].astype(str).str.strip().isin(["", "nan"]) == False)].copy()
clean_pairs = clean_pairs.rename(columns={"doc": "doctor_id", "fac_honin": "facility_id"})

fac_to_doc = dict(zip(clean_pairs["facility_id"], clean_pairs["doctor_id"]))
doc_to_fac = dict(zip(clean_pairs["doctor_id"], clean_pairs["facility_id"]))
clean_fac_ids = set(clean_pairs["facility_id"])
clean_doc_ids = set(clean_pairs["doctor_id"])
# doctor_master: 解析対象に応じてソース変更
_dm_src = rw_list if INCLUDE_ONLY_RW else fac_doc_list
doctor_master = _dm_src.rename(columns={"doc": "doctor_id", "fac_honin": "facility_id"})

print(f"\n  [クリーン1:1ペア] {len(clean_doc_ids)} 施設 / {len(clean_doc_ids)} 医師")

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

# Two-way within transformation (FWL): avoids large dense dummy matrices
# Equivalent to including unit + time fixed effects as explicit dummies
def _twoway_demean(df, cols, unit_col="unit_id", time_col="month_index"):
    out = {}
    for col in cols:
        s = df[col].astype(float)
        grand_m = s.mean()
        unit_m = s.groupby(df[unit_col]).transform("mean")
        time_m = s.groupby(df[time_col]).transform("mean")
        out[col] = (s - unit_m - time_m + grand_m).values
    return pd.DataFrame(out, index=df.index)

_dm = _twoway_demean(panel_r, ["amount", "did"])
X_twfe = pd.DataFrame({"did": _dm["did"]})
model_twfe = sm.OLS(_dm["amount"], X_twfe).fit(
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
#    CONTENT_TYPES (webiner, e_contents, Web講演会) は処置変数 → 除外
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
_dm_r = _twoway_demean(panel_robust, ["amount", "did", "mr_activity_count"])
X_robust = pd.DataFrame({"did": _dm_r["did"], "mr_activity": _dm_r["mr_activity_count"]})
y_robust = _dm_r["amount"].values

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
# Part 5c: 共変量構築 (Doubly Robust推定用)
# ================================================================
print("\n" + "=" * 70)
print(" Part 5c: 共変量構築 (DR推定用)")
print("=" * 70)

# analysis_fac_idsをベースに各共変量を施設単位で整理
_cov = pd.DataFrame(index=sorted(analysis_fac_ids))
_cov.index.name = "facility_id"

# 1. 処置前平均売上 (wash-out期間 month 0,1 = 全ユニットで処置前)
_pre_sales = (
    panel_base[panel_base["month_index"] < WASHOUT_MONTHS]
    .groupby("facility_id")["amount"].mean()
)
_cov["cov_baseline_sales"] = _pre_sales

# 2. 施設属性 (facility_attribute_修正.csv, fac_honin単位で取得)
_fac_attr = fac_df.drop_duplicates("fac_honin").set_index("fac_honin")
_cov["cov_is_hospital"] = _fac_attr["施設区分名"].map(
    {"病院": 1.0, "診療所": 0.0}
)
_cov["cov_beds"] = _fac_attr["許可病床数_合計"]
_uhp_dum = pd.get_dummies(
    _fac_attr["UHP区分名"], prefix="cov_uhp", drop_first=True, dtype=float
)
# 列名を安全な識別子に (ハイフン除去)
_uhp_dum.columns = [c.replace("-", "_") for c in _uhp_dum.columns]
for _c in _uhp_dum.columns:
    _cov[_c] = _uhp_dum[_c]

# 3. 処置前MR活動量 (wash-out期間内の平均)
_mr_pre = (
    mr_counts[mr_counts["month_index"] < WASHOUT_MONTHS]
    .groupby("facility_id")["mr_activity_count"].mean()
)
_cov["cov_mr_pre"] = _mr_pre

# 4. 医師歴 (doctor_attribute → fac_to_docで施設にマッピング)
if "医師歴" in doc_attr_df.columns:
    _doc_exp_map = doc_attr_df.set_index("doc")["医師歴"]
    _cov["cov_exp_years"] = pd.Series(fac_to_doc).map(_doc_exp_map)

_cov = _cov.fillna(0)

# 5. 標準化 (z-score)
_cov_means = _cov.mean()
_cov_stds  = _cov.std().replace(0, 1)
cov_std    = (_cov - _cov_means) / _cov_stds
COV_COLS   = list(cov_std.columns)

# 6. panel_r に共変量列を追加 (施設固定値)
for _col in COV_COLS:
    _col_map = cov_std[_col].to_dict()
    panel_r[_col] = panel_r["facility_id"].map(_col_map).fillna(0)

print(f"  共変量 ({len(COV_COLS)}個): {COV_COLS}")
print(f"  対象施設数: {len(cov_std)}")

# ================================================================
# Part 5d: TWFE-DR推定 (Doubly Robust: IPW重み付きWLS + 共変量制御)
# ================================================================
print("\n" + "=" * 70)
print(" Part 5d: TWFE-DR推定 (IPW重み付きWLS + 共変量制御)")
print("=" * 70)

# 1. 施設クロスセクション: 処置インジケータと共変量
_cs_df = panel_r.drop_duplicates("facility_id").set_index("facility_id")
_cs_D  = _cs_df["treated"].values.astype(float)
_cs_X  = cov_std.reindex(_cs_df.index).fillna(0).values
_cs_fac_ids = _cs_df.index.tolist()

# 2. 傾向スコア推定 P(treated=1 | COV_COLS)
_lr_ps = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
try:
    _lr_ps.fit(_cs_X, _cs_D)
    _ps_hat = np.clip(_lr_ps.predict_proba(_cs_X)[:, 1], 0.01, 0.99)
except Exception as _e:
    print(f"  傾向スコア推定失敗: {_e} → 等重みを使用")
    _ps_hat = np.full(len(_cs_D), _cs_D.mean())

# 3. ATT型IPW重み: 処置群=1, 対照群=p̂/(1-p̂)
_ipw_raw = np.where(_cs_D == 1, 1.0, _ps_hat / (1.0 - _ps_hat))
_ipw_map = dict(zip(_cs_fac_ids, _ipw_raw))
panel_r["ipw_weight"] = panel_r["facility_id"].map(_ipw_map).fillna(1.0)

# 4. 加重WLS TWFE + 共変量制御 (Doubly Robust)
# within変換後にIPW重みを適用 (IPW重みは施設固定のためunit FEは一致)
_dm_dr = _twoway_demean(panel_r, ["amount", "did"] + COV_COLS)
X_twfe_dr = pd.DataFrame({"did": _dm_dr["did"]})
for _c in COV_COLS:
    X_twfe_dr[_c] = _dm_dr[_c].values
model_twfe_dr = sm.WLS(_dm_dr["amount"], X_twfe_dr, weights=panel_r["ipw_weight"].values).fit(
    cov_type="cluster", cov_kwds={"groups": panel_r["unit_id"].values}
)

beta_dr_twfe  = model_twfe_dr.params["did"]
se_twfe_dr    = model_twfe_dr.bse["did"]
pval_twfe_dr  = model_twfe_dr.pvalues["did"]
ci_twfe_dr    = model_twfe_dr.conf_int().loc["did"]
sig_twfe_dr   = (
    "***" if pval_twfe_dr < 0.001 else "**" if pval_twfe_dr < 0.01
    else "*" if pval_twfe_dr < 0.05 else "n.s."
)

print(f"\n  DID推定量(DR) : {beta_dr_twfe:.2f}")
print(f"  SE           : {se_twfe_dr:.2f}")
print(f"  p値          : {pval_twfe_dr:.6f} {sig_twfe_dr}")
print(f"  95%CI        : [{ci_twfe_dr[0]:.2f}, {ci_twfe_dr[1]:.2f}]")
print(f"\n  [比較] TWFE(無調整)={beta:.2f}  →  TWFE-DR={beta_dr_twfe:.2f}"
      f"  (差: {beta_dr_twfe - beta:+.2f})")

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
# Part 6b: CS-DR推定 (Doubly Robust, 共変量調整)
# ================================================================
print("\n" + "=" * 70)
print(" Part 6b: CS-DR推定 (Doubly Robust, 共変量調整)")
print("=" * 70)
print(f"  共変量: {COV_COLS}")

att_gt_dr, cs_dyn_dr, cs_overall_dr, se_dr = run_cs_dr_with_bootstrap(
    panel_r, COV_COLS, n_boot=200, label="DR"
)

if cs_overall_dr is not None:
    z_dr   = cs_overall_dr / se_dr if se_dr > 0 else float("inf")
    p_dr   = 2 * (1 - stats.norm.cdf(abs(z_dr)))
    sig_dr = (
        "***" if p_dr < 0.001 else "**" if p_dr < 0.01
        else "*" if p_dr < 0.05 else "n.s."
    )
    print(f"\n  ATT-DR(全体) : {cs_overall_dr:.2f}")
    print(f"  SE           : {se_dr:.2f}")
    print(f"  p値          : {p_dr:.6f} {sig_dr}")
    print(f"\n  [比較] CS(無調整)={cs_overall:.2f}  →  CS-DR={cs_overall_dr:.2f}"
          f"  (差: {cs_overall_dr - cs_overall:+.2f})")
    print(f"\n  [動的効果 (DR)]")
    print(f"  {'ET':>5} {'ATT':>8} {'SE':>8} {'95%CI':>20}")
    print(f"  {'-' * 45}")
    for _, r in cs_dyn_dr.iterrows():
        et = int(r["event_time"])
        marker = " <-" if et == 0 else ""
        print(
            f"  {'e=' + str(et):>5} {r['att']:>8.2f} {r['se']:>8.2f}"
            f" [{r['ci_lo']:>7.2f}, {r['ci_hi']:>7.2f}]{marker}"
        )
else:
    print("  -> DR推定不可 (フォールバック: CS無調整を使用)")
    cs_overall_dr = cs_overall
    se_dr         = se_overall
    sig_dr        = sig_cs
    p_dr          = p_cs
    cs_dyn_dr     = cs_dyn_all.copy()

# ================================================================
# Part 7: CS推定 (チャネル別)
# ================================================================
print("\n" + "=" * 70)
print(" Part 7: CS チャネル別推定")
print("=" * 70)

control_fac_ids = {doc_to_fac[d] for d in control_doc_ids}
channel_results = {}
channel_dr_results = {}

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

    # DR推定: COV_COLSをch_panelに付与して実施
    for _col in COV_COLS:
        _col_map = cov_std[_col].to_dict()
        ch_panel[_col] = ch_panel["facility_id"].map(_col_map).fillna(0)
    dr_res = run_cs_dr_with_bootstrap(ch_panel, COV_COLS, n_boot=100, label=f"{ch}-DR")
    att_gt_ch_dr, dyn_ch_dr, overall_ch_dr, se_ch_dr = dr_res
    if overall_ch_dr is not None:
        z_ch_dr  = overall_ch_dr / se_ch_dr if se_ch_dr > 0 else float("inf")
        p_ch_dr  = 2 * (1 - stats.norm.cdf(abs(z_ch_dr)))
        sig_ch_dr = (
            "***" if p_ch_dr < 0.001 else "**" if p_ch_dr < 0.01
            else "*" if p_ch_dr < 0.05 else "n.s."
        )
        print(f"  ATT-DR={overall_ch_dr:.2f}, SE={se_ch_dr:.2f}, p={p_ch_dr:.4f} {sig_ch_dr}")
        channel_dr_results[ch] = {
            "dynamic": dyn_ch_dr, "overall": overall_ch_dr,
            "se": se_ch_dr, "p": p_ch_dr, "sig": sig_ch_dr, "n_treated": n_ch_t,
        }
    else:
        print(f"  -> DR推定不可 ({ch})")

# ================================================================
# Part 8: 推定結果比較
# ================================================================
print("\n" + "=" * 70)
print(" Part 8: 推定結果の比較")
print("=" * 70)

print(f"\n  {'手法':<25} {'ATT':>8} {'SE':>8} {'p値':>10} {'有意性':>6}")
print(f"  {'-' * 62}")
print(f"  {'TWFE (全体)':<25} {beta:>8.2f} {se_twfe:>8.2f} {pval_twfe:>10.6f} {sig_twfe:>6}")
print(f"  {'TWFE-DR (全体, 共変量調整)':<25} {beta_dr_twfe:>8.2f} {se_twfe_dr:>8.2f} {pval_twfe_dr:>10.6f} {sig_twfe_dr:>6}")
print(f"  {'CS (全体)':<25} {cs_overall:>8.2f} {se_overall:>8.2f} {p_cs:>10.6f} {sig_cs:>6}")
print(f"  {'CS-DR (全体, 共変量調整)':<25} {cs_overall_dr:>8.2f} {se_dr:>8.2f} {p_dr:>10.6f} {sig_dr:>6}")
for ch in CONTENT_TYPES:
    if ch in channel_results:
        r = channel_results[ch]
        lbl = f"CS ({ch})"
        print(f"  {lbl:<25} {r['overall']:>8.2f} {r['se']:>8.2f} {r['p']:>10.6f} {r['sig']:>6}")
for ch in CONTENT_TYPES:
    if ch in channel_dr_results:
        r = channel_dr_results[ch]
        lbl = f"CS-DR ({ch})"
        print(f"  {lbl:<25} {r['overall']:>8.2f} {r['se']:>8.2f} {r['p']:>10.6f} {r['sig']:>6}")

print(f"\n  DGPの真の効果:")
print(f"    webiner: 18, e_contents: 10, Web講演会: 22")
print(f"    + 月次成長 1.0/月 (視聴継続中)")
print(f"    + 停止後減衰 -1.5/月 (猶予2ヶ月)")
print(f"    x 属性modifier (地域/施設タイプ/経験年数/診療科)")

# ================================================================
# Part 9: 可視化
# ================================================================
print("\n" + "=" * 70)
print(" Part 9: 可視化")
print("=" * 70)

fig, axes = plt.subplots(3, 2, figsize=(14, 18))
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
# CS (無調整)
ax.fill_between(cs_dyn_all["event_time"], cs_dyn_all["ci_lo"], cs_dyn_all["ci_hi"],
                alpha=0.15, color="steelblue")
ax.plot(cs_dyn_all["event_time"], cs_dyn_all["att"], "o-",
        color="steelblue", ms=4, label="CS (無調整)")
# CS-DR (共変量調整)
ax.fill_between(cs_dyn_dr["event_time"], cs_dyn_dr["ci_lo"], cs_dyn_dr["ci_hi"],
                alpha=0.15, color="darkorange")
ax.plot(cs_dyn_dr["event_time"], cs_dyn_dr["att"], "s--",
        color="darkorange", ms=4, label="CS-DR (共変量調整)")
ax.set_xlabel("イベント時間 (月)")
ax.set_ylabel("ATT")
ax.set_title("(b) CS動的効果 (全体): 無調整 vs DR")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ch_colors  = {"webiner": "#1f77b4", "e_contents": "#ff7f0e", "Web講演会": "#2ca02c"}
ch_markers = {"webiner": "o", "e_contents": "s", "Web講演会": "^"}

# (c) CS動的効果 チャネル別 (無調整)
ax = axes[1, 0]
ax.axhline(0, color="black", lw=0.8)
ax.axvline(-0.5, color="red", ls="--", lw=0.8, alpha=0.7)
for ch in CONTENT_TYPES:
    if ch not in channel_results:
        continue
    dyn = channel_results[ch]["dynamic"]
    c, m = ch_colors[ch], ch_markers[ch]
    ax.fill_between(dyn["event_time"], dyn["ci_lo"], dyn["ci_hi"], alpha=0.1, color=c)
    ax.plot(dyn["event_time"], dyn["att"], f"{m}-", color=c, ms=4, label=ch)
ax.set_xlabel("イベント時間 (月)")
ax.set_ylabel("ATT")
ax.set_title("(c) CS動的効果 (チャネル別, 無調整)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (d) CS-DR動的効果 チャネル別
ax = axes[1, 1]
ax.axhline(0, color="black", lw=0.8)
ax.axvline(-0.5, color="red", ls="--", lw=0.8, alpha=0.7)
for ch in CONTENT_TYPES:
    if ch not in channel_dr_results:
        continue
    dyn = channel_dr_results[ch]["dynamic"]
    c, m = ch_colors[ch], ch_markers[ch]
    ax.fill_between(dyn["event_time"], dyn["ci_lo"], dyn["ci_hi"], alpha=0.1, color=c)
    ax.plot(dyn["event_time"], dyn["att"], f"{m}--", color=c, ms=4, label=ch)
ax.set_xlabel("イベント時間 (月)")
ax.set_ylabel("ATT")
ax.set_title("(d) CS-DR動的効果 (チャネル別, 共変量調整)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (e) チャネル別ATT比較: CS vs CS-DR グループ棒グラフ
ax = axes[2, 0]
ch_names = [ch for ch in CONTENT_TYPES if ch in channel_results]
n_ch = len(ch_names)
x_pos = np.arange(n_ch)
bar_w = 0.35
ch_atts_cs = [channel_results[ch]["overall"] for ch in ch_names]
ch_ses_cs  = [channel_results[ch]["se"] for ch in ch_names]
ch_atts_dr = [channel_dr_results[ch]["overall"] if ch in channel_dr_results else 0 for ch in ch_names]
ch_ses_dr  = [channel_dr_results[ch]["se"] if ch in channel_dr_results else 0 for ch in ch_names]
bars_cs = ax.bar(x_pos - bar_w / 2, ch_atts_cs, bar_w,
                 color=[ch_colors[ch] for ch in ch_names], alpha=0.8,
                 edgecolor="gray", label="CS (無調整)")
bars_dr = ax.bar(x_pos + bar_w / 2, ch_atts_dr, bar_w,
                 color=[ch_colors[ch] for ch in ch_names], alpha=0.4,
                 edgecolor="gray", hatch="//", label="CS-DR (調整後)")
ax.errorbar(x_pos - bar_w / 2, ch_atts_cs,
            yerr=[1.96 * s for s in ch_ses_cs], fmt="none", color="black", capsize=4)
ax.errorbar(x_pos + bar_w / 2, ch_atts_dr,
            yerr=[1.96 * s for s in ch_ses_dr], fmt="none", color="black", capsize=4)
ax.set_xticks(list(x_pos))
ax.set_xticklabels(ch_names, fontsize=9)
ax.set_ylabel("ATT")
ax.set_title("(e) チャネル別ATT比較: CS vs CS-DR")
ax.axhline(cs_overall, color="steelblue", ls="--", lw=0.8,
           label=f"CS全体={cs_overall:.1f}")
ax.axhline(cs_overall_dr, color="darkorange", ls="--", lw=0.8,
           label=f"CS-DR全体={cs_overall_dr:.1f}")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")
for i, ch in enumerate(ch_names):
    for xoff, atts, ses, results in [
        (-bar_w / 2, ch_atts_cs, ch_ses_cs, channel_results),
        (+bar_w / 2, ch_atts_dr, ch_ses_dr, channel_dr_results),
    ]:
        if ch in results:
            sig = results[ch]["sig"]
            yoff = atts[i] + 1.96 * ses[i] + 0.3
            ax.text(i + xoff, yoff, sig, ha="center", fontsize=8)

# (f) TWFE vs TWFE-DR 全体比較
ax = axes[2, 1]
methods = ["TWFE\n(無調整)", "TWFE-DR\n(共変量調整)", "CS\n(無調整)", "CS-DR\n(共変量調整)"]
atts    = [beta, beta_dr_twfe, cs_overall, cs_overall_dr]
ses     = [se_twfe, se_twfe_dr, se_overall, se_dr]
sigs    = [sig_twfe, sig_twfe_dr, sig_cs, sig_dr]
colors  = ["#4472C4", "#4472C4", "#ED7D31", "#ED7D31"]
alphas  = [0.9, 0.45, 0.9, 0.45]
hatches = ["", "//", "", "//"]
x_f = np.arange(len(methods))
for i, (att, se, sig, col, alp, hat) in enumerate(
        zip(atts, ses, sigs, colors, alphas, hatches)):
    ax.bar(i, att, color=col, alpha=alp, edgecolor="gray", hatch=hat)
    ax.errorbar(i, att, yerr=1.96 * se, fmt="none", color="black", capsize=5)
    ax.text(i, att + 1.96 * se + 0.3, sig, ha="center", fontsize=9)
ax.set_xticks(list(x_f))
ax.set_xticklabels(methods, fontsize=8)
ax.set_ylabel("ATT")
ax.set_title("(f) 全体ATT比較: TWFE vs TWFE-DR vs CS vs CS-DR")
ax.axhline(0, color="black", lw=0.5)
ax.grid(True, alpha=0.3, axis="y")

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
  TWFE-DR (共変量調整): {beta_dr_twfe:.2f} (SE={se_twfe_dr:.2f}, p={pval_twfe_dr:.4f}) {sig_twfe_dr}
  CS (無調整)        : {cs_overall:.2f} (SE={se_overall:.2f}, p={p_cs:.4f}) {sig_cs}
  CS-DR (共変量調整) : {cs_overall_dr:.2f} (SE={se_dr:.2f}, p={p_dr:.4f}) {sig_dr}
  DR共変量           : {COV_COLS}

  === チャネル別効果 (CS推定) ===""")
for ch in CONTENT_TYPES:
    if ch in channel_results:
        r = channel_results[ch]
        print(f"  {ch:<12}: ATT={r['overall']:.2f} (SE={r['se']:.2f}, p={r['p']:.4f}) {r['sig']}  N={r['n_treated']}")
print()

# ================================================================
# JSON結果保存
# ================================================================
import json  # noqa: E402


results_dir = os.path.join(SCRIPT_DIR, "results")
os.makedirs(results_dir, exist_ok=True)

# 除外フロー情報
exclusion_flow = {
    "total_delivery_rows": n_sales_all,
    "ent_delivery_rows": len(daily),
    "total_rw_list": n_rw_all,
    # Step 1: 施設内医師数==1 フィルタ
    "single_staff_facilities": len(single_staff_fac),
    "multi_staff_facilities": len(multi_staff_fac),
    "after_step1_doctors": len(after_step1),
    "excluded_step1_doctors": len(all_docs) - len(after_step1),
    # Step 2: 所属施設数==1 フィルタ
    "after_step2_doctors": len(after_step2),
    "multi_fac_doctors": len(after_step1) - len(after_step2),
    # Step 3: RW医師フィルタ + 1:1確認
    "ent_rw_doctors": len(after_step3),
    "non_rw_excluded": non_rw_excluded,
    "total_viewing_rows": n_digital_all + n_activity_all,
    "viewing_after_filter": len(viewing),
    "total_doctors": len(candidate_docs),
    "total_facilities": len({_doc_to_honin[d] for d in candidate_docs}),
    "clean_pairs": len(fac_to_doc),
    "washout_excluded": len(washout_viewers),
    "late_excluded": len(late_adopters),
    "final_treated": len(treated_doc_ids),
    "final_control": len(control_doc_ids),
    "final_total": len(analysis_fac_ids),
    "filter_single_fac_doctor": FILTER_SINGLE_FAC_DOCTOR,
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

# TWFE-DR結果
twfe_dr_result = {
    "att": float(beta_dr_twfe),
    "se": float(se_twfe_dr),
    "p": float(pval_twfe_dr),
    "ci_lo": float(ci_twfe_dr[0]),
    "ci_hi": float(ci_twfe_dr[1]),
    "sig": sig_twfe_dr,
    "covariates": COV_COLS,
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

# CS-DR全体結果
cs_dr_result = {
    "att": float(cs_overall_dr),
    "se": float(se_dr),
    "p": float(p_dr),
    "ci_lo": float(cs_overall_dr - 1.96 * se_dr),
    "ci_hi": float(cs_overall_dr + 1.96 * se_dr),
    "sig": sig_dr,
    "covariates": COV_COLS,
    "dynamic": cs_dyn_dr[["event_time", "att", "se", "ci_lo", "ci_hi"]].to_dict("records"),
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

# チャネル別DR結果
ch_dr_results_json = {}
for ch in CONTENT_TYPES:
    if ch in channel_dr_results:
        r = channel_dr_results[ch]
        ch_dr_results_json[ch] = {
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
    "twfe_dr": twfe_dr_result,
    "twfe_robust": twfe_robust_result,
    "cs_overall": cs_result,
    "cs_overall_dr": cs_dr_result,
    "cs_channel": ch_results_json,
    "cs_channel_dr": ch_dr_results_json,
    "cohort_distribution": cohort_json,
    "descriptive_stats": desc_json,
    "n_treated": n_treated,
    "n_control": n_control,
    "n_total": n_total,
    "include_only_rw": INCLUDE_ONLY_RW,
}

json_path = os.path.join(results_dir, "did_results.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(did_results_json, f, ensure_ascii=False, indent=2)
print(f"  結果をJSON保存: {json_path}")
