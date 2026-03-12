"""
===================================================================
Staggered DID分析 ver2: デジタルコンテンツ視聴の効果検証（複数医師施設対応）
===================================================================
ver1 との違い:
  - 複数医師が所属する施設も解析対象に含める
  - 分析粒度: 施設 (facility_honin) レベル
  - 処置: 施設内いずれかの医師が初回視聴した月 (バイナリ)
  - 複数施設所属医師は平均納入額最大の施設を主施設として割り当て
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
MR_ACTIVITY_TYPES = ["面談", "面談_アポ", "説明会"]  # MR活動種別 (非デジタル対面活動)

# ===================================================================
# 共変量設定 (ver2: 施設レベル属性 + 施設医師数)
# -------------------------------------------------------------------
# カテゴリ共変量 (one-hot 化)
COV_CAT_COLS = [
    "UHP区分名称",        # 施設: UHPセグメント (U/H/P/雑)
    "baseline_cat",    # 施設: ベースライン売上カテゴリ
]
# 連続共変量 (z-score 標準化)
COV_CONT_COLS = ["n_docs"]  # 施設あたり医師数

# 医師クインタイル → 施設平均スコア共変量
# Z=0, L=1, M=2, H=3, VH=4 に変換して施設内医師の平均を連続共変量として使用
# カラムが存在しない場合は自動スキップ
QUINTILE_MAP = {"Z": 0, "L": 1, "M": 2, "H": 3, "VH": 4}
DOCTOR_QUINTILE_COLS = [
    "MR_VISIT_F2F_ER_QUINTILE_FINAL",   # MR面談関与度
    "OWNED_MEDIA_ER_QUINTILE_FINAL",    # オウンドメディア関与度
]
# ===================================================================

# ファイル名
FILE_RW_LIST = "rw_list.csv"
FILE_SALES = "sales.csv"
FILE_DIGITAL = "デジタル視聴データ.csv"
FILE_ACTIVITY = "活動データ.csv"
FILE_FACILITY_MASTER = "facility_attribute_修正.csv"
FILE_DOCTOR_ATTR = "doctor_attribute.csv"
FILE_FAC_DOCTOR_LIST = "施設医師リスト.csv"

# 解析集団フィルタパラメータ
INCLUDE_ONLY_RW     = False# True: RW医師のみ / False: 制限なし or 非RWのみ
INCLUDE_ONLY_NON_RW = False  # True: 非RW医師のみ (INCLUDE_ONLY_RW=Falseのときのみ有効)
EXCLUDE_ZERO_SALES_FACILITIES = False  # True: 全期間納入が0の施設を解析対象から除外
FILTER_SINGLE_FAC_DOCTOR = False  # True: 1施設1医師の施設のみを対象（複数医師施設を除外）
UHP_RANK = {"U": 0, "H": 1, "P": 2, "雑": 3}  # 数値小さいほど上位 (U>H>P>雑)

# 出力ファイル名サフィックス (フラグ設定に応じて自動生成)
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
_single_sfx = "_single" if FILTER_SINGLE_FAC_DOCTOR else ""
_suffix = _pop_sfx + _zero_sfx + _single_sfx

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
        # 3Dテンソル回避: 列ごとに計算 → ピークメモリ (n_boot, n_units) のみ
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
print(" Staggered DID分析 ver2: デジタルコンテンツ視聴の効果検証（複数医師施設対応）")
print("=" * 70)

# 1. RW医師リスト (除外フローで使用; 絞り込みはPart 2で実施)
rw_list = pd.read_csv(os.path.join(DATA_DIR, FILE_RW_LIST))
n_rw_all = len(rw_list)

# 2. 売上データ (日付・実績・品目コードが文字列)
sales_raw = pd.read_csv(os.path.join(DATA_DIR, FILE_SALES), dtype=str)
sales_raw["実績"] = pd.to_numeric(sales_raw["実績"], errors="coerce").fillna(0)
n_sales_all = len(sales_raw)
daily = sales_raw[sales_raw["品目コード"].str.strip() == ENT_PRODUCT_CODE].copy()
daily["日付"] = pd.to_datetime(daily["日付"], format="mixed")
daily = daily.rename(columns={
    "日付": "delivery_date",
    "施設（本院に合算）コード": "facility_id",
    "実績": "amount",
})
daily["facility_id"] = daily["facility_id"].astype(str).str.strip()

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
fac_doc_list["fac_honin"] = fac_doc_list["fac_honin"].astype(str).str.strip()

# [Step 1] facility_attribute_修正.csv 読み込み (施設属性用)
fac_df = pd.read_csv(os.path.join(DATA_DIR, FILE_FACILITY_MASTER))
fac_df["fac_honin"] = fac_df["fac_honin"].astype(str).str.strip()

# [Step 2] doctor_attribute.csv 読み込み (医師属性用)
doc_attr_df = pd.read_csv(os.path.join(DATA_DIR, FILE_DOCTOR_ATTR))

_doc_to_fac   = dict(zip(fac_doc_list["doc"], fac_doc_list["fac"]))
_doc_to_honin = dict(zip(fac_doc_list["doc"], fac_doc_list["fac_honin"]))
all_docs = set(fac_doc_list["doc"])

# --- 主施設割り当て (最適化版): 1施設所属は直接割り当て、複数施設所属のみ売上ベース ---
_doc_fac_list = fac_doc_list[["doc", "fac_honin"]].drop_duplicates()
_doc_fac_count = _doc_fac_list.groupby("doc")["fac_honin"].nunique()
_single_fac_docs = set(_doc_fac_count[_doc_fac_count == 1].index)
_multi_fac_docs  = set(_doc_fac_count[_doc_fac_count >  1].index)
print(f"  1施設所属: {len(_single_fac_docs)}名, 複数施設所属: {len(_multi_fac_docs)}名")

# 1施設所属 → 直接割り当て（売上計算不要）
_single_assign = (
    _doc_fac_list[_doc_fac_list["doc"].isin(_single_fac_docs)]
    .drop_duplicates("doc")
    .set_index("doc")["fac_honin"]
)

# 複数施設所属 → 平均納入額最大の施設を主施設に
_doc_fac_list_multi = _doc_fac_list[_doc_fac_list["doc"].isin(_multi_fac_docs)]
_sales_by_fac = (
    daily.groupby("facility_id")["amount"].mean()
    .reset_index()
    .rename(columns={"facility_id": "fac_honin", "amount": "avg_sales"})
)
_doc_fac_sales = _doc_fac_list_multi.merge(_sales_by_fac, on="fac_honin", how="left")
_doc_fac_sales["avg_sales"] = _doc_fac_sales["avg_sales"].fillna(0)

_multi_assign = (
    _doc_fac_sales.sort_values("avg_sales", ascending=False)
    .groupby("doc")["fac_honin"].first()
)

# 全施設0の場合: UHP区分最上位の施設を主施設に（複数施設所属のみ対象）
_fac_uhp = fac_df.drop_duplicates("fac_honin").set_index("fac_honin")["UHP区分名称"] \
    if "UHP区分名称" in fac_df.columns else pd.Series(dtype=str)
_zero_sum = _doc_fac_sales.groupby("doc")["avg_sales"].sum()
_zero_docs_set = set(_zero_sum[_zero_sum == 0].index)
if _zero_docs_set:
    _zero_df = _doc_fac_sales[_doc_fac_sales["doc"].isin(_zero_docs_set)].copy()
    _zero_df["_uhp_rank"] = _zero_df["fac_honin"].map(_fac_uhp).map(
        lambda x: UHP_RANK.get(str(x), 99) if pd.notna(x) else 99
    )
    _zero_best = _zero_df.sort_values("_uhp_rank").groupby("doc")["fac_honin"].first()
    _multi_assign.update(_zero_best)

# 結合: 1施設所属 + 複数施設所属
_doc_primary_all = pd.concat([_single_assign, _multi_assign])

doc_primary_fac = _doc_primary_all  # doc → fac_honin (主施設)

# Step 3: 施設レベルのRW/非RWフィルタ（fac_honinベース）
# ※ 医師IDで絞ると「RW医師の主施設 ≠ rw_list.fac_honin」な施設が混入するため
#   rw_list.fac_honin で施設を確定してから、その施設に所属する全医師を解析対象とする
rw_fac_honins_str = set(rw_list["fac_honin"].astype(str).str.strip())

# --- 施設→医師リスト (主施設ベース、全医師から構築してからフィルタ) ---
_prim_df = pd.DataFrame({"doc": doc_primary_fac.index, "fac": doc_primary_fac.values})
fac_to_docs = _prim_df.groupby("fac")["doc"].agg(list).to_dict()

if INCLUDE_ONLY_RW:
    fac_to_docs = {
        fac: docs for fac, docs in fac_to_docs.items()
        if str(fac).strip() in rw_fac_honins_str
    }
    print(f"  [Step 3] RWフィルタ(fac_honin): {len(fac_to_docs)} 施設")
elif INCLUDE_ONLY_NON_RW:
    fac_to_docs = {
        fac: docs for fac, docs in fac_to_docs.items()
        if str(fac).strip() not in rw_fac_honins_str
    }
    print(f"  [Step 3] 非RWフィルタ(fac_honin): {len(fac_to_docs)} 施設")
else:
    print(f"  [Step 3] スキップ (全施設): {len(fac_to_docs)} 施設")

# 解析対象医師 = フィルタ済み施設の全所属医師
analysis_docs_all = set(d for docs in fac_to_docs.values() for d in docs)

# --- 施設医師数 ---
n_docs_map = {fac: len(docs) for fac, docs in fac_to_docs.items()}

print(f"\n  主施設割り当て完了:")
print(f"    解析対象医師数: {len(analysis_docs_all)}")
print(f"    解析対象施設数: {len(fac_to_docs)}")
_multi = sum(1 for docs in fac_to_docs.values() if len(docs) > 1)
print(f"    複数医師施設: {_multi} 施設")

# 1施設1医師フィルタ（フラグで制御）
if FILTER_SINGLE_FAC_DOCTOR:
    fac_to_docs = {fac: docs for fac, docs in fac_to_docs.items() if len(docs) == 1}
    n_docs_map  = {fac: len(docs) for fac, docs in fac_to_docs.items()}
    print(f"  [1施設1医師フィルタ] 複数医師施設を除外 → 残 {len(fac_to_docs)} 施設")

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

# 医師クインタイル施設平均スコア計算 (fac_to_docs確定後)
_fac_quintile_means: dict = {}
for _qcol in DOCTOR_QUINTILE_COLS:
    if _qcol in doc_attr_df.columns:
        _qnum = doc_attr_df[["doc", _qcol]].copy()
        _qnum[_qcol + "_num"] = _qnum[_qcol].map(QUINTILE_MAP)
        _fac_q: dict = {}
        for _fac, _docs in fac_to_docs.items():
            _vals = _qnum[_qnum["doc"].isin(_docs)][_qcol + "_num"].dropna()
            _fac_q[_fac] = float(_vals.mean()) if len(_vals) > 0 else float("nan")
        _fac_quintile_means[_qcol] = _fac_q
        _n_valid = sum(1 for v in _fac_q.values() if v == v)  # notna count
        print(f"  [{_qcol}] 施設平均スコア算出: 非欠損施設 {_n_valid}/{len(fac_to_docs)}")

# --- 視聴データに主施設IDを付与 (解析対象医師のみ) ---
viewing_all = viewing[viewing["doctor_id"].isin(analysis_docs_all)].copy()
viewing_all["facility_id"] = viewing_all["doctor_id"].map(doc_primary_fac)

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

# 1. 活動データからMR活動 (面談, 面談_アポ, 説明会) を抽出
mr_activity = activity_raw[
    (activity_raw["品目コード"] == ENT_PRODUCT_CODE)
    & (activity_raw["活動種別"].isin(MR_ACTIVITY_TYPES))
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

print(f"\n  MR活動レコード数 (面談・面談_アポ・説明会): {len(mr_activity):,}")
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
# Part 5c: 共変量構築 (ver2: 施設属性 + 施設医師数)
# ================================================================
print("\n" + "=" * 70)
print(" Part 5c: 共変量構築 (ver2: 施設属性 + 施設医師数)")
print("=" * 70)

_cov = pd.DataFrame(index=sorted(analysis_fac_ids))
_cov.index.name = "facility_id"

# 施設医師数 (n_docs)
_cov["n_docs"] = pd.Series(n_docs_map).reindex(_cov.index, fill_value=1)
print(f"  n_docs: 平均={_cov['n_docs'].mean():.1f}, 最大={_cov['n_docs'].max()}")

# MR活動ベースライン（処置前期間: month_index < WASHOUT_MONTHS の月平均）
# ※ MR_ACTIVITY_TYPES（面談・面談_アポ・説明会）に限定
# ※ MR活動量の多い施設は処置確率にも影響するため傾向スコア共変量として追加
_mr_bl_raw = activity_raw[
    (activity_raw["品目コード"] == ENT_PRODUCT_CODE)
    & (activity_raw["活動種別"].isin(MR_ACTIVITY_TYPES))
].copy()
if len(_mr_bl_raw) > 0:
    _mr_bl_raw["活動日_dt"] = pd.to_datetime(_mr_bl_raw["活動日_dt"], format="mixed")
    _mr_bl_raw["_midx"] = (
        (_mr_bl_raw["活動日_dt"].dt.year - 2023) * 12
        + _mr_bl_raw["活動日_dt"].dt.month - 4
    )
    _mr_bl_raw["_fac"] = _mr_bl_raw["fac_honin"].astype(str).str.strip()
    _mr_bl = (
        _mr_bl_raw[_mr_bl_raw["_midx"] < WASHOUT_MONTHS]
        .groupby("_fac").size()
        .div(max(WASHOUT_MONTHS, 1))   # 月平均に換算
    )
    _cov["fac_mr_activity_baseline"] = _mr_bl.reindex(_cov.index, fill_value=0)
else:
    _cov["fac_mr_activity_baseline"] = 0.0
print(f"  fac_mr_activity_baseline: 平均={_cov['fac_mr_activity_baseline'].mean():.2f}, "
      f"最大={_cov['fac_mr_activity_baseline'].max():.1f}, "
      f"非ゼロ施設={(_cov['fac_mr_activity_baseline'] > 0).sum()}/{len(_cov)}")

# 医師クインタイル施設平均スコア (連続共変量)
for _qcol in DOCTOR_QUINTILE_COLS:
    if _qcol in _fac_quintile_means:
        _cname = _qcol + "_mean"
        _cov[_cname] = pd.Series(_fac_quintile_means[_qcol]).reindex(_cov.index)
        _gmean = _cov[_cname].mean()
        _cov[_cname] = _cov[_cname].fillna(_gmean if _gmean == _gmean else 2.0)
        print(f"  {_cname}: 平均={_cov[_cname].mean():.2f}, 欠損→平均補完")

# 施設属性 (UHP区分名, 施設区分名)
_fac_attr_idx = fac_df.drop_duplicates("fac_honin").set_index("fac_honin")
for _cat in COV_CAT_COLS:
    if _cat == "baseline_cat":
        _pre_sales_ser = (
            panel_base[panel_base["month_index"] < WASHOUT_MONTHS]
            .groupby("facility_id")["amount"].mean()
            .reindex(_cov.index, fill_value=0)
        )
        _pos = _pre_sales_ser[_pre_sales_ser > 0]
        if len(_pos) == 0:
            _cat_ser = pd.Series(["0以下"] * len(_cov), index=_cov.index)
        else:
            _q33, _q67 = _pos.quantile(1/3), _pos.quantile(2/3)
            if _q33 == _q67:
                _bc_bins, _bc_labels = [-np.inf, 0, _q67, np.inf], ["0以下", "低", "高"]
            else:
                _bc_bins, _bc_labels = [-np.inf, 0, _q33, _q67, np.inf], ["0以下", "低", "中", "高"]
            _cat_ser = pd.cut(_pre_sales_ser, bins=_bc_bins, labels=_bc_labels, include_lowest=True)
        print(f"  baseline_cat: {sorted(str(v) for v in _cat_ser.dropna().unique())}")
    elif _cat in _fac_attr_idx.columns:
        _cat_ser = _fac_attr_idx[_cat].reindex(_cov.index)
    else:
        print(f"  {_cat}: 列なし → スキップ")
        continue
    _dum = pd.get_dummies(_cat_ser, prefix=_cat, drop_first=True, dtype=float)
    _dum.columns = [c.replace("-", "_").replace(" ", "_") for c in _dum.columns]
    for _c in _dum.columns:
        _cov[_c] = _dum[_c]

_cov = _cov.fillna(0)
_cov_means = _cov.mean()
_cov_stds  = _cov.std().replace(0, 1)
cov_std    = (_cov - _cov_means) / _cov_stds
COV_COLS   = list(cov_std.columns)

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
# Coverage パネル構築 (施設視聴率: 累積視聴医師数 / 施設総医師数)
# ================================================================
_first_view_per_doc = (
    viewing_all[viewing_all["facility_id"].isin(analysis_fac_ids) &
                viewing_all["month_index"].between(0, N_MONTHS - 1)]
    .groupby(["facility_id", "doctor_id"])["month_index"].min()
    .reset_index(name="first_view_month")
)

_all_fac_months = pd.MultiIndex.from_product(
    [sorted(analysis_fac_ids), range(N_MONTHS)],
    names=["facility_id", "month_index"]
).to_frame(index=False)

_fvd = _first_view_per_doc.copy()
_fvd_expanded = _all_fac_months.merge(
    _fvd[["facility_id", "first_view_month"]], on="facility_id", how="left"
)
_fvd_expanded["viewed_by_month"] = (
    _fvd_expanded["first_view_month"] <= _fvd_expanded["month_index"]
).astype(int)
_cum_viewed = (
    _fvd_expanded.groupby(["facility_id", "month_index"])["viewed_by_month"]
    .sum().reset_index(name="n_viewed_docs")
)

_n_docs_ser = pd.Series({fac: len(docs) for fac, docs in fac_to_docs.items()}, name="n_docs")
_cum_viewed["n_docs_total"] = _cum_viewed["facility_id"].map(_n_docs_ser)
_cum_viewed["coverage"] = (
    _cum_viewed["n_viewed_docs"] / _cum_viewed["n_docs_total"].clip(lower=1)
)
coverage_panel = _cum_viewed[["facility_id", "month_index", "coverage"]]

# ================================================================
# Part 5e: Coverage TWFE 推定（連続処置強度）
# ================================================================
print("\n" + "=" * 70)
print(" Part 5e: Coverage TWFE（施設視聴率を連続処置として推定）")
print("=" * 70)

panel_cov = panel.merge(coverage_panel, on=["facility_id", "month_index"], how="left")
panel_cov["coverage"] = panel_cov["coverage"].fillna(0.0)
panel_cov = panel_cov.set_index(["unit_id", "month_index"])

try:
    from linearmodels import PanelOLS
    _formula_cov = "amount ~ coverage + EntityEffects + TimeEffects"
    _mod_cov = PanelOLS.from_formula(_formula_cov, data=panel_cov)
    _res_cov = _mod_cov.fit(cov_type="clustered", cluster_entity=True)
    beta_cov = float(_res_cov.params["coverage"])
    se_cov   = float(_res_cov.std_errors["coverage"])
    pval_cov = float(_res_cov.pvalues["coverage"])
    ci_cov   = _res_cov.conf_int().loc["coverage"].values.tolist()
    sig_cov  = "***" if pval_cov < 0.001 else "**" if pval_cov < 0.01 else "*" if pval_cov < 0.05 else "n.s."
    print(f"  Coverage ATT: {beta_cov:.4f}")
    print(f"  SE          : {se_cov:.4f}")
    print(f"  p値         : {pval_cov:.6f} {sig_cov}")
    print(f"  95%CI       : [{ci_cov[0]:.4f}, {ci_cov[1]:.4f}]")
    print(f"  解釈: Coverage が 1（= 0→100% 全員視聴）変化したときの月次売上への効果 = {beta_cov:.2f}円/月")
    coverage_twfe = {
        "att": beta_cov, "se": se_cov, "p": pval_cov,
        "ci_lo": ci_cov[0], "ci_hi": ci_cov[1], "sig": sig_cov,
        "note": "coverage=0→1の変化に対応する売上変化(円/月)"
    }
except Exception as e:
    print(f"  Coverage TWFE 推定失敗: {e}")
    coverage_twfe = None

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

channel_results = {}
channel_dr_results = {}

for ch in CONTENT_TYPES:
    print(f"\n  --- {ch} ---")
    # チャネル別 cohort_month: 施設のそのチャネルでの初回視聴月
    _ch_view = viewing_all[viewing_all["channel_category"] == ch].copy()
    _ch_first = (
        _ch_view[
            (_ch_view["month_index"] >= WASHOUT_MONTHS) &
            (_ch_view["month_index"] <= LAST_ELIGIBLE_MONTH) &
            _ch_view["facility_id"].notna() &
            ~_ch_view["facility_id"].isin(washout_fac_ids)
        ]
        .groupby("facility_id")["month_index"].min()
    )

    if len(_ch_first) == 0:
        print(f"  -> 該当なし")
        continue

    ch_treated_facs = set(_ch_first.index)
    use_facs = ch_treated_facs | control_fac_ids

    ch_panel = panel_base[panel_base["facility_id"].isin(use_facs)].copy()
    ch_panel = ch_panel.drop(columns=["cohort_month", "treated"], errors="ignore")
    ch_panel["unit_id"] = ch_panel["facility_id"]

    # cohort_month を施設単位でマージ
    _ch_cohort_df = _ch_first.reset_index()
    _ch_cohort_df.columns = ["facility_id", "cohort_month"]
    ch_panel = ch_panel.merge(
        _ch_cohort_df, on="facility_id", how="left"
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
    "Staggered DID ver2: デジタルコンテンツ視聴の効果\n"
    "(複数医師施設対応, wash-out/遅延視聴施設除外)",
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
out_path = os.path.join(SCRIPT_DIR, f"staggered_did_results_v2{_suffix}.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  図を保存: {out_path}")

# ================================================================
# Coverage サンプル可視化（複数医師施設）
# ================================================================
print("\n[Coverage サンプル可視化]")

_multi_treated = [
    fac for fac in treated_fac_ids
    if n_docs_map.get(fac, 1) >= 2
]
_sample_facs = sorted(_multi_treated, key=lambda f: -n_docs_map.get(f, 1))[:3]

if not _sample_facs:
    print("  複数医師の処置施設なし - スキップ")
else:
    _ncols = len(_sample_facs)
    fig_cov, axes_cov = plt.subplots(
        2, _ncols, figsize=(6 * _ncols, 9),
        gridspec_kw={"height_ratios": [2, 1]},
        sharex="col"
    )
    if _ncols == 1:
        axes_cov = axes_cov.reshape(2, 1)

    _month_labels = [m.strftime("%Y/%m") for m in months]
    _xticks_sparse = list(range(0, N_MONTHS, 6))

    # 医師ごとのR/W区分マップ (seg="R" → R医師, "W" → W医師, それ以外 → -)
    _rw_seg_map = dict(zip(rw_list["doc"], rw_list["seg"].fillna("")))

    for col_i, fac in enumerate(_sample_facs):
        ax_top = axes_cov[0, col_i]
        ax_bot = axes_cov[1, col_i]

        _docs_in_fac = fac_to_docs.get(fac, [])
        _n_total = len(_docs_in_fac)
        _cohort = int(_first_view.get(fac, np.nan)) if fac in _first_view.index else None

        # --- Coverage 折れ線 ---
        _cov_fac = (
            coverage_panel[coverage_panel["facility_id"] == fac]
            .set_index("month_index")["coverage"]
            .reindex(range(N_MONTHS), fill_value=0.0)
        )
        ax_top.plot(range(N_MONTHS), _cov_fac.values,
                    color="#1565C0", linewidth=2, label="Coverage")
        ax_top.fill_between(range(N_MONTHS), _cov_fac.values, alpha=0.15, color="#1565C0")

        # --- 各医師の初回視聴月（縦線） ---
        _colors_doc = plt.cm.Set2(np.linspace(0, 1, max(_n_total, 1)))
        _doc_labels = []
        _n_r = sum(1 for d in _docs_in_fac if _rw_seg_map.get(d, "") == "R")
        _n_w = sum(1 for d in _docs_in_fac if _rw_seg_map.get(d, "") == "W")
        for di, doc in enumerate(_docs_in_fac):
            _doc_rows = _first_view_per_doc[
                (_first_view_per_doc["facility_id"] == fac) &
                (_first_view_per_doc["doctor_id"] == doc)
            ]
            if len(_doc_rows) == 0:
                continue
            _fvm = int(_doc_rows["first_view_month"].iloc[0])
            _c = _colors_doc[di]
            _doc_type = _rw_seg_map.get(doc, "") or "-"
            ax_top.axvline(_fvm, color=_c, linestyle="--", alpha=0.8, linewidth=1.5)
            ax_top.text(_fvm + 0.2, 0.05 + 0.12 * di,
                        f"医師{di+1}({_doc_type})初回", fontsize=7, color=_c, va="bottom")
            _doc_labels.append(f"医師{di+1}({_doc_type})初回 (month={_fvm})")

        # --- 施設コホート月（太い点線） ---
        if _cohort is not None:
            ax_top.axvline(_cohort, color="red", linestyle=":", linewidth=2.5,
                           label=f"施設初回視聴 (month={_cohort})")

        # --- ウォッシュアウト期間シェード ---
        ax_top.axvspan(0, WASHOUT_MONTHS, alpha=0.08, color="gray", label="washout")

        ax_top.set_ylim(-0.05, 1.15)
        ax_top.set_ylabel("Coverage（視聴率）")
        ax_top.set_title(f"施設 {str(fac)[:12]}\n医師数={_n_total}名 (R:{_n_r}, W:{_n_w})", fontsize=10)
        ax_top.legend(fontsize=7, loc="upper left")
        ax_top.set_xticks(_xticks_sparse)
        ax_top.set_xticklabels([_month_labels[i] for i in _xticks_sparse],
                                rotation=45, fontsize=7)
        ax_top.grid(axis="y", alpha=0.3)

        # --- 月次売上（棒グラフ） ---
        _sales_fac = (
            panel_base[panel_base["facility_id"] == fac]
            .set_index("month_index")["amount"]
            .reindex(range(N_MONTHS), fill_value=0)
        )
        _bar_colors = ["#FF8F00" if _cov_fac.iloc[mi] > 0 else "#B0BEC5"
                       for mi in range(N_MONTHS)]
        ax_bot.bar(range(N_MONTHS), _sales_fac.values, color=_bar_colors,
                   width=0.8, alpha=0.8)
        if _cohort is not None:
            ax_bot.axvline(_cohort, color="red", linestyle=":", linewidth=2.5)
        ax_bot.axvspan(0, WASHOUT_MONTHS, alpha=0.08, color="gray")
        ax_bot.set_ylabel("月次売上（円）")
        ax_bot.set_xlabel("月")
        ax_bot.set_xticks(_xticks_sparse)
        ax_bot.set_xticklabels([_month_labels[i] for i in _xticks_sparse],
                                rotation=45, fontsize=7)
        ax_bot.yaxis.get_major_formatter().set_scientific(False)
        ax_bot.grid(axis="y", alpha=0.3)

    fig_cov.suptitle(
        "Coverage（施設視聴率）月次推移 - サンプル複数医師施設\n"
        "棒グラフ: 月次売上（橙=視聴後、灰=未視聴）　点線赤: 施設初回視聴月　破線: 各医師初回視聴月",
        fontsize=10, y=1.01
    )
    plt.tight_layout()
    _cov_sample_path = os.path.join(SCRIPT_DIR, f"coverage_sample_v2{_suffix}.png")
    fig_cov.savefig(_cov_sample_path, dpi=150, bbox_inches="tight")
    plt.close(fig_cov)
    print(f"  図保存: {_cov_sample_path}")
    print(f"  サンプル施設: {_sample_facs}")

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

# 除外フロー情報 (ver2: 施設ベース)
exclusion_flow = {
    "total_delivery_rows": n_sales_all,
    "ent_delivery_rows": len(daily),
    "total_rw_list": n_rw_all,
    "total_doctors": len(all_docs),
    "analysis_doctors": len(analysis_docs_all),
    "total_facilities": len(fac_to_docs),
    "multi_doctor_facilities": _multi,
    "washout_excluded_facilities": len(washout_fac_ids),
    "washout_excluded_doctors": sum(len(fac_to_docs.get(f, [])) for f in washout_fac_ids),
    "final_treated": len(treated_fac_ids),
    "final_treated_doctors": sum(len(fac_to_docs.get(f, [])) for f in treated_fac_ids),
    "final_control": len(control_fac_ids),
    "final_control_doctors": sum(len(fac_to_docs.get(f, [])) for f in control_fac_ids),
    "final_total": len(analysis_fac_ids),
    "final_total_doctors": sum(len(fac_to_docs.get(f, [])) for f in analysis_fac_ids),
    "include_only_rw": INCLUDE_ONLY_RW,
    "exclude_zero_sales": EXCLUDE_ZERO_SALES_FACILITIES,
    "total_viewing_rows": len(viewing),       # viewing（ENT絞り込み済み、施設ID付与前）の行数
    "viewing_after_filter": len(viewing_all), # 施設ID付与済みの視聴データ行数
    "version": "v2",
}

# 施設内医師構成カテゴリ別集計（モードに応じて分類定義が変わる）
_rw_doc_set_all = set(rw_list["doc"])

def _get_rw_breakdowns(fac_ids):
    """施設をRW/非RW構成でカテゴリ分けし、施設数・医師数を返す"""
    if INCLUDE_ONLY_RW:
        cats = ["1RW先", "複数RW先"]
        counts = {k: {"facs": 0, "docs": 0} for k in cats}
        for f in fac_ids:
            docs = fac_to_docs.get(f, [])
            key = "1RW先" if len(docs) == 1 else "複数RW先"
            counts[key]["facs"] += 1
            counts[key]["docs"] += len(docs)
    elif INCLUDE_ONLY_NON_RW:
        cats = ["1非RW先", "複数非RW先"]
        counts = {k: {"facs": 0, "docs": 0} for k in cats}
        for f in fac_ids:
            docs = fac_to_docs.get(f, [])
            key = "1非RW先" if len(docs) == 1 else "複数非RW先"
            counts[key]["facs"] += 1
            counts[key]["docs"] += len(docs)
    else:
        # RW＋非RW混在：施設の医師構成で5分類（1医師先は1RW先+1非RW先の合計）
        cats = ["1RW先", "1非RW先", "RW非RW複数先", "RW複数先", "非RW複数先"]
        counts = {k: {"facs": 0, "docs": 0} for k in cats}
        for f in fac_ids:
            docs = fac_to_docs.get(f, [])
            n_rw  = sum(1 for d in docs if d in _rw_doc_set_all)
            n_non = len(docs) - n_rw
            if n_rw == 1 and n_non == 0:
                key = "1RW先"
            elif n_rw == 0 and n_non == 1:
                key = "1非RW先"
            elif n_rw >= 1 and n_non >= 1:
                key = "RW非RW複数先"
            elif n_rw >= 2 and n_non == 0:
                key = "RW複数先"
            else:  # n_rw == 0, n_non >= 2
                key = "非RW複数先"
            counts[key]["facs"] += 1
            counts[key]["docs"] += len(docs)
    return counts

exclusion_flow["doc_filter_label"]      = ("RW先" if INCLUDE_ONLY_RW else "非RW先" if INCLUDE_ONLY_NON_RW else "医師先")
exclusion_flow["rw_breakdown_treated"]  = _get_rw_breakdowns(treated_fac_ids)
exclusion_flow["rw_breakdown_control"]  = _get_rw_breakdowns(control_fac_ids)
# 後方互換キー（CONSORT図の旧参照用）
_t_bd = exclusion_flow["rw_breakdown_treated"]
_c_bd = exclusion_flow["rw_breakdown_control"]
exclusion_flow["treated_single_rw"]      = sum(v["facs"] for k, v in _t_bd.items() if k.startswith("1"))
exclusion_flow["treated_single_rw_docs"] = sum(v["docs"] for k, v in _t_bd.items() if k.startswith("1"))
exclusion_flow["treated_multi_rw"]       = sum(v["facs"] for k, v in _t_bd.items() if not k.startswith("1"))
exclusion_flow["treated_multi_rw_docs"]  = sum(v["docs"] for k, v in _t_bd.items() if not k.startswith("1"))
exclusion_flow["control_single_rw"]      = sum(v["facs"] for k, v in _c_bd.items() if k.startswith("1"))
exclusion_flow["control_single_rw_docs"] = sum(v["docs"] for k, v in _c_bd.items() if k.startswith("1"))
exclusion_flow["control_multi_rw"]       = sum(v["facs"] for k, v in _c_bd.items() if not k.startswith("1"))
exclusion_flow["control_multi_rw_docs"]  = sum(v["docs"] for k, v in _c_bd.items() if not k.startswith("1"))

# 除外された施設ID一覧
excluded_ids = {
    "washout": sorted([str(f) for f in washout_fac_ids]),
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
    "coverage_twfe": coverage_twfe,
}

json_path = os.path.join(results_dir, f"did_results_v2{_suffix}.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(did_results_json, f, ensure_ascii=False, indent=2)
print(f"  結果をJSON保存: {json_path}")
