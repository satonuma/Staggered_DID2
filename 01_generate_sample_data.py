"""
===================================================================
サンプルデータ生成: Staggered DID分析用
デジタルコンテンツ視聴（webiner, e_contents, Web講演会）の効果検証
===================================================================
設計:
  - 売上データ: 日別・施設別 (doctor_idなし, 整数額)
  - 施設ごとに月間売上回数が異なる (月1~5回)
  - 1施設に複数医師所属あり / 1医師が複数施設所属あり
  - 視聴パターン: 未視聴 / wash-out視聴 / 単発 / 定常 / 離脱 / 遅延
  - 観測期間: 2023/4 ~ 2025/12 (33ヶ月)
  - CATE用属性: ベースライン納入額カテゴリ
    → 処置効果に異質性あり (modifier で乗算)

出力形式: 本番データに準拠
  - rw_list.csv: RW医師リスト (ENT絞込済、品目カラムなし)
  - sales.csv: 売上実績 (日付・実績が文字列)
  - デジタル視聴データ.csv: webiner/e_contentsの視聴ログ
  - 活動データ.csv: Web講演会+その他の活動記録
===================================================================
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

# === パラメータ ===
START_DATE = "2023-04-01"
N_MONTHS = 33  # 2023/4 ~ 2025/12
NOISE_SD = 18
PER_DOCTOR_BASE = 100
TREND = 1.2

# === 品目コード (パラメータ) ===
ENT_PRODUCT_CODE = "00001"   # ENTの品目コード (5桁文字列)
PRODUCT_CODES = {
    "ENT": "00001", "CNS": "00002", "GI": "00003",
    "CV": "00004", "その他": "00005",
}

CONTENT_TYPES = ["webiner", "e_contents", "Web講演会"]
CHANNEL_EFFECTS = {"webiner": 18, "e_contents": 10, "Web講演会": 22}
CHANNEL_GROWTH = 1.0    # 視聴継続中の月次成長
DECAY_RATE = 1.5        # 視聴停止後の月次減衰
GRACE_MONTHS = 2        # 視聴停止後の猶予期間

# 活動種別マッピング (チャネル → 活動種別コード)
ACTIVITY_TYPE_MAP = {
    "webiner":    ["AT01", "AT02", "AT03"],
    "e_contents": ["AT04", "AT05", "AT06"],
    "Web講演会":  ["AT07", "AT08", "AT09"],
    "面談":       ["AT13", "AT14"],
    "面談_アポ":  ["AT15", "AT16"],
    "説明会":     ["AT17", "AT18"],
    "その他":     ["AT10", "AT11", "AT12"],
}
ACTIVITY_CHANNEL_FILTER = "Web講演会"  # 活動データから抽出する活動種別

PRODUCTS = ["ENT", "CNS", "GI", "CV", "その他"]
PRODUCT_BASE_RATIO = {"CNS": 0.65, "GI": 0.55, "CV": 0.45, "その他": 0.30}
RW_FLAG_RATE = 0.85     # ENT医師のうちRW対象の割合
ENT_DOCTOR_RATE = 0.75  # 全医師のうちENT品目の割合

# CATE用: 処置効果の異質性 (modifier を乗算)
REGION_MODIFIER = {"都市部": 0.75, "郊外": 1.00, "地方": 1.40}
TYPE_MODIFIER = {"病院": 0.85, "クリニック": 1.20}
EXP_MODIFIER = {"若手": 1.30, "中堅": 1.00, "ベテラン": 0.70}
SPEC_MODIFIER = {"内科": 1.15, "外科": 0.85, "その他": 1.00}

# 施設構成
N_SINGLE = 120
N_DOUBLE = 40
N_TRIPLE = 25
N_QUAD = 15
DOC_COUNTS = [1]*N_SINGLE + [2]*N_DOUBLE + [3]*N_TRIPLE + [4]*N_QUAD
N_FACILITIES = len(DOC_COUNTS)

N_MULTI_FAC_DOCS = 25   # 複数施設所属医師
TREAT_RATE = 0.62        # 視聴する医師の割合

# 視聴パターン配分（視聴医師内）
PATTERN_WEIGHTS = {
    "wash_out":   0.10,
    "one_shot":   0.20,
    "continuous":  0.35,
    "lapsed":     0.25,
    "late":       0.10,
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data")
# ロック回避: 既存CSVが開かれている場合は別ディレクトリに出力
_locked = False
for _fn in ["rw_list.csv", "sales.csv", "デジタル視聴データ.csv", "活動データ.csv"]:
    _fp = os.path.join(OUTPUT_DIR, _fn)
    if os.path.exists(_fp):
        try:
            with open(_fp, "a"):
                pass
        except (PermissionError, OSError):
            _locked = True
            break
if _locked:
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data2")
    print(f"[注意] data/ のCSVがロックされているため {OUTPUT_DIR} に出力します")
os.makedirs(OUTPUT_DIR, exist_ok=True)

months = pd.date_range(start=START_DATE, periods=N_MONTHS, freq="MS")

# === 1. 施設マスター (内部用) ===
facility_ids = [f"F{i:03d}" for i in range(1, N_FACILITIES + 1)]
facility_effects = np.random.normal(0, 15.0, N_FACILITIES)
fac_effect_map = dict(zip(facility_ids, facility_effects))

regions = np.random.choice(
    ["都市部", "郊外", "地方"], N_FACILITIES, p=[0.40, 0.35, 0.25]
)
fac_types = np.random.choice(
    ["病院", "クリニック"], N_FACILITIES, p=[0.30, 0.70]
)

# 施設ごとの月間納入回数 (1~5回, 施設の特性)
fac_delivery_freq = {
    fid: int(np.random.choice([1, 2, 3, 4, 5], p=[0.15, 0.30, 0.25, 0.20, 0.10]))
    for fid in facility_ids
}

facilities = pd.DataFrame({
    "facility_id": facility_ids,
    "facility_name": [f"施設{i}" for i in range(1, N_FACILITIES + 1)],
    "region": regions,
    "facility_type": fac_types,
    "monthly_delivery_freq": [fac_delivery_freq[f] for f in facility_ids],
})
fac_region_map = dict(zip(facility_ids, regions))
fac_type_map = dict(zip(facility_ids, fac_types))
fac_honin_name_map = dict(zip(facilities["facility_id"], facilities["facility_name"]))

# fac (DCF施設コード): 大半はfac_honinと同じ、約10%が異なる
fac_map = {}
for i, fid in enumerate(facility_ids):
    if i % 10 == 0:
        fac_map[fid] = f"G{fid[1:]}"
    else:
        fac_map[fid] = fid

# fac_name_map: facility_id → fac列の施設名
fac_name_map = {}
for fid in facility_ids:
    fc = fac_map[fid]
    if fc == fid:
        fac_name_map[fid] = fac_honin_name_map[fid]
    else:
        fac_name_map[fid] = f"分院{fc[1:]}"

# === 2. 医師マスター (内部用) ===
doc_records = []
doc_counter = 0
for fac_id, n_docs in zip(facility_ids, DOC_COUNTS):
    for _ in range(n_docs):
        doc_counter += 1
        doc_records.append({"doctor_id": f"D{doc_counter:04d}", "facility_id": fac_id})

# 複数施設所属
multi_fac_doc_ids = np.random.choice(
    [r["doctor_id"] for r in doc_records], N_MULTI_FAC_DOCS, replace=False
)
existing_facs = {r["doctor_id"]: r["facility_id"] for r in doc_records}
for doc_id in multi_fac_doc_ids:
    other_facs = [f for f in facility_ids if f != existing_facs[doc_id]]
    second_fac = np.random.choice(other_facs)
    doc_records.append({"doctor_id": doc_id, "facility_id": second_fac})

doctors = pd.DataFrame(doc_records)
unique_doc_ids = doctors["doctor_id"].unique()
N_DOCTORS = len(unique_doc_ids)

doc_effects = {d: np.random.normal(0, 20.0) for d in unique_doc_ids}

# 医師属性
doc_exp_years = {d: int(np.random.uniform(1, 36)) for d in unique_doc_ids}
doc_specialties = {
    d: np.random.choice(["内科", "外科", "その他"], p=[0.40, 0.25, 0.35])
    for d in unique_doc_ids
}

def exp_category(years):
    if years <= 10:
        return "若手"
    elif years <= 20:
        return "中堅"
    else:
        return "ベテラン"

doc_exp_cat = {d: exp_category(y) for d, y in doc_exp_years.items()}

# 医師名マップ
doc_name_map = {d: f"医師{int(d[1:]):04d}" for d in unique_doc_ids}

# primary facility
primary_fac_map = {}
for rec in doc_records:
    if rec["doctor_id"] not in primary_fac_map:
        primary_fac_map[rec["doctor_id"]] = rec["facility_id"]

# 処置効果 modifier
doc_modifier = {}
for d in unique_doc_ids:
    pf = primary_fac_map[d]
    m = (REGION_MODIFIER[fac_region_map[pf]]
         * TYPE_MODIFIER[fac_type_map[pf]]
         * EXP_MODIFIER[doc_exp_cat[d]]
         * SPEC_MODIFIER[doc_specialties[d]])
    doc_modifier[d] = m

# 品目・RWフラグ割当
doc_product = {}
doc_rw_flag = {}
for d in unique_doc_ids:
    if np.random.random() < ENT_DOCTOR_RATE:
        doc_product[d] = "ENT"
        doc_rw_flag[d] = "RW" if np.random.random() < RW_FLAG_RATE else ""
    else:
        doc_product[d] = np.random.choice(["CNS", "GI", "CV", "その他"])
        doc_rw_flag[d] = ""

# === 3. 視聴パターンとチャネル割当 ===
n_treated = int(N_DOCTORS * TREAT_RATE)
treated_ids = np.random.choice(unique_doc_ids, n_treated, replace=False)

patterns = list(PATTERN_WEIGHTS.keys())
pattern_probs = list(PATTERN_WEIGHTS.values())
assigned_patterns = np.random.choice(patterns, n_treated, p=pattern_probs)

doctor_info = {}
for doc_id, pattern in zip(treated_ids, assigned_patterns):
    n_ch = np.random.choice([1, 2, 3], p=[0.30, 0.40, 0.30])
    chs = np.random.choice(CONTENT_TYPES, n_ch, replace=False)

    ch_info = {}
    if pattern == "wash_out":
        for ch in chs:
            first_m = np.random.randint(0, 2)
            last_m = min(first_m + np.random.randint(6, 25), N_MONTHS - 1)
            ch_info[ch] = (first_m, last_m)
    elif pattern == "one_shot":
        for ch in chs:
            first_m = np.random.randint(2, 27)
            last_m = first_m + np.random.randint(0, 2)
            ch_info[ch] = (first_m, min(last_m, N_MONTHS - 1))
    elif pattern == "continuous":
        for ch in chs:
            first_m = np.random.randint(2, 20)
            ch_info[ch] = (first_m, N_MONTHS - 1)
    elif pattern == "lapsed":
        for ch in chs:
            first_m = np.random.randint(2, 15)
            duration = np.random.randint(6, 19)
            last_m = min(first_m + duration, N_MONTHS - 1)
            ch_info[ch] = (first_m, last_m)
    elif pattern == "late":
        for ch in chs:
            first_m = np.random.randint(30, N_MONTHS)
            ch_info[ch] = (first_m, N_MONTHS - 1)

    doctor_info[doc_id] = {"pattern": pattern, "channels": ch_info}

# === 4. 日別視聴ログ (本番形式) ===
viewing_records = []
for doc_id, info in doctor_info.items():
    fac_ids_for_doc = doctors.loc[doctors["doctor_id"] == doc_id, "facility_id"].values
    primary_fac = fac_ids_for_doc[0]
    prod_code = PRODUCT_CODES[doc_product[doc_id]]

    for ch, (first_m, last_m) in info["channels"].items():
        first_date = months[first_m]
        first_offset = np.random.randint(0, min(28, first_date.days_in_month))
        viewing_records.append({
            "活動日_dt": (first_date + pd.Timedelta(days=first_offset)).strftime("%Y-%m-%d"),
            "品目コード": prod_code,
            "活動種別": ch,
            "活動種別コード": np.random.choice(ACTIVITY_TYPE_MAP[ch]),
            "dcf_fac": primary_fac,
            "fac_honin": primary_fac,
            "fac": fac_map[primary_fac],
            "dcf_doc": doc_id,
            "doc": doc_id,
            "doc_name": doc_name_map[doc_id],
        })

        active_months = last_m - first_m
        if active_months <= 0:
            continue

        if info["pattern"] in ("continuous", "wash_out"):
            n_extra = np.random.randint(active_months // 2, active_months + 1)
        elif info["pattern"] == "lapsed":
            n_extra = np.random.randint(active_months // 3, active_months)
        elif info["pattern"] == "one_shot":
            n_extra = np.random.randint(0, 3)
        else:
            n_extra = np.random.randint(0, max(1, active_months))

        if n_extra > 0 and active_months > 0:
            end_date = months[min(last_m, N_MONTHS - 1)] + pd.offsets.MonthEnd(0)
            span_days = max((end_date - first_date).days - 28, 1)
            offsets = sorted(np.random.choice(
                range(28, 28 + span_days), min(n_extra, span_days), replace=False
            ))
            for off in offsets:
                vd = first_date + pd.Timedelta(days=int(off))
                if vd <= end_date:
                    viewing_records.append({
                        "活動日_dt": vd.strftime("%Y-%m-%d"),
                        "品目コード": prod_code,
                        "活動種別": ch,
                        "活動種別コード": np.random.choice(ACTIVITY_TYPE_MAP[ch]),
                        "dcf_fac": primary_fac,
                        "fac_honin": primary_fac,
                        "fac": fac_map[primary_fac],
                        "dcf_doc": doc_id,
                        "doc": doc_id,
                        "doc_name": doc_name_map[doc_id],
                    })

# その他活動の記録を追加（処置効果なし → ノイズ）
n_sonota_viewers = int(n_treated * 0.25)
sonota_viewer_ids = np.random.choice(treated_ids, n_sonota_viewers, replace=False)
for doc_id in sonota_viewer_ids:
    primary_fac = primary_fac_map[doc_id]
    prod_code = PRODUCT_CODES[doc_product[doc_id]]
    n_views = np.random.randint(1, 6)
    for _ in range(n_views):
        view_month = np.random.randint(0, N_MONTHS)
        month_start = months[view_month]
        day_offset = np.random.randint(0, month_start.days_in_month)
        vd = month_start + pd.Timedelta(days=day_offset)
        viewing_records.append({
            "活動日_dt": vd.strftime("%Y-%m-%d"),
            "品目コード": prod_code,
            "活動種別": "その他",
            "活動種別コード": np.random.choice(ACTIVITY_TYPE_MAP["その他"]),
            "dcf_fac": primary_fac,
            "fac_honin": primary_fac,
            "fac": fac_map[primary_fac],
            "dcf_doc": doc_id,
            "doc": doc_id,
            "doc_name": doc_name_map[doc_id],
        })

# MR活動の記録を追加（面談・面談_アポ・説明会）— 全ENT医師対象
ent_doc_ids = [d for d in unique_doc_ids if doc_product[d] == "ENT"]
mr_type_choices = ["面談", "面談_アポ", "説明会"]
mr_type_probs = [0.60, 0.25, 0.15]

for doc_id in ent_doc_ids:
    primary_fac = primary_fac_map[doc_id]
    prod_code = PRODUCT_CODES["ENT"]
    # 月ごとに0〜3回程度 (Poisson λ=1.0)
    for t in range(N_MONTHS):
        n_mr = np.random.poisson(1.0)
        if n_mr == 0:
            continue
        n_mr = min(n_mr, 3)
        month_start = months[t]
        for _ in range(n_mr):
            mr_type = np.random.choice(mr_type_choices, p=mr_type_probs)
            day_offset = np.random.randint(0, month_start.days_in_month)
            vd = month_start + pd.Timedelta(days=day_offset)
            viewing_records.append({
                "活動日_dt": vd.strftime("%Y-%m-%d"),
                "品目コード": prod_code,
                "活動種別": mr_type,
                "活動種別コード": np.random.choice(ACTIVITY_TYPE_MAP[mr_type]),
                "dcf_fac": primary_fac,
                "fac_honin": primary_fac,
                "fac": fac_map[primary_fac],
                "dcf_doc": doc_id,
                "doc": doc_id,
                "doc_name": doc_name_map[doc_id],
            })

viewing_df = pd.DataFrame(viewing_records)

# === 5. 日別施設別売上データ (本番形式) ===
fac_doctors_grp = doctors.groupby("facility_id")
daily_records = []

for fac_id, grp in fac_doctors_grp:
    fac_eff = fac_effect_map[fac_id]
    base_freq = fac_delivery_freq[fac_id]

    for t in range(N_MONTHS):
        facility_total = 0.0

        for _, doc_row in grp.iterrows():
            d_id = doc_row["doctor_id"]
            base = PER_DOCTOR_BASE
            trend = TREND * t
            d_eff = doc_effects[d_id]

            treatment = 0.0
            modifier = doc_modifier[d_id]
            if d_id in doctor_info:
                for ch, (first_m, last_m) in doctor_info[d_id]["channels"].items():
                    if t < first_m:
                        continue
                    eff = CHANNEL_EFFECTS[ch]
                    months_since_first = t - first_m
                    months_since_last = max(0, t - last_m)

                    if months_since_last <= GRACE_MONTHS:
                        treatment += modifier * (eff + CHANNEL_GROWTH * months_since_first)
                    else:
                        peak = eff + CHANNEL_GROWTH * (last_m - first_m + GRACE_MONTHS)
                        decay_months = months_since_last - GRACE_MONTHS
                        treatment += modifier * max(0, peak - DECAY_RATE * decay_months)

            noise = np.random.normal(0, NOISE_SD)
            facility_total += max(0, base + d_eff + fac_eff + trend + treatment + noise)

        facility_total = int(round(facility_total))
        if facility_total <= 0:
            continue

        month_start = months[t]
        n_days = month_start.days_in_month

        # 月間納入回数 (施設固有 ± 1のゆらぎ)
        n_del = max(1, base_freq + np.random.choice([-1, 0, 0, 0, 1]))
        n_del = min(n_del, n_days)

        # 納入日: 間隔をある程度均等にする
        if n_del == 1:
            del_days = [np.random.randint(0, n_days)]
        else:
            interval = n_days / n_del
            del_days = []
            for k in range(n_del):
                center = int(interval * (k + 0.5))
                jitter = np.random.randint(-2, 3)
                day = max(0, min(n_days - 1, center + jitter))
                del_days.append(day)
            del_days = sorted(set(del_days))
            n_del = len(del_days)

        # 金額配分 (整数)
        weights = np.random.dirichlet(np.ones(n_del) * 3)
        amts = np.round(facility_total * weights).astype(int)
        amts[-1] = facility_total - amts[:-1].sum()

        for i, d in enumerate(del_days):
            a = max(0, int(amts[i]))
            if a > 0:
                daily_records.append({
                    "日付": (month_start + pd.Timedelta(days=int(d))).strftime("%Y%m%d"),
                    "施設（本院に合算）コード": fac_id,
                    "DCF施設コード": fac_id,
                    "品目コード": PRODUCT_CODES["ENT"],
                    "実績": a,
                })

# 他品目(CNS, GI, CV, その他)の売上レコードを簡易生成
for fac_id in facility_ids:
    base_freq = fac_delivery_freq[fac_id]
    for t in range(N_MONTHS):
        month_start = months[t]
        n_days = month_start.days_in_month
        for prod in ["CNS", "GI", "CV", "その他"]:
            ratio = PRODUCT_BASE_RATIO[prod]
            n_del = max(1, min(2, base_freq - 1))
            for _ in range(n_del):
                amt = int(max(0, np.random.normal(PER_DOCTOR_BASE * ratio, NOISE_SD)))
                if amt > 0:
                    day = np.random.randint(0, n_days)
                    daily_records.append({
                        "日付": (month_start + pd.Timedelta(days=int(day))).strftime("%Y%m%d"),
                        "施設（本院に合算）コード": fac_id,
                        "DCF施設コード": fac_id,
                        "品目コード": PRODUCT_CODES[prod],
                        "実績": amt,
                    })

delivery_df = pd.DataFrame(daily_records)

# === 6. CSV保存 ===
# rw_list.csv: ENT医師のみ (品目カラムなし、本番と同じ)
ent_doctors = doctors[doctors["doctor_id"].map(doc_product) == "ENT"].copy()
rw_list_out = pd.DataFrame({
    "doc": ent_doctors["doctor_id"].values,
    "doc_name": [doc_name_map[d] for d in ent_doctors["doctor_id"]],
    "fac_honin": ent_doctors["facility_id"].values,
    "fac_honin_name": [fac_honin_name_map[f] for f in ent_doctors["facility_id"]],
    "fac": [fac_map[f] for f in ent_doctors["facility_id"]],
    "fac_name": [fac_name_map[f] for f in ent_doctors["facility_id"]],
    "seg": [doc_rw_flag[d] for d in ent_doctors["doctor_id"]],
})
rw_list_out.to_csv(
    os.path.join(OUTPUT_DIR, "rw_list.csv"), index=False, encoding="utf-8-sig")

# facility_attribute.csv: 施設マスター + 施設内医師数 + 施設属性
# doctors DataFrame には複数施設所属の行も含まれるため，施設ごとに正確な医師数が集計される
facility_doc_counts = (
    doctors.groupby("facility_id")["doctor_id"]
    .nunique()
    .rename("施設内医師数")
    .reset_index()
)
facility_doc_counts_dict = dict(zip(facility_doc_counts["facility_id"], facility_doc_counts["施設内医師数"]))

uhp_options = ["UHP-A", "UHP-B", "UHP-C", "非UHP"]
management_orgs = [f"医療法人{chr(0x611B + i % 20)}{chr(0x548C + i % 15)}会" for i in range(N_FACILITIES)]
fac_kubun_map = {"病院": "病院", "クリニック": "診療所"}

def _gen_beds(fac_type):
    if fac_type == "病院":
        return int(np.random.choice([100, 200, 300, 500], p=[0.35, 0.35, 0.20, 0.10]))
    return int(np.random.choice([0, 5, 10, 19], p=[0.35, 0.30, 0.25, 0.10]))

facility_attribute_out = pd.DataFrame({
    "dcf_fac":        [fac_map[f] for f in facility_ids],
    "fac_honin":      facility_ids,
    "fac_honin_name": [fac_honin_name_map[f] for f in facility_ids],
    "施設区分名":     [fac_kubun_map[fac_type_map[f]] for f in facility_ids],
    "UHP区分名":      np.random.choice(uhp_options, N_FACILITIES, p=[0.20, 0.25, 0.25, 0.30]),
    "経営体名":       management_orgs,
    "許可病床数_合計": [_gen_beds(fac_type_map[f]) for f in facility_ids],
    "施設内医師数":   [facility_doc_counts_dict.get(f, 0) for f in facility_ids],
})
facility_attribute_out["施設内医師数"] = facility_attribute_out["施設内医師数"].astype(int)
facility_attribute_out["許可病床数_合計"] = facility_attribute_out["許可病床数_合計"].astype(int)
facility_attribute_out.to_csv(
    os.path.join(OUTPUT_DIR, "facility_attribute.csv"), index=False, encoding="utf-8-sig")

# doctor_attribute.csv: 医師属性マスター
def _get_channel_pref(doc_id):
    if doc_id not in doctor_info:
        return "なし"
    channels = list(doctor_info[doc_id]["channels"].keys())
    if len(channels) >= 2:
        return "マルチ"
    return channels[0] if channels else "なし"

def _get_doctor_segment(doc_id):
    if doc_rw_flag.get(doc_id, "") == "RW":
        return "Key Account" if doc_exp_years.get(doc_id, 0) > 20 else "Potential"
    return "Maintenance" if doc_product.get(doc_id, "") == "ENT" else "Non-Target"

_graduation_age = 24
doctor_attribute_out = pd.DataFrame({
    "doc":                      list(unique_doc_ids),
    "DCF医師コード":            [f"DCF{int(d[1:]):06d}" for d in unique_doc_ids],
    "doc_name":                 [doc_name_map[d] for d in unique_doc_ids],
    "年齢":                     [doc_exp_years[d] + _graduation_age for d in unique_doc_ids],
    "卒業時年齢":               [_graduation_age] * len(unique_doc_ids),
    "医師歴":                   [doc_exp_years[d] for d in unique_doc_ids],
    "DIGITAL_CHANNEL_PREFERENCE": [_get_channel_pref(d) for d in unique_doc_ids],
    "DOCTOR_SEGEMNT":           [_get_doctor_segment(d) for d in unique_doc_ids],
})
doctor_attribute_out.to_csv(
    os.path.join(OUTPUT_DIR, "doctor_attribute.csv"), index=False, encoding="utf-8-sig")

# sales.csv: 全品目 (実績を文字列に変換)
sales_out = delivery_df.copy()
sales_out["実績"] = sales_out["実績"].astype(str)
sales_out.to_csv(
    os.path.join(OUTPUT_DIR, "sales.csv"), index=False, encoding="utf-8-sig")

# デジタル視聴データ.csv: webiner + e_contents
digital_df = viewing_df[viewing_df["活動種別"].isin(["webiner", "e_contents"])].copy()
digital_cols = ["活動日_dt", "品目コード", "活動種別", "活動種別コード",
                "dcf_fac", "fac_honin", "fac", "dcf_doc", "doc", "doc_name"]
digital_df[digital_cols].to_csv(
    os.path.join(OUTPUT_DIR, "デジタル視聴データ.csv"), index=False, encoding="utf-8-sig")

# 活動データ.csv: Web講演会 + 面談 + 面談_アポ + 説明会 + その他 (活動種別コードが先)
activity_df = viewing_df[viewing_df["活動種別"].isin(
    ["Web講演会", "面談", "面談_アポ", "説明会", "その他"]
)].copy()
activity_cols = ["活動日_dt", "品目コード", "活動種別コード", "活動種別",
                 "dcf_fac", "fac_honin", "fac", "dcf_doc", "doc"]
activity_df[activity_cols].to_csv(
    os.path.join(OUTPUT_DIR, "活動データ.csv"), index=False, encoding="utf-8-sig")

# === 7. サマリー ===
multi_fac_set = set(multi_fac_doc_ids)
single_fac_ids = [f for f, n in zip(facility_ids, DOC_COUNTS) if n == 1]
clean_pairs = [
    f for f in single_fac_ids
    if doctors.loc[doctors["facility_id"] == f, "doctor_id"].values[0] not in multi_fac_set
]

print("=" * 60)
print(" サンプルデータ生成完了 (本番形式)")
print("=" * 60)
print(f"観測期間      : {months[0].strftime('%Y-%m')} ~ {months[-1].strftime('%Y-%m')} ({N_MONTHS}ヶ月)")
print(f"施設数        : {N_FACILITIES} (1医師:{N_SINGLE} / 2医師:{N_DOUBLE} / 3医師:{N_TRIPLE} / 4医師:{N_QUAD})")
print(f"医師数        : {N_DOCTORS} (うち複数施設所属: {N_MULTI_FAC_DOCS})")
print(f"全視聴医師    : {n_treated}")

print(f"\n視聴パターン別:")
for pat in patterns:
    n = sum(1 for info in doctor_info.values() if info["pattern"] == pat)
    print(f"  {pat:<12}: {n}名")

print(f"\n施設別月間売上回数:")
freq_dist = pd.Series([fac_delivery_freq[f] for f in facility_ids]).value_counts().sort_index()
for freq, cnt in freq_dist.items():
    print(f"  月{freq}回: {cnt}施設")

print(f"\nDGP処置効果の設計:")
print(f"  チャネル基本効果: {CHANNEL_EFFECTS}")
print(f"  月次成長: +{CHANNEL_GROWTH}/月, 停止後減衰: -{DECAY_RATE}/月 (猶予{GRACE_MONTHS}ヶ月)")
print(f"  modifier (乗算):")
for dim, mods in [("地域", REGION_MODIFIER), ("施設タイプ", TYPE_MODIFIER),
                  ("経験年数", EXP_MODIFIER), ("診療科", SPEC_MODIFIER)]:
    print(f"    {dim}: " + ", ".join(f"{k}={v}" for k, v in mods.items()))

mod_vals = list(doc_modifier.values())
print(f"  modifier: min={min(mod_vals):.3f}, mean={np.mean(mod_vals):.3f}, max={max(mod_vals):.3f}")

print(f"\n1施設1医師 AND 1医師1施設 : {len(clean_pairs)} 施設")
n_clean_treated = sum(
    1 for f in clean_pairs
    if doctors.loc[doctors["facility_id"] == f, "doctor_id"].values[0] in doctor_info
)
print(f"  うち視聴あり: {n_clean_treated}, 視聴なし: {len(clean_pairs) - n_clean_treated}")

n_ent_doctors = sum(1 for d in unique_doc_ids if doc_product[d] == "ENT")
n_rw_doctors = sum(1 for d in unique_doc_ids if doc_rw_flag[d] == "RW")

print(f"\n品目・RWフラグ:")
print(f"  ENT医師数     : {n_ent_doctors} / {N_DOCTORS} ({n_ent_doctors/N_DOCTORS*100:.1f}%)")
print(f"  RWフラグ医師数: {n_rw_doctors} / {n_ent_doctors} ENT医師 ({n_rw_doctors/n_ent_doctors*100:.1f}%)")

print(f"\n品目別売上行数:")
for prod_name in PRODUCTS:
    prod_code = PRODUCT_CODES[prod_name]
    n_rows = len(delivery_df[delivery_df["品目コード"] == prod_code])
    print(f"  {prod_name:<6} ({prod_code}): {n_rows:>8,} 行")

print(f"\n売上データ統計 (ENT品目):")
ent_delivery = delivery_df[delivery_df["品目コード"] == PRODUCT_CODES["ENT"]]
print(f"  1件あたり金額: 平均{ent_delivery['実績'].mean():.0f}, 中央値{ent_delivery['実績'].median():.0f}, "
      f"min={ent_delivery['実績'].min()}, max={ent_delivery['実績'].max()}")

print(f"\n活動種別マッピング:")
for cat, codes in ACTIVITY_TYPE_MAP.items():
    print(f"  {cat:<12}: {len(codes)} コード ({', '.join(codes)})")

n_digital_views = len(digital_df)
n_activity_views = len(activity_df)
n_web_lecture = len(activity_df[activity_df["活動種別"] == "Web講演会"])
n_mendan = len(activity_df[activity_df["活動種別"] == "面談"])
n_mendan_apo = len(activity_df[activity_df["活動種別"] == "面談_アポ"])
n_setsumeikai = len(activity_df[activity_df["活動種別"] == "説明会"])
n_sonota_views = len(activity_df[activity_df["活動種別"] == "その他"])
print(f"\n視聴・活動データ:")
print(f"  デジタル視聴データ: {n_digital_views:,} 行 (webiner + e_contents)")
print(f"  活動データ: {n_activity_views:,} 行")
print(f"    Web講演会: {n_web_lecture:,} / 面談: {n_mendan:,} / 面談_アポ: {n_mendan_apo:,} / 説明会: {n_setsumeikai:,} / その他: {n_sonota_views:,}")

print(f"\nfac/fac_honin 不一致:")
n_diff_fac = sum(1 for fid in facility_ids if fac_map[fid] != fid)
print(f"  {n_diff_fac} / {N_FACILITIES} 施設で fac != fac_honin")

print(f"\n出力ファイル:")
print(f"  rw_list.csv              : {len(rw_list_out):>8} 行 (ENT医師, seg=RWフラグ)")
print(f"  facility_attribute.csv   : {len(facility_attribute_out):>8} 行 (施設マスター+属性)")
print(f"  doctor_attribute.csv     : {len(doctor_attribute_out):>8} 行 (医師属性マスター)")
print(f"  sales.csv                : {len(delivery_df):>8,} 行 (日別施設別, 全品目)")
print(f"  デジタル視聴データ.csv    : {n_digital_views:>8,} 行 (webiner + e_contents)")
print(f"  活動データ.csv            : {n_activity_views:>8,} 行 (Web講演会 + 面談 + 面談_アポ + 説明会 + その他)")
print(f"\nカラム名:")
print(f"  rw_list            : doc, doc_name, fac_honin, fac_honin_name, fac, fac_name, seg")
print(f"  facility_attribute : dcf_fac, fac_honin, fac_honin_name, 施設区分名, UHP区分名, 経営体名, 許可病床数_合計, 施設内医師数")
print(f"  doctor_attribute   : doc, DCF医師コード, doc_name, 年齢, 卒業時年齢, 医師歴, DIGITAL_CHANNEL_PREFERENCE, DOCTOR_SEGEMNT")
print(f"  sales              : 日付, 施設（本院に合算）コード, DCF施設コード, 品目コード, 実績")
print(f"  デジタル視聴        : 活動日_dt, 品目コード, 活動種別, 活動種別コード, dcf_fac, fac_honin, fac, dcf_doc, doc, doc_name")
print(f"  活動データ          : 活動日_dt, 品目コード, 活動種別コード, 活動種別, dcf_fac, fac_honin, fac, dcf_doc, doc")
