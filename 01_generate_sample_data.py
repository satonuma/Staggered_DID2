"""
===================================================================
サンプルデータ生成: Staggered DID分析用
デジタルコンテンツ視聴（Webinar, e-contents, web講演会）の効果検証
===================================================================
設計:
  - 納入データ: 日別・施設別 (doctor_idなし, 整数額)
  - 施設ごとに月間納入回数が異なる (月1~5回)
  - 1施設に複数医師所属あり / 1医師が複数施設所属あり
  - 視聴パターン: 未視聴 / wash-out視聴 / 単発 / 定常 / 離脱 / 遅延
  - 観測期間: 2023/4 ~ 2025/12 (33ヶ月)
  - CATE用属性: 地域/施設タイプ/経験年数/診療科
    → 処置効果に異質性あり (modifier で乗算)
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

CONTENT_TYPES = ["Webinar", "e-contents", "web講演会"]
CHANNEL_EFFECTS = {"Webinar": 18, "e-contents": 10, "web講演会": 22}
CHANNEL_GROWTH = 1.0    # 視聴継続中の月次成長
DECAY_RATE = 1.5        # 視聴停止後の月次減衰
GRACE_MONTHS = 2        # 視聴停止後の猶予期間

# チャネルマスタ: 細分化チャネル → 大分類へのマッピング
CHANNEL_MASTER_DATA = [
    {"channel_id": "CH01", "channel_name": "Web講演会 LIVE",        "channel_category": "web講演会"},
    {"channel_id": "CH02", "channel_name": "Web講演会 オンデマンド", "channel_category": "web講演会"},
    {"channel_id": "CH03", "channel_name": "Web講演会 アーカイブ",  "channel_category": "web講演会"},
    {"channel_id": "CH04", "channel_name": "Webinar 社内講演",      "channel_category": "Webinar"},
    {"channel_id": "CH05", "channel_name": "Webinar 学術講演",      "channel_category": "Webinar"},
    {"channel_id": "CH06", "channel_name": "Webinar セミナー",      "channel_category": "Webinar"},
    {"channel_id": "CH07", "channel_name": "e-contents 動画",       "channel_category": "e-contents"},
    {"channel_id": "CH08", "channel_name": "e-contents PDF資料",    "channel_category": "e-contents"},
    {"channel_id": "CH09", "channel_name": "e-contents インタラクティブ", "channel_category": "e-contents"},
    {"channel_id": "CH10", "channel_name": "メルマガ配信",          "channel_category": "その他"},
    {"channel_id": "CH11", "channel_name": "アンケート回答",        "channel_category": "その他"},
    {"channel_id": "CH12", "channel_name": "資材ダウンロード",      "channel_category": "その他"},
]
# 大分類 → 細分化チャネルIDの逆引き
CATEGORY_TO_CHANNELS = {}
for _ch in CHANNEL_MASTER_DATA:
    CATEGORY_TO_CHANNELS.setdefault(_ch["channel_category"], []).append(_ch["channel_id"])

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
for _fn in ["delivery_data.csv", "facility_master.csv", "rw_doctor_list.csv", "doctor_master.csv", "viewing_logs.csv", "channel_master.csv"]:
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

# === 1. 施設マスター ===
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

# === 2. 医師マスター ===
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
        doc_rw_flag[d] = "対象" if np.random.random() < RW_FLAG_RATE else ""
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

# === 4. 日別視聴ログ ===
viewing_records = []
for doc_id, info in doctor_info.items():
    fac_ids_for_doc = doctors.loc[doctors["doctor_id"] == doc_id, "facility_id"].values
    primary_fac = fac_ids_for_doc[0]

    for ch, (first_m, last_m) in info["channels"].items():
        first_date = months[first_m]
        first_offset = np.random.randint(0, min(28, first_date.days_in_month))
        viewing_records.append({
            "view_date": (first_date + pd.Timedelta(days=first_offset)).strftime("%Y-%m-%d"),
            "doctor_id": doc_id, "facility_id": primary_fac,
            "channel_id": np.random.choice(CATEGORY_TO_CHANNELS[ch]),
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
                        "view_date": vd.strftime("%Y-%m-%d"),
                        "doctor_id": doc_id, "facility_id": primary_fac,
                        "channel_id": np.random.choice(CATEGORY_TO_CHANNELS[ch]),
                    })

# その他チャネルの視聴記録を追加（処置効果なし）
sonota_channels = CATEGORY_TO_CHANNELS["その他"]
n_sonota_viewers = int(n_treated * 0.25)
sonota_viewer_ids = np.random.choice(treated_ids, n_sonota_viewers, replace=False)
for doc_id in sonota_viewer_ids:
    primary_fac = primary_fac_map[doc_id]
    n_views = np.random.randint(1, 6)
    for _ in range(n_views):
        view_month = np.random.randint(0, N_MONTHS)
        month_start = months[view_month]
        day_offset = np.random.randint(0, month_start.days_in_month)
        vd = month_start + pd.Timedelta(days=day_offset)
        viewing_records.append({
            "view_date": vd.strftime("%Y-%m-%d"),
            "doctor_id": doc_id,
            "facility_id": primary_fac,
            "channel_id": np.random.choice(sonota_channels),
        })

viewing_df = pd.DataFrame(viewing_records)

# === 5. 日別施設別納入データ ===
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
            # 均等間隔 + ランダムジッター
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
                    "delivery_date": (month_start + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d"),
                    "facility_id": fac_id,
                    "品目": "ENT",
                    "amount": a,
                })

# 他品目(CNS, GI, CV, その他)の納入レコードを簡易生成
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
                        "delivery_date": (month_start + pd.Timedelta(days=int(day))).strftime("%Y-%m-%d"),
                        "facility_id": fac_id,
                        "品目": prod,
                        "amount": amt,
                    })

delivery_df = pd.DataFrame(daily_records)

# === 6. CSV保存 ===
doctors["品目"] = doctors["doctor_id"].map(doc_product)
doctors["rw_flag"] = doctors["doctor_id"].map(doc_rw_flag)
doctors["experience_years"] = doctors["doctor_id"].map(doc_exp_years)
doctors["experience_cat"] = doctors["doctor_id"].map(doc_exp_cat)
doctors["specialty"] = doctors["doctor_id"].map(doc_specialties)

# RW医師リスト (施設-医師-品目-RWフラグのみ)
doctors[["doctor_id", "facility_id", "品目", "rw_flag"]].to_csv(
    os.path.join(OUTPUT_DIR, "rw_doctor_list.csv"), index=False, encoding="utf-8-sig")

# 医師マスタ (属性情報, 1医師1行)
doctor_master_df = doctors.drop_duplicates(subset="doctor_id", keep="first")
doctor_master_df[["doctor_id", "experience_years", "experience_cat", "specialty"]].to_csv(
    os.path.join(OUTPUT_DIR, "doctor_master.csv"), index=False, encoding="utf-8-sig")

# 施設マスタ (monthly_delivery_freqは除外 — 納入データから算出する)
facilities[["facility_id", "facility_name", "region", "facility_type"]].to_csv(
    os.path.join(OUTPUT_DIR, "facility_master.csv"), index=False, encoding="utf-8-sig")

# チャネルマスタ
pd.DataFrame(CHANNEL_MASTER_DATA).to_csv(
    os.path.join(OUTPUT_DIR, "channel_master.csv"), index=False, encoding="utf-8-sig")

delivery_df.to_csv(
    os.path.join(OUTPUT_DIR, "delivery_data.csv"), index=False, encoding="utf-8-sig")
viewing_df.to_csv(
    os.path.join(OUTPUT_DIR, "viewing_logs.csv"), index=False, encoding="utf-8-sig")

# === 7. サマリー ===
multi_fac_set = set(multi_fac_doc_ids)
single_fac_ids = [f for f, n in zip(facility_ids, DOC_COUNTS) if n == 1]
clean_pairs = [
    f for f in single_fac_ids
    if doctors.loc[doctors["facility_id"] == f, "doctor_id"].values[0] not in multi_fac_set
]

print("=" * 60)
print(" サンプルデータ生成完了")
print("=" * 60)
print(f"観測期間      : {months[0].strftime('%Y-%m')} ~ {months[-1].strftime('%Y-%m')} ({N_MONTHS}ヶ月)")
print(f"施設数        : {N_FACILITIES} (1医師:{N_SINGLE} / 2医師:{N_DOUBLE} / 3医師:{N_TRIPLE} / 4医師:{N_QUAD})")
print(f"医師数        : {N_DOCTORS} (うち複数施設所属: {N_MULTI_FAC_DOCS})")
print(f"全視聴医師    : {n_treated}")

print(f"\n視聴パターン別:")
for pat in patterns:
    n = sum(1 for info in doctor_info.values() if info["pattern"] == pat)
    print(f"  {pat:<12}: {n}名")

print(f"\n施設別月間納入回数:")
freq_dist = pd.Series([fac_delivery_freq[f] for f in facility_ids]).value_counts().sort_index()
for freq, cnt in freq_dist.items():
    print(f"  月{freq}回: {cnt}施設")

print(f"\nCATE用属性:")
for attr, vals in [("region", list(regions)), ("facility_type", list(fac_types))]:
    vc = pd.Series(vals).value_counts()
    print(f"  {attr}: " + ", ".join(f"{k}={v}" for k, v in vc.items()))
for attr, mapping in [("experience_cat", doc_exp_cat), ("specialty", doc_specialties)]:
    vc = pd.Series(list(mapping.values())).value_counts()
    print(f"  {attr}: " + ", ".join(f"{k}={v}" for k, v in vc.items()))
mod_vals = list(doc_modifier.values())
print(f"  modifier: min={min(mod_vals):.3f}, mean={np.mean(mod_vals):.3f}, max={max(mod_vals):.3f}")

print(f"\nDGP処置効果の設計:")
print(f"  チャネル基本効果: {CHANNEL_EFFECTS}")
print(f"  月次成長: +{CHANNEL_GROWTH}/月, 停止後減衰: -{DECAY_RATE}/月 (猶予{GRACE_MONTHS}ヶ月)")
print(f"  modifier (乗算):")
for dim, mods in [("地域", REGION_MODIFIER), ("施設タイプ", TYPE_MODIFIER),
                  ("経験年数", EXP_MODIFIER), ("診療科", SPEC_MODIFIER)]:
    print(f"    {dim}: " + ", ".join(f"{k}={v}" for k, v in mods.items()))

print(f"\n1施設1医師 AND 1医師1施設 : {len(clean_pairs)} 施設")
n_clean_treated = sum(
    1 for f in clean_pairs
    if doctors.loc[doctors["facility_id"] == f, "doctor_id"].values[0] in doctor_info
)
print(f"  うち視聴あり: {n_clean_treated}, 視聴なし: {len(clean_pairs) - n_clean_treated}")

n_ent_doctors = sum(1 for d in unique_doc_ids if doc_product[d] == "ENT")
n_rw_doctors = sum(1 for d in unique_doc_ids if doc_rw_flag[d] == "対象")

print(f"\n品目・RWフラグ:")
print(f"  ENT医師数     : {n_ent_doctors} / {N_DOCTORS} ({n_ent_doctors/N_DOCTORS*100:.1f}%)")
print(f"  RWフラグ医師数: {n_rw_doctors} / {n_ent_doctors} ENT医師 ({n_rw_doctors/n_ent_doctors*100:.1f}%)")

print(f"\n品目別納入行数:")
for prod in PRODUCTS:
    n_rows = len(delivery_df[delivery_df["品目"] == prod])
    print(f"  {prod:<6}: {n_rows:>8,} 行")

print(f"\n納入データ統計 (ENT品目):")
ent_delivery = delivery_df[delivery_df["品目"] == "ENT"]
print(f"  1件あたり金額: 平均{ent_delivery['amount'].mean():.0f}, 中央値{ent_delivery['amount'].median():.0f}, "
      f"min={ent_delivery['amount'].min()}, max={ent_delivery['amount'].max()}")

print(f"\nチャネルマスタ:")
print(f"  大分類 → 細分化チャネル数:")
for cat, chs in CATEGORY_TO_CHANNELS.items():
    print(f"    {cat:<12}: {len(chs)} チャネル ({', '.join(chs)})")

n_sonota_views = len(viewing_df[viewing_df["channel_id"].isin(CATEGORY_TO_CHANNELS["その他"])])
print(f"  その他チャネル視聴レコード: {n_sonota_views} 行")

print(f"\n出力ファイル:")
print(f"  delivery_data.csv   : {len(delivery_df):>8,} 行 (日別施設別, 品目付き)")
print(f"  viewing_logs.csv    : {len(viewing_df):>8,} 行 (日別医師別, channel_id付き)")
print(f"  rw_doctor_list.csv  : {len(doctors):>8} 行 (施設-医師-品目-RWフラグのみ)")
print(f"  doctor_master.csv   : {len(doctor_master_df):>8} 行 (医師属性)")
print(f"  facility_master.csv : {N_FACILITIES:>8} 行 (施設属性)")
print(f"  channel_master.csv  : {len(CHANNEL_MASTER_DATA):>8} 行 (チャネルマスタ)")
print(f"\nカラム名:")
print(f"  delivery_data  : delivery_date, facility_id, 品目, amount")
print(f"  viewing_logs   : view_date, doctor_id, facility_id, channel_id")
print(f"  rw_doctor_list : doctor_id, facility_id, 品目, rw_flag")
print(f"  doctor_master  : doctor_id, experience_years, experience_cat, specialty")
print(f"  facility_master: facility_id, facility_name, region, facility_type")
print(f"  channel_master : channel_id, channel_name, channel_category")
