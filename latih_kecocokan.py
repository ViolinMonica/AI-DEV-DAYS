"""
Latih dan evaluasi model kecocokan (T2), validasi independen (T4).

    python tarik_fitur.py          # sekali, sekitar 3 jam
    python latih_kecocokan.py [path/ke/NUSA_CORP - DATA.xlsx]

Label TIDAK memakai skor_aturan. Satu-satunya komoditas proposal yang punya
data panen tingkat kabupaten se-Indonesia adalah cabai rawit (2024), jadi hanya
cabai_rawit yang punya model ML. Sembilan tanaman lain memakai skor_aturan yang
dipetakan ke skala probabilitas yang sama (isotonic, dipelajari dari cabai rawit).

Menulis:
  data/kecocokan_latih.csv           fitur + label per kabupaten
  ml_lib/model_kecocokan.joblib      model + kalibrasi (dibaca NusaCropModel)
  laporan/kecocokan_hasil.json       semua angka evaluasi
"""

import json
import os
import re
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ml_lib import ARTIFACT_DIR
from ml_lib.climate_features import derive
from ml_lib.rule_based_scorer import load_profiles, score_all_crops, score_crop

XLSX = sys.argv[1] if len(sys.argv) > 1 else os.path.join("..", "NUSA_CORP - DATA.xlsx")
FITUR_KAB = os.path.join("data", "fitur_kab.jsonl")
SQL = os.path.join("data", "mentah", "wilayah_level_1_2.sql")

# Hanya fitur yang tersedia di input recommend(). NDVI sengaja tidak dipakai.
FITUR = ["ph", "temp_c", "elevation_m", "rainfall_mm", "rain_wet3", "rain_dry3",
         "bulan_kering", "sand_persen", "clay_persen"]
LUAS_MIN_HA = 100      # luas panen cabai + padi minimal supaya share bermakna
N_LIPATAN = 5
ACAK = 42

# Nama di berkas BPS yang ejaannya berbeda dari Kepmendagri.
ALIAS = {"KOTA:PADANGSIDEMPUAN": "KOTA:PADANGSIDIMPUAN",
         "KAB:OGANKOMERING": "KAB:OGANKOMERINGILIR"}
ALIAS_PROV = {"DI zogzakarta": "DI Yogyakarta", "Papua Barat Daza": "Papua Barat Daya",
              "Daerah Istimewa Yogyakarta": "DI Yogyakarta"}

# crop_code -> nama komoditas di sheet provinsi 2019-2023
KOMODITAS_PROV = {"singkong": "Ubi kayu", "ubi_jalar": "Ubi jalar", "jagung": "Jagung",
                  "kacang_tanah": "Kacang tanah", "kacang_hijau": "Kacang hijau"}


def kunci_kab(nama):
    s = nama.upper()
    for a, b in (("KABUPATEN", "KAB."), ("ADMINISTRASI", ""), ("ADM.", ""), ("KEP.", "KEPULAUAN")):
        s = s.replace(a, b)
    s = re.sub(r"[^A-Z. ]", "", s).strip()
    jenis = "KOTA" if s.startswith("KOTA ") else "KAB"
    k = jenis + ":" + re.sub(r"^(KAB\.?|KOTA)\s+", "", s).replace(" ", "").replace(".", "")
    return ALIAS.get(k, k)


def nama_provinsi():
    out = {}
    with open(SQL, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"^\('(\d\d)','([^']*)'", line)
            if m:
                out[m[1]] = m[2]
    return out


def luas_wilayah():
    """kode Kemendagri -> luas wilayah (km2) dari tabel wilayah."""
    pola = re.compile(r"^\('(\d\d\.\d\d)','[^']*','[^']*',\s*[-\d.]+,\s*[-\d.]+,"
                      r"\s*[-\d.]+,\s*\d+,\s*([\d.]+),")
    with open(SQL, encoding="utf-8") as f:
        return {m[1]: float(m[2]) for m in map(pola.match, f) if m}


def muat_fitur():
    with open(FITUR_KAB, encoding="utf-8") as f:
        d = pd.DataFrame([json.loads(l) for l in f if l.strip()])
    d = d[d["ph"].notna()].reset_index(drop=True)
    musim = pd.DataFrame([derive(m) for m in d["monthly_rain"]])
    d = pd.concat([d, musim], axis=1)
    prov = nama_provinsi()
    d["provinsi"] = d["kode_kemendagri"].str[:2].map(prov)
    d["kunci"] = d["nama"].map(kunci_kab)
    d["luas_km2"] = d["kode_kemendagri"].map(luas_wilayah())
    return d


def tambah_label_cabai(d):
    l = pd.read_excel(XLSX, sheet_name="1a Label Panen", header=4)
    l = l[l["kode_bps_kab"].notna() & (l["status_baris"] != "CONTOH")]
    l["kunci"] = l["nama_kabupaten_kota"].map(kunci_kab)
    luas = l.pivot_table(index="kunci", columns="komoditas", values="luas_panen_ha", aggfunc="first")
    prod = l[l["komoditas"] == "Cabai rawit"].set_index("kunci")["produksi_ton"]
    d = d.join(luas[["Cabai rawit", "Padi (sawah+ladang)"]].rename(
        columns={"Cabai rawit": "luas_cabai", "Padi (sawah+ladang)": "luas_padi"}), on="kunci")
    d = d.join(prod.rename("produksi_cabai"), on="kunci")
    d = d[d["luas_cabai"].notna()].copy()
    # L1 (Definisi_Label), disesuaikan: pembagi hortikultura belum ada di paket
    # data, jadi dipakai cabai + padi (satu-satunya dua komoditas tingkat kabupaten).
    # Wilayah dengan luas panen sangat kecil (kota) dibuang: share-nya ditentukan
    # beberapa hektare saja dan melompat ke 0 atau 1.
    total = d["luas_cabai"] + d["luas_padi"].fillna(0)
    d = d[total >= LUAS_MIN_HA].copy()
    d["share_cabai"] = d["luas_cabai"] / total[total >= LUAS_MIN_HA]
    # L4: hadir = share di atas median nasional (ambang dari distribusi, bukan tebakan)
    d["hadir"] = (d["share_cabai"] >= d["share_cabai"].median()).astype(int)
    # Label pembanding: intensitas = ha cabai rawit per km2 wilayah, di atas median
    d["intensitas_cabai"] = d["luas_cabai"] / d["luas_km2"]
    d["hadir_intensitas"] = (d["intensitas_cabai"] >= d["intensitas_cabai"].median()).astype(int)
    d["yield_cabai"] = d["produksi_cabai"] / d["luas_cabai"].where(d["luas_cabai"] > 0)
    return d.reset_index(drop=True)


def lahan(r):
    return {"ph": r["ph"], "temp_c": r["temp_c"], "elevation_m": r["elevation_m"],
            "soil_texture": r["soil_texture"], "monthly_rain": r["monthly_rain"],
            **derive(r["monthly_rain"])}


# ---------------------------------------------------------------- model

def buat_lr():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=1000, class_weight="balanced"))


def buat_xgb(monoton_ph):
    # pH dikunci tidak-menurun: simulasi kapur menilai ulang tanaman dengan pH
    # lebih tinggi, dan cabai rawit peka aluminium (acid_tolerance rendah).
    kunci = tuple(1 if f == "ph" and monoton_ph else 0 for f in FITUR)
    return XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                         monotone_constraints=kunci, random_state=ACAK, n_jobs=1)


def terkalibrasi(dasar):
    # kalibrasi di dalam lipatan latih (CV dalam), jadi lipatan uji tidak bocor
    return CalibratedClassifierCV(dasar, method="isotonic", cv=3)


KANDIDAT = {
    "Logistic regression": lambda: terkalibrasi(buat_lr()),
    "XGBoost": lambda: terkalibrasi(buat_xgb(False)),
    "XGBoost, pH monoton": lambda: terkalibrasi(buat_xgb(True)),
}


def metrik(y, p, share):
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "spearman_share": float(spearmanr(p, share, nan_policy="omit").statistic)}


def cv_spasial(d, X, y, skor_aturan, kol_share="share_cabai"):
    """GroupKFold per provinsi. Semua pembanding dinilai pada lipatan yang sama."""
    rng = np.random.default_rng(ACAK)
    pred = {n: np.zeros(len(d)) for n in list(KANDIDAT) + ["Aturan (skor_aturan)", "Acak"]}
    lipatan = np.zeros(len(d), dtype=int)
    for k, (tr, te) in enumerate(GroupKFold(N_LIPATAN).split(X, y, d["provinsi"])):
        lipatan[te] = k
        for n, buat in KANDIDAT.items():
            pred[n][te] = buat().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(skor_aturan[tr], y[tr])
        pred["Aturan (skor_aturan)"][te] = iso.predict(skor_aturan[te])
        pred["Acak"][te] = rng.random(len(te))
    tabel = {}
    for n, p in pred.items():
        per = [metrik(y[lipatan == k], p[lipatan == k], d[kol_share][lipatan == k])
               for k in range(N_LIPATAN)]
        tabel[n] = {m: float(np.mean([x[m] for x in per])) for m in per[0]}
        tabel[n]["roc_auc_sd"] = float(np.std([x["roc_auc"] for x in per]))
        if n == "Aturan (skor_aturan)":
            # AUC dan korelasi aturan dihitung dari skor mentah; isotonic hanya untuk Brier
            tabel[n]["roc_auc"] = float(np.mean(
                [roc_auc_score(y[lipatan == k], skor_aturan[lipatan == k]) for k in range(N_LIPATAN)]))
            tabel[n]["spearman_share"] = float(np.mean(
                [spearmanr(skor_aturan[lipatan == k], d[kol_share][lipatan == k]).statistic
                 for k in range(N_LIPATAN)]))
    return tabel, pred


# ---------------------------------------------------------------- validasi provinsi

def label_provinsi():
    """L1: rata-rata 2019-2023 dari luas panen komoditas / jumlah luas panen 6 komoditas."""
    p = pd.read_excel(XLSX, sheet_name="1a Prov 2019-2023", header=4)
    p = p[(p["indikator"] == "luas_panen") & (p["status_baris"] != "CONTOH")].copy()
    p["provinsi"] = p["provinsi"].replace(ALIAS_PROV)
    tahun = [c for c in p.columns if str(c)[:4] in ("2019", "2020", "2021", "2022", "2023")]
    panjang = p.melt(id_vars=["provinsi", "komoditas"], value_vars=tahun, value_name="luas").dropna()
    panjang["total"] = panjang.groupby(["provinsi", "variable"])["luas"].transform("sum")
    panjang = panjang[panjang["total"] > 0]
    panjang["share"] = panjang["luas"] / panjang["total"]
    return panjang.groupby(["provinsi", "komoditas"])["share"].mean().unstack()


def validasi_provinsi(fitur, nilai):
    """Per komoditas pangan: korelasi peringkat skor vs share antarprovinsi, dan
    Top-3 di 5 provinsi sentra (share tertinggi). `nilai(land)` -> {crop_code: skor}."""
    share = label_provinsi()
    num = ["ph", "temp_c", "elevation_m", "sand_persen", "silt_persen", "clay_persen"]
    baris = {}
    for prov, g in fitur.groupby("provinsi"):
        med = g[num].median()
        hujan = np.median(np.array(g["monthly_rain"].tolist()), axis=0)
        tekstur = g["soil_texture"].mode().iloc[0]
        land = {"ph": med["ph"], "temp_c": med["temp_c"], "elevation_m": med["elevation_m"],
                "soil_texture": tekstur, "monthly_rain": list(hujan), **derive(list(hujan)),
                "sand_persen": med["sand_persen"], "clay_persen": med["clay_persen"]}
        baris[prov.replace("Daerah Istimewa Yogyakarta", "DI Yogyakarta")
                  .replace("Daerah Khusus Ibukota Jakarta", "DKI Jakarta")] = nilai(land)
    skor = pd.DataFrame(baris).T
    sama = skor.index.intersection(share.index)
    out = {"n_provinsi": int(len(sama)), "per_tanaman": {}}
    kena, total = 0, 0
    for kode, nama in KOMODITAS_PROV.items():
        s = share.loc[sama, nama].dropna()
        rho = spearmanr(skor.loc[s.index, kode], s).statistic
        sentra = s.nlargest(5).index
        top3 = int(sum(kode in skor.loc[p].nlargest(3).index for p in sentra))
        kena, total = kena + top3, total + len(sentra)
        out["per_tanaman"][kode] = {"spearman": float(rho), "top3_di_5_sentra": top3}
    out["top3_sentra"] = kena / total
    out["top3_acak"] = 3 / skor.shape[1]
    out["provinsi_tak_cocok"] = sorted(set(skor.index) ^ set(share.index))
    return out


# ---------------------------------------------------------------- utama

def main():
    os.makedirs("laporan", exist_ok=True)
    profil = load_profiles()
    cabai = next(p for p in profil if p["crop_code"] == "cabai_rawit")

    fitur = muat_fitur()
    d = tambah_label_cabai(fitur)
    lands = [lahan(r) for _, r in d.iterrows()]
    d["skor_aturan_cabai"] = [score_crop(l, cabai)[0] for l in lands]
    X, y = d[FITUR].to_numpy(dtype=float), d["hadir"].to_numpy()
    d.drop(columns=["monthly_rain", "hujan_per_tahun", "geser_tanah"]).to_csv(
        "data/kecocokan_latih.csv", index=False)
    print(f"{len(fitur)} titik kabupaten berfitur; {len(d)} berlabel cabai rawit; "
          f"{d['provinsi'].nunique()} provinsi; hadir={y.mean():.2f}; "
          f"median share={d['share_cabai'].median():.4f}")

    tabel, pred = cv_spasial(d, X, y, d["skor_aturan_cabai"].to_numpy())
    print(pd.DataFrame(tabel).T.round(3).to_string(), "\n")
    # kepekaan terhadap definisi label
    tabel_int, _ = cv_spasial(d, X, d["hadir_intensitas"].to_numpy(),
                              d["skor_aturan_cabai"].to_numpy(), "intensitas_cabai")
    print("label pembanding (intensitas per km2):")
    print(pd.DataFrame(tabel_int).T.round(3).to_string(), "\n")

    # Yang dipasang ke aplikasi harus lulus uji kewajaran pH (simulasi kapur
    # menilai ulang tanaman dengan pH baru), jadi dipakai varian pH monoton.
    auc_terbaik = max(KANDIDAT, key=lambda n: tabel[n]["roc_auc"])
    terbaik = "XGBoost, pH monoton"
    p_acuan = float(y.mean())

    # Skor hibrida (yang dipakai aplikasi): skor_aturan x koreksi_ml(p). Dinilai
    # dengan prediksi luar-lipatan, jadi tidak ada kabupaten yang menilai dirinya.
    from ml_lib.predict import koreksi_ml
    aturan_cabai = d["skor_aturan_cabai"].to_numpy()
    hibrida = np.minimum(100, aturan_cabai * koreksi_ml(pred[terbaik], p_acuan))
    lipatan = np.zeros(len(d), dtype=int)
    for k, (_, te) in enumerate(GroupKFold(N_LIPATAN).split(X, y, d["provinsi"])):
        lipatan[te] = k

    def auc_lipatan(skor, label):
        return float(np.mean([roc_auc_score(label[lipatan == k], skor[lipatan == k])
                              for k in range(N_LIPATAN)]))

    y_int = d["hadir_intensitas"].to_numpy()
    hibrida_cv = {
        "roc_auc_share": {"aturan": auc_lipatan(aturan_cabai, y), "hibrida": auc_lipatan(hibrida, y)},
        "roc_auc_intensitas": {"aturan": auc_lipatan(aturan_cabai, y_int),
                               "hibrida": auc_lipatan(hibrida, y_int)},
        "spearman_share": {
            "aturan": float(spearmanr(aturan_cabai, d["share_cabai"]).statistic),
            "hibrida": float(spearmanr(hibrida, d["share_cabai"]).statistic)},
    }
    print("cabai rawit, aturan vs hibrida:", json.dumps(hibrida_cv))

    # Top-K: di kabupaten sentra cabai rawit (desil teratas share), apakah
    # cabai_rawit masuk 3 besar dari 10 tanaman?
    sentra = (d["share_cabai"] >= d["share_cabai"].quantile(0.9)).to_numpy()
    kena_aturan, kena_hibrida = [], []
    for l, s, h in zip(lands, sentra, hibrida):
        if not s:
            continue
        skor = {r["crop_code"]: r["score"] for r in score_all_crops(l, profil)}
        kena_aturan.append("cabai_rawit" in sorted(skor, key=skor.get, reverse=True)[:3])
        skor["cabai_rawit"] = h
        kena_hibrida.append("cabai_rawit" in sorted(skor, key=skor.get, reverse=True)[:3])
    topk_cabai = {"n_sentra": int(sentra.sum()), "top3_aturan": float(np.mean(kena_aturan)),
                  "top3_hibrida": float(np.mean(kena_hibrida)), "top3_acak": 0.3}
    print("Top-3 cabai rawit di kabupaten sentra:", topk_cabai)

    # model akhir: dilatih di semua data
    model = KANDIDAT[terbaik]().fit(X, y)
    paket = {"fitur": FITUR, "model": {"cabai_rawit": model}, "p_acuan": p_acuan,
             "rentang": [(float(np.nanpercentile(X[:, j], 1)), float(np.nanpercentile(X[:, j], 99)))
                         for j in range(len(FITUR))],
             "model_terpilih": terbaik}
    joblib.dump(paket, os.path.join(ARTIFACT_DIR, "model_kecocokan.joblib"))
    print("AUC terbaik:", auc_terbaik, "| model terpasang:", terbaik)

    # Uji kewajaran pH pada model akhir: titik pH <= 5, naikkan ke 5,0/5,5/6,0
    turun = 0
    masam = d[d["ph"] <= 5.0]
    for _, r in masam.iterrows():
        x = np.repeat(r[FITUR].to_numpy(dtype=float)[None], 4, axis=0)
        x[:, FITUR.index("ph")] = [4.5, 5.0, 5.5, 6.0]
        p = model.predict_proba(x)[:, 1]
        # toleransi 1e-6: kalibrasi isotonic memberi derau float32 sekitar 1e-8
        turun += bool((np.diff(p) < -1e-6).any())
    uji_ph = {"n_titik": int(len(masam)), "n_skor_turun": int(turun)}
    print("uji kewajaran pH (cabai_rawit ML):", uji_ph)

    # Validasi independen tingkat provinsi (5 komoditas pangan)
    # saklar bawaan mati, jadi dinyalakan di sini hanya untuk evaluasi skor_akhir
    import ml_lib.predict as predict
    predict.PAKAI_SKOR_ML = True
    m = predict.NusaCropModel()
    assert m.ml is not None, "model_kecocokan.joblib gagal dimuat"

    def nilai_aturan(land):
        return {r["crop_code"]: r["score"] for r in score_all_crops(land, profil)}

    def nilai_akhir(land):
        return dict(zip([p["crop_code"] for p in profil], m._skor_peringkat([land], None)[0]))

    val = {"aturan": validasi_provinsi(fitur, nilai_aturan),
           "skor_akhir": validasi_provinsi(fitur, nilai_akhir)}
    for n, v in val.items():
        print(f"validasi provinsi [{n}]: Top-3 sentra {v['top3_sentra']:.2f} "
              f"(acak {v['top3_acak']:.2f}), n={v['n_provinsi']}")
        print("  ", {k: (round(x["spearman"], 2), x["top3_di_5_sentra"])
                     for k, x in v["per_tanaman"].items()})

    # T4: lebar ketidakpastian pH SoilGrids dan variasi hujan (untuk confidence)
    sd_ph = float(((fitur["ph_q95"] - fitur["ph_q05"]) / 3.29).median())
    cv = fitur["hujan_per_tahun"].map(lambda h: np.std(list(h.values())) / np.mean(list(h.values())))
    print(f"SoilGrids: median sd pH = {sd_ph:.2f}; median CV hujan tahunan = {cv.median():.3f}")

    hasil = {"n_titik_fitur": int(len(fitur)), "n_berlabel": int(len(d)),
             "n_provinsi": int(d["provinsi"].nunique()),
             "median_share": float(d["share_cabai"].median()),
             "cv_spasial": tabel, "cv_spasial_label_intensitas": tabel_int,
             "auc_terbaik": auc_terbaik,
             "topk_cabai": topk_cabai, "model_terpilih": terbaik,
             "hibrida_cv": hibrida_cv, "p_acuan": p_acuan,
             "uji_ph": uji_ph, "validasi_provinsi": val,
             "sd_ph_soilgrids_median": sd_ph, "cv_hujan_median": float(cv.median()),
             "rentang_fitur": dict(zip(FITUR, paket["rentang"]))}
    with open("laporan/kecocokan_hasil.json", "w", encoding="utf-8") as f:
        json.dump(hasil, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
