"""Evaluasi "bisa hidup": di tempat tanaman TERBUKTI dibudidayakan (data panen BPS),
apakah sistem menyatakannya bisa hidup (pH, hujan, dan suhu di dalam batas toleransi)?

    python evaluasi_bisa_hidup.py

Ini mengukur sensitivitas saja. Luas panen nol bukan bukti tanaman tidak bisa
hidup, jadi tidak ada negatif sejati untuk menghitung spesifisitas.
Menulis data/cek_bisa_hidup.csv.
"""
import warnings

import numpy as np
import pandas as pd

import latih_kecocokan as L
from ml_lib.climate_features import derive
from ml_lib.rule_based_scorer import load_profiles, score_crop

warnings.filterwarnings("ignore")
profil = {p["crop_code"]: p for p in load_profiles()}
KUNCI = ("ph", "rainfall", "temp")


def nilai(land, kode):
    skor, b = score_crop(land, profil[kode])
    return skor, all(b[k] > 0 for k in KUNCI), {k: b[k] for k in KUNCI}


# ---- kabupaten: cabai rawit (2024), ubi kayu & ubi jalar (Sumbar 2024)
f = L.muat_fitur()
l = pd.read_excel(L.XLSX, sheet_name="1a Label Panen", header=4)
l = l[l["kode_bps_kab"].notna()]
l["kunci"] = l["nama_kabupaten_kota"].map(L.kunci_kab)
NAMA = {"Cabai rawit": "cabai_rawit", "Ubi kayu": "singkong", "Ubi jalar": "ubi_jalar"}
baris = []
for _, r in l[l["komoditas"].isin(NAMA)].merge(f, on="kunci").iterrows():
    skor, hidup, rinci = nilai(L.lahan(r), NAMA[r["komoditas"]])
    baris.append({"tingkat": "kabupaten", "tanaman": NAMA[r["komoditas"]], "luas": r["luas_panen_ha"],
                  "skor": skor, "hidup": hidup, **rinci})

# ---- provinsi: 5 pangan (rata-rata 2019-2023), terong & kacang panjang (2023-2024)
num = ["ph", "temp_c", "elevation_m"]
lahan_prov = {}
for prov, g in f.groupby("provinsi"):
    hujan = list(np.median(np.array(g["monthly_rain"].tolist()), axis=0))
    med = g[num].median()
    nama = prov.replace("Daerah Istimewa Yogyakarta", "DI Yogyakarta").replace(
        "Daerah Khusus Ibukota Jakarta", "DKI Jakarta")
    lahan_prov[nama] = {"ph": med["ph"], "temp_c": med["temp_c"], "elevation_m": med["elevation_m"],
                        "soil_texture": g["soil_texture"].mode().iloc[0],
                        "monthly_rain": hujan, **derive(hujan)}
p = pd.read_excel(L.XLSX, sheet_name="1a Prov 2019-2023", header=4)
p = p[p["indikator"] == "luas_panen"].copy()
p["provinsi"] = p["provinsi"].replace(L.ALIAS_PROV)
tahun = [c for c in p.columns if str(c)[:4] in ("2019", "2020", "2021", "2022", "2023")]
p["luas"] = p[tahun].mean(axis=1)
PROV = {v: k for k, v in L.KOMODITAS_PROV.items()}
q = pd.read_excel(L.XLSX, sheet_name="1a Provinsi", header=None)
hdr = next(i for i in range(8) if "komoditas" in q.iloc[i].astype(str).tolist())
q = pd.read_excel(L.XLSX, sheet_name="1a Provinsi", header=hdr)
q = q[(q["indikator"] == "luas_panen") & q["komoditas"].isin(["Terong", "Kacang panjang"])]
q = q.groupby(["provinsi", "komoditas"])["nilai"].mean().reset_index().rename(columns={"nilai": "luas"})
PROV.update({"Terong": "terong", "Kacang panjang": "kacang_panjang"})
for _, r in pd.concat([p[["provinsi", "komoditas", "luas"]], q]).iterrows():
    if r["komoditas"] in PROV and r["provinsi"] in lahan_prov and pd.notna(r["luas"]):
        skor, hidup, rinci = nilai(lahan_prov[r["provinsi"]], PROV[r["komoditas"]])
        baris.append({"tingkat": "provinsi", "tanaman": PROV[r["komoditas"]], "luas": r["luas"],
                      "skor": skor, "hidup": hidup, **rinci})

d = pd.DataFrame(baris)
d.to_csv("data/cek_bisa_hidup.csv", index=False)
print("pasangan wilayah x tanaman:", len(d))
for amb in (1, 10, 100):
    t = d[d["luas"] >= amb]
    print(f"\nluas panen >= {amb} ha: n={len(t)}  dinyatakan bisa hidup {t['hidup'].mean():.1%}  "
          f"skor>=50 {np.mean(t['skor'] >= 50):.1%}")
    print(t.groupby(["tingkat", "tanaman"]).agg(n=("hidup", "size"), bisa_hidup=("hidup", "mean"),
                                                skor50=("skor", lambda s: (s >= 50).mean())).round(2).to_string())
t = d[(d["luas"] >= 10) & ~d["hidup"]]
print("\npenyebab gagal (luas >= 10 ha):", {k: int((t[k] == 0).sum()) for k in KUNCI}, "dari", len(t))
z = d[d["luas"] <= 0]
print(f"luas panen 0 (bukan bukti tidak bisa hidup): n={len(z)} dinyatakan bisa hidup {z['hidup'].mean():.1%}")
