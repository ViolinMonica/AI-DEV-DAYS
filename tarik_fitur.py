"""
Tarik fitur lahan untuk satu titik per kabupaten/kota (514), dengan pipeline
yang sama dengan BE: SoilGrids 5-15 cm, Open-Meteo ERA5 2022-2024, DEM Open-Meteo.

    python tarik_fitur.py

Titik = koordinat ibu kota kabupaten dari cahyadsn/wilayah (MIT, Kepmendagri
300.2.2-2138/2025). Paket data belum punya poligon berkode BPS maupun masker
lahan pertanian (butir A1, A2), jadi satu titik ini MEWAKILI kabupaten secara
kasar; itu keterbatasan yang ditulis di laporan.

Hasil ditambahkan ke data/fitur_kab.jsonl dan bisa dilanjutkan kalau terputus.
SoilGrids membatasi 5 permintaan/menit, jadi satu putaran penuh sekitar 2-3 jam.
Baris GAGAL di log ditarik lagi dengan menjalankan ulang script ini.
"""

import json
import os
import random
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from ml_lib.climate_features import monthly_from_daily
from ml_lib.soilgrids_adapter import usda_texture_class

SQL = os.path.join("data", "mentah", "wilayah_level_1_2.sql")
SQL_URL = "https://raw.githubusercontent.com/cahyadsn/wilayah/master/db/wilayah_level_1_2.sql"
OUT = os.path.join("data", "fitur_kab.jsonl")
# Batas wajar SoilGrids 5 permintaan/menit: 2 pekerja x 1 permintaan per 25 detik.
PEKERJA = 2
JEDA = 25.0

BARIS = re.compile(r"^\('(\d\d\.\d\d)','([^']*)','([^']*)',\s*([-\d.]+),\s*([-\d.]+),")


def get(url, coba=4):
    for i in range(coba):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.load(r)
        except Exception as e:      # jaringan/429/5xx: tunggu lalu ulang
            galat = e
            time.sleep(20 * (i + 1))
    raise galat


def daftar_kab():
    if not os.path.exists(SQL):
        os.makedirs(os.path.dirname(SQL), exist_ok=True)
        urllib.request.urlretrieve(SQL_URL, SQL)
    out = []
    with open(SQL, encoding="utf-8") as f:
        for line in f:
            m = BARIS.match(line)
            if m:
                out.append({"kode_kemendagri": m[1], "nama": m[2], "ibukota": m[3],
                            "lat": float(m[4]), "lon": float(m[5])})
    return out


def tanah(lat, lon):
    """SoilGrids 5-15 cm: rata-rata + kuantil pH. Geser titik kalau pikselnya kosong."""
    for dlat, dlon in ((0, 0), (0.03, 0), (0, 0.03), (-0.03, 0), (0, -0.03)):
        t0 = time.time()
        j = get("https://rest.isric.org/soilgrids/v2.0/properties/query"
                f"?lon={lon + dlon}&lat={lat + dlat}"
                "&property=phh2o&property=sand&property=silt&property=clay"
                "&property=soc&property=nitrogen&depth=5-15cm"
                "&value=mean&value=Q0.05&value=Q0.95")
        v = {}
        for layer in j["properties"]["layers"]:
            f = layer["unit_measure"]["d_factor"]
            for k, x in layer["depths"][0]["values"].items():
                v[f"{layer['name']}_{k}"] = None if x is None else x / f
        # tiap pekerja paling cepat 1 permintaan per JEDA detik
        time.sleep(max(0.0, JEDA - (time.time() - t0)))
        if v.get("phh2o_mean") is not None and v.get("sand_mean") is not None:
            return {
                "ph": v["phh2o_mean"], "ph_q05": v["phh2o_Q0.05"], "ph_q95": v["phh2o_Q0.95"],
                "sand_persen": v["sand_mean"], "silt_persen": v["silt_mean"],
                "clay_persen": v["clay_mean"],
                # soc SoilGrids dalam g/kg setelah dibagi d_factor
                "organic_carbon_gkg": v.get("soc_mean"), "nitrogen_gkg": v.get("nitrogen_mean"),
                "soil_texture": usda_texture_class(v["sand_mean"], v["silt_mean"], v["clay_mean"]),
                "geser_tanah": [dlat, dlon],
            }
    return None


def iklim(lat, lon):
    j = get("https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}&start_date=2022-01-01&end_date=2024-12-31"
            "&daily=precipitation_sum,temperature_2m_mean&timezone=auto")
    d = j["daily"]
    suhu = [t for t in d["temperature_2m_mean"] if t is not None]
    # hujan per tahun, untuk variasi antartahun (dipakai confidence)
    per_tahun = {}
    for t, v in zip(d["time"], d["precipitation_sum"]):
        if v is not None:
            per_tahun[t[:4]] = per_tahun.get(t[:4], 0.0) + v
    return {"monthly_rain": [round(x, 1) for x in monthly_from_daily(d["time"], d["precipitation_sum"])],
            "temp_c": round(sum(suhu) / len(suhu), 2),
            "hujan_per_tahun": {k: round(v) for k, v in per_tahun.items()},
            "elevation_m": j.get("elevation")}


def main():
    sudah = set()
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            sudah = {json.loads(l)["kode_kemendagri"] for l in f if l.strip()}
    kab = [k for k in daftar_kab() if k["kode_kemendagri"] not in sudah]
    # urutan acak: kalau terputus, yang sudah terkumpul tetap tersebar se-Indonesia
    random.Random(0).shuffle(kab)
    print(f"{len(sudah)} sudah ada, {len(kab)} akan ditarik", flush=True)
    kunci = threading.Lock()

    def kerja(k):
        try:
            t = tanah(k["lat"], k["lon"])
            k.update(iklim(k["lat"], k["lon"]))
            k.update(t or {"ph": None})
        except Exception as e:
            print("GAGAL", k["kode_kemendagri"], k["nama"], repr(e)[:120], flush=True)
            return
        with kunci:
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(k, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(PEKERJA) as ex:
        list(ex.map(kerja, kab))
    print("selesai", flush=True)


if __name__ == "__main__":
    main()
