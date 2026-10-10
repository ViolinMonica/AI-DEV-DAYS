"""
Perbarui profil syarat tumbuh di seed.json dari sumber yang bisa dilacak.

    python perbarui_profil.py

Aturan pengisian, berurutan:
  1. Sheet `3 Param Tanaman` di NUSA_CORP - DATA.xlsx kalau kolomnya terisi
     (sumber Indonesia per baris; lihat sheet `Sumber`).
  2. Basis data FAO EcoCrop (EcoCrop_DB.csv, salinan OpenCLIM/ecocrop di
     GitHub) untuk suhu, pH, dan ketinggian maksimal.
  3. Selain itu nilai lama seed.json dibiarkan.

Yang SENGAJA tidak diambil dari EcoCrop:
  - Curah hujan. Angka EcoCrop adalah hujan tahunan, sedangkan scorer memakai
    rainfall_* sebagai kebutuhan air satu siklus tanam. Hanya kacang hijau
    (angka per bulan dari Ditjen Tanaman Pangan) dan kemiri (tanaman tahunan)
    yang diubah.
  - Ketinggian maksimal terong: EcoCrop mencatat 50 m, tidak masuk akal.

seed.json lama disimpan di arsip_model_lama/seed_lama.json.
"""

import json
import os
import shutil

SEED = os.path.join("ml_lib", "seed.json")
LAMA = os.path.join("arsip_model_lama", "seed_lama.json")

ECOCROP = "FAO EcoCrop (EcoCrop_DB.csv)"


def suhu(lo, opt_lo, opt_hi, hi):
    return {"temp_min": lo, "temp_optimal_min": opt_lo, "temp_optimal_max": opt_hi, "temp_max": hi}


def ph(lo, opt_lo, opt_hi, hi):
    return {"ph_min": lo, "ph_optimal_min": opt_lo, "ph_optimal_max": opt_hi, "ph_max": hi}


# crop_code -> (perubahan, sumber). Komentar menyebut asal tiap angka.
PROFIL = {
    "singkong": ({
        **suhu(10, 20, 29, 35),          # EcoCrop Manihot esculenta (= sheet 3 Param Tanaman)
        **ph(4.0, 5.5, 8.0, 9.0),        # idem
        "elevation_min": 0,              # batas bawah 10 m di seed lama tanpa sumber
        "elevation_max": 2000,           # EcoCrop ALTMX
    }, f"{ECOCROP}, Manihot esculenta, kode 1420"),
    "ubi_jalar": ({
        **suhu(10, 21, 27, 38),          # optimal: Rukmana 2007 via Pasaribu dkk. 2018; batas: EcoCrop
        **ph(4.0, 5.0, 7.0, 8.7),        # EcoCrop Ipomoea batatas
        "elevation_max": 2800,           # EcoCrop ALTMX
    }, f"{ECOCROP}, Ipomoea batatas, kode 1265; suhu optimal Rukmana (2007) dalam Pasaribu dkk. (2018)"),
    "jagung": ({
        **ph(4.5, 5.0, 7.0, 8.5),        # EcoCrop Zea mays (suhu sudah sama)
        "elevation_max": 4000,           # EcoCrop ALTMX
    }, f"{ECOCROP}, Zea mays, kode 2175"),
    "sorgum": ({
        **suhu(10, 24, 35, 40),          # FAO 2000 kelas dataran rendah, Ramirez-Villegas dkk. 2013 Tabel 3
        **ph(5.0, 5.5, 7.5, 8.0),        # EcoCrop Sorghum bicolor
        "elevation_max": 2500,           # EcoCrop ALTMX
    }, f"{ECOCROP}, Sorghum bicolor, kode 48747; suhu Ramirez-Villegas dkk. (2013)"),
    "kacang_tanah": ({
        **suhu(10, 22, 32, 45),          # EcoCrop Arachis hypogaea
        **ph(4.5, 5.5, 6.5, 8.5),
        "elevation_max": 1650,
    }, f"{ECOCROP}, Arachis hypogaea, kode 2199"),
    "kacang_hijau": ({
        **suhu(8, 25, 27, 40),           # optimal: Ditjen Tanaman Pangan 2016; batas: EcoCrop
        **ph(4.3, 5.8, 7.0, 8.3),        # optimal: Ditjen Tanaman Pangan 2016; batas: EcoCrop
        # hujan optimal 50-200 mm/bulan (Ditjen Tanaman Pangan 2016) x siklus 3 bulan
        "rainfall_min": 150, "rainfall_optimal_min": 150, "rainfall_optimal_max": 600,
        "elevation_max": 2000,           # EcoCrop ALTMX
    }, f"Ditjen Tanaman Pangan (2016) via BSIP Banten; batas dari {ECOCROP}, Vigna radiata, kode 2150"),
    "kacang_panjang": ({
        **suhu(14, 20, 35, 38),          # EcoCrop Vigna unguiculata ssp. sesquipedalis
        **ph(4.3, 5.5, 7.0, 7.5),
        "elevation_max": 1000,
    }, f"{ECOCROP}, Vigna unguiculata ssp. sesquipedalis, kode 10834"),
    # Cabai rawit sengaja TIDAK memakai anjuran BRMP Banten di sheet (suhu 25-32,
    # 0-500 mdpl). Data panen BPS 2024 menunjukkan sentranya banyak di dataran
    # tinggi; dengan anjuran itu cabai rawit masuk 3 besar di 0 dari 37 kabupaten
    # sentra. Dipakai EcoCrop, dan rentang ketinggian optimal dikosongkan karena
    # tidak ada sumbernya (seed lama 0-800 tanpa rujukan).
    "cabai_rawit": ({
        **suhu(7, 18, 30, 40),           # EcoCrop Capsicum frutescens
        **ph(4.3, 5.5, 6.8, 8.3),
        "elevation_optimal_min": None, "elevation_optimal_max": None,
        "elevation_max": 2000,           # EcoCrop ALTMX
    }, f"{ECOCROP}, Capsicum frutescens, kode 621"),
    "terong": ({
        **suhu(9, 22, 30, 40),           # optimal: Ditjen Hortikultura 2010; batas: EcoCrop
        # pH dari EcoCrop. Sheet menulis "pH sekitar 5-6" (Ditjen Hortikultura 2010),
        # tetapi dengan optimal 5,0-6,0 skor terong TURUN saat tanah dikapur dari
        # 5,5 ke 6,0, bertentangan dengan uji kewajaran pH untuk tanaman peka masam.
        **ph(4.3, 5.5, 6.8, 8.5),
        "elevation_optimal_max": 1000,   # Ditjen Hortikultura 2010: sampai 1000 mdpl
    }, f"Ditjen Hortikultura (2010) via BSIP Banten; batas dari {ECOCROP}, Solanum melongena, kode 1965"),
    "kemiri": ({
        **suhu(18, 20, 27, 28),          # Balittro (optimal) dan Krisnawati dkk. 2011 CIFOR (batas)
        **ph(5.0, 5.0, 8.0, 8.0),        # Krisnawati dkk. 2011: pH 5-8, tanpa rentang optimal
        # hujan tahunan (tanaman tahunan): optimal Balittro, batas Krisnawati dkk. 2011
        "rainfall_min": 640, "rainfall_optimal_min": 1500, "rainfall_optimal_max": 2400,
        "rainfall_max": 4290,
        "elevation_optimal_min": 0, "elevation_optimal_max": 800,   # 0-800 mdpl di Indonesia
    }, "Ferry, Towaha & Rusli (Balittro); Krisnawati, Kallio & Kanninen (2011) CIFOR"),
}


# Batas pH BAWAH tidak diambil dari EcoCrop. Batas EcoCrop (4,0-5,0) adalah
# batas tanaman masih hidup; dengan itu pada pH 4,5 hanya 2 dari 10 tanaman yang
# dinilai tidak layak dan terong (peka masam) naik ke peringkat 1, sehingga
# simulasi kapur nyaris tidak mengubah peringkat. Dipertahankan batas seed lama
# (tanpa rujukan per angka; ditulis sebagai keterbatasan di LAPORAN_ML.md).
# Satu-satunya yang berubah: terong 5,5 -> 5,0, batas bawah "pH sekitar 5-6"
# dari Ditjen Hortikultura (2010) di sheet 3 Param Tanaman.
PH_MIN = {"terong": 5.0}


def main():
    os.makedirs(os.path.dirname(LAMA), exist_ok=True)
    if not os.path.exists(LAMA):
        shutil.copy(SEED, LAMA)
    with open(LAMA, encoding="utf-8") as f:      # selalu mulai dari profil lama
        data = json.load(f)
    for d in data:
        if d["model"] != "crops.cropprofile":
            continue
        fields = d["fields"]
        ubah, sumber = PROFIL[fields["crop_code"]]
        # Batas pH bawah: lihat PH_MIN di atas.
        ubah = dict(ubah, ph_min=PH_MIN.get(fields["crop_code"], fields["ph_min"]))
        for k, baru in ubah.items():
            assert k in fields, k
            if fields[k] != baru:
                print(f"{fields['crop_code']:15} {k:22} {fields[k]} -> {baru}")
            fields[k] = baru
        fields["reference"] = sumber
        # urutan batas harus masuk akal
        for p in ("ph", "temp", "rainfall"):
            lo, a, b, hi = (fields[f"{p}_min"], fields[f"{p}_optimal_min"],
                            fields[f"{p}_optimal_max"], fields[f"{p}_max"])
            assert lo <= a <= b <= hi, (fields["crop_code"], p, lo, a, b, hi)
    for path in (SEED, "seed.json"):             # salinan di root repo ikut diperbarui
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    main()
