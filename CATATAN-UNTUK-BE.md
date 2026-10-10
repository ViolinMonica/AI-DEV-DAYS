# Catatan dari tim ML untuk BE

Per 10 Oktober 2026. Menjawab `RINGKASAN-UNTUK-ML.md`. Rincian dan angka ada di `LAPORAN_ML.md`.

## Yang perlu dilakukan BE

1. **Salin seluruh folder `ml_lib/`** (ganti yang lama). Isinya sekarang: `predict.py`, `pengapuran.py`, `rule_based_scorer.py`, `climate_features.py`, `soilgrids_adapter.py`, `usda_texture.py`, `seed.json`, `model_kecocokan.joblib`. Berkas XGBoost lama (`model_xgb.json`, `preprocessor.joblib`, `label_encoder.joblib`, `preprocess.py`, `feature_names.txt`) sudah tidak dipakai dan boleh dihapus dari sisi BE.
2. **Dependensi**: `numpy`, `joblib`, `scikit-learn`, `xgboost` (dipakai saat memuat `model_kecocokan.joblib`). `pandas` tidak lagi dibutuhkan saat inference. Versi yang dipakai saat melatih ada di `ml_lib/requirements.txt`. Model disimpan dengan scikit-learn 1.9 dan xgboost 3.4; kalau versi BE berbeda dan model gagal dimuat, `recommend()` otomatis jalan tanpa `skor_akhir`.
3. **Ganti aturan tingkat kepercayaan** (lihat bagian confidence di bawah).
4. **Kirim `organic_carbon` ke `simulasi()` hanya kalau berasal dari uji tanah**, jangan dari SoilGrids.
5. **Muat ulang `seed.json` ke tabel Crop.** Syarat tumbuh kesepuluh tanaman diperbarui dari sumber yang bisa dilacak (64 nilai; struktur dan `crop_code` tidak berubah, kolom `reference` kini menyebut sumbernya). Batas pH bawah tidak berubah kecuali terong (5,5 menjadi 5,0); yang berubah terutama rentang suhu, pH optimal, dan ketinggian maksimal. Rinciannya di `LAPORAN_ML.md` bagian "Pembaruan profil tanaman".

## Simulasi kapur: `ml_lib.pengapuran.simulasi()`

Kontrak sesuai permintaan, tanpa perubahan signature. Jawaban lima pertanyaan:

1. Data iklim tidak diperlukan; cukup pH, tekstur, C-organik.
2. Al-dd diestimasi dari pH dan tekstur di dalam fungsi. Kalau `al_dd` dikirim, nilai itu dipakai (`al_dd_sumber: "terukur"`).
3. `tekstur_kelas=None` tidak error; dipakai `loam` dan ada catatan untuk petani. Form FE tidak perlu mewajibkan tekstur.
4. Jenis kapur ditangani di dalam fungsi. `dosis_ton_ha` selalu berat **produk**, jadi bisa langsung dikali harga per kg.
5. Batas: dosis 0-20 t/ha, target pH 4,0-7,5. Target yang butuh lebih dari 20 t/ha dilempar sebagai `ValueError` berbahasa Indonesia.

Field tambahan yang dikembalikan: `mode`, `metode`, `jenis_kapur`, `dosis_setara_caco3_ton_ha`, `al_dd_awal`, `al_dd_baru`, `al_dd_sumber`, `dosis_penetral_al_ton_ha`, `estimasi_lama_efek_tahun`, `tekstur_dipakai`, `tekstur_diasumsikan`, `organic_carbon_dipakai`, `organic_carbon_diasumsikan`, `organic_carbon_dibatasi`.

Hal yang perlu diketahui:

- **Dosis mode target bisa besar.** Contoh: pH 4,8 ke 6,0 pada clay loam butuh sekitar 10 t dolomit/ha. Saran: di FE tonjolkan `dosis_penetral_al_ton_ha` (2,0 t/ha pada contoh yang sama) sebagai kebutuhan minimal, dan tampilkan dosis mode target sebagai angka untuk mencapai pH yang diminta.
- **C-organik dipotong di 25 g/kg.** SoilGrids memberi median 53 g/kg, sedangkan data percobaan kapur 6-21 g/kg; tanpa batas itu dosis kira-kira dua kali lipat. Kalau terpotong, `organic_carbon_dibatasi` bernilai `True` dan ada catatan untuk petani.
- Menerima variasi penulisan: `clay_loam`, `Clay Loam`, `kapur tohor`.

## Rekomendasi: `NusaCropModel().recommend()`

Output wajib tidak berubah artinya: `crop_code`, `skor_aturan`, `rincian`, `mulai_tanam`, `musim_tanam_mm`, `confidence`.

### Yang berubah

| Hal | Sebelum | Sekarang |
|---|---|---|
| `confidence` | Probabilitas XGBoost, nyaris biner, `None` untuk terong | Kestabilan skor 0-1, terisi untuk 10 tanaman, tidak pernah `None` |
| `skor_akhir` | Tidak ada | Dikirim untuk 10 tanaman (skor hibrida aturan x koreksi ML) |
| `skor_per_bulan` | Tidak ada | Dikirim, 12 angka, skala sama dengan `skor_akhir` |
| `sumber_skor_tanaman` | Tidak ada | `"ml"` (saat ini hanya `cabai_rawit`) atau `"aturan"` |
| `di_luar_rentang_latih` | Tidak ada | `True` kalau input di luar rentang data latih model (hanya muncul untuk tanaman `"ml"`) |

Karena `skor_akhir` dan `skor_per_bulan` sekarang dikirim, BE akan memakai keduanya untuk urutan dan kalender tanam, dan tidak perlu lagi memanggil `recommend(mulai=0..11)`.

Untuk sembilan tanaman tanpa model, `skor_akhir` sama persis dengan `skor_aturan`. Untuk cabai rawit, skor aturannya digeser paling banyak 30% oleh model.

### Parameter tambahan yang sekarang diterima

`sand_persen`, `clay_persen`, `ph_q05`, `ph_q95`, `ph_terukur`. Semua opsional dan punya default, jadi mekanisme pengecekan signature di BE akan mengirimnya otomatis.

Tidak ditambahkan karena tidak dipakai model: `ndvi_aktual`, `kemiringan_derajat`, `nitrogen_gkg`, `organic_carbon_gkg`, `silt_persen`, `et0_tahunan_mm`. Bulk density juga tidak perlu ditarik.

`recommend_dua()` meneruskan parameter tambahan yang sama. Logika `tanam_sekarang` masih identik dengan `recommend(mulai=...)`, jadi BE tidak perlu beralih.

### Confidence dan tingkat kepercayaan

`confidence` mengukur seberapa stabil skor dan peringkat tanaman kalau pH, curah hujan, dan tekstur digoyang sebesar ketidakpastiannya. Ini **bukan** probabilitas tanaman cocok.

Usulan aturan baru untuk BE, menggantikan aturan lama:

| Tingkat | `confidence` |
|---|---|
| Tinggi | >= 0,75 |
| Sedang | 0,50 sampai < 0,75 |
| Rendah | < 0,50 |

- Kualitas data pH sudah masuk lewat `ph_terukur` dan `ph_q05`/`ph_q95`, jadi tidak perlu aturan terpisah "uji tanah > API". Dengan pH uji tanah, 78% dari 3 rekomendasi teratas jatuh di "tinggi"; dengan pH SoilGrids 28%.
- Tetap turunkan satu tingkat kalau iklim atau elevasi berasal dari data cadangan.
- Hapus aturan "turun satu tingkat kalau tidak masuk 3 besar XGBoost".

### Saklar

`PAKAI_SKOR_ML` di `ml_lib/predict.py` (bawaan `True`). Kalau `False` atau model gagal dimuat, `skor_akhir` dan `skor_per_bulan` tidak dikirim dan BE kembali ke `skor_aturan` seperti sebelumnya.

## Jawaban butir prioritas di Bagian B

| Butir | Status |
|---|---|
| 1. Output wajib dijaga | Ya |
| 2. Confidence untuk 10 tanaman | Ya |
| 3. Confidence tidak biner | Ya |
| 4. `skor_akhir` hibrida | Ya: `skor_aturan x koreksi ML` (0,7-1,3) |
| 5. Aturan tingkat kepercayaan | Diganti, lihat tabel ambang di atas |
| 6. Validasi data independen | Sudah, dengan data panen BPS dan spatial CV per provinsi; hasilnya di laporan |
| 7. NDVI | Dihapus dari fitur |
| 8. Fitur vs proposal | 9 fitur model, bukan 16; paper disesuaikan |
| 9. Komoditas baru | Tidak ada; tetap 10 `crop_code` yang sama |

## Koreksi untuk dokumen BE

Model lama punya 9 kelas (termasuk `kacang_hijau`), bukan 8. Hanya `terong` yang confidence-nya `None`.
