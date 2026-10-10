# Laporan ML NUSA-CROP

Status per 10 Oktober 2026: T1-T6 selesai, dengan data lengkap 514 kabupaten (484 punya data tanah SoilGrids).

Ringkasan keputusan: `PAKAI_SKOR_ML = True` dengan skor **hibrida** (`skor_akhir = skor_aturan x koreksi ML`, koreksi dibatasi 0,7-1,3). Saat ini hanya cabai rawit yang dikoreksi ML; sembilan tanaman lain `skor_akhir`-nya sama dengan `skor_aturan`. `confidence` terisi untuk 10 tanaman. Profil tanaman di `seed.json` sudah diperbarui dari sumber yang bisa dilacak (bagian "Pembaruan profil tanaman"); semua angka T2-T4 di bawah memakai profil baru.

Kalimat siap tempel dan daftar angka untuk proposal ada di bagian terakhir, "Untuk proposal".

## T1. Model simulasi kapur

Berkas: `ml_lib/pengapuran.py` (satu file, hanya pustaka standar), `test_pengapuran.py` (10 test, lulus), `latih_pengapuran.py` (ekstraksi data, perbandingan model, fit parameter).

Cara mengulang: `python latih_pengapuran.py "../NUSA_CORP - DATA.xlsx"` lalu `python test_pengapuran.py`.

### Data yang dipakai

Sheet `1c Pengapuran` berisi 109 baris (106 `SEBAGIAN`, 3 `DIBACA`, 0 `CONTOH`). Untuk model pH dipakai **30 baris dari 8 studi** (28 `SEBAGIAN`, 2 `DIBACA`): baris berdosis lebih dari nol yang punya pH sesudah kapur dan dosis absolut. Semua baris itu belum dicek ke dokumen aslinya.

| Studi | Baris | Jenis percobaan | Asal dosis setara CaCO3 |
|---|---|---|---|
| S08 | 3 | Pot kacang tanah, Maluku | g/kg x 0,93 x 2,4 (konversi) |
| S09 | 4 | Inkubasi lab 2 minggu, Padang | g/kg x CCE murni x 2,4 (konversi) |
| S10 | 2 | Sawah, Kutai Kartanegara | kalsit x 1,00 |
| S14 | 2 | Lapangan kacang tanah, Sitiung | NP 85% dilaporkan |
| S15 | 5 | Pot padi sawah, Jember | kalsit x 1,00 |
| S20 | 2 | Lapangan kacang hijau, Bengkulu | dolomit x 0,93 |
| S22 | 6 | Lapangan jagung, Sembawa (0-20 cm) | dinyatakan sebagai CaCO3 |
| S32 | 6 | Lapangan sorgum, Jonggol | NP 98% dihitung dari CaO dan MgO |

Rentang data: dosis 0,4-3,0 t CaCO3/ha pada percobaan bertanaman (S09 hasil konversi 3,8-7,7), pH awal 4,4-5,8. pH awal diisi pH kontrol blok yang sama, bukan pH sebelum percobaan.

Studi yang dibuang: S13 (pH KCl), S36 dan S33 (dosis hanya kelipatan Al-dd), S35 (duplikat sekunder S12), serta S01, S12, S21, S23, S24, S34 (tidak ada pH sesudah kapur). Lapisan 20-40 cm S22 juga dibuang karena kapurnya diberikan di permukaan.

Anomali sumber yang dibiarkan apa adanya:
- S22: pH sebelum olah tanah 5,34 lebih tinggi dari kontrol (4,44-4,51), dan satu perlakuan (OTM 1,65 t/ha) pH-nya turun lagi ke 4,53. Kapurnya dilarik, jadi dosis efektif di barisan tidak pasti.
- S15 dan S10 adalah padi sawah tergenang; penggenangan ikut menaikkan pH.
- S09 baru 2 minggu inkubasi, netralisasi belum tuntas.
- S20: Al-dd naik setelah dikapur 3,2 t/ha. Baris itu tidak dipakai untuk Al-dd.
- Tekstur hanya ada di 1 studi (S15) dan C-organik di 3 studi. Sisanya memakai nilai default.

### Perbandingan model (leave-one-study-out)

| Model | RMSE pH | MAE pH | MAE rata-rata per studi | Monoton | Bisa dibalik | pH maks di grid 0-20 t/ha |
|---|---|---|---|---|---|---|
| A. pH tidak berubah | 0,59 | 0,46 | 0,43 | ya | tidak | - |
| B. Rumus brief tanpa training | 0,64 | 0,51 | 0,51 | ya | ya | 7,80 |
| C1. Parametrik, fit k (kuadrat terkecil) | 0,57 | 0,41 | 0,37 | ya | ya | 7,76 |
| C2. Parametrik, fit k dan plafon | 0,54 | 0,40 | 0,36 | ya | ya | 6,80 |
| **C3. Parametrik, k = median per studi** | **0,48** | **0,37** | **0,34** | ya | ya | 7,77 |
| D1. Regresi linear | 0,69 | 0,58 | 0,50 | ya | ya | 6,78 |
| D2. Regresi polinomial | 1,73 | 0,84 | 0,76 | **tidak** | tidak | 7,21 |
| E. XGBoost monoton | 0,62 | 0,51 | 0,47 | ya | **tidak** (bertangga) | 6,02 |

Plot: `laporan/pengapuran_pred_vs_ukur.png`. Angka lengkap: `laporan/pengapuran_hasil.json`.

**Model terpilih: C3.** Bentuknya rumus Baseline B dengan satu pengali daya sangga:

```
BC      = 2,56 x (BC_TEKSTUR[tekstur] + 0,04 x C_organik_gkg)
R       = 7,8 - ph_awal
ph_baru = ph_awal + R x (1 - exp(-D / (BC x R)))
```

Alasan: error terkecil di antara model yang memenuhi semua syarat, satu parameter, monoton, berplafon 7,8, dan punya invers tertutup. C2 ditolak karena plafon hasil fit menabrak batas bawah 6,8, sehingga target pH 7 tidak bisa dihitung.

Yang perlu diketahui tentang angka ini:
- **Keunggulannya atas "pH tidak berubah" tipis** (RMSE 0,48 vs 0,59). Dengan 8 studi, model ini lebih tepat disebut rumus terkalibrasi daripada model ML.
- **k antarstudi 0,6 sampai 9,4** (S15 0,55; S08 1,02; S22 1,84; S20 2,24; S10 2,88; S32 3,79; S14 4,15; S09 9,44). Dosis untuk satu lahan bisa meleset beberapa kali lipat.
- **C3 ditambahkan setelah melihat hasil C1.** Fit kuadrat terkecil ternyata ditentukan satu studi: k = 2,72 dengan semua data, tetapi 1,56 tanpa S09. Median per studi (2,56) tidak bergantung pada satu studi. Ini pilihan pasca-hoc dan harus disebut begitu di paper.
- Rumus brief tanpa training (B) memprediksi kenaikan pH terlalu besar, yaitu dosis terlalu kecil.
- Urutan antarkelas tekstur dan koefisien C-organik **tidak dilatih**; tetap dari tabel brief.

Contoh keluaran: pH 4,8 ke 6,0 pada clay loam dengan C-organik 14 g/kg butuh 9,96 t dolomit/ha. Dosis sebesar ini konsisten dengan data lapangan (S10: 2 t/ha hanya menaikkan pH 0,4; S14: 1,7 t/ha hanya 0,2), tetapi jauh di atas dosis yang biasa dipakai petani. Catatan untuk petani sudah mengarahkan ke dosis penetral aluminium yang lebih kecil (2,0 t/ha pada contoh ini).

### Cek syarat

- Monoton dan terbatas: dicek di grid dosis 0-20 t/ha, pH awal 4,0-6,0, 12 tekstur dan None (script), dan di `test_monoton`.
- Dua arah: dosis hasil mode target dimasukkan ke mode dosis, selisih pH < 0,02 untuk 6 tekstur x 3 jenis kapur x 4 pasangan pH (`test_dua_arah_konsisten`).
- Di luar rentang data: kurva tetap naik menuju 7,8 dan tidak melewatinya; catatan "hasil kurang pasti" muncul kalau dosis setara CaCO3 > 3 t/ha.

### Estimasi Al-dd

60 pasangan pH H2O dan Al-dd dari 10 grup (1c awal dan sesudah kapur, 2a). S13 (pH KCl) dan S23 (Al-dd 30 cmol/kg, polibag) dibuang.

| Model | RMSE | MAE (cmol(+)/kg) |
|---|---|---|
| Rumus brief: 2,0 x (5,5 - pH)^1,3 | 1,41 | 0,98 |
| Fit a dan b | 1,39 | 1,01 |
| Rata-rata tanpa pH | 1,44 | 1,16 |

**Dipakai: rumus brief.** Fit dari data tidak lebih baik, dan keduanya hampir tidak mengalahkan tebakan rata-rata. Al-dd dari pH saja adalah perkiraan kasar (meleset sekitar 1 cmol(+)/kg); faktor tekstur 0,4-1,5 tidak bisa diuji karena data tidak punya tekstur. Kalau `al_dd` dikirim, nilai itu dipakai dan diturunkan sebanding dengan kurva estimasi.

`dosis_penetral_al_ton_ha` = Al-dd x 1,2 t CaCO3/ha per cmol(+)/kg, dibagi CCE. Faktor 1,2 berasal dari stoikiometri sheet `Rumus_Bantu` dengan asumsi lapisan olah 20 cm dan bobot isi 1,2 g/cm3.

### Lama efek kapur

Interpolasi linear dari tiga titik (dosis setara CaCO3, tahun): (0; 0), (0,5; 2), (2; 5), dibatasi 5 tahun dan dibulatkan ke 0,5.
- 0,5 t/ha sekitar 2 tahun dan 2 t/ha 5 tahun atau lebih: studi Ultisol kaolinitik (P20-P21). Lokasi studi itu belum diverifikasi.
- Sitiung (`3i Efek Residual`): 0,32 t CaCO3/ha kembali ke Al awal dalam sekitar 3 tahun; 1,9 dan 5,5 t/ha masih menahan Al di 66% dan 49% nilai awal pada bulan ke-42.
- Lampung: 8 t/ha, Al masih mendekati nol pada bulan ke-21.

Tidak ada pengamatan lebih dari 3,5 tahun, jadi angka 5 tahun adalah batas, bukan hasil ukur.

### Nilai CCE

| Jenis | CCE | Sumber |
|---|---|---|
| `dolomit` | 0,93 | 4 sampel dolomit lokal, rata-rata 29,5% CaO dan 16,3% MgO (Santi & Goenadi 2012; P30-P31) |
| `kalsit` | 0,85 | Syarat mutu minimal CaCO3+MgCO3 85% (Dariah dkk. 2015; P34); kapur program 85% (Amien 1991) |
| `kapur_tohor` | 1,70 | Nilai awal brief; tidak ada data mutu di paket data |

Ini berbeda dari nilai awal brief (dolomit dan kalsit 1,00). NP terukur produk di pasaran tidak ada di paket data; sheet `Kebutuhan VS Status` sendiri meminta NP tidak diisi 100% tanpa sumber.

## Jawaban untuk 5 pertanyaan BE (Bagian A)

1. **Data iklim tidak diperlukan.** Cukup pH, tekstur, dan C-organik. Bulk density tidak diminta.
2. **Al-dd diestimasi dari pH dan tekstur di dalam fungsi**, lalu dihitung ulang pada pH baru. Kalau `al_dd` dikirim, dipakai dan ditandai `"terukur"`.
3. **Tekstur None tidak error.** Dipakai `loam` dengan catatan untuk petani. Form FE tidak perlu mewajibkan tekstur.
4. **Jenis kapur ditangani di dalam fungsi.** `dosis_ton_ha` selalu berat produk.
5. **Batas yang dijaga:** dosis 0-20 t/ha, target pH 4,0-7,5; target yang butuh lebih dari 20 t/ha ditolak dengan `ValueError`. Peringatan muncul untuk pH baru di atas 6,5 dan dosis di atas 3 t/ha. Batas pH maksimal per tanaman tidak ada di fungsi ini.

## Hal yang perlu diputuskan tim

- **CCE dolomit 0,93 dan kalsit 0,85** menggantikan 1,00 dari brief (sudah disetujui 9 Oktober 2026). Dosis produk naik 8% dan 18%.
- **Besarnya dosis mode target.** Lihat contoh 9,96 t/ha di atas. Kalau FE menampilkan angka ini sebagai rekomendasi, sebaiknya `dosis_penetral_al_ton_ha` yang ditonjolkan.

## Keterbatasan (untuk paper)

- 30 baris dari 8 studi; 28 di antaranya berstatus `SEBAGIAN` dan belum dicek ke dokumen asli.
- Dua studi adalah padi sawah tergenang dan satu studi inkubasi laboratorium 2 minggu.
- Dosis 7 dari 30 baris dikonversi dari g/kg tanah dengan asumsi 2,4 juta kg tanah/ha.
- Tekstur dan C-organik hampir tidak ada di data; pengaruhnya berasal dari tabel, bukan hasil latih.
- Waktu sejak aplikasi tidak dimodelkan.
- Estimasi Al-dd dari pH lemah.
- C-organik dari SoilGrids jauh lebih tinggi daripada nilai laboratorium di data latih (median 53 vs 12 g/kg), jadi `simulasi()` memotongnya di 25 g/kg dan menandai `organic_carbon_dibatasi`. Sebaiknya BE hanya mengirim `organic_carbon` kalau berasal dari uji tanah.

---

## Pembaruan profil tanaman (`seed.json`)

Berkas: `perbarui_profil.py` (semua perubahan beserta asal tiap angka). Profil lama tersimpan di `arsip_model_lama/seed_lama.json`; untuk kembali, salin berkas itu ke `ml_lib/seed.json` dan `seed.json`.

**Aturan pengisian:** sheet `3 Param Tanaman` kalau kolomnya terisi; kalau tidak, basis data FAO EcoCrop (`EcoCrop_DB.csv`, salinan OpenCLIM/ecocrop) untuk suhu, pH optimal dan pH atas, serta ketinggian maksimal; selain itu nilai lama dibiarkan. Sebanyak 64 nilai berubah di 10 tanaman, dan kolom `reference` sekarang menyebut sumber tiap profil.

**Yang sengaja tidak mengikuti aturan itu:**
- **Curah hujan tidak diambil dari EcoCrop.** Angka EcoCrop adalah hujan tahunan, sedangkan scorer memakainya sebagai kebutuhan air satu siklus tanam. Hanya kacang hijau (50-200 mm/bulan, Ditjen Tanaman Pangan) dan kemiri (tanaman tahunan) yang diubah.
- **Cabai rawit memakai EcoCrop, bukan anjuran BRMP Banten di sheet** (suhu 25-32, 0-500 mdpl). Dengan anjuran itu cabai rawit masuk 3 besar di 0 dari 37 kabupaten sentra. Ini pilihan yang dibuat setelah melihat data panen.
- **pH terong memakai EcoCrop (optimal 5,5-6,8), bukan "sekitar 5-6" di sheet.** Dengan optimal 5,0-6,0 skor terong turun saat tanah dikapur dari 5,5 ke 6,0, bertentangan dengan uji kewajaran pH.
- **Ketinggian maksimal terong** di EcoCrop tercatat 50 m (tidak masuk akal), jadi nilai lama dipertahankan.
- Kacang tanah, jagung, dan kacang panjang kosong di sheet, jadi seluruhnya dari EcoCrop.

**Batas pH bawah tetap dari profil lama** (diputuskan 10 Oktober 2026), kecuali terong yang turun dari 5,5 ke 5,0 mengikuti "pH sekitar 5-6" di sheet. Tiga pilihan sudah diukur:

| Ukuran (aturan saja) | Profil lama | Baru, batas pH bawah EcoCrop | **Baru, batas pH bawah lama (dipasang)** |
|---|---|---|---|
| 5 komoditas pangan masuk 3 besar di provinsi sentra | 36% (9 dari 25) | 40% (10 dari 25) | **44% (11 dari 25)** |
| Ubi jalar, dari 5 sentra | 3 | 3 | 5 |
| Kacang tanah, dari 5 sentra | 0 | 3 | 2 |
| Singkong, dari 5 sentra | 2 | 0 | 0 |
| Cabai rawit, ROC-AUC label share | 0,42 | 0,45 | 0,44 |
| Cabai rawit, ROC-AUC label intensitas | 0,72 | 0,74 | 0,75 |
| Cabai rawit masuk 3 besar di 37 kabupaten sentra | 24% | 14% | 8% |
| Grobogan pH 4,5: tanaman berskor di bawah 25 | 7 dari 10 | 2 dari 10 | 7 dari 10 |
| Grobogan: skor terong pH 4,5 lalu pH 6,0 | 20,6 lalu 92,4 | 75,0 lalu 93,1 | 20,5 lalu 93,1 |

Alasan pilihan:
- Batas pH bawah EcoCrop (4,0-5,0) adalah batas tanaman masih hidup. Dengan itu, pada pH 4,5 hanya dua tanaman yang dinilai tidak layak, terong (peka masam) naik ke peringkat 1, dan simulasi kapur nyaris tidak mengubah skor.
- Varian yang dipasang paling baik di validasi provinsi dan mempertahankan penalti tanah masam yang tegas.

Yang harus ditulis jujur di paper:
- **Batas pH bawah sembilan tanaman tidak punya rujukan per angka** (warisan `seed.json` lama).
- **Profil baru tidak jelas lebih akurat.** Selisih 9, 10, dan 11 dari 25 terlalu kecil untuk disebut perbaikan, singkong turun dari 2 ke 0, dan cabai rawit di kabupaten sentra turun dari 24% ke 8% (dikoreksi skor hibrida menjadi 32%).
- Keuntungan pastinya: suhu, pH optimal, dan ketinggian kini punya sumber.
- Kolom `acid_tolerance` dan `al_tolerance` di `seed.json` tidak dipakai scorer.

BE perlu memuat ulang `seed.json` ke tabel Crop.

## T2. Model kecocokan berbasis ML

Cara mengulang: `python tarik_fitur.py` (sekitar 4 jam, bisa dilanjutkan kalau terputus), lalu `python latih_kecocokan.py`.

Berkas: `tarik_fitur.py` (fitur per kabupaten), `latih_kecocokan.py` (label, spatial CV, validasi provinsi, model), `ml_lib/model_kecocokan.joblib`, `laporan/kecocokan_hasil.json`.

### Keputusan: `PAKAI_SKOR_ML = True`, skor hibrida

Tim memutuskan tetap memakai ML (10 Oktober 2026). Rumus yang dipasang:

```
skor_akhir = min(100, skor_aturan x koreksi_ml)
koreksi_ml = p / p_acuan, dibatasi 0,7 sampai 1,3       (tanaman yang punya model)
koreksi_ml = 1                                           (tanaman tanpa model)
```

`p` = probabilitas terkalibrasi dari model tanaman itu, `p_acuan` = rata-rata label di data latih (0,5). `skor_aturan` dihitung untuk bulan `mulai` yang diminta, jadi musim tanam dan batas keras pH, hujan, dan suhu tetap dari aturan. `skor_per_bulan` memakai rumus yang sama untuk tiap bulan.

Saat ini hanya `cabai_rawit` yang punya model, karena itu satu-satunya komoditas proposal dengan data panen tingkat kabupaten se-Indonesia. Untuk sembilan tanaman lain `skor_akhir` sama dengan `skor_aturan` (`sumber_skor_tanaman: "aturan"`).

**Hasil hibrida dibanding aturan saja** (prediksi luar-lipatan, spatial CV per provinsi):

| Ukuran | Aturan | Hibrida | Acak |
|---|---|---|---|
| Cabai rawit, ROC-AUC label share | 0,44 | 0,54 | 0,5 |
| Cabai rawit, ROC-AUC label intensitas | 0,75 | 0,70 | 0,5 |
| Cabai rawit masuk 3 besar di 37 kabupaten sentra | 8% | 32% | 30% |
| 5 komoditas pangan masuk 3 besar di provinsi sentra (25 kasus) | 44% | 32% | 30% |

Dibaca apa adanya: hibrida **memperbaiki cabai rawit tetapi merugikan tanaman lain**. Untuk cabai rawit ia lebih baik pada label share dan di kabupaten sentra (8% menjadi 32%), sedikit lebih buruk pada label intensitas. Di validasi provinsi ia kehilangan tiga kasus (11 menjadi 8 dari 25: dua kacang tanah, satu ubi jalar), karena cabai rawit yang dinaikkan menggeser tanaman itu dari 3 besar. Belum ada bukti hibrida lebih baik secara keseluruhan; yang bisa diklaim adalah ML sudah berada di jalur skor, tidak merusak respons terhadap pH, dan siap diperluas begitu label tanaman lain tersedia.

Batas 0,7-1,3 ditetapkan, bukan dilatih.

### Rumus brief yang dicoba dan ditolak

Rumus di brief, `skor_akhir = 100 x p_kalibrasi x f_musim` dengan skor aturan sembilan tanaman dipetakan ke probabilitas lewat kalibrasi isotonic dari cabai rawit, sudah dijalankan dan gagal:

- Kalibrasinya hampir datar: skor aturan 20-90 semuanya dipetakan ke 0,50 (0-10 ke 0,42; 100 ke 0,67).
- Akibatnya `skor_akhir` nyaris tidak bergantung pada pH. Contoh Grobogan: terong tetap 50,0 untuk pH 4,5 sampai 6,0, padahal skor aturannya naik dari 20,6 ke 92,4, sehingga simulasi kapur tidak mengubah peringkat.
- Di validasi provinsi hasilnya 20% (5 dari 25), di bawah tebakan acak 30%.

Kalau ingin kembali ke aturan murni, set `PAKAI_SKOR_ML = False`; `recommend()` lalu tidak mengirim `skor_akhir`.

### Data dan label

- Titik fitur: satu titik per kabupaten (koordinat ibu kota kabupaten dari cahyadsn/wilayah, lisensi MIT), dengan pipeline BE: SoilGrids 5-15 cm, Open-Meteo ERA5 2022-2024, DEM Open-Meteo. Paket data belum punya poligon berkode BPS dan masker lahan pertanian, jadi fitur tidak diagregasi di lahan tani.
- Nama kabupaten BPS dan Kepmendagri dicocokkan lewat nama (514 dari 514 cocok setelah 2 alias ejaan). Kodenya tidak bisa dipakai karena berbeda sistem.
- Label (L1 dan L4 di `Definisi Label`, disesuaikan): `share = luas panen cabai rawit / (cabai rawit + padi)`, `hadir = share >= median`. Pembagi kelompok hortikultura belum ada di paket data. Wilayah dengan luas panen cabai + padi di bawah 100 ha dibuang.
- Terpakai: 370 kabupaten berlabel di 36 provinsi (semua baris berstatus `DIBACA`), median share 1,2%. Dari 514 kabupaten, 30 tidak punya data tanah SoilGrids di titiknya dan 114 lagi tersisih oleh batas 100 ha.
- Fitur: `ph`, `temp_c`, `elevation_m`, `rainfall_mm`, `rain_wet3`, `rain_dry3`, `bulan_kering`, `sand_persen`, `clay_persen`. NDVI tidak dipakai.

### Spatial CV (GroupKFold 5 lipatan per provinsi), cabai rawit

| Model | ROC-AUC (sd antarlipatan) | PR-AUC | Brier | Korelasi peringkat dengan share |
|---|---|---|---|---|
| Logistic regression | 0,65 (0,03) | 0,66 | 0,249 | 0,27 |
| XGBoost | 0,68 (0,07) | 0,66 | 0,239 | 0,32 |
| XGBoost, pH monoton (dipasang) | 0,65 (0,08) | 0,62 | 0,244 | 0,28 |
| Aturan (`skor_aturan`) | 0,44 (0,05) | 0,48 | 0,267 | -0,12 |
| Acak | 0,48 (0,07) | 0,53 | 0,340 | -0,07 |

Probabilitas dikalibrasi isotonic di dalam lipatan latih. Pada label ini ML jelas di atas aturan, dan **skor aturan cabai rawit berkorelasi negatif dengan share** (AUC 0,44, di bawah 0,5): makin tinggi skor aturannya, makin kecil porsi cabai rawit terhadap padi.

Dengan label pembanding (intensitas = ha cabai rawit per km2 wilayah, di atas median), selisihnya hilang: XGBoost 0,75, XGBoost pH monoton 0,73, logistic regression 0,73, **aturan 0,75**, acak 0,53 (sd antarlipatan 0,11-0,15). Jadi "ML lebih baik dari aturan" bukan kesimpulan yang tahan terhadap pilihan label. Label share mengukur cabai rawit relatif terhadap padi (lahan kering vs sawah), sedangkan label intensitas mengukur seberapa banyak cabai ditanam di wilayah itu.

Top-3 di kabupaten sentra cabai rawit (desil teratas share, 37 kabupaten): cabai rawit masuk 3 besar aturan di **8%**, dengan skor hibrida 32% (acak 30%). Dengan profil lama angka aturan 24%. Rendahnya angka aturan adalah kelemahan nyata: di sentra dataran tinggi, tanaman lain mengungguli cabai rawit.

### Uji kewajaran pH

- Model cabai rawit yang dipasang memakai `monotone_constraints` positif pada pH: 0 dari 71 titik masam (pH <= 5,0) yang skornya turun saat pH dinaikkan 4,5 ke 5,0 / 5,5 / 6,0. Kendala itu menurunkan AUC dari 0,68 ke 0,65. Logistic regression tidak dipasang karena pada data sebagian koefisien pH-nya negatif (13 dari 17 titik turun).
- `skor_aturan`: lulus (`test_kenaikan_ph_tidak_menurunkan_skor`, 6 kombinasi lahan).
- `skor_akhir` hibrida: lulus pada kombinasi yang sama, dan tetap responsif terhadap pH. Contoh Grobogan, cabai rawit: 15,2 (pH 4,5), 53,4 (pH 5,0), 66,4 (pH 5,5), 75,5 (pH 6,0).

## T3. Confidence untuk 10 tanaman

`confidence` sekarang terisi untuk kesepuluh tanaman, tidak biner, dan tidak bergantung pada XGBoost lama.

**Definisi.** Input digoyang 100 kali sebesar ketidakpastiannya, lalu semua tanaman dinilai ulang:
- pH: seragam +-0,2 kalau `ph_terukur`; kalau `ph_q05` dan `ph_q95` dikirim, normal dengan simpangan baku (q95 - q05) / 3,29 dan dipotong di rentang itu; selain itu simpangan baku 0,88 (median SoilGrids di 484 titik kabupaten).
- Curah hujan: semua bulan dikali satu faktor lognormal dengan CV 0,14 (median variasi antartahun 2022-2024 di titik yang sama).
- Tekstur: peluang 0,3 pindah ke salah satu dari 2 kelas USDA terdekat.

`confidence = 0,5 x (bagian simulasi dengan skor bergeser <= 10 poin) + 0,5 x (bagian simulasi dengan peringkat bergeser <= 1)`.

Benih acak tetap, jadi input yang sama selalu memberi angka yang sama. Ini ukuran kestabilan penilaian, bukan probabilitas tanaman cocok. Waktu hitung sekitar 0,03 detik per panggilan (0,07 dengan ML).

**Sebaran pada 484 titik kabupaten (3 tanaman teratas):**

| Sumber pH | P5 | Median | P95 | Tinggi | Sedang | Rendah |
|---|---|---|---|---|---|---|
| SoilGrids (q05-q95 dikirim) | 0,31 | 0,62 | 0,88 | 28% | 47% | 26% |
| Uji tanah (`ph_terukur`) | 0,50 | 0,92 | 1,00 | 78% | 17% | 5% |

**Usulan ambang untuk BE:** tinggi >= 0,75; sedang 0,50-0,75; rendah < 0,50. Kualitas data pH tidak perlu digabung terpisah, karena sudah masuk lewat `ph_terukur` dan `ph_q05`/`ph_q95`. Yang masih perlu aturan BE: turunkan satu tingkat kalau iklim atau elevasi berasal dari data cadangan. Aturan lama "turun satu tingkat kalau tidak masuk 3 besar XGBoost" sebaiknya dihapus.

## T4. Validasi independen

**Tingkat provinsi, 5 komoditas pangan.** Label: rata-rata 2019-2023 share luas panen di antara 6 komoditas. Fitur: median titik kabupaten per provinsi (37 provinsi; DKI Jakarta tidak punya share pangan selain padi). Top-3 = di 5 provinsi sentra tiap komoditas, apakah komoditas itu masuk 3 besar dari 10 tanaman.

| Skor | Top-3 di sentra | Acak |
|---|---|---|
| `skor_aturan` | 44% (11 dari 25) | 30% |
| `skor_akhir` hibrida (dipasang) | 32% (8 dari 25) | 30% |
| `skor_akhir` rumus brief (ditolak) | 20% (5 dari 25) | 30% |

Rincian aturan per komoditas (dari 5 sentra): ubi jalar 5, jagung 4, kacang tanah 2, singkong 0, kacang hijau 0. Hanya 25 kasus, jadi angkanya kasar. Korelasi peringkat skor aturan dengan share antarprovinsi lemah: kacang hijau 0,25; kacang tanah 0,22; singkong 0,02; ubi jalar -0,05; jagung -0,08.

Angka ini jauh di bawah Top-3 92% pada 24 sentra di paper. Keduanya tidak sebanding langsung (titik sentra pilihan vs median titik ibu kota kabupaten per provinsi), tetapi paper sebaiknya tidak menyandarkan klaim akurasi pada angka 92% saja.

**pH SoilGrids vs terukur.** Dari 14 baris `2a Tanah`, hanya 9 berkoordinat dan semuanya satu koordinat (Lamandau, 3 profil). Di titik itu SoilGrids 5-15 cm memberi pH 4,8 (rentang 5-95%: 3,7-6,4); pH terukur lapisan atas 4,8; 4,9; 5,1. Selisih rata-rata -0,13 dan ketiganya di dalam rentang. Ini satu lokasi, jadi belum bisa disebut validasi. Sheet `4_Validasi` hanya berisi baris `CONTOH`.

**C-organik SoilGrids tidak bisa dipakai apa adanya.** Di Lamandau SoilGrids memberi 189 g/kg, terukur 2-9 g/kg; median titik kabupaten 53 g/kg.

## T5. Fitur final

| Nama | Satuan | Sumber | Dipakai untuk |
|---|---|---|---|
| `ph` | pH H2O | Uji tanah atau SoilGrids 5-15 cm | Aturan, model ML |
| `temp_c` | derajat C | Open-Meteo ERA5 2022-2024 | Aturan, model ML |
| `elevation_m` | mdpl | SRTM / DEM Open-Meteo | Aturan, model ML |
| `soil_texture` | kelas USDA | Uji tanah atau fraksi SoilGrids | Aturan |
| `monthly_rain` (12 nilai) | mm/bulan | Open-Meteo ERA5 2022-2024 | Aturan (jendela tanam) |
| `rainfall_mm`, `rain_wet3`, `rain_dry3` | mm | Turunan `monthly_rain` | Model ML |
| `bulan_kering` | bulan < 100 mm | Turunan `monthly_rain` | Aturan (penalti kemarau), model ML |
| `sand_persen`, `clay_persen` | % | SoilGrids atau uji tanah; kalau kosong dari titik tengah kelas tekstur | Model ML |
| `ph_q05`, `ph_q95`, `ph_terukur` | pH, bool | SoilGrids, BE | Confidence |

Jadi 5 input dasar dan 9 fitur model, bukan 16 seperti di proposal. Al-dd, kemiringan, bulk density, NDVI, nitrogen, C-organik, dan ET0 **tidak dipakai**; paper perlu disesuaikan.

## T6. Uji integrasi

`test_rekomendasi.py` (10 test, lulus), dijalankan untuk mode aturan dan mode ML:
- kontrak output wajib untuk 10 tanaman, dengan dan tanpa `mulai`;
- `recommend()` jalan tanpa parameter tambahan, dan dengan semuanya;
- saklar mati: `skor_akhir` dan `skor_per_bulan` tidak ada, urutan mengikuti `skor_aturan`;
- hibrida: tanaman tanpa model punya `skor_akhir` sama dengan `skor_aturan`, tanaman ber-model bergeser paling banyak 30%;
- saklar hidup: keduanya terisi untuk 10 tanaman, `skor_per_bulan[mulai]` sama dengan `skor_akhir`, tanaman tahunan sama di semua bulan;
- `ph_baru` dari `simulasi()` dimasukkan ke `recommend()`: faktor pH semua tanaman tidak turun, dan terong berubah dari tidak layak (pH 4,6, di bawah batas 5,0) menjadi layak (pH 5,8).

## Perubahan signature dan perilaku

- `recommend()` menerima parameter baru, semua opsional: `sand_persen`, `clay_persen`, `ph_q05`, `ph_q95`, `ph_terukur`. BE otomatis mengirimnya karena namanya ada di signature. Parameter lain di daftar BE (`ndvi_aktual`, `kemiringan_derajat`, `nitrogen_gkg`, `organic_carbon_gkg`, `silt_persen`, `et0_tahunan_mm`) sengaja tidak ditambahkan karena tidak dipakai.
- `sort_by` bawaan berubah dari `"aturan"` ke `"akhir"` (urut `skor_akhir` kalau ada, kalau tidak `skor_aturan`). `sort_by="ml"` dihapus.
- `confidence` berubah arti: dari probabilitas kelas XGBoost menjadi kestabilan Monte Carlo. Tidak pernah `None` lagi.
- Key baru saat ML aktif: `skor_akhir`, `skor_per_bulan`, `sumber_skor_tanaman`, `di_luar_rentang_latih`.
- `recommend_dua()` meneruskan parameter tambahan ke `recommend()`.
- Properti `kelas_model` diganti `tanaman_ml`.
- XGBoost lama tidak dimuat lagi. Berkas `model_xgb.json`, `label_encoder.joblib`, `preprocessor.joblib`, `preprocess.py`, dan `feature_names.txt` dipindah dari `ml_lib/` ke `arsip_model_lama/`.
- `simulasi()` mengembalikan key tambahan `organic_carbon_dibatasi`.
- Koreksi dokumen: model lama punya 9 kelas (termasuk `kacang_hijau`), bukan 8; hanya `terong` yang confidence-nya `None`.

## Keterbatasan T2-T4 (untuk paper)

- Label panen mencerminkan di mana tanaman **biasa ditanam** (pasar, kebiasaan, akses), bukan semata di mana tanaman **cocok**.
- Label kabupaten hanya ada untuk cabai rawit; sorgum dan kemiri tidak punya label di tingkat mana pun; terong dan kacang panjang hanya provinsi 2023-2024 dan hanya dipakai di uji "bisa hidup".
- Satu titik (ibu kota kabupaten) mewakili satu kabupaten; variasi di dalam kabupaten hilang dan titiknya belum tentu lahan pertanian.
- Pembagi share bukan kelompok hortikultura, melainkan cabai rawit + padi.
- Rentang ketidakpastian pH SoilGrids lebar (median simpangan baku 0,88 unit pH), sehingga confidence untuk data API memang rendah sampai sedang.
- 30 dari 514 titik ibu kota kabupaten tidak punya data tanah SoilGrids (piksel kota atau perairan) dan tidak ikut dihitung.

---

## Untuk proposal

Bagian ini merangkum cara membingkai sistem, kalimat siap tempel, dan angka yang boleh dikutip. Semua angka berasal dari bagian T1-T4 di atas.

### Pembingkaian

Sistem dirancang ML-first dan saat ini dibatasi data, bukan teknologi. Tiga hal yang mendukung:

1. **Jalur ML sudah terpasang dan tervalidasi.** Model dilatih dari data panen BPS 370 kabupaten dan diuji dengan spatial cross-validation, bukan dari label buatan sendiri.
2. **ML menambal kelemahan aturan di tempat yang datanya ada.** Di kabupaten sentra cabai rawit, cabai rawit masuk 3 besar di 32% kasus dengan koreksi ML, dibanding 8% dengan aturan saja. Catatan: 32% itu baru setara tebakan acak (30%), jadi yang bisa diklaim adalah "memperbaiki aturan", bukan "akurat".
3. **Hambatannya spesifik dan bisa diatasi dengan pendanaan:** label panen tingkat kabupaten baru tersedia untuk 1 dari 10 komoditas.

**Pesan utama untuk proposal:** model ML ini adalah hasil pelatihan terbaik yang bisa dicapai dengan data yang tersedia saat ini, dan pengembangan selanjutnya akan lebih baik dengan dataset yang lebih baik.

Dasar klaim "terbaik dengan data yang tersedia", supaya bisa dijawab kalau ditanya juri:
- Tiga model dibandingkan pada lipatan spatial CV yang sama: logistic regression (ROC-AUC 0,65), XGBoost (0,68), dan XGBoost dengan pH monoton (0,65).
- Selisih ketiganya lebih kecil dari simpangan antarlipatan (0,03-0,08), jadi mengganti atau menyetel algoritma tidak akan mengubah hasil secara berarti. Yang membatasi adalah data.
- Yang dipasang adalah XGBoost dengan pH monoton, bukan yang AUC-nya tertinggi, karena hanya varian itu yang dijamin tidak menurunkan skor saat tanah dikapur.
- Penyetelan hyperparameter sudah dicoba (36 kombinasi jumlah pohon, kedalaman, laju belajar, dan `min_child_weight` pada XGBoost pH monoton, spatial CV yang sama): ROC-AUC berkisar 0,62-0,66 dan akurasi 54-62%. Kombinasi terbaik (0,663; 61,6%) hanya 0,014 AUC dan 2 poin akurasi di atas konfigurasi terpasang (0,649; 59,7%), jauh lebih kecil dari simpangan antarlipatan (0,08). Konfigurasi terpasang tidak diganti, karena memilih pemenang dari 36 percobaan pada data uji yang sama hanya akan melaporkan angka yang terlalu optimistis.

**Akurasi saat ini (cabai rawit, spatial CV, ambang 0,5):** model terpasang 59,7% (simpangan antarlipatan 4,5 poin); skor aturan 49,7%; tebakan acak 50% karena labelnya seimbang. Jadi model benar di sekitar 6 dari 10 kabupaten, 10 poin di atas aturan dan tebakan acak.

Batas klaimnya: "terbaik yang dapat dicapai dengan data ini", bukan "akurat". Akurasi 60% itu rendah secara mutlak, dan itulah yang membenarkan kebutuhan dataset yang lebih baik.

Yang sebaiknya tidak ditulis: "rekomendasi dihasilkan oleh model ML". Sembilan dari sepuluh tanaman saat ini sepenuhnya dari aturan.

### Kalimat siap tempel

**Arsitektur.**
> Soilmate memakai arsitektur hibrida: skor kesesuaian berbasis aturan agronomi yang dikoreksi model machine learning terlatih dari data panen BPS. Skor akhir tiap tanaman adalah skor aturan dikali faktor koreksi model, yang dibatasi 0,7 sampai 1,3. Saat ini koreksi ML aktif untuk satu komoditas (cabai rawit), satu-satunya yang memiliki data panen tingkat kabupaten secara nasional. Pada komoditas itu, koreksi ML menaikkan ketepatan rekomendasi di kabupaten sentra dari 8% menjadi 32% (tebakan acak 30%), dan akurasi model 60% dibanding 50% untuk skor aturan.

**Status model dan pengembangan lanjutan.**
> Model yang dipasang adalah hasil pelatihan terbaik yang dapat dicapai dengan data yang tersedia saat ini. Tiga algoritma dan 36 kombinasi hyperparameter dibandingkan dengan spatial cross-validation; hasilnya setara dalam batas galat (ROC-AUC 0,62 sampai 0,68, akurasi sekitar 60% dibanding 50% untuk skor aturan dan tebakan acak), yang menunjukkan bahwa kinerja dibatasi oleh data, bukan oleh pilihan algoritma. Pengembangan selanjutnya akan lebih baik dengan dataset yang lebih baik: label panen tingkat kabupaten untuk seluruh komoditas, deret beberapa tahun, dan data hasil panen tingkat petak dari pengguna aplikasi.

**Validasi.**
> Model divalidasi dengan spatial cross-validation lima lipatan per provinsi pada 370 kabupaten di 36 provinsi, sehingga kabupaten uji selalu berasal dari provinsi yang tidak dipakai melatih. Untuk cabai rawit, model mencapai ROC-AUC 0,65, dibanding 0,44 untuk skor aturan dan 0,48 untuk tebakan acak.

**Simulasi kapur.**
> Simulasi pengapuran memakai kurva jenuh daya sangga tanah yang dikalibrasi dari 30 hasil percobaan pada 8 studi pengapuran di Indonesia. Pada validasi leave-one-study-out, galat prediksi pH sesudah pengapuran adalah RMSE 0,48 dan MAE 0,37 unit pH. Dosis dihitung dalam berat produk untuk dolomit, kalsit, dan kapur tohor.

**Tingkat kepercayaan.**
> Tiap rekomendasi disertai tingkat kepercayaan yang dihitung dengan simulasi Monte Carlo: pH, curah hujan, dan tekstur digoyang sebesar ketidakpastiannya, lalu diukur seberapa stabil skor dan peringkat tanaman. Dengan pH hasil uji tanah, 78% rekomendasi tiga teratas berada di tingkat kepercayaan tinggi; dengan pH dari SoilGrids hanya 28%. Ini menunjukkan nilai uji tanah bagi petani.

**Keterbatasan dan pendanaan.**
> Keterbatasan utama adalah data, bukan arsitektur. Pada validasi tingkat provinsi untuk lima komoditas pangan, skor aturan menempatkan komoditas yang benar di tiga besar pada 44% provinsi sentra dan skor hibrida pada 32% (tebakan acak 30%), karena koreksi ML baru tersedia untuk satu komoditas. Pendanaan lanjutan akan dipakai untuk memperluas data latih ke sembilan komoditas lain, sehingga peran ML bertambah tanpa perubahan arsitektur.

### Akurasi kecocokan: belum 80-85%

Untuk pertanyaan "apakah tanaman ini memang cocok di lahan ini", **akurasi 80-85% belum tercapai dan tidak boleh diklaim**. Angka yang ada:

| Yang diukur | Hasil | Tebakan acak |
|---|---|---|
| Model ML menebak cabai rawit banyak ditanam atau tidak | 60% | 50% |
| Tanaman yang benar masuk 3 besar rekomendasi, skor aturan | 44% | 30% |
| Sama, skor hibrida | 32% | 30% |

Penyebabnya data, bukan model: tidak ada label yang menyatakan "tanaman X cocok di lahan Y". Yang ada hanya data panen per kabupaten, yang mengukur di mana tanaman biasa ditanam, dengan satu titik mewakili satu kabupaten. Tiga algoritma dan 36 kombinasi hyperparameter semuanya mentok di sekitar 60%.

Cara menulisnya di proposal: akurasi saat ini 60% (di atas aturan dan tebakan acak, keduanya 50%), dan 80-85% adalah **sasaran pengembangan** yang mensyaratkan label kecocokan sesungguhnya, yaitu hasil panen per petak dengan pH terukur dan lokasi tepat untuk banyak tanaman. Data itu yang mulai dicatat aplikasi.

Angka 87% di bawah menjawab pertanyaan yang lebih mudah ("bisa hidup atau tidak") dan tidak boleh dipakai sebagai akurasi kecocokan.

### Uji "tanaman bisa hidup" (sensitivitas)

Berkas: `evaluasi_bisa_hidup.py`, keluaran `data/cek_bisa_hidup.csv`.

**Pertanyaannya:** di wilayah tempat suatu tanaman terbukti dibudidayakan menurut data panen BPS, apakah sistem menyatakan tanaman itu bisa hidup? "Bisa hidup" berarti pH, curah hujan musim tanam, dan suhu semuanya berada di dalam batas toleransi tanaman (tidak ada faktor kunci yang bernilai nol). Ambangnya ditetapkan sebelum dihitung dan tidak disetel.

| Bukti budidaya | Pasangan wilayah x tanaman | Dinyatakan bisa hidup |
|---|---|---|
| Luas panen >= 10 ha | 677 | **87%** |
| Luas panen >= 100 ha | 455 | **92%** |
| Tingkat kabupaten saja, >= 10 ha | 439 | 80% |
| Tingkat kabupaten saja, >= 100 ha | 237 | 85% |
| Tingkat provinsi saja, >= 10 ha | 238 | 99,6% |

Cakupan: 8 dari 10 tanaman. Kabupaten: cabai rawit (2024, nasional), ubi kayu dan ubi jalar (2024, Sumatera Barat). Provinsi: jagung, kacang tanah, kacang hijau, ubi kayu, ubi jalar (rata-rata 2019-2023), terong dan kacang panjang (2023-2024). Sorgum dan kemiri tidak punya data panen.

Per tanaman di tingkat kabupaten (>= 10 ha): cabai rawit 78% (413 kabupaten), ubi kayu 100% (16), ubi jalar 100% (10). Dari 91 kegagalan, 56 karena curah hujan di luar batas dan 46 karena pH di bawah batas (sebagian dua-duanya); tidak ada yang karena suhu.

**Kalimat siap tempel.**
> Kemampuan sistem menilai apakah tanaman bisa hidup diuji terhadap data panen BPS: pada 677 pasangan wilayah dan tanaman yang terbukti dibudidayakan (luas panen minimal 10 ha), sistem menyatakan tanaman tersebut bisa hidup pada 87% kasus, dan 92% pada wilayah dengan luas panen minimal 100 ha. Pada tingkat kabupaten saja angkanya 80% dan 85%.

**Batas klaim, wajib ikut ditulis:**
- **Ini sensitivitas, bukan akurasi penuh.** Yang diukur hanya "tanaman yang terbukti tumbuh dinyatakan bisa hidup". Kebalikannya tidak bisa diukur, karena luas panen nol bukan bukti tanaman tidak bisa hidup. Sistem yang selalu menjawab "bisa hidup" akan mendapat 100% pada ukuran ini.
- Sebagai pembanding kasar: di 31 kabupaten/kota tanpa panen cabai rawit, sistem menyatakan bisa hidup pada 45%, jauh di bawah 78% di kabupaten yang menanamnya. Jadi sistem memang membedakan, tetapi 31 wilayah itu kebanyakan kota dan bukan negatif sejati.
- **Angka tingkat provinsi (99,6%) menggelembungkan angka gabungan.** Kondisi median satu provinsi hampir selalu berada di dalam batas toleransi. Angka yang paling ketat adalah tingkat kabupaten: 80% dan 85%.
- Sebagian kegagalan berasal dari data, bukan aturan: pH diambil dari satu titik SoilGrids di ibu kota kabupaten, dengan ketidakpastian sekitar 0,9 unit pH.
- Ini mengukur "bisa hidup", bukan "paling cocok". Ketepatan peringkat (tanaman yang benar masuk 3 besar) tetap 44% di validasi provinsi.

### Angka yang boleh dikutip

| Bagian | Angka |
|---|---|
| Simulasi kapur | 30 baris dari 8 studi; RMSE 0,48 pH dan MAE 0,37 pH (leave-one-study-out); pembanding "pH tidak berubah" RMSE 0,59; 7 model dibandingkan |
| Setara CaCO3 | Dolomit 0,93; kalsit 0,85; kapur tohor 1,70 |
| Data model kecocokan | 514 kabupaten ditarik, 370 berlabel di 36 provinsi; 9 fitur |
| Cabai rawit, ROC-AUC | XGBoost 0,68; model terpasang (pH monoton) 0,65; aturan 0,44; acak 0,48 |
| Tanaman bisa hidup (sensitivitas) | 87% dari 677 pasangan wilayah x tanaman berluas panen >= 10 ha; 92% untuk >= 100 ha; tingkat kabupaten saja 80% dan 85% |
| Cabai rawit, akurasi | Model terpasang 59,7%; aturan 49,7%; acak 50% |
| Penyetelan hyperparameter | 36 kombinasi; ROC-AUC 0,62-0,66; terbaik hanya +0,014 dari konfigurasi terpasang |
| Cabai rawit di 37 kabupaten sentra | Masuk 3 besar: aturan 8%, hibrida 32%, acak 30% |
| Validasi provinsi, 5 komoditas pangan | Masuk 3 besar di provinsi sentra (25 kasus): aturan 44%, hibrida 32%, acak 30% |
| Confidence | 100 simulasi per rekomendasi; tingkat tinggi di 78% (pH uji tanah) dan 28% (pH SoilGrids) |
| Profil tanaman | 64 nilai diperbarui dari FAO EcoCrop dan sumber Indonesia |

### Angka lama yang harus diganti atau dibuang

| Angka lama | Alasan |
|---|---|
| F1 0,937 | Dari XGBoost lama yang dilatih dengan label aturannya sendiri; model itu sudah tidak dipakai |
| Top-3 92% di 24 sentra | Tidak bisa dihitung ulang (`evaluate_sanity.py` tidak ada di repo ini) dan profil tanaman sudah berubah, jadi tidak berlaku untuk sistem sekarang |
| 16 fitur | Model sekarang memakai 9 fitur |
| 8 dari 10 tanaman dikenali model | Sekarang ML mengoreksi 1 tanaman; 9 lainnya dari aturan |

### Peta jalan data (bagian pendanaan)

| Data | Sumber | Gunanya |
|---|---|---|
| Luas panen dan produksi per kabupaten untuk 9 komoditas lain | Publikasi BPS "Kabupaten Dalam Angka", Satu Data Pertanian | Model ML per tanaman |
| Deret 5 tahun per kabupaten | Sumber yang sama | Label lebih stabil, tidak bergantung satu tahun |
| Hasil panen tingkat petak dari pengguna | Sudah mulai dicatat aplikasi | Label yang mengukur cocok, bukan sekadar biasa ditanam |
| Batas kabupaten dan masker lahan pertanian | BIG, ESA WorldCover | Fitur dari lahan tani, bukan satu titik ibu kota kabupaten |
| Uji tanah berkoordinat (pH, Al-dd) | Balittanah, survei lapangan | Validasi SoilGrids dan model kapur |

Baris ketiga adalah argumen terkuat: makin banyak petani memakai aplikasi, makin baik modelnya. Menambah tanaman ke jalur ML cukup dengan melatih satu model lagi di `latih_kecocokan.py`; kode BE tidak berubah.

### Yang tetap harus disebut

- Angka yang kurang menguntungkan (aturan 8% di sentra cabai rawit, hibrida 32% vs aturan 44% di validasi provinsi) justru menguatkan pembingkaian "dibatasi data". Kalau disembunyikan lalu ketahuan, kesannya berubah menjadi menutupi kelemahan.
- Label panen mengukur di mana tanaman biasa ditanam, bukan semata di mana tanaman cocok.
- Batas koreksi ML (0,7-1,3) dan batas pH bawah sembilan tanaman adalah pilihan desain, bukan hasil latih atau kutipan sumber.
- Model kapur lebih tepat disebut rumus terkalibrasi daripada model ML, dan dosisnya bisa meleset jauh untuk satu lahan (pengali daya sangga antarstudi 0,6 sampai 9,4).
