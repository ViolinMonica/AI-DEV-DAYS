"""Test kontrak recommend() dan integrasi dengan simulasi kapur (T2, T3, T6).

Jalankan: `python test_rekomendasi.py` (atau pytest).
"""

import ml_lib.predict as predict
from ml_lib.pengapuran import simulasi
from ml_lib.predict import NusaCropModel

HUJAN = [287, 266, 311, 218, 118, 98, 45, 31, 76, 152, 311, 285]     # Grobogan
LAHAN = dict(ph=5.5, temp_c=27.3, elevation_m=31, soil_texture="clay loam", monthly_rain=HUJAN)
SEPULUH = {"singkong", "ubi_jalar", "jagung", "sorgum", "kacang_tanah", "kacang_hijau",
           "kacang_panjang", "cabai_rawit", "terong", "kemiri"}
BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]

def buat(pakai_ml):
    lama = predict.PAKAI_SKOR_ML
    predict.PAKAI_SKOR_ML = pakai_ml
    try:
        return NusaCropModel()
    finally:
        predict.PAKAI_SKOR_ML = lama


def tanpa_ml():
    return buat(False)


# Mode ML diuji walau saklar bawaannya mati (M.ml None kalau model belum dilatih).
M = buat(True)


def cek_kontrak_wajib(hasil):
    assert {h["crop_code"] for h in hasil} == SEPULUH
    for h in hasil:
        assert 0 <= h["skor_aturan"] <= 100
        assert set(h["rincian"]) == {"ph", "rainfall", "temp", "elevation", "texture"}
        assert all(0 <= v <= 1 for v in h["rincian"].values())
        assert h["mulai_tanam"] in BULAN
        assert h["musim_tanam_mm"] >= 0
        assert type(h["confidence"]) is float and 0 <= h["confidence"] <= 1


def test_kontrak_wajib():
    for m in (M, tanpa_ml()):
        cek_kontrak_wajib(m.recommend(**LAHAN))
        cek_kontrak_wajib(m.recommend(**LAHAN, mulai=5))
        assert len(m.recommend(**LAHAN, top_k=3)) == 3
        assert m.recommend(**LAHAN, mulai="Jun") == m.recommend(**LAHAN, mulai=5)


def test_confidence_semua_tanaman_tidak_biner():
    for m in (M, tanpa_ml()):
        c = [h["confidence"] for h in m.recommend(**LAHAN)]
        assert len(c) == 10 and all(x is not None for x in c)
        assert len(set(c)) >= 4 and any(0.05 < x < 0.95 for x in c)
        assert m.recommend(**LAHAN) == m.recommend(**LAHAN)      # deterministik


def test_confidence_naik_kalau_ph_terukur():
    for m in (M, tanpa_ml()):
        api = sum(h["confidence"] for h in m.recommend(**LAHAN, ph_q05=4.2, ph_q95=7.0))
        terukur = sum(h["confidence"] for h in m.recommend(**LAHAN, ph_terukur=True))
        assert terukur > api


def test_saklar_bawaan_hidup():
    assert predict.PAKAI_SKOR_ML is True and NusaCropModel().ml is not None


def test_hibrida_hanya_mengoreksi_tanaman_ml():
    """Tanaman tanpa model: skor_akhir = skor_aturan. Tanaman ber-model: skor
    aturan digeser paling banyak 30%."""
    if M.ml is None:
        return
    for mulai in (None, 5):
        for h in M.recommend(**LAHAN, mulai=mulai):
            if h["sumber_skor_tanaman"] == "aturan":
                assert h["skor_akhir"] == h["skor_aturan"]
            else:
                assert (predict.KOREKSI_ML_MIN * h["skor_aturan"] - 0.01 <= h["skor_akhir"]
                        <= min(100, predict.KOREKSI_ML_MAKS * h["skor_aturan"]) + 0.01)


def test_saklar_mati():
    hasil = tanpa_ml().recommend(**LAHAN)
    assert all("skor_akhir" not in h and "skor_per_bulan" not in h for h in hasil)
    skor = [h["skor_aturan"] for h in hasil]
    assert skor == sorted(skor, reverse=True)


def test_parameter_tambahan_opsional():
    for m in (M, tanpa_ml()):
        cek_kontrak_wajib(m.recommend(**LAHAN, sand_persen=33.0, clay_persen=34.0,
                                      ph_q05=4.8, ph_q95=6.3, ph_terukur=False))
        dua = m.recommend_dua(**LAHAN, bulan_sekarang="Agu", ph_terukur=True)
        assert dua["bulan_sekarang"] == "Agu"
        cek_kontrak_wajib(dua["kondisi_lahan"])
        cek_kontrak_wajib(dua["tanam_sekarang"])


def test_skor_ml():
    if M.ml is None:
        return
    hasil = M.recommend(**LAHAN)
    akhir = [h["skor_akhir"] for h in hasil]
    assert akhir == sorted(akhir, reverse=True)
    siklus = {p["crop_code"]: p["cycle_months"] for p in M.profiles}
    for h in hasil:
        assert 0 <= h["skor_akhir"] <= 100 and len(h["skor_per_bulan"]) == 12
        assert h["sumber_skor_tanaman"] in ("ml", "aturan")
        # tanpa `mulai`, skor_akhir = bulan tanam terbaik
        assert abs(h["skor_akhir"] - max(h["skor_per_bulan"])) < 0.02
        if siklus[h["crop_code"]] >= 12:
            assert len(set(h["skor_per_bulan"])) == 1
    for bulan in (0, 5, 11):
        for h in M.recommend(**LAHAN, mulai=bulan):
            assert abs(h["skor_akhir"] - h["skor_per_bulan"][bulan]) < 0.02
    aturan = [h["skor_aturan"] for h in M.recommend(**LAHAN, sort_by="aturan")]
    assert aturan == sorted(aturan, reverse=True)


def test_kenaikan_ph_tidak_menurunkan_skor():
    """Uji kewajaran pH: tanah masam yang dikapur tidak boleh membuat tanaman
    yang peka kemasaman jadi lebih buruk."""
    peka = {p["crop_code"] for p in M.profiles if p["acid_tolerance"] == "rendah"}
    for m in (M, tanpa_ml()):
        kunci = "skor_akhir" if m.ml is not None else "skor_aturan"
        for tekstur in ("sandy loam", "clay loam", "clay"):
            for suhu, tinggi in ((27.3, 31), (22.0, 700)):
                sebelum = None
                for ph in (4.5, 5.0, 5.5, 6.0):
                    skor = {h["crop_code"]: h[kunci] for h in m.recommend(
                        ph=ph, temp_c=suhu, elevation_m=tinggi, soil_texture=tekstur,
                        monthly_rain=HUJAN)}
                    if sebelum:
                        for kode in peka:
                            assert skor[kode] >= sebelum[kode] - 0.01, (kode, ph, tekstur)
                    sebelum = skor


def test_integrasi_kapur():
    """ph_baru dari simulasi() masuk ke recommend(); perubahan peringkat masuk akal."""
    kapur = simulasi(ph_awal=4.6, tekstur_kelas="clay loam", target_ph=5.8)
    for m in (M, tanpa_ml()):
        kunci = "skor_akhir" if m.ml is not None else "skor_aturan"
        dasar = dict(LAHAN, ph=4.6)
        lama = {h["crop_code"]: h for h in m.recommend(**dasar, ph_terukur=True)}
        baru = {h["crop_code"]: h for h in m.recommend(**dict(dasar, ph=kapur["ph_baru"]),
                                                       ph_terukur=True)}
        for kode in SEPULUH:
            assert baru[kode]["rincian"]["ph"] >= lama[kode]["rincian"]["ph"]
        # terong peka masam: faktor pH dan skornya harus naik jelas setelah dikapur
        assert lama["terong"]["rincian"]["ph"] < 0.5 < baru["terong"]["rincian"]["ph"]
        assert baru["terong"][kunci] > lama["terong"][kunci] + 10


if __name__ == "__main__":
    print("skor ML aktif:", M.ml is not None)
    for nama, fn in sorted(globals().items()):
        if nama.startswith("test_") and callable(fn):
            fn()
            print("lulus", nama)
