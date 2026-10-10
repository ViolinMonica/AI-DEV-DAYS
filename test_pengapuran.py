"""Test simulasi kapur. Jalankan: `python test_pengapuran.py` (atau pytest)."""

import json
import os

from ml_lib import pengapuran
from ml_lib.pengapuran import simulasi

TEKSTUR = ["sand", "sandy loam", "loam", "clay loam", "clay", None]


def gagal(**kw):
    try:
        simulasi(**kw)
    except ValueError as e:
        assert str(e)
        return True
    return False


def test_contoh_be():
    for h in (simulasi(ph_awal=4.8, tekstur_kelas="clay loam", organic_carbon=14.0, target_ph=6.0),
              simulasi(ph_awal=5.2, tekstur_kelas=None, dosis_ton_ha=2.0, jenis_kapur="kalsit")):
        assert type(h["ph_baru"]) is float and type(h["dosis_ton_ha"]) is float
        assert type(h["catatan"]) is list and all(type(c) is str for c in h["catatan"])
        assert h["ph_baru"] == round(h["ph_baru"], 2)
    assert gagal(ph_awal=6.5, tekstur_kelas="loam", target_ph=6.0)
    assert gagal(ph_awal=5.0, tekstur_kelas="loam")


def test_error():
    assert gagal(ph_awal=5.0, tekstur_kelas="loam", dosis_ton_ha=1.0, target_ph=6.0)
    assert gagal(ph_awal=5.0, tekstur_kelas="lempung", dosis_ton_ha=1.0)
    assert gagal(ph_awal=5.0, tekstur_kelas="loam", dosis_ton_ha=1.0, jenis_kapur="gipsum")
    assert gagal(ph_awal=4.0, tekstur_kelas="clay", organic_carbon=40.0, target_ph=7.5)  # > 20 t/ha
    assert gagal(ph_awal=5.0, tekstur_kelas="loam", dosis_ton_ha=-1.0)
    assert gagal(ph_awal=5.0, tekstur_kelas="loam", dosis_ton_ha=25.0)
    assert gagal(ph_awal=float("nan"), tekstur_kelas="loam", dosis_ton_ha=1.0)
    assert gagal(ph_awal="5", tekstur_kelas="loam", dosis_ton_ha=1.0)
    assert gagal(ph_awal=2.0, tekstur_kelas="loam", dosis_ton_ha=1.0)
    assert gagal(ph_awal=5.0, tekstur_kelas="loam", target_ph=8.0)


def test_variasi_penulisan():
    a = simulasi(ph_awal=5.0, tekstur_kelas="clay loam", dosis_ton_ha=2.0, jenis_kapur="kapur_tohor")
    b = simulasi(ph_awal=5.0, tekstur_kelas="Clay_Loam", dosis_ton_ha=2.0, jenis_kapur="Kapur Tohor")
    assert a == b and a["tekstur_dipakai"] == "clay loam" and a["jenis_kapur"] == "kapur_tohor"


def test_monoton():
    for tek in TEKSTUR:
        for ph0 in (3.5, 4.5, 5.5, 6.5, 8.5):
            ph = [simulasi(ph_awal=ph0, tekstur_kelas=tek, dosis_ton_ha=d)["ph_baru"]
                  for d in (0, 0.5, 1, 2, 4, 8, 20)]
            assert ph[0] == ph0
            assert all(y >= x for x, y in zip(ph, ph[1:])), (tek, ph0, ph)
            assert ph[-1] <= max(ph0, pengapuran.PH_PLAFON)


def test_dua_arah_konsisten():
    for tek in TEKSTUR:
        for jenis in pengapuran.CCE:
            for ph0, target in ((4.2, 5.0), (4.8, 5.5), (5.0, 6.0), (5.5, 6.5)):
                t = simulasi(ph_awal=ph0, tekstur_kelas=tek, target_ph=target, jenis_kapur=jenis)
                assert t["ph_baru"] == target
                d = simulasi(ph_awal=ph0, tekstur_kelas=tek, dosis_ton_ha=t["dosis_ton_ha"],
                             jenis_kapur=jenis)
                assert abs(d["ph_baru"] - target) < 0.02, (tek, jenis, ph0, target)


def test_urutan_agronomi():
    def dosis(**kw):
        return simulasi(ph_awal=4.8, target_ph=5.5, **kw)["dosis_ton_ha"]
    assert dosis(tekstur_kelas="clay") > dosis(tekstur_kelas="loam") > dosis(tekstur_kelas="sand")
    assert dosis(tekstur_kelas="loam", jenis_kapur="kapur_tohor") < dosis(tekstur_kelas="loam")
    assert (dosis(tekstur_kelas="loam", organic_carbon=30.0)
            > dosis(tekstur_kelas="loam", organic_carbon=5.0))


def test_c_organik_dibatasi():
    # C-organik SoilGrids (median 53 g/kg) tidak boleh menggandakan dosis
    biasa = simulasi(ph_awal=4.8, tekstur_kelas="loam", organic_carbon=25.0, target_ph=5.5)
    tinggi = simulasi(ph_awal=4.8, tekstur_kelas="loam", organic_carbon=189.0, target_ph=5.5)
    assert tinggi["dosis_ton_ha"] == biasa["dosis_ton_ha"]
    assert tinggi["organic_carbon_dibatasi"] and not biasa["organic_carbon_dibatasi"]
    assert tinggi["organic_carbon_dipakai"] == 25.0


def test_al_dd():
    est = simulasi(ph_awal=4.5, tekstur_kelas="clay", dosis_ton_ha=2.0)
    assert est["al_dd_sumber"] == "estimasi dari pH"
    assert 0 <= est["al_dd_baru"] < est["al_dd_awal"]
    ukur = simulasi(ph_awal=4.5, tekstur_kelas="clay", dosis_ton_ha=2.0, al_dd=3.0)
    assert ukur["al_dd_sumber"] == "terukur" and ukur["al_dd_awal"] == 3.0
    assert 0 <= ukur["al_dd_baru"] < 3.0
    # 1 x Al-dd dalam berat produk
    assert abs(ukur["dosis_penetral_al_ton_ha"] - 3.0 * 1.2 / pengapuran.CCE["dolomit"]) < 0.01
    assert simulasi(ph_awal=6.0, tekstur_kelas="loam", dosis_ton_ha=1.0)["al_dd_awal"] == 0.0


def test_catatan():
    def teks(**kw):
        return " ".join(simulasi(**kw)["catatan"])
    assert "tidak masam" in teks(ph_awal=6.8, tekstur_kelas="loam", dosis_ton_ha=1.0)
    assert "sarung tangan" in teks(ph_awal=5.0, tekstur_kelas="loam", dosis_ton_ha=1.0,
                                   jenis_kapur="kapur_tohor")
    assert "tidak diisi" in teks(ph_awal=5.0, tekstur_kelas=None, dosis_ton_ha=1.0)
    assert "dibagi" in teks(ph_awal=5.0, tekstur_kelas="loam", dosis_ton_ha=4.0)
    assert "5,5-6,5" in teks(ph_awal=5.0, tekstur_kelas="sand", target_ph=7.0)
    assert simulasi(ph_awal=5.0, tekstur_kelas="loam", organic_carbon=12.0, al_dd=1.0,
                    dosis_ton_ha=1.0)["catatan"] == [
        "Untuk menetralkan aluminium saja, perkiraan kebutuhannya sekitar 1,3 ton per hektare."]


def test_parameter_sama_dengan_hasil_latih():
    """Konstanta di modul harus sama dengan keluaran latih_pengapuran.py."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "laporan", "pengapuran_hasil.json")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        h = json.load(f)
    assert abs(pengapuran.K_BC - h["parametrik_k"]) < 0.005
    assert pengapuran.PH_PLAFON == h["parametrik_plafon"]
    assert abs(pengapuran.OC_DEFAULT_GKG - h["oc_median_gkg"]) < 0.05


if __name__ == "__main__":
    for nama, fn in sorted(globals().items()):
        if nama.startswith("test_") and callable(fn):
            fn()
            print("lulus", nama)
