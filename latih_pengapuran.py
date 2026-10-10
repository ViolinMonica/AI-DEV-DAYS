"""
Latih dan bandingkan model simulasi kapur (T1).

Jalankan dari root repo:

    python latih_pengapuran.py [path/ke/NUSA_CORP - DATA.xlsx]

Membaca sheet `1c Pengapuran` dan `2a Tanah`, menulis:
  data/pengapuran_latih.csv        baris yang dipakai (dengan asal-usul dosis)
  data/aldd_latih.csv              pasangan pH-Al-dd
  laporan/pengapuran_hasil.json    tabel leave-one-study-out + parameter terpilih
  laporan/pengapuran_pred_vs_ukur.png

Model terpilih: C3 (k = median dari k tiap studi). Parameter terpilih DISALIN ke konstanta di ml_lib/pengapuran.py (BE meminta satu
file tanpa berkas pendamping). test_pengapuran.py memeriksa keduanya sama.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from ml_lib.pengapuran import (BC_TEKSTUR, BC_PER_OC, PH_PLAFON_AWAL,
                               FAKTOR_AL_TEKSTUR, TEKSTUR_DEFAULT, _ph_dari_dosis)

XLSX = sys.argv[1] if len(sys.argv) > 1 else os.path.join("..", "NUSA_CORP - DATA.xlsx")

# t CaCO3/ha per (g CaCO3/kg tanah): lapisan olah 20 cm, bobot isi 1,2 g/cm3
# = 2,4 juta kg tanah/ha. Asumsi yang sama dengan sheet Rumus_Bantu.
T_HA_PER_G_KG = 2.4

# Daya netralisasi (setara CaCO3) untuk studi yang tidak melaporkan NP.
CCE_MURNI = {"cao": 1.785, "ca(oh)2": 1.351, "caco3": 1.0, "dolomit": 1.086}
CCE_DOLOMIT_LAPANG = 0.93   # sama dengan ml_lib.pengapuran.CCE["dolomit"]

# Studi yang dipakai untuk model pH, dan alasan studi lain dibuang.
DIBUANG = {
    "S01": "pH akhir tidak diambil", "S12": "tidak ada pH akhir (hanya hasil umbi)",
    "S13": "pH KCl, tidak boleh dicampur dengan pH H2O",
    "S21": "tidak ada pH", "S23": "tidak ada pH akhir", "S24": "sekunder, tanpa pH",
    "S33": "dosis absolut tidak dilaporkan", "S34": "tidak ada pH akhir",
    "S35": "duplikat sekunder dari S12, tanpa pH",
    "S36": "dosis hanya sebagai kelipatan Al-dd (faktor konversi tidak baku)",
}


def _f(x):
    try:
        v = float(x)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


def baca_1c():
    df = pd.read_excel(XLSX, sheet_name="1c Pengapuran", header=3)
    df = df[df["study_id"].notna()]
    return df[df["status_baris"] != "CONTOH"].reset_index(drop=True)


def dosis_caco3(r):
    """(t CaCO3/ha, keterangan asal) untuk satu baris, atau (None, alasan)."""
    sid, jenis = r["study_id"], str(r["jenis_kapur"]).lower()
    caco3, produk = _f(r["dosis_setara_caco3_t_ha"]), _f(r["dosis_produk_t_ha"])
    g_kg = _f(r["dosis_g_produk_per_kg_tanah"])
    if caco3 is not None:
        return caco3, "kolom setara CaCO3 (NP dilaporkan)"
    if produk is not None:
        if "dolomit" in jenis:
            return produk * CCE_DOLOMIT_LAPANG, "produk x 0,93 (NP dolomit tidak dilaporkan)"
        return produk, "produk x 1,00 (kalsit, kemurnian tidak dilaporkan)"
    if g_kg is not None and sid in ("S08", "S09"):
        if sid == "S08":
            cce = CCE_DOLOMIT_LAPANG
        else:
            cce = next((v for k, v in CCE_MURNI.items() if k in jenis), 1.0)
        return g_kg * cce * T_HA_PER_G_KG, f"g/kg x CCE {cce} x 2,4 (asumsi 20 cm, BD 1,2)"
    return None, "tidak ada dosis absolut"


def susun_data_ph(df):
    """Baris berdosis > 0 dengan pH kontrol blok yang sama sebagai pH awal."""
    baris, kontrol = [], None
    for _, r in df.iterrows():
        sid = r["study_id"]
        if sid in DIBUANG or _f(r["ph_awal"]) is None:
            continue           # ph_awal kosong = lapisan 20-40 cm S22 (kapur di permukaan)
        d, asal = dosis_caco3(r)
        if d is None:
            continue
        if d == 0:
            # Kontrol membuka blok baru. pH kontrol sesudah musim lebih tepat
            # daripada pH sebelum percobaan (S22: 5,34 sebelum olah vs 4,5 kontrol).
            kontrol = _f(r["ph_akhir"]) if _f(r["ph_akhir"]) is not None else _f(r["ph_awal"])
            continue
        if _f(r["ph_akhir"]) is None:
            continue
        ph0 = kontrol if kontrol is not None else _f(r["ph_awal"])
        liat = _f(r["liat_pct"])
        oc = _f(r["c_organik_awal_pct"])
        baris.append({
            "study_id": sid, "status_baris": r["status_baris"],
            "ph_awal": ph0, "dosis_caco3": round(d, 3), "ph_akhir": _f(r["ph_akhir"]),
            # liat 33% tanpa fraksi pasir: kelas USDA terdekat clay loam (S15)
            "tekstur": "clay loam" if liat is not None else None,
            "oc_gkg": None if oc is None else oc * 10,
            "asal_dosis": asal, "jenis_kapur": r["jenis_kapur"],
        })
    return pd.DataFrame(baris)


def susun_data_al(df):
    """Pasangan (pH H2O, Al-dd) dari 1c (awal dan sesudah kapur) dan 2a."""
    p = []
    for _, r in df.iterrows():
        sid = r["study_id"]
        if sid in ("S13", "S23"):   # S13 pH KCl; S23 Al-dd 30 cmol/kg (polibag, pencilan)
            continue
        if _f(r["ph_awal"]) is not None and _f(r["al_dd_awal_cmol_kg"]) is not None:
            p.append((sid, _f(r["ph_awal"]), _f(r["al_dd_awal_cmol_kg"]), "awal"))
        # S20 dosis 3,2 t/ha: Al naik setelah dikapur (anomali sumber) -> tidak dipakai
        anomali = sid == "S20" and _f(r["dosis_produk_t_ha"]) == 3.2
        if (_f(r["ph_akhir"]) is not None and _f(r["al_dd_akhir_cmol_kg"]) is not None
                and not anomali):
            p.append((sid, _f(r["ph_akhir"]), _f(r["al_dd_akhir_cmol_kg"]), "sesudah"))
    t = pd.read_excel(XLSX, sheet_name="2a Tanah", header=3)
    for _, r in t.iterrows():
        if _f(r["ph_h2o"]) is not None and _f(r["al_dd_cmol_kg"]) is not None:
            p.append((str(r["titik_id"])[:3], _f(r["ph_h2o"]), _f(r["al_dd_cmol_kg"]), "2a"))
    return pd.DataFrame(p, columns=["grup", "ph", "al_dd", "asal"]).drop_duplicates()


# ---------------------------------------------------------------- model pH

def _bc(d, k=1.0):
    tek = d["tekstur"].fillna(TEKSTUR_DEFAULT).map(BC_TEKSTUR)
    return k * (tek + BC_PER_OC * d["oc_gkg"].fillna(OC_MEDIAN))


def pred_param(d, k, plafon):
    return np.array([_ph_dari_dosis(p, D, b, plafon)
                     for p, D, b in zip(d["ph_awal"], d["dosis_caco3"], _bc(d, k))])


def fit_param(d, fit_plafon):
    def res(t):
        return pred_param(d, t[0], t[1] if fit_plafon else PH_PLAFON_AWAL) - d["ph_akhir"]
    if fit_plafon:
        # plafon dibatasi di atas pH terukur tertinggi supaya R tetap positif
        s = least_squares(res, [1.0, 7.0], bounds=([0.05, 6.8], [20, 8.3]))
        return s.x[0], s.x[1]
    s = least_squares(res, [1.0], bounds=([0.05], [20]))
    return s.x[0], PH_PLAFON_AWAL


def fit_param_median(d):
    """k = median dari k tiap studi: tiap studi berbobot sama, jadi tidak
    ditarik oleh satu studi ekstrem (S09 inkubasi 2 minggu, S15 pot sawah)."""
    ks = [fit_param(d[d["study_id"] == g], False)[0] for g in d["study_id"].unique()]
    return float(np.median(ks)), PH_PLAFON_AWAL


def fitur_linear(d, kuadrat):
    cols = [np.ones(len(d)), d["ph_awal"], d["dosis_caco3"]]
    if kuadrat:
        cols += [d["dosis_caco3"] ** 2, d["dosis_caco3"] * d["ph_awal"]]
    return np.column_stack(cols)


def buat_model():
    """nama -> (fit(train) -> state, predict(state, test))."""
    from xgboost import XGBRegressor

    def fit_lin(kuadrat):
        return lambda tr: np.linalg.lstsq(fitur_linear(tr, kuadrat),
                                          tr["ph_akhir"], rcond=None)[0]

    def fit_xgb(tr):
        m = XGBRegressor(n_estimators=60, max_depth=2, learning_rate=0.1,
                         monotone_constraints=(0, 1), random_state=42, n_jobs=1)
        return m.fit(tr[["ph_awal", "dosis_caco3"]], tr["ph_akhir"])

    return {
        "A. pH tidak berubah": (lambda tr: None, lambda s, te: te["ph_awal"].to_numpy()),
        "B. Rumus tanpa training": (lambda tr: None,
                                    lambda s, te: pred_param(te, 1.0, PH_PLAFON_AWAL)),
        "C1. Parametrik, fit k": (lambda tr: fit_param(tr, False),
                                  lambda s, te: pred_param(te, *s)),
        "C2. Parametrik, fit k dan plafon": (lambda tr: fit_param(tr, True),
                                             lambda s, te: pred_param(te, *s)),
        "C3. Parametrik, k median per studi": (fit_param_median,
                                               lambda s, te: pred_param(te, *s)),
        "D1. Regresi linear": (fit_lin(False), lambda s, te: fitur_linear(te, False) @ s),
        "D2. Regresi polinomial": (fit_lin(True), lambda s, te: fitur_linear(te, True) @ s),
        "E. XGBoost monoton": (fit_xgb,
                               lambda s, te: s.predict(te[["ph_awal", "dosis_caco3"]])),
    }


def loso(d, fit, pred, kol_y="ph_akhir", kol_g="study_id"):
    out = np.zeros(len(d))
    for g in d[kol_g].unique():
        uji = (d[kol_g] == g).to_numpy()
        out[uji] = pred(fit(d[~uji]), d[uji])
    e = out - d[kol_y].to_numpy()
    per_studi = pd.Series(np.abs(e)).groupby(d[kol_g].to_numpy()).mean()
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
            # rata-rata MAE per studi: supaya studi berbaris banyak tidak mendominasi
            "mae_per_studi": float(per_studi.mean())}, out


def cek_syarat(fit, pred, d):
    """Syarat wajib pada grid dosis 0-20 t/ha, model dilatih di semua data."""
    s = fit(d)
    dosis = np.linspace(0, 20, 81)
    monoton, naik_tegas, maks = True, True, -np.inf
    for ph0 in (4.0, 4.5, 5.0, 5.5, 6.0):
        for tek in list(BC_TEKSTUR) + [None]:
            g = pd.DataFrame({"ph_awal": ph0, "dosis_caco3": dosis,
                              "tekstur": tek, "oc_gkg": OC_MEDIAN})
            y = np.asarray(pred(s, g), dtype=float)
            dy = np.diff(y)
            monoton &= bool((dy >= -1e-9).all())
            naik_tegas &= bool((dy > 1e-9).all())
            maks = max(maks, y.max())
    return {"monoton": monoton, "bisa_dibalik": naik_tegas,
            "ph_maks_di_grid": round(float(maks), 2), "terbatas": bool(maks <= 8.5)}


# ---------------------------------------------------------------- model Al-dd

def al_bentuk(ph, a, b, p0=5.5):
    return a * np.clip(p0 - np.asarray(ph, dtype=float), 0, None) ** b


def fit_al(tr):
    s = least_squares(lambda t: al_bentuk(tr["ph"], *t) - tr["al_dd"],
                      [2.0, 1.3], bounds=([0.01, 0.3], [50, 4]))
    return s.x


def main():
    global OC_MEDIAN
    os.makedirs("data", exist_ok=True)
    os.makedirs("laporan", exist_ok=True)

    df = baca_1c()
    d = susun_data_ph(df)
    # median per studi dari studi yang melaporkan C-organik
    OC_MEDIAN = float(d.dropna(subset=["oc_gkg"]).groupby("study_id")["oc_gkg"]
                      .first().median())
    d.to_csv("data/pengapuran_latih.csv", index=False)

    print(f"1c Pengapuran: {len(df)} baris, status {df['status_baris'].value_counts().to_dict()}")
    print(f"Dipakai untuk model pH: {len(d)} baris dari {d['study_id'].nunique()} studi")
    print(d.groupby(["study_id", "status_baris"]).size().to_string())
    print(f"Rentang dosis: {d['dosis_caco3'].min()}-{d['dosis_caco3'].max()} t CaCO3/ha; "
          f"pH awal {d['ph_awal'].min()}-{d['ph_awal'].max()}; C-organik median {OC_MEDIAN} g/kg\n")

    model = buat_model()
    tabel, prediksi = {}, {}
    for nama, (fit, pred) in model.items():
        skor, prediksi[nama] = loso(d, fit, pred)
        tabel[nama] = {**skor, **cek_syarat(fit, pred, d)}
    print(pd.DataFrame(tabel).T.to_string(float_format=lambda v: f"{v:.3f}"), "\n")

    k, plafon = fit_param(d, False)
    k2, plafon2 = fit_param(d, True)
    print(f"C1 semua data: k={k:.4f} (plafon {plafon});  C2: k={k2:.4f} plafon={plafon2:.3f}")
    k_per_studi = {g: float(fit_param(d[d["study_id"] == g], False)[0])
                   for g in d["study_id"].unique()}
    print("k per studi:", {g: round(v, 2) for g, v in k_per_studi.items()})
    # Sensitivitas terhadap studi yang dosisnya dikonversi dari g/kg (S08, S09)
    # dan inkubasi 2 minggu yang belum tuntas (S09).
    sensitivitas = {
        "tanpa S09": float(fit_param(d[d["study_id"] != "S09"], False)[0]),
        "tanpa S08 dan S09": float(fit_param(d[~d["study_id"].isin(["S08", "S09"])], False)[0]),
        "median k per studi": float(np.median(list(k_per_studi.values()))),
    }
    print("sensitivitas k:", {g: round(v, 2) for g, v in sensitivitas.items()}, "\n")

    # Al-dd dari pH
    a = susun_data_al(df)
    a.to_csv("data/aldd_latih.csv", index=False)
    f_loam = FAKTOR_AL_TEKSTUR[TEKSTUR_DEFAULT]
    al_tabel = {
        "Rumus brief (2,0 x (5,5-pH)^1,3)": loso(
            a, lambda tr: None, lambda s, te: f_loam * al_bentuk(te["ph"], 2.0, 1.3),
            "al_dd", "grup")[0],
        "Fit a dan b (p0 = 5,5)": loso(
            a, fit_al, lambda s, te: al_bentuk(te["ph"], *s), "al_dd", "grup")[0],
        "Rata-rata (tanpa pH)": loso(
            a, lambda tr: tr["al_dd"].mean(),
            lambda s, te: np.full(len(te), s), "al_dd", "grup")[0],
    }
    al_a, al_b = fit_al(a)
    print(f"Al-dd: {len(a)} pasangan dari {a['grup'].nunique()} grup")
    print(pd.DataFrame(al_tabel).T.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"Fit semua data: Al = {al_a:.3f} x (5,5 - pH)^{al_b:.3f}\n")

    hasil = {
        "n_baris_1c": int(len(df)),
        "status_1c": df["status_baris"].value_counts().to_dict(),
        "n_latih": int(len(d)),
        "baris_per_studi": d.groupby("study_id").size().to_dict(),
        "status_latih": d["status_baris"].value_counts().to_dict(),
        "dibuang": DIBUANG,
        "rentang_dosis_caco3": [float(d["dosis_caco3"].min()), float(d["dosis_caco3"].max())],
        "oc_median_gkg": OC_MEDIAN,
        "loso_ph": tabel,
        "model_terpilih": TERPILIH,
        "parametrik_k": sensitivitas["median k per studi"],
        "parametrik_plafon": float(plafon),
        "c1_k": float(k),
        "k_per_studi": k_per_studi, "sensitivitas_k": sensitivitas,
        "c2_k": float(k2), "c2_plafon": float(plafon2),
        "n_al": int(len(a)), "loso_al": al_tabel,
        "al_fit_a": float(al_a), "al_fit_b": float(al_b),
    }
    with open("laporan/pengapuran_hasil.json", "w", encoding="utf-8") as f:
        json.dump(hasil, f, indent=1, ensure_ascii=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.2), sharex=True, sharey=True)
    for ax, nama in zip(axs, ("B. Rumus tanpa training", TERPILIH)):
        for g, sub in d.groupby("study_id"):
            ax.scatter(sub["ph_akhir"], prediksi[nama][sub.index], label=g, s=28)
        lim = [4.3, 7.0]
        ax.plot(lim, lim, color="0.5", lw=1)
        ax.set_title(f"{nama}\nRMSE {tabel[nama]['rmse']:.2f}, MAE {tabel[nama]['mae']:.2f}",
                     fontsize=10)
        ax.set_xlabel("pH terukur sesudah kapur")
    axs[0].set_ylabel("pH terprediksi (leave-one-study-out)")
    axs[1].legend(fontsize=8, title="studi")
    fig.tight_layout()
    fig.savefig("laporan/pengapuran_pred_vs_ukur.png", dpi=140)


OC_MEDIAN = None
TERPILIH = "C3. Parametrik, k median per studi"

if __name__ == "__main__":
    main()
