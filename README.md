# Land-cover classification pipeline

Pipeline ini mengunduh data **Google Satellite Embedding V1 Annual**, melakukan pelatihan model jaringan saraf (PyTorch), dan menghasilkan peta klasifikasi tutupan lahan sesuai shapefile yang ditempatkan pada folder `shp`.

## Struktur folder

```
GEE-Code/
├── main.py                     # Entry point pipeline
├── requirements.txt            # Dependensi Python
├── src/gee_landcover_pipeline/ # Modul utilitas pipeline
└── shp/
    ├── picker/                # Shapefile training, satu kelas per berkas
    └── location/              # Shapefile area of interest untuk inferensi
```

> **Catatan:** Tiap shapefile pada folder `shp/picker/` merepresentasikan satu kelas. Nama berkas (tanpa ekstensi) dipetakan otomatis ke kode kelas berikut dan disisipkan sebagai kolom `class_id` serta `class_name` saat proses unduh sampel dijalankan:
>
> | Kode | Nama shapefile      | Kelas output         |
> |-----:|---------------------|----------------------|
> | 0    | `forest`            | Forest               |
> | 1    | `mangrove`          | Mangrove             |
> | 2    | `built`             | Built-up             |
> | 3    | `bareland`          | Bareland             |
> | 4    | `water`             | Water                |
> | 5    | `vegetation`        | Vegetation           |
> | 6    | `low vegetation`    | Low Vegetation       |
> | 7    | `forest plantation` | Forest Plantation    |
> | 8    | `palm plantation`   | Palm Plantation      |
>
> Jika struktur nama berbeda, perbarui pemetaan pada konstanta `SHAPEFILE_CLASS_MAP` di `src/gee_landcover_pipeline/config.py`. Shapefile inferensi pada folder `shp/location/` cukup berisi geometri area of interest.

## Persiapan lingkungan

1. Buat dan aktifkan lingkungan Python (virtualenv/conda) sesuai kebutuhan.
2. Instal dependensi:

   ```bash
   pip install -r requirements.txt
   ```

3. Salin file kunci service account Google Earth Engine (`ee-faizfajrice-f9b66d6b94e8.json`) ke direktori `key/` pada root proyek. Nama file dan akun layanan harus cocok dengan nilai bawaan pada `initialize_earth_engine()` atau disesuaikan melalui parameter fungsi tersebut.

## Menjalankan pipeline

Letakkan shapefile training ke dalam `shp/picker/` dan shapefile area of interest (tanpa label) pada `shp/location/`. Setelah dependensi terpasang dan kredensial Earth Engine tersedia, jalankan:

```bash
python main.py --shapefile-dir shp
```

Secara default pipeline akan:

1. Mengunduh embedding untuk setiap shapefile dan tahun (default 2023) ke direktori `data/` dalam format GeoParquet.
2. Menggunakan data berlabel untuk melatih model klasifikasi (`output/landcover_classifier.pt`). Riwayat pelatihan tersimpan sebagai `output/landcover_classifier.history.json`.
3. Melakukan inferensi pada shapefile tanpa label dan menulis hasil prediksi ke folder `output/` (`*_predictions.geojson`).

### Opsi penting

- `--years`: daftar tahun yang akan diunduh, misal `--years 2022 2023`.
- `--sample-per-class`: jumlah sampel per kelas untuk training.
- `--max-inference-pixels`: batas jumlah piksel yang digunakan untuk inferensi per shapefile.
- `--target-property`: nama kolom label pada shapefile training (default `class_id`).
- `--output-format`: format hasil prediksi (`geojson`, `gpkg`, `parquet`).
- `--no-probabilities`: tidak menuliskan probabilitas per kelas pada hasil inferensi.

Untuk melihat semua opsi yang tersedia:

```bash
python main.py --help
```

## Output

- `data/*.parquet` : embedding hasil sampling untuk training/inference.
- `output/landcover_classifier.pt` : bobot model PyTorch.
- `output/landcover_classifier.metadata.json` : metadata model dan konfigurasi.
- `output/landcover_classifier.history.json` : riwayat loss/akurasi selama training.
- `output/*_predictions.geojson` : hasil inferensi tutupan lahan, termasuk kolom `predicted_class`, `predicted_name`, `predicted_color`, `confidence`, dan (opsional) `prob_<kode>` untuk probabilitas per kelas.

## Catatan tambahan

- Sampling Earth Engine menggunakan `stratifiedSample`, sehingga area training yang sangat luas mungkin memerlukan penyesuaian parameter `--sample-per-class`, `--tile-scale`, atau penggunaan ROI yang lebih kecil.
- Model multi-layer perceptron sederhana disediakan sebagai baseline; arsitektur dapat dimodifikasi melalui parameter `--hidden-dims`, `--dropout`, dan `--num-epochs`.
- Pastikan proyeksi shapefile adalah WGS84 (`EPSG:4326`). Script akan melempar error jika CRS tidak terdefinisi.
