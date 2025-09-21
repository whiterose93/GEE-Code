# Land-cover classification pipeline

Pipeline ini mengunduh data **Google Satellite Embedding V1 Annual**, melakukan pelatihan model jaringan saraf (PyTorch), dan menghasilkan peta klasifikasi tutupan lahan sesuai shapefile yang ditempatkan pada folder `shp`.

## Struktur folder

```
GEE-Code/
├── main.py                     # Entry point pipeline
├── requirements.txt            # Dependensi Python
├── src/gee_landcover_pipeline/ # Modul utilitas pipeline
└── shp/                        # Letakkan shapefile di sini
```

> **Catatan:** Berkas shapefile untuk training harus memiliki kolom atribut `class_id` (atau nama kolom yang ditetapkan melalui argumen `--target-property`) yang memuat kode kelas sesuai daftar berikut:
>
> | Kode | Kelas                |
> |-----:|----------------------|
> | 10   | Tree cover           |
> | 20   | Shrubland            |
> | 30   | Grassland            |
> | 40   | Cropland             |
> | 50   | Built-up             |
> | 60   | Bare                 |
> | 80   | Water                |
> | 90   | Herbaceous wetland   |
> | 95   | Mangroves            |
>
> Shapefile inference (tanpa label) cukup berisi geometri area of interest.

## Persiapan lingkungan

1. Buat dan aktifkan lingkungan Python (virtualenv/conda) sesuai kebutuhan.
2. Instal dependensi:

   ```bash
   pip install -r requirements.txt
   ```

3. Autentikasi Google Earth Engine:

   ```bash
   earthengine authenticate
   ```

## Menjalankan pipeline

Letakkan semua shapefile pada direktori `shp/`. Setelah dependensi terpasang dan autentikasi Earth Engine selesai, jalankan:

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
