# DHWprj - Coral Bleaching Forecasting and Monitoring

Du an xay dung pipeline du bao va giam sat nguy co tay trang san ho quanh Lizard Island, Great Barrier Reef bang du lieu NOAA Coral Reef Watch va cac mo hinh Machine Learning/Deep Learning.

## 1. Trang thai du lieu

Du lieu cot loi da san sang trong repo:

- `data/dhw_5km_3231_0c35_4219_U1739212814032.nc`: NOAA CRW daily 5 km, giai doan 2010-01-01 den 2025-02-08, gom SST, SST anomaly, HotSpot, DHW, Bleaching Alert Area.
- `data/GBR_ct5km_MMM_v3.1.nc`: climatology/MMM cho khu vuc GBR.
- `data/zipfolder/`: shapefile Great Barrier Reef.
- `data/processed/`: du lieu da tien xu ly tu lan chay pipeline gan nhat.

Du lieu mo rong chua co san trong repo va dang de o che do tuy chon:

- Chlorophyll-a va PAR tu MODIS/NASA OceanColor.
- Salinity va ocean currents tu Copernicus Marine/GLORYS.
- ENSO index tu NOAA CPC ONI hoac MEI.

Pipeline hien tai van chay duoc voi bo NOAA CRW co san. Khi bo sung du lieu mo rong, dat file CSV vao `data/external/` va cau hinh trong `configs/default.yaml`.

## 2. Cau truc du an

```text
configs/
  default.yaml              # Cau hinh chinh cho data, feature, split, model, training
  data_sources.yaml         # Goi y nguon du lieu mo rong
coral_bleaching_pipeline/
  cli.py                    # Command line interface
  data.py                   # Doc NetCDF/CSV, ghep feature, tao supervised dataset
  features.py               # Alert level, lag, rolling, calendar, adjacency
  train.py                  # Orchestrate prepare, EDA, train, evaluate
  visualize.py              # Bieu do EDA va evaluation
  models/
    classical.py            # Random Forest, XGBoost, LightGBM
    deep.py                 # LSTM, GRU, CNN-LSTM, TFT-lite, ST-GNN
data/
  processed/                # Du lieu da tao boi pipeline
docs/
  pipeline_deep_learning.md # Tai lieu nghien cuu va thiet ke chi tiet
reports/
  eda/                      # Hinh phan tich du lieu
  evaluation/               # Leaderboard, confusion matrix, prediction CSV
artifacts/
  models/                   # Checkpoint/model da train
scripts/
  run_pipeline.py           # Entry point phu
tests/
```

## 3. Cai dat moi truong

Tao virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Cai package co ban:

```bash
pip install -r requirements.txt
```

Neu train tren GPU thue, cai PyTorch CUDA phu hop voi image GPU truoc, sau do:

```bash
pip install -r requirements-gpu.txt
```

## 4. Chay nhanh de kiem tra

Lenh nay chay toan bo pipeline voi cau hinh nho de smoke test:

```bash
python -m coral_bleaching_pipeline.cli all --fast-dev-run --models random_forest,lstm,st_gnn
```

Ket qua se duoc ghi vao:

- `data/processed/`
- `reports/eda/`
- `reports/evaluation/`
- `artifacts/models/`

## 5. Chay tung buoc

Buoc 1 - Tao du lieu da tien xu ly:

```bash
python -m coral_bleaching_pipeline.cli prepare
```

Buoc 2 - Truc quan va phan tich du lieu:

```bash
python -m coral_bleaching_pipeline.cli eda
```

Buoc 3 - Huan luyen va danh gia tat ca model duoc bat trong config:

```bash
python -m coral_bleaching_pipeline.cli train
```

Buoc 4 - Chay het prepare, EDA va train:

```bash
python -m coral_bleaching_pipeline.cli all
```

## 6. Chon model de train

Chi train cac baseline tabular:

```bash
python -m coral_bleaching_pipeline.cli train --models random_forest,xgboost,lightgbm
```

Chi train deep learning:

```bash
python -m coral_bleaching_pipeline.cli train --models lstm,gru,cnn_lstm,tft,st_gnn
```

Chi train mot model:

```bash
python -m coral_bleaching_pipeline.cli train --models st_gnn
```

Neu `lightgbm` hoac `xgboost` chua duoc cai, pipeline se thong bao va bo qua model do.

## 7. Cau hinh quan trong

File cau hinh chinh: `configs/default.yaml`.

Mot so tham so nen chinh khi train that:

- `targets.forecast_horizon_days`: so ngay du bao truoc, mac dinh 28.
- `targets.sequence_length_days`: do dai chuoi dau vao cho deep learning, mac dinh 90.
- `split.train_end` va `split.val_end`: moc chia train/validation/test theo ngay target.
- `training.epochs`: so epoch.
- `training.patience`: early stopping patience.
- `training.batch_size`: batch size.
- `training.amp`: bat/tat mixed precision tren GPU.
- `models.enabled`: danh sach model mac dinh.

## 8. Bo sung du lieu MODIS, Copernicus, ENSO

Dat CSV vao `data/external/`.

Neu bien ap dung chung cho toan khu vuc, CSV chi can:

```text
time,enso_index
2020-01-01,0.5
2020-01-02,0.5
```

Neu bien theo tung node/diem luoi, CSV can:

```text
time,node_id,chlorophyll_a,par,salinity,current_u,current_v
2020-01-01,0,0.12,45.0,34.7,0.03,-0.02
2020-01-01,1,0.11,44.5,34.8,0.02,-0.01
```

Ten bien optional da duoc pipeline nhan:

- `chlorophyll_a`
- `par`
- `salinity`
- `current_u`
- `current_v`
- `current_speed`
- `enso_index`

## 9. Dau ra danh gia

Sau khi train, xem:

- `reports/evaluation/leaderboard.csv`: bang so sanh model.
- `reports/evaluation/*_predictions.csv`: du bao tung model.
- `reports/evaluation/*_confusion_matrix.csv`: confusion matrix cho alert level.
- `reports/evaluation/*_classification_report.csv`: classification report.
- `reports/evaluation/training_curves/`: loss train/validation.

Metrics chinh:

- Regression: MAE, RMSE, R2 cho DHW.
- Classification: accuracy, balanced accuracy, macro F1, weighted F1 cho alert level.
- Event metrics: precision, recall, F1 cho alert >= Alert Level 1.

## 10. Ghi chu ve Git

Repo nay duoc cau hinh de day len remote moi:

```bash
git remote set-url origin https://github.com/coderkhongodo/DHWprj.git
git config user.name coderkhongodo
git config user.email coderkhongodo@users.noreply.github.com
```

