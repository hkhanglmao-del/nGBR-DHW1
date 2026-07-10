# Pipeline dự báo và giám sát tẩy trắng san hô bằng Machine Learning/Deep Learning

Tài liệu này mô tả nhánh Python mới bổ sung cho dự án `DHWprj`. Phần R Markdown hiện có vẫn là workflow tái lập phân tích DHW/BAA quanh Lizard Island; nhánh Python mới dùng cùng dữ liệu NOAA CRW để xây dựng pipeline huấn luyện, so sánh mô hình và mở rộng sang các biến hải dương học khác.

## 1. Hiện trạng dữ liệu trong repo

File `data/dhw_5km_3231_0c35_4219_U1739212814032.nc` là NetCDF NOAA Coral Reef Watch, có 5,513 ngày từ 2010-01-01 đến 2025-02-08, với lưới 3 latitude x 2 longitude quanh Lizard Island. Các biến sẵn có:

- `CRW_SST`: Sea Surface Temperature.
- `CRW_SSTANOMALY`: SST anomaly.
- `CRW_HOTSPOT`: coral bleaching HotSpot.
- `CRW_DHW`: Degree Heating Weeks.
- `CRW_BAA`: Bleaching Alert Area.
- `CRW_BAA_7D_MAX`: 7-day maximum BAA.

Pipeline hiện dùng dữ liệu này để tạo hai dạng học:

- **Node-level/tabular**: mỗi ô lưới là một node, thêm lag/rolling features cho Random Forest, XGBoost, LightGBM.
- **Sequence/spatio-temporal**: chuỗi thời gian trung bình không gian cho LSTM/GRU/CNN-LSTM/TFT-lite, và tensor `time x node x feature` cho ST-GNN.

## 2. Nguồn dữ liệu nên ghép thêm

| Nhóm biến | Nguồn gợi ý | Cách dùng trong pipeline |
|---|---|---|
| SST, SSTA, HotSpot, DHW, BAA | NOAA CRW daily 5 km products: https://coastwatch.pfeg.noaa.gov/erddap/info/NOAA_DHW/index.html | Nguồn chính, daily, lưới 5 km. |
| Chlorophyll-a, PAR | NASA OceanColor MODIS Aqua L3/L4: https://oceandata.sci.gsfc.nasa.gov/l3/ | Tải daily hoặc 8-day mapped, log-transform chlorophyll-a, nội suy/aggregate về lưới CRW. |
| Salinity, currents | Copernicus GLORYS Global Ocean Physics Reanalysis: https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description | Dùng surface/shallowest level; thêm `current_speed = sqrt(u^2 + v^2)` và hướng dòng chảy. |
| ENSO | NOAA CPC ONI: https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php hoặc NOAA PSL MEI.v2: https://psl.noaa.gov/enso/mei/ | Monthly hoặc bi-monthly index, forward-fill/interpolate sang daily. |

Khi có dữ liệu ngoài, đặt CSV trong `data/external/`. Nếu CSV chỉ có `time` và một hoặc nhiều biến, pipeline sẽ broadcast biến đó cho mọi node. Nếu CSV có `time,node_id`, pipeline sẽ merge theo từng node. Với NetCDF lớn, nên tiền xử lý về CSV/Parquet theo lưới CRW trước để tránh mỗi lần train phải đọc file vệ tinh nặng.

## 3. Bài toán dự báo và giám sát

Pipeline tạo hai target đồng thời:

- **Regression**: dự báo `CRW_DHW` sau `forecast_horizon_days`, mặc định 28 ngày.
- **Classification/monitoring**: dự báo `alert_level` sau cùng horizon.

Alert level được tính từ `CRW_HOTSPOT` và `CRW_DHW`. Mặc định dùng scheme `noaa_extended`:

- 0: No Stress.
- 1: Watch.
- 2: Warning.
- 3: Alert Level 1, DHW 4-8.
- 4: Alert Level 2, DHW 8-12.
- 5: Alert Level 3, DHW 12-16.
- 6: Alert Level 4, DHW 16-20.
- 7: Alert Level 5, DHW >= 20.

Nếu muốn bám legacy BAA 0-4, đổi `data.alert_scheme` thành `legacy_baa` trong config.

## 4. Feature engineering

Các feature được tạo tự động:

- Biến gốc NOAA CRW.
- Biến lịch mùa vụ: `doy_sin`, `doy_cos`, `month_sin`, `month_cos`.
- Lag theo ngày: mặc định 1, 3, 7, 14, 28, 56.
- Rolling mean/max: mặc định 7, 14, 28, 56 ngày.
- Optional: `chlorophyll_a`, `par`, `salinity`, `current_u`, `current_v`, `current_speed`, `enso_index`.

Nguyên tắc chống leakage:

- Lag/rolling chỉ dùng quá khứ, rolling đã shift 1 ngày trước khi tính.
- Split train/validation/test dựa trên **ngày của target dự báo** (`target_time_h{horizon}`), không dựa trên ngày đầu vào.
- Scaling cho deep learning fit trên train split rồi mới transform validation/test.

## 5. Mô hình

Pipeline hỗ trợ:

- **Random Forest**: baseline mạnh, ổn định với dữ liệu nhỏ và lag/rolling features.
- **XGBoost**: baseline boosting chính, thường rất mạnh cho tabular time-series features.
- **LightGBM**: optional, tự bỏ qua nếu môi trường chưa cài.
- **LSTM/GRU**: học phụ thuộc thời gian dài trên chuỗi daily.
- **CNN-LSTM**: CNN bắt pattern ngắn hạn, LSTM bắt động lực dài hơn.
- **TFT-lite**: biến thể attention dependency-light, dùng projection + positional embedding + Transformer encoder + multi-task heads. Nếu cần TFT đầy đủ đa horizon với interpretability sâu hơn, có thể thay bằng PyTorch Forecasting sau khi dữ liệu đủ lớn.
- **ST-GNN**: graph temporal model dùng ma trận kề theo khoảng cách địa lý giữa các node CRW; không cần `torch-geometric`.

Các deep model đều là multi-task: một head dự báo DHW, một head dự báo alert class.

## 6. Đánh giá và so sánh

Mỗi model xuất:

- Regression: MAE, RMSE, R2.
- Classification: accuracy, balanced accuracy, macro F1, weighted F1.
- Event metrics: precision/recall/F1 cho sự kiện alert >= `evaluation.alert_positive_threshold`, mặc định Alert Level 1 trở lên.
- Prediction CSV, confusion matrix, classification report.
- Leaderboard: `reports/evaluation/leaderboard.csv`.

Khuyến nghị khi train thật:

- Chạy thêm rolling-origin backtest theo năm, ví dụ train đến 2018, test 2019; train đến 2019, test 2020; ...
- Báo cáo riêng cho các năm bleaching mạnh như 2016, 2017, 2020, 2022, 2024.
- Ưu tiên event recall khi mục tiêu là cảnh báo sớm; ưu tiên RMSE nếu mục tiêu là dự báo cường độ DHW.

## 7. Huấn luyện GPU

Cài môi trường:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Với GPU, cài PyTorch CUDA theo image GPU trước, sau đó:

```bash
pip install -r requirements-gpu.txt
```

Chạy pipeline:

```bash
python -m coral_bleaching_pipeline.cli prepare
python -m coral_bleaching_pipeline.cli eda
python -m coral_bleaching_pipeline.cli train
```

Chạy nhanh để kiểm thử:

```bash
python -m coral_bleaching_pipeline.cli all --fast-dev-run --models random_forest,lstm,st_gnn
```

Chỉ chạy một nhóm model:

```bash
python -m coral_bleaching_pipeline.cli train --models xgboost,lightgbm
python -m coral_bleaching_pipeline.cli train --models lstm,gru,cnn_lstm,tft,st_gnn
```

Các tối ưu GPU đã có:

- Tự chọn `cuda` nếu PyTorch thấy GPU.
- AMP/mixed precision qua `training.amp`.
- `pin_memory` cho DataLoader khi có CUDA.
- Gradient clipping.
- Early stopping theo validation loss.
- Checkpoint model tốt nhất trong `artifacts/models/`.
- Progress bar terminal bằng `tqdm`.

## 8. Kết quả đầu ra

- `data/processed/node_features.csv`: dữ liệu node-level sau khi ghép feature/target.
- `data/processed/aggregate_features.csv`: dữ liệu trung bình không gian.
- `data/processed/tabular_supervised.csv`: dữ liệu tabular có lag/rolling.
- `reports/eda/*.png`: trực quan dữ liệu.
- `reports/evaluation/leaderboard.csv`: bảng so sánh model.
- `reports/evaluation/*_predictions.csv`: dự báo từng model.
- `reports/evaluation/*_confusion_matrix.csv`: confusion matrix.
- `reports/evaluation/training_curves/*.png`: loss train/validation.
- `artifacts/models/*.pt` hoặc `.joblib`: checkpoint/model đã train.

## 9. Hướng mở rộng nghiên cứu

- Thêm dữ liệu bleaching survey/in-situ nếu có để chuyển target từ proxy thermal stress sang bleaching severity thực địa.
- Dùng multi-horizon output 7/14/28/56 ngày thay vì một horizon.
- Thêm uncertainty: quantile regression, MC dropout hoặc ensemble.
- Thêm spatial context rộng hơn toàn GBR để ST-GNN có nhiều node hơn; 6 node hiện tại đủ smoke test nhưng chưa khai thác hết lợi thế GNN.
- Với MODIS chlorophyll/PAR, nên kiểm tra cloud gaps; dùng 8-day composite hoặc rolling imputation để giảm missingness.
- Với Copernicus, thử surface và shallow-depth features vì san hô không chỉ chịu nhiệt ở skin SST.
