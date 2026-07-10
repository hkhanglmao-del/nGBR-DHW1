# Hướng Dẫn Học Và Phân Tích: Model Tabular Và Time Series

Tài liệu này đi kèm notebook `modeling_tabular_timeseries_lecture.ipynb`.

- Notebook dùng để chạy code, xem bảng và tạo biểu đồ.
- File Markdown này dùng để giải thích, dẫn dắt bài học và hướng dẫn học sinh phân tích từng biểu đồ.

Bài này tiếp nối bài dữ liệu tẩy trắng san hô. Nếu bài trước trả lời "các biến trong dữ liệu nghĩa là gì", thì bài này trả lời "mô hình học từ dữ liệu đó như thế nào".

## Mục Tiêu Học Tập

Sau bài học, người học cần hiểu được:

1. Sự khác nhau giữa cách học của model tabular và model time series.
2. Vì sao dữ liệu chuỗi thời gian không nên chia train/test ngẫu nhiên.
3. Lag feature, rolling feature và sequence window có ý nghĩa gì.
4. Các nhóm model trong repo: Random Forest, XGBoost, LightGBM, LSTM, GRU, CNN-LSTM, TFT-lite và ST-GNN.
5. Cách đọc các metric: MAE, RMSE, R2, accuracy, balanced accuracy, macro F1, event precision, event recall và event F1.
6. Vì sao một model có RMSE tốt vẫn có thể dự báo kém các cảnh báo hiếm như Alert Level 1/2.

## 1. Bài Toán Model Hóa

Ta có dữ liệu theo ngày quanh Lizard Island. Mỗi ngày có các biến như SST, anomaly, HotSpot, DHW, BAA. Mục tiêu là dùng thông tin ở ngày `t` và quá khứ để dự báo sau 28 ngày:

- `target_CRW_DHW_h28`: DHW tại ngày `t + 28`.
- `target_alert_level_h28`: mức cảnh báo tại ngày `t + 28`.

Vì vậy đây là bài toán **multi-task**:

- Hồi quy để dự báo giá trị DHW.
- Phân loại để dự báo mức cảnh báo.

## 2. Tabular Model Là Gì?

Model tabular học từ một bảng 2 chiều: mỗi dòng là một mẫu, mỗi cột là một feature.

Vấn đề là dữ liệu gốc của ta là chuỗi thời gian. Nếu chỉ đưa giá trị hôm nay vào model, model không biết những ngày trước đó xảy ra gì. Vì vậy pipeline tạo thêm:

- **Lag features**: giá trị của biến trong quá khứ, ví dụ `CRW_DHW_lag7` là DHW trước đó 7 ngày.
- **Rolling features**: thống kê trên một cửa sổ quá khứ, ví dụ trung bình DHW 28 ngày gần đây.
- **Calendar features**: biến mùa vụ như sin/cos của ngày trong năm.

Sau feature engineering, chuỗi thời gian được biến thành một bảng tabular. Các model như Random Forest, XGBoost và LightGBM học từ bảng này.

### Điểm Mạnh Của Tabular Model

- Chạy nhanh hơn nhiều model deep learning.
- Mạnh với dữ liệu bảng và số lượng mẫu vừa phải.
- Dễ xem feature importance.
- Ít yêu cầu tuning phức tạp hơn neural network.

### Điểm Yếu Của Tabular Model

- Không tự hiểu thứ tự thời gian nếu ta không tạo lag/rolling features.
- Cửa sổ quá khứ phải được thiết kế thủ công.
- Có nguy cơ leakage nếu tạo feature từ tương lai hoặc chia dữ liệu sai cách.

## 3. Time Series Model Là Gì?

Model time series giữ nguyên thứ tự ngày trong đầu vào. Thay vì biến 90 ngày quá khứ thành nhiều cột lag/rolling, ta đưa cả cửa sổ 90 ngày vào model dưới dạng:

```text
samples x sequence_length x features
```

Trong repo này, `sequence_length_days = 90`. Nghĩa là mỗi mẫu đầu vào là chuỗi 90 ngày trước ngày `t`, và nhãn là trạng thái tại `t + 28`.

### Điểm Mạnh Của Time Series Model

- Giữ được thứ tự ngày trong cửa sổ đầu vào.
- Có thể học pattern tăng/giảm theo thời gian.
- LSTM/GRU có bộ nhớ tuần tự.
- Transformer/TFT có thể học quan hệ xa trong cửa sổ.

### Điểm Yếu Của Time Series Model

- Cần nhiều dữ liệu và tuning hơn.
- Dễ overfit nếu dữ liệu nhỏ hoặc lớp hiếm.
- Khó giải thích hơn model cây.
- Chạy chậm hơn tabular model.

## 4. Các Model Trong Repo

### Random Forest

Random Forest dùng nhiều cây quyết định học song song. Mỗi cây nhìn một phần dữ liệu và một phần feature. Dự đoán cuối cùng là tổng hợp của nhiều cây.

Khi giảng, có thể nói: Random Forest giống "hội đồng nhiều cây quyết định". Một cây có thể sai, nhưng nhiều cây bỏ phiếu hoặc lấy trung bình sẽ ổn định hơn.

### XGBoost

XGBoost là gradient boosting trên cây quyết định. Các cây được học tuần tự, cây sau cố gắng sửa lỗi của cây trước.

Điểm cần nhấn mạnh: XGBoost thường rất mạnh trên dữ liệu bảng, đặc biệt khi feature engineering tốt.

### LightGBM

LightGBM cũng là gradient boosting trên cây quyết định, tối ưu để train nhanh và hiệu quả. Trong pipeline này, classifier dùng `class_weight="balanced"` để phần nào xử lý mất cân bằng lớp.

### LSTM

LSTM là mạng hồi tiếp chuyên cho chuỗi. Nó có cơ chế cổng để giữ hoặc quên thông tin qua thời gian.

Cách nói đơn giản: LSTM đọc chuỗi 90 ngày theo thứ tự và cố nhớ những gì quan trọng cho dự báo sau 28 ngày.

### GRU

GRU giống LSTM nhưng gọn hơn, ít cổng hơn. GRU thường train nhanh hơn và đôi khi hiệu quả tương đương.

### CNN-LSTM

CNN-LSTM dùng CNN để bắt pattern cục bộ trong chuỗi, sau đó LSTM đọc chuỗi đã được trích xuất đặc trưng.

Ví dụ: CNN có thể phát hiện các đoạn tăng nóng ngắn hạn, rồi LSTM dùng thông tin đó để dự báo.

### TFT-Lite

TFT-lite trong repo là phiên bản gọn của ý tưởng Temporal Fusion Transformer. Nó dùng attention để học quan hệ trong cửa sổ 90 ngày.

Điểm cần giảng: attention giúp model không chỉ nhớ ngày cuối cùng, mà có thể học ngày nào trong 90 ngày quan trọng hơn.

### ST-GNN

ST-GNN là spatio-temporal graph neural network. Nó không chỉ đọc thời gian mà còn lan truyền thông tin giữa 6 ô lưới dựa trên adjacency matrix.

Điểm cần giảng: nếu các ô lưới gần nhau, stress nhiệt ở một ô có thể liên quan đến ô bên cạnh. ST-GNN cố gắng học cả quan hệ thời gian và quan hệ không gian.

## 5. Vì Sao Phải Chia Train/Validation/Test Theo Thời Gian?

Với chuỗi thời gian, không nên shuffle ngẫu nhiên vì sẽ làm dữ liệu tương lai rơi vào train set. Khi đó model có thể gián tiếp học thông tin của tương lai, làm kết quả đánh giá quá lạc quan.

Trong repo:

| Split | Ý nghĩa |
|---|---|
| Train | học tham số model |
| Validation | chọn model, early stopping, tuning |
| Test | đánh giá cuối cùng trên giai đoạn tương lai |

Pipeline chia theo `target_time_h28`, không phải chỉ theo `time`, vì nhãn dự báo nằm ở tương lai 28 ngày.

## 6. Các Metric Cần Hiểu

### MAE

MAE là sai số tuyệt đối trung bình. Nếu MAE = 0.4 DHW, nghĩa là trung bình dự báo lệch khoảng 0.4 DHW.

MAE dễ hiểu vì cùng đơn vị với DHW.

### RMSE

RMSE là căn bậc hai của sai số bình phương trung bình. RMSE phạt lỗi lớn mạnh hơn MAE.

Nếu muốn model tránh các sai số lớn ở đỉnh stress, RMSE là metric quan trọng.

### R2

R2 cho biết model giải thích được bao nhiêu biến thiên của mục tiêu. R2 càng gần 1 càng tốt. Nếu R2 thấp, model chưa nắm được quy luật dữ liệu tốt.

### Accuracy

Accuracy là tỷ lệ dự báo đúng lớp cảnh báo.

Nhược điểm: nếu dữ liệu mất cân bằng mạnh, model đoán lớp phổ biến vẫn có accuracy cao. Với bài toán cảnh báo tẩy trắng, accuracy không đủ.

### Balanced Accuracy

Balanced accuracy lấy trung bình recall của từng lớp. Metric này công bằng hơn khi lớp mất cân bằng.

### Macro F1

Macro F1 tính F1 cho từng lớp rồi lấy trung bình. Mỗi lớp có trọng số ngang nhau, nên lớp hiếm cũng quan trọng như lớp phổ biến.

### Event Precision, Recall, F1

Trong repo, sự kiện cảnh báo cao được định nghĩa là:

```text
alert_level >= 3
```

Tức là Alert Level 1 hoặc cao hơn.

- **Event precision**: trong các lần model báo sự kiện, bao nhiêu lần đúng.
- **Event recall**: trong các sự kiện thật, model bắt được bao nhiêu.
- **Event F1**: cân bằng giữa precision và recall.

Với hệ thống cảnh báo, event recall thường rất quan trọng: bỏ sót một đợt cảnh báo cao có thể nghiêm trọng hơn báo động giả.

## 7. Hướng Dẫn Đọc Bảng Và Biểu Đồ Trong Notebook

### Bảng Tổng Quan Dữ Liệu Và Leaderboard

Bảng đầu notebook cho biết:

- `tabular_supervised.csv` có 32,574 dòng và 106 cột.
- Tabular model dùng 95 feature columns.
- Có 36 lag features và 48 rolling features.
- Test set có target time từ 2022-12-31 đến 2025-02-08.

Cần nhấn mạnh: số cột của tabular data lớn hơn dữ liệu gốc vì pipeline đã tạo thêm nhiều feature từ quá khứ.

### Hình 01: Tabular Và Time Series Học Khác Nhau Như Thế Nào

File hình: `reports/modeling_lecture_visuals/01_tabular_vs_timeseries_overview.png`

Hình này so sánh hai con đường học:

- Tabular: biến quá khứ thành các cột lag/rolling.
- Time series: giữ nguyên cửa sổ 90 ngày.

Câu hỏi gợi ý:

- Nếu không tạo lag/rolling, tabular model có biết tuần trước nóng thế nào không?
- Nếu dùng time series model, vì sao thứ tự ngày lại quan trọng?

Thông điệp cần rút ra: cùng một bài toán, ta có thể biểu diễn đầu vào theo hai cách khác nhau. Cách biểu diễn quyết định model học được gì.

### Hình 02: Chia Train/Validation/Test Theo Thời Gian

File hình: `reports/modeling_lecture_visuals/02_temporal_train_val_test_split.png`

Hình này cho thấy target DHW theo thời gian, với ba vùng train, validation và test.

Cần nhìn vào:

- Train nằm ở quá khứ.
- Validation nằm sau train.
- Test nằm ở giai đoạn mới nhất.
- Các mốc chia dựa trên `target_time_h28`.

Thông điệp cần rút ra: đánh giá model phải mô phỏng tình huống thật, tức là dùng quá khứ để dự báo tương lai.

### Hình 03: Lag Và Rolling Features Cho Tabular Model

File hình: `reports/modeling_lecture_visuals/03_tabular_lag_rolling_features.png`

Hình này minh họa cách biến chuỗi thời gian thành feature cho model bảng.

Cần nhìn vào:

- Ngày `t` là ngày model có thông tin đầu vào.
- Các điểm `lag 1`, `lag 7`, `lag 28`, `lag 56` là giá trị quá khứ được biến thành cột.
- Các vùng rolling 7/28/56 ngày là cửa sổ để tính trung bình hoặc cực đại.
- Nhãn nằm ở `t + 28`.

Thông điệp cần rút ra: tabular model không đọc chuỗi trực tiếp. Ta phải tự "đóng gói" lịch sử thành các cột.

### Hình 04: Sequence Window 90 Ngày

File hình: `reports/modeling_lecture_visuals/04_sequence_window_heatmap.png`

Hình này là heatmap của một cửa sổ 90 ngày đầu vào cho model time series.

Cần nhìn vào:

- Trục ngang là ngày trong cửa sổ 90 ngày.
- Trục dọc là các biến như SST, anomaly, HotSpot, DHW, BAA 7-day max.
- Màu thể hiện giá trị chuẩn hóa trong cửa sổ.

Thông điệp cần rút ra: time series model nhìn thấy toàn bộ diễn biến theo thứ tự ngày, thay vì chỉ nhìn các cột lag/rolling đã tóm tắt.

### Hình 05: Các Nhóm Model Trong Pipeline

File hình: `reports/modeling_lecture_visuals/05_model_families.png`

Hình này chia model thành ba nhóm:

- Tabular: Random Forest, XGBoost, LightGBM.
- Time series: LSTM, GRU, CNN-LSTM, TFT-lite.
- Spatio-temporal: ST-GNN.

Thông điệp cần rút ra: không phải model nào cũng nhìn dữ liệu theo cùng một cách. ST-GNN còn nhìn thêm quan hệ không gian giữa các ô lưới.

### Hình 06: Leaderboard RMSE, Macro F1 Và Event F1

File hình: `reports/modeling_lecture_visuals/06_leaderboard_rmse_macro_event.png`

Hình này so sánh model theo ba góc nhìn:

- RMSE cho dự báo DHW.
- Macro F1 cho phân loại mức cảnh báo.
- Event F1 cho cảnh báo cao Alert Level 1/2.

Cần nhìn vào:

- TFT có RMSE tốt nhất trong leaderboard hiện tại.
- CNN-LSTM có macro F1 tốt nhất trong nhóm deep hiện tại.
- Event F1 còn thấp, nghĩa là bắt sự kiện cảnh báo cao vẫn khó.

Thông điệp cần rút ra: "model tốt nhất" phụ thuộc vào metric. Nếu mục tiêu là dự báo DHW, có thể chọn theo RMSE. Nếu mục tiêu là cảnh báo sự kiện hiếm, phải nhìn event recall/F1.

### Hình 07: Dự Báo DHW Theo Thời Gian Trên Test Set

File hình: `reports/modeling_lecture_visuals/07_test_predictions_timeseries.png`

Hình này so sánh DHW thật với dự báo của model tabular tốt nhất theo RMSE và model deep tốt nhất theo RMSE.

Cần nhìn vào:

- Model có theo kịp các đỉnh DHW không?
- Model có dự báo quá cao hoặc quá thấp ở giai đoạn ít stress không?
- Các đỉnh năm 2024 có được bắt tốt không?

Thông điệp cần rút ra: metric tổng hợp chỉ cho một con số. Chuỗi dự báo theo thời gian giúp thấy model sai ở thời điểm nào.

### Hình 08: Predicted Vs Actual

File hình: `reports/modeling_lecture_visuals/08_predicted_vs_actual_scatter.png`

Mỗi điểm là một dự báo. Trục x là DHW thật, trục y là DHW dự báo.

Cần nhìn vào:

- Điểm càng gần đường chéo thì dự báo càng đúng.
- Điểm nằm trên đường chéo nghĩa là model dự báo cao hơn thật.
- Điểm nằm dưới đường chéo nghĩa là model dự báo thấp hơn thật.

Thông điệp cần rút ra: scatter giúp phát hiện bias. Ví dụ model có thể dự báo tốt vùng thấp nhưng khó bắt vùng DHW cao.

### Hình 09: Confusion Matrix Chuẩn Hóa

File hình: `reports/modeling_lecture_visuals/09_confusion_matrices.png`

Confusion matrix cho biết lớp thật và lớp dự báo.

Cần nhìn vào:

- Đường chéo chính là dự báo đúng.
- Các ô ngoài đường chéo là nhầm lẫn.
- Chuẩn hóa theo hàng giúp đọc recall của từng lớp.

Thông điệp cần rút ra: model thường dự báo tốt No Stress và Watch hơn các lớp hiếm như Warning hoặc Alert Level 1.

### Hình 10: Event Precision, Recall Và F1

File hình: `reports/modeling_lecture_visuals/10_event_precision_recall_f1.png`

Hình này chỉ tập trung vào sự kiện cảnh báo cao:

```text
Alert Level 1 hoặc Alert Level 2, tức numeric alert_level >= 3
```

Cần nhìn vào:

- Recall thấp nghĩa là model bỏ sót nhiều sự kiện thật.
- Precision thấp nghĩa là model báo động nhiều lần nhưng sai nhiều.
- F1 cân bằng hai yếu tố trên.

Thông điệp cần rút ra: với bài toán cảnh báo, event metrics thường quan trọng hơn accuracy.

### Hình 11: Training Curves Của Deep Models

File hình: `reports/modeling_lecture_visuals/11_deep_training_curves.png`

Hình này vẽ validation loss theo epoch.

Cần nhìn vào:

- Loss giảm nghĩa là model học tốt hơn trên validation set.
- Loss tăng lại hoặc không giảm nghĩa là model có thể overfit hoặc không cải thiện.
- Early stopping giúp dừng train khi validation loss không còn tốt hơn.

Thông điệp cần rút ra: deep learning không chỉ là gọi model. Ta cần theo dõi quá trình học để biết model đang học thật hay chỉ overfit.

### Hình 12: Feature Importance Của Random Forest

File hình: `reports/modeling_lecture_visuals/12_tabular_feature_importance.png`

Hình này cho biết Random Forest regressor dùng feature nào nhiều nhất khi dự báo DHW.

Cần nhìn vào:

- Các rolling feature thường quan trọng vì DHW là stress tích lũy.
- Các biến liên quan BAA, DHW, HotSpot và SST quá khứ thường xuất hiện cao.
- Feature importance giúp giải thích model tabular dễ hơn model deep.

Thông điệp cần rút ra: model tabular không chỉ đưa ra dự báo, mà còn giúp ta hỏi "model đang dựa vào tín hiệu nào".

## 8. Câu Hỏi Thảo Luận Trên Lớp

1. Vì sao tabular model cần lag và rolling features?
2. Vì sao time series model có thể giữ lại nhiều thông tin hơn về thứ tự ngày?
3. Nếu shuffle dữ liệu thời gian ngẫu nhiên, điều gì có thể sai?
4. Model có RMSE thấp nhất có chắc là model cảnh báo tốt nhất không?
5. Vì sao accuracy không đủ cho bài toán cảnh báo tẩy trắng?
6. Event recall thấp có ý nghĩa gì trong bối cảnh quản lý rạn san hô?
7. Feature importance của Random Forest cho ta biết điều gì và không cho ta biết điều gì?
8. ST-GNN khác LSTM/GRU ở điểm nào?

## 9. Kết Luận Cần Chốt

1. Tabular model học từ bảng feature đã được thiết kế thủ công, đặc biệt là lag và rolling features.
2. Time series model học trực tiếp từ cửa sổ nhiều ngày, giữ lại thứ tự thời gian.
3. Chia dữ liệu theo thời gian là bắt buộc để tránh leakage và mô phỏng đúng bài toán dự báo tương lai.
4. Đánh giá model phải nhìn nhiều metric: RMSE cho DHW, macro F1 cho phân loại, event recall/F1 cho cảnh báo cao.
5. Trong bài toán tẩy trắng san hô, lớp cảnh báo cao rất hiếm nhưng lại quan trọng nhất, nên một model có accuracy cao vẫn có thể chưa đủ tốt cho mục tiêu cảnh báo.
