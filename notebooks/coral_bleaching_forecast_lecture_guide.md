# Hướng Dẫn Học Và Phân Tích: Dự Báo Tẩy Trắng San Hô Quanh Lizard Island

Tài liệu này đi kèm notebook `coral_bleaching_forecast_lecture.ipynb`.

- Notebook dùng để chạy code, xem bảng và tạo biểu đồ.
- File Markdown này dùng để giải thích, dẫn dắt bài học và hướng dẫn học sinh phân tích từng biểu đồ.

Khi giảng, nên mở notebook và file này song song. Học sinh có thể đọc file này như một tài liệu học độc lập để hiểu dữ liệu và ý nghĩa từng thành phần.

## Mục Tiêu Học Tập

Sau bài học, người học cần nắm được:

1. Dự báo tẩy trắng san hô là bài toán dự báo stress nhiệt của biển trong tương lai gần.
2. Vì sao nhiệt độ mặt biển cao và kéo dài làm tăng nguy cơ tẩy trắng san hô.
3. Ý nghĩa của các biến SST, SST anomaly, HotSpot, DHW, BAA và BAA 7-day max.
4. Cách đọc các biểu đồ để nhận ra năm stress nhiệt cao, khu vực rủi ro và mất cân bằng lớp cảnh báo.

## Dữ Liệu Đang Phân Tích

Dữ liệu trong repo đã được xử lý sẵn ở `data/processed`.

| Thành phần | Ý nghĩa |
|---|---|
| Đơn vị quan sát | Một ngày tại một ô lưới |
| Không gian | 6 ô lưới 5 km quanh Lizard Island, Great Barrier Reef |
| Thời gian | 2010-01-01 đến 2025-02-08 |
| Số dòng node-level | 33,078 |
| Đầu vào chính | `CRW_SST`, `CRW_SSTANOMALY`, `CRW_HOTSPOT`, `CRW_DHW`, `CRW_BAA`, `CRW_BAA_7D_MAX` |
| Đầu ra dự báo | `target_CRW_DHW_h28`, `target_alert_level_h28` |

Lưu ý: hậu tố `h28` nghĩa là nhãn ở thời điểm sau 28 ngày. Ví dụ tại ngày `t`, mô hình dùng thông tin hiện tại và quá khứ để dự báo DHW và mức cảnh báo ở ngày `t + 28`.

## 1. Mở Đầu Bài Toán

### Tẩy Trắng San Hô Là Gì?

San hô sống cộng sinh với các vi tảo nằm trong mô san hô. Các vi tảo này giúp san hô có màu sắc và cung cấp một phần năng lượng. Khi nước biển nóng bất thường và kéo dài, san hô bị stress nhiệt. Mối quan hệ cộng sinh bị rối loạn, san hô có thể mất tảo cộng sinh, nhìn trắng hơn và suy yếu.

Trong bài này, ta không mô phỏng trực tiếp toàn bộ cơ chế sinh học của san hô. Ta dùng dữ liệu nhiệt biển và các chỉ số cảnh báo của NOAA Coral Reef Watch để theo dõi nguy cơ tẩy trắng.

### Vì Sao Nhiệt Độ Biển Cao Gây Rủi Ro?

Một ngày nóng chưa chắc gây tẩy trắng nghiêm trọng. Rủi ro tăng mạnh khi nhiệt độ vượt ngưỡng và kéo dài qua nhiều ngày hoặc nhiều tuần. Vì vậy dữ liệu cần có cả:

- Biến mô tả trạng thái hiện tại, như SST và HotSpot.
- Biến mô tả stress tích lũy, như DHW.
- Biến cảnh báo dễ diễn giải, như BAA.

Ví dụ đời thường:

- **SST** giống nhiệt độ biển hôm nay.
- **SST anomaly** giống hôm nay nóng hơn bình thường bao nhiêu.
- **DHW** giống tổng số ngày bị nóng kéo dài.

### Vì Sao Dự Báo Trước 28 Ngày?

Dự báo trước 28 ngày giúp có thời gian:

- Theo dõi sát hơn các vùng đang tăng rủi ro.
- Lên kế hoạch khảo sát thực địa.
- Cảnh báo và ưu tiên nguồn lực quản lý.
- Chuẩn bị truyền thông về đợt stress nhiệt.

Trong notebook, bài toán có hai đầu ra:

- **Dự báo DHW sau 28 ngày**: giá trị liên tục, là bài toán hồi quy.
- **Dự báo mức cảnh báo sau 28 ngày**: nhãn rời rạc, là bài toán phân loại.

## 2. Ý Nghĩa Từng Đặc Trưng

### SST: Sea Surface Temperature

`CRW_SST` là nhiệt độ mặt biển, đơn vị độ C.

Cách hiểu đơn giản: SST là nhiệt độ biển hôm nay. SST cao thường xuất hiện vào mùa hè hoặc đầu thu ở Nam bán cầu. Tuy nhiên, SST cao chưa đủ để kết luận có nguy cơ tẩy trắng, vì mỗi mùa trong năm có nền nhiệt khác nhau.

Khi phân tích, nên hỏi học sinh: nhiệt độ 29 độ C có luôn nguy hiểm không? Câu trả lời là không nhất thiết. Cần so với mức bình thường của đúng khu vực và đúng thời điểm trong năm.

### SST Anomaly

`CRW_SSTANOMALY` là độ lệch của SST so với mức bình thường, đơn vị độ C.

Cách hiểu đơn giản: SST anomaly cho biết hôm nay nóng hơn hoặc lạnh hơn bình thường bao nhiêu. Nếu anomaly dương, nước biển nóng hơn mức nền. Nếu anomaly âm, nước biển lạnh hơn mức nền.

Vai trò của biến này là phát hiện bất thường theo mùa, không chỉ nhìn nhiệt độ tuyệt đối.

### HotSpot

`CRW_HOTSPOT` là mức vượt ngưỡng stress nhiệt, đơn vị độ C.

Cách hiểu đơn giản: HotSpot là phần nhiệt độ vượt qua ngưỡng nguy hiểm đối với san hô. HotSpot gắn trực tiếp hơn với nguy cơ tẩy trắng so với SST thuần túy.

Khi HotSpot dương, khu vực bắt đầu có dấu hiệu nóng bất thường. Khi HotSpot đạt từ khoảng 1 độ C trở lên, stress nhiệt trở nên đáng chú ý hơn.

### DHW: Degree Heating Weeks

`CRW_DHW` là Degree Heating Weeks, đơn vị độ C-weeks.

DHW đo stress nhiệt tích lũy qua nhiều tuần. Nếu HotSpot là "hôm nay nóng vượt ngưỡng bao nhiêu", thì DHW là "nhiều ngày nóng vượt ngưỡng cộng lại thành bao nhiêu".

Cần nhấn mạnh:

- DHW càng cao thì stress nhiệt càng mạnh và kéo dài.
- Khoảng 4 DHW thường gắn với Alert Level 1.
- Khoảng 8 DHW thường gắn với Alert Level 2 trong hệ BAA 0-4 của dữ liệu này.

Ví dụ đời thường: một ngày làm việc quá sức có thể chưa sao, nhưng làm quá sức nhiều tuần liên tiếp thì nguy cơ suy kiệt tăng lên. DHW cũng là ý tưởng tích lũy áp lực như vậy.

### BAA: Bleaching Alert Area

`CRW_BAA` biến thông tin stress nhiệt thành mức cảnh báo dễ đọc hơn.

| BAA | Tên mức | Diễn giải ngắn |
|---:|---|---|
| 0 | No Stress | Chưa có stress nhiệt đáng kể |
| 1 | Watch | Có dấu hiệu nóng, cần theo dõi |
| 2 | Warning | Đang vượt ngưỡng nóng, rủi ro tăng |
| 3 | Alert Level 1 | Stress tích lũy cao, nguy cơ tẩy trắng đáng kể |
| 4 | Alert Level 2 | Stress rất cao, nguy cơ nghiêm trọng |

### BAA 7-Day Max

`CRW_BAA_7D_MAX` là mức BAA cao nhất trong 7 ngày gần nhất.

Biến này hữu ích vì cảnh báo có thể tăng nhanh trong vài ngày. Nếu chỉ nhìn BAA của riêng ngày hiện tại, ta có thể bỏ lỡ một đỉnh cảnh báo vừa xảy ra. BAA 7-day max giữ lại ký ức ngắn hạn của mức cảnh báo cao nhất trong tuần.

## 3. Hướng Dẫn Đọc Bảng Và Biểu Đồ Trong Notebook

### Bảng Tổng Quan Dữ Liệu

Bảng đầu tiên cho biết kích thước dữ liệu, khoảng thời gian, các cột quan trọng và tỷ lệ thiếu của nhãn dự báo.

Cần nhìn vào:

- Số ô lưới là 6, nghĩa là mỗi ngày có tối đa 6 quan sát.
- Khoảng thời gian 2010-2025 đủ dài để thấy các năm stress nhiệt lớn.
- Các cột target có hậu tố `h28`, nghĩa là nhãn của 28 ngày sau.

Thông điệp cần rút ra: dữ liệu này phù hợp để minh họa bài toán dự báo theo thời gian, từ hiện tại và quá khứ gần để dự báo trạng thái sau 28 ngày.

### Hình 01: Khung Bài Toán Dự Báo

File hình: `reports/lecture_visuals/01_forecast_problem_diagram.png`

Hình này mô tả dòng chảy thông tin:

1. Đầu vào là dữ liệu ngày `t` và quá khứ.
2. Mô hình học quan hệ theo thời gian.
3. Đầu ra là DHW và mức cảnh báo ở ngày `t + 28`.

Cần nhìn vào:

- Đầu vào không chỉ có SST mà có nhiều biến mô tả stress nhiệt.
- Đầu ra gồm một giá trị liên tục và một nhãn phân loại.

Câu hỏi gợi ý:

- Vì sao cùng một bộ dữ liệu có thể dùng cho cả hồi quy và phân loại?
- Nếu chỉ dự báo BAA mà không dự báo DHW, ta mất thông tin gì?

### Hình 02: Cách Hiểu Nhãn 28 Ngày

File hình: `reports/lecture_visuals/02_28_day_target_alignment.png`

Hình này so sánh DHW quan sát tại ngày `t` với nhãn dự báo DHW tại ngày `t + 28`.

Cần nhìn vào:

- Đường liền là DHW tại ngày hiện tại.
- Đường đứt là DHW sau 28 ngày, được dùng làm nhãn.
- Mũi tên 28 ngày cho thấy khoảng cách giữa thời điểm có đầu vào và thời điểm cần dự báo.

Thông điệp cần rút ra: tại ngày `t`, mô hình không được nhìn thấy tương lai. Mô hình chỉ dùng thông tin đến ngày `t` để dự báo ngày `t + 28`.

### Hình 03: HotSpot Từng Ngày Và DHW Tích Lũy

File hình: `reports/lecture_visuals/03_dhw_accumulation_intuition.png`

Đây là hình minh họa sư phạm, không phải phép tính NOAA đầy đủ. Mục đích là giúp học sinh hiểu trực giác về tích lũy stress.

Cần nhìn vào:

- Cột vàng là HotSpot từng ngày.
- Đường đỏ là DHW minh họa.
- Khi HotSpot vượt ngưỡng trong nhiều ngày, DHW tăng dần.

Thông điệp cần rút ra: DHW không tăng chỉ vì một ngày nóng. Nó tăng khi stress nhiệt kéo dài.

### Bảng Ý Nghĩa Đặc Trưng

Bảng này là từ điển dữ liệu cho 6 biến chính.

Cần nhìn vào:

- Đơn vị của từng biến.
- Ví dụ đời thường để nhớ nghĩa của biến.
- Vai trò của biến trong dự báo.

Thông điệp cần rút ra: mỗi cột đại diện cho một khía cạnh của stress nhiệt, gồm hiện tại, bất thường, vượt ngưỡng, tích lũy và cảnh báo.

### Bảng Mức BAA

Bảng này giải thích các mức 0-4 của Bleaching Alert Area.

Cần nhìn vào:

- No Stress và Watch là các mức phổ biến hơn.
- Alert Level 1 và Alert Level 2 là các mức nghiêm trọng hơn và hiếm hơn.
- Các ngưỡng 4 DHW và 8 DHW giúp liên hệ DHW với cảnh báo.

Thông điệp cần rút ra: BAA là cách biến thông tin nhiệt thành ngôn ngữ cảnh báo để người quản lý và cộng đồng dễ hiểu hơn.

### Hình 04: BAA Và BAA 7-Day Max

File hình: `reports/lecture_visuals/04_baa_vs_7day_max.png`

Hình này phân biệt BAA trong ngày và BAA cao nhất trong 7 ngày gần nhất.

Cần nhìn vào:

- Nếu BAA trong ngày giảm, BAA 7-day max có thể vẫn còn cao trong vài ngày.
- Đường 7-day max thường mượt hơn và ít dao động hơn.

Thông điệp cần rút ra: BAA 7-day max giúp không bỏ qua các đỉnh cảnh báo ngắn hạn. Đây là một ví dụ tốt về đặc trưng có ký ức ngắn hạn trong chuỗi thời gian.

### Hình 05: Từ SST Đến DHW Trong Đợt Stress 2024

File hình: `reports/lecture_visuals/05_feature_progression_2024.png`

Hình này đặt 4 chuỗi thời gian lên cùng một giai đoạn: SST, SST anomaly, HotSpot và DHW.

Cần nhìn vào:

- SST cho thấy nhiệt độ mặt biển thay đổi theo ngày.
- SST anomaly cho thấy mức bất thường so với bình thường.
- HotSpot cho thấy phần vượt ngưỡng stress.
- DHW tăng chậm hơn vì nó là biến tích lũy.

Câu hỏi gợi ý:

- Khi SST giảm, DHW có giảm ngay không? Vì sao?
- Biến nào phản ứng nhanh, biến nào phản ứng chậm?

Thông điệp cần rút ra: SST và HotSpot gần với hiện tại, còn DHW mang ký ức của nhiều tuần trước.

### Hình 06: Bản Đồ 6 Ô Lưới Quanh Lizard Island

File hình: `reports/lecture_visuals/06_lizard_island_grid_map.png`

Hình này cho thấy 6 ô lưới 5 km quanh Lizard Island, tô màu theo DHW cực đại trong giai đoạn 2010-2025.

Cần nhìn vào:

- Mỗi ô là một node trong dữ liệu.
- Màu đỏ đậm hơn nghĩa là DHW cực đại cao hơn.
- Node 1 có max DHW cao nhất trong dữ liệu node-level, khoảng 9.51.

Thông điệp cần rút ra: ngay cả trong một vùng nhỏ, các ô lưới có thể có lịch sử stress nhiệt khác nhau. Vì vậy yếu tố không gian vẫn quan trọng.

### Hình 07: Chuỗi DHW Dài Hạn Và Các Năm Stress Cao

File hình: `reports/lecture_visuals/07_dhw_timeseries_key_years.png`

Hình này là chuỗi DHW trung bình của 6 ô lưới từ 2010 đến 2025.

Cần nhìn vào:

- Đỉnh stress nhiệt lớn nhất nằm ở năm 2016 và 2017, gần hoặc vượt ngưỡng 8 DHW.
- Năm 2024 cũng nổi bật, đạt khoảng 6.24 DHW ở trung bình vùng.
- Các đường ngang 4 và 8 DHW là mốc diễn giải cảnh báo.

Thông điệp cần rút ra: stress nhiệt không xuất hiện đều mỗi năm. Nó tập trung thành các đợt và có một số năm nổi bật, đặc biệt 2016, 2017 và 2024 trong bộ dữ liệu này.

### Hình 08: DHW Cực Đại Từng Năm

File hình: `reports/lecture_visuals/08_annual_max_dhw.png`

Hình cột này rút gọn chuỗi thời gian thành một giá trị mỗi năm: DHW cực đại trong năm.

Cần nhìn vào:

- 2016: max DHW trung bình vùng khoảng 8.79.
- 2017: max DHW trung bình vùng khoảng 8.75.
- 2024: max DHW trung bình vùng khoảng 6.24.
- Một số năm khác như 2020, 2021, 2022 cũng có stress đáng chú ý nhưng thấp hơn.

Thông điệp cần rút ra: biểu đồ cột giúp so sánh năm với năm nhanh hơn chuỗi thời gian đầy đủ, nhưng làm mất thông tin về thời điểm và độ dài của đợt stress trong năm.

### Hình 09: Phóng To Các Năm 2016, 2017, 2024

File hình: `reports/lecture_visuals/09_key_years_zoom.png`

Hình này phóng to nửa đầu năm của các năm stress cao.

Cần nhìn vào:

- Đỉnh DHW thường xuất hiện khoảng tháng 3-4.
- 2016 và 2017 đạt gần hoặc vượt mốc 8 DHW ở trung bình 6 ô lưới.
- 2024 không cao bằng 2016/2017 nhưng vẫn vượt mốc Alert Level 1.

Thông điệp cần rút ra: phóng to theo mùa giúp thấy đợt stress có hình dạng như thế nào: bắt đầu tăng, đạt đỉnh, rồi giảm.

### Hình 10: Phân Bố Lớp Cảnh Báo Bị Lệch

File hình: `reports/lecture_visuals/10_alert_distribution_imbalance.png`

Hình này cho thấy bài toán phân loại mức cảnh báo bị mất cân bằng lớp mạnh.

Tỷ lệ xấp xỉ trong dữ liệu hiện tại:

| Mức cảnh báo | Tỷ lệ |
|---|---:|
| No Stress | 74.07% |
| Watch | 20.41% |
| Warning | 3.95% |
| Alert Level 1 | 1.40% |
| Alert Level 2 | 0.17% |

Cần nhìn vào:

- Đa số mẫu là No Stress.
- Alert Level 1 và Alert Level 2 rất hiếm.
- Đồ thị bên phải dùng thang log để nhìn được các lớp hiếm.

Câu hỏi gợi ý:

- Nếu mô hình luôn đoán No Stress, accuracy có thể cao không?
- Vì sao accuracy không đủ để đánh giá bài toán này?
- Vì sao cần quan tâm recall hoặc F1 cho lớp cảnh báo cao?

Thông điệp cần rút ra: với bài toán cảnh báo, lớp hiếm lại thường là lớp quan trọng nhất. Khi đánh giá mô hình, cần dùng metric phù hợp như balanced accuracy, macro F1 hoặc recall cho Alert Level 1/2.

### Hình 11: Tương Quan Và Quan Hệ Giữa Các Biến

File hình: `reports/lecture_visuals/11_feature_relationships.png`

Hình này gồm heatmap tương quan và scatter giữa SST anomaly, HotSpot, DHW và BAA.

Cần nhìn vào:

- Một số biến liên quan chặt vì cùng mô tả stress nhiệt.
- SST anomaly và HotSpot không phải cùng một khái niệm.
- Điểm có DHW lớn hơn và màu cảnh báo cao hơn thường tập trung ở vùng stress nhiệt lớn.

Thông điệp cần rút ra: tương quan giúp nhận ra biến nào có quan hệ gần nhau, nhưng không nên kết luận nhân quả chỉ từ tương quan. Trong bài toán dự báo, thứ tự thời gian và cách tạo nhãn 28 ngày rất quan trọng.

### Hình 12: Mùa Vụ Stress Nhiệt

File hình: `reports/lecture_visuals/12_monthly_seasonality.png`

Hình này gồm DHW trung bình, DHW cực đại và tỷ lệ ngày Alert Level 1/2 theo tháng.

Cần nhìn vào:

- DHW cao tập trung vào tháng 2-5.
- Tháng 3 và 4 là giai đoạn đỉnh trong bộ dữ liệu này.
- Các tháng 8-10 hầu như không có stress DHW đáng kể.

Thông điệp cần rút ra: stress nhiệt có tính mùa vụ. Khi xây dựng mô hình, các biến lịch như tháng, ngày trong năm hoặc biến sin/cos theo mùa có thể giúp mô hình học quy luật mùa vụ.

## 4. Câu Hỏi Thảo Luận Trên Lớp

1. Vì sao SST cao chưa đủ để kết luận nguy cơ tẩy trắng?
2. Sự khác nhau giữa SST anomaly và HotSpot là gì?
3. Vì sao DHW tăng chậm hơn HotSpot?
4. Nếu BAA 7-day max cao hơn BAA trong ngày, điều đó nói lên điều gì?
5. Năm nào trong dữ liệu có stress nhiệt cao nhất? Dựa vào biểu đồ nào để trả lời?
6. Vì sao Alert Level 2 rất quan trọng dù chỉ chiếm khoảng 0.17%?
7. Khi dự báo trước 28 ngày, đầu vào và nhãn dự báo nằm ở hai thời điểm nào?

## 5. Kết Luận Cần Chốt

1. Tẩy trắng san hô trong dữ liệu này được tiếp cận như bài toán dự báo stress nhiệt của biển.
2. SST và HotSpot cho thấy tình trạng nhiệt hiện tại, còn DHW cho thấy stress đã tích lũy qua nhiều tuần.
3. BAA là ngôn ngữ cảnh báo, giúp biến chỉ số nhiệt thành mức rủi ro dễ hiểu.
4. Dự báo 28 ngày cần dự báo cả giá trị DHW và mức cảnh báo, trong khi lớp cảnh báo cao rất hiếm nên cần đánh giá mô hình cẩn thận.
