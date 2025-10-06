
# Tổng Quan về Machine Learning

## 1. Định nghĩa và Ứng dụng

- **Machine learning (ML)** là lĩnh vực cho phép máy tính học từ dữ liệu để dự đoán hoặc phân loại mà không cần biết rõ quy trình tạo ra dữ liệu đó.
- ML tập trung vào việc xấp xỉ hàm số (function approximation) để dự đoán giá trị tương lai dựa trên dữ liệu đã học.
- Ứng dụng ML rất đa dạng: lọc thư rác, xếp hạng tìm kiếm web, tối ưu hóa lộ trình, phát hiện gian lận, nhận diện khuôn mặt, dự đoán giá nhà, v.v.


## 2. Mô hình hóa và Đơn giản hóa

- **Mô hình (model)** là sự trừu tượng hóa các đặc điểm quan trọng của thực tế, bỏ qua các chi tiết không cần thiết.
- Mô hình tốt giúp đơn giản hóa hiện tượng phức tạp, dễ hiểu và dễ phân tích.


## 3. Tham số và Siêu tham số

- **Tham số (parameters):** Được học từ dữ liệu, ví dụ: hệ số trong hồi quy tuyến tính.
- **Siêu tham số (hyperparameters):** Được thiết lập trước khi huấn luyện, ví dụ: số lượng cây trong random forest, hệ số phạt trong regularization.


## 4. Các loại học có giám sát

- **Hồi quy (Regression):** Dự đoán giá trị liên tục (ví dụ: giá nhà, doanh thu).
- **Phân loại (Classification):** Dự đoán nhãn rời rạc (ví dụ: spam/không spam, khách hàng rời bỏ).


## 5. Quá trình Huấn luyện và Dự đoán

- Mô hình được huấn luyện trên dữ liệu quá khứ để học mối quan hệ giữa đặc trưng (features) và biến mục tiêu (target).
- Dùng tham số đã học để dự đoán cho dữ liệu mới.
- Cần tránh overfitting bằng cách đảm bảo mô hình tổng quát hóa tốt.


## 6. Hàm mất mát và Cập nhật tham số

- **Hàm mất mát (loss function):** Đo lường độ chính xác của dự đoán, hướng dẫn cập nhật tham số để giảm lỗi.
- Ví dụ:
    - Hồi quy: Mean Squared Error (MSE)

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$
    - Phân loại: Cross-Entropy Loss


## 7. Diễn giải (Interpretation) vs. Dự đoán (Prediction)

- **Diễn giải:** Hiểu cơ chế mô hình, ý nghĩa các đặc trưng ảnh hưởng đến biến mục tiêu.
    - Ví dụ: Phân tích nhân khẩu học khách hàng để hiểu yếu tố thúc đẩy doanh số.
- **Dự đoán:** Tối ưu độ chính xác dự đoán, thường dùng mô hình phức tạp hơn, ít dễ hiểu.
    - Ví dụ: Dự đoán khách hàng rời bỏ mà không cần biết lý do cụ thể.
- **Trade-off:** Mô hình càng phức tạp càng khó diễn giải nhưng có thể dự đoán tốt hơn. Lựa chọn mô hình tùy mục tiêu bài toán.


## 8. Ví dụ: Dự đoán Giá Nhà

- Dữ liệu gồm các đặc trưng như chất lượng nhà, vị trí, số tầng...
- Mô hình hồi quy tuyến tính:

$$
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_p X_p
$$
    - $Y$: Giá nhà dự đoán
    - $X_i$: Đặc trưng thứ i
    - $\beta_i$: Hệ số hồi quy
- $\beta_0$: Giá trị dự đoán khi tất cả đặc trưng bằng 0
- $\beta_1$: Giá trị tăng thêm khi tăng 1 đơn vị đặc trưng $X_1$


## 9. Fitting và Đánh giá mô hình

- Tìm $\beta_0, \beta_1, ..., \beta_p$ bằng cách tối thiểu hóa hàm mất mát (thường là MSE).
- Đường hồi quy tốt nhất là đường thẳng giảm tổng bình phương sai số giữa giá trị thực và giá trị dự đoán.
- **R-squared ($R^2$)** đo lường phần phương sai được giải thích bởi mô hình:

$$
R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}
$$
    - $\bar{y}$: Giá trị trung bình của $y$
    - $R^2$ càng cao, mô hình càng giải thích tốt dữ liệu.


## 10. Quy trình thực hiện hồi quy tuyến tính với Python (sklearn)

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

- Đánh giá mô hình bằng MSE hoặc $R^2$:

```python
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

