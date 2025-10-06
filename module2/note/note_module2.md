

# Tổng Quan về Data Splitting, Cross-Validation và Mở Rộng Linear Regression

## 1. Data Splitting (Chia Tập Dữ Liệu)

- **Mục đích:** Đánh giá khả năng tổng quát hóa của mô hình trên dữ liệu chưa từng thấy.
- **Các bước cơ bản:**
    - Chia dữ liệu thành hai phần: training set (tập huấn luyện) và test set (tập kiểm tra).
    - Training set dùng để huấn luyện mô hình, test set dùng để đánh giá.
- **Ký hiệu:**
    - \$ X_{train}, Y_{train} \$: Đặc trưng và nhãn của tập huấn luyện.
    - \$ X_{test}, Y_{test} \$: Đặc trưng và nhãn của tập kiểm tra.


### Công thức chia dữ liệu với sklearn:

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

- **test_size:** Tỉ lệ phần trăm dữ liệu dành cho test set (ví dụ: 0.2 là 20%).
- **random_state:** Đảm bảo kết quả chia dữ liệu có thể lặp lại.


### Các phương pháp chia dữ liệu khác:

- **ShuffleSplit:** Chia nhiều lần ngẫu nhiên.
- **StratifiedShuffleSplit:** Đảm bảo phân phối nhãn giống nhau ở cả train và test (quan trọng với phân loại mất cân bằng).


## 2. Cross-Validation (CV)

- **Khái niệm:** Chia dữ liệu thành nhiều phần (folds), lần lượt dùng mỗi phần làm test set, các phần còn lại làm training set.
- **K-Fold Cross-Validation:**
    - Chia dữ liệu thành \$ K \$ phần bằng nhau.
    - Lặp lại \$ K \$ lần, mỗi lần chọn một fold làm test set, \$ K-1 \$ fold còn lại làm training set.
    - Trung bình hóa kết quả các lần để có đánh giá tổng quát.


### Công thức tổng quát:

$$
\text{CV Error} = \frac{1}{K} \sum_{k=1}^{K} \text{Error}_k
$$

- \$ Error_k \$: Sai số trên fold thứ \$ k \$.


### Các biến thể:

- **Stratified K-Fold:** Đảm bảo tỉ lệ nhãn đồng đều ở mỗi fold.
- **Leave-One-Out (LOO):** Mỗi lần chỉ để lại 1 mẫu làm test set, phù hợp với tập dữ liệu nhỏ.


## 3. Model Complexity và Error

- **Underfitting:** Mô hình quá đơn giản, không học được quy luật dữ liệu (bias cao, variance thấp).
- **Overfitting:** Mô hình quá phức tạp, học cả nhiễu của dữ liệu (bias thấp, variance cao).
- **Goal:** Tìm điểm cân bằng giữa bias và variance để tổng sai số thấp nhất.


### Biểu đồ minh họa:

- Đường cong U ngược: Training error giảm dần khi tăng độ phức tạp, test error giảm rồi tăng lại do overfitting.


## 4. Đánh Giá Mô Hình

- **Sai số phổ biến:**
    - Hồi quy: Mean Squared Error (MSE), Mean Absolute Error (MAE), \$ R^2 \$.
    - Phân loại: Accuracy, Precision, Recall, F1-score.


### Công thức MSE:

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$

### Công thức \$ R^2 \$:

$$
R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}
$$

## 5. Mở Rộng Linear Regression

- **Polynomial Features:** Thêm các bậc cao hơn của đặc trưng để mô hình hóa quan hệ phi tuyến.
    - Ví dụ: \$ X, X^2, X^3 \$
- **Interaction Terms:** Nhân các đặc trưng với nhau để mô hình hóa tương tác giữa các biến.
    - Ví dụ: \$ X_1 \times X_2 \$


### Công thức hồi quy đa thức:

$$
Y = \beta_0 + \beta_1 X + \beta_2 X^2 + ... + \beta_p X^p
$$

### Tạo polynomial features với sklearn:

```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
```


## 6. Chọn Dạng Hàm (Functional Form)

- **Phân tích tương quan:** Tạo các đặc trưng mới, kiểm tra tương quan với biến mục tiêu để chọn đặc trưng phù hợp.
- **Feature selection:** Sử dụng các phương pháp như kiểm định thống kê, regularization, hoặc recursive feature elimination (RFE).


## 7. Khung Đánh Giá Mô Hình

- Các nguyên tắc này áp dụng cho cả hồi quy và phân loại.
- Luôn chú ý đến bias-variance trade-off.
- Các mô hình phức tạp hơn (logistic regression, decision tree, v.v.) cũng cần sử dụng các kỹ thuật biến đổi đặc trưng và regularization để kiểm soát độ phức tạp và lỗi.


## 8. Tổng Kết Quy Trình

1. Chia dữ liệu thành train/test (và validation nếu cần).
2. Huấn luyện mô hình trên training set.
3. Đánh giá mô hình trên test set bằng các chỉ số phù hợp.
4. Sử dụng cross-validation để có đánh giá tổng quát hơn.
5. Mở rộng mô hình bằng polynomial features, interaction terms nếu cần.
6. Chọn đặc trưng và dạng hàm tối ưu dựa trên phân tích dữ liệu và kết quả đánh giá.

