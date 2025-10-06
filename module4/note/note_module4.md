

# Model Complexity, Error, and Regularization

## 1. Model Complexity and Error

- **Mục tiêu:** Tối thiểu hóa cả training error và test error bằng cách chọn độ phức tạp mô hình phù hợp.
- **Underfitting:** Mô hình quá đơn giản, bias cao, variance thấp, lỗi lớn trên cả train và test.
- **Overfitting:** Mô hình quá phức tạp, bias thấp, variance cao, lỗi train nhỏ nhưng lỗi test lớn.


### Biểu đồ minh họa mối quan hệ giữa lỗi và độ phức tạp

```
Error
  ^
  |   \         .
  |    \      .   .
  |     \  .       .
  |      .           .
  |----------------------> Model Complexity
         Underfit   Overfit
```


## 2. Bias và Variance

- **Bias:** Xu hướng mô hình bỏ qua quy luật thực tế do quá đơn giản.
- **Variance:** Độ nhạy của mô hình với biến động nhỏ trong dữ liệu huấn luyện.
- **Bias-Variance Tradeoff:**

$$
\text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}
$$
- **Mục tiêu:** Chọn mô hình có bias và variance đều thấp.


## 3. Nguồn gốc lỗi mô hình

- **Bias cao:** Underfitting.
- **Variance cao:** Overfitting.
- **Irreducible error:** Nhiễu ngẫu nhiên không thể loại bỏ.


## 4. Regularization (Chuẩn hóa)

- **Mục đích:** Giảm overfitting bằng cách phạt các hệ số lớn, giúp mô hình tổng quát hóa tốt hơn.


### Ridge Regression (L2 Regularization)

- **Hàm mất mát:**

$$
J(\beta) = \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^p \beta_j^2
$$
- **Tác dụng:** Shrink tất cả hệ số về gần 0 nhưng không triệt tiêu hoàn toàn.
- **Cần chuẩn hóa đặc trưng:**

$$
x' = \frac{x - \mu}{\sigma}
$$

với \$ \mu \$ là trung bình, \$ \sigma \$ là độ lệch chuẩn.


### LASSO Regression (L1 Regularization)

- **Hàm mất mát:**

$$
J(\beta) = \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^p |\beta_j|
$$
- **Tác dụng:** Đưa một số hệ số về đúng 0 (feature selection tự động).


### Elastic Net

- **Hàm mất mát:**

$$
J(\beta) = \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda_1 \sum_{j=1}^p |\beta_j| + \lambda_2 \sum_{j=1}^p \beta_j^2
$$
- **Kết hợp ưu điểm của Ridge và LASSO.**


## 5. Ảnh hưởng của Lambda ($\lambda$)

- **$\lambda$ nhỏ:** Ít regularization, mô hình dễ overfit.
- **$\lambda$ lớn:** Regularization mạnh, mô hình dễ underfit.
- **Chọn $\lambda$ tối ưu:** Dùng cross-validation để tìm giá trị cân bằng bias-variance.


## 6. Feature Selection và RFE

- **Regularization:** LASSO tự động loại bỏ đặc trưng không quan trọng.
- **RFE (Recursive Feature Elimination):**
    - Lặp lại quá trình huấn luyện, loại bỏ đặc trưng ít quan trọng nhất ở mỗi vòng.
    - Dùng với các mô hình có thuộc tính `coef_` hoặc `feature_importances_`.


### Triển khai RFE với sklearn

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
model = LinearRegression()
rfe = RFE(model, n_features_to_select=5)
rfe.fit(X_train, y_train)
```


### RFECV (RFE + Cross-Validation)

```python
from sklearn.feature_selection import RFECV
rfecv = RFECV(model, cv=5)
rfecv.fit(X_train, y_train)
```


## 7. Tổng kết

- Chọn mô hình phù hợp dựa trên bias-variance tradeoff.
- Sử dụng regularization (Ridge, LASSO, Elastic Net) để kiểm soát độ phức tạp.
- Dùng RFE hoặc regularization để chọn đặc trưng quan trọng.
- Luôn đánh giá mô hình bằng cross-validation để đảm bảo tổng quát hóa tốt.

