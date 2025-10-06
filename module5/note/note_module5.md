

# Regularization: Analytical, Geometric, and Probabilistic Views

## 1. Analytical View

- **Regularization** là kỹ thuật thêm penalty vào hàm mất mát để kiểm soát độ lớn của các hệ số hồi quy.
- **L1 (LASSO):** Phạt tổng giá trị tuyệt đối của các hệ số.
- **L2 (Ridge):** Phạt tổng bình phương các hệ số.
- **Ý nghĩa:** Hệ số nhỏ → đặc trưng ít ảnh hưởng đến biến mục tiêu, hệ số lớn → mô hình nhạy cảm, variance cao.


### Công thức tổng quát:

- **Ridge Regression (L2):**

$$
J(\beta) = \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^p \beta_j^2
$$
- **LASSO Regression (L1):**

$$
J(\beta) = \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^p |\beta_j|
$$


## 2. Geometric View

- **Ridge:** Constraint vùng chọn hệ số là hình tròn (Euclidean ball), các hệ số bị "ép" về gần 0 nhưng hiếm khi bằng 0.
- **LASSO:** Constraint vùng chọn hệ số là hình thoi (diamond), dễ tạo ra các hệ số đúng bằng 0 (feature selection).


### Biểu diễn hình học:

- **Ridge:**

$$
\sum_{j=1}^p \beta_j^2 \leq t
$$

(Vùng constraint là hình tròn trong không gian hệ số)
- **LASSO:**

$$
\sum_{j=1}^p |\beta_j| \leq t
$$

(Vùng constraint là hình thoi)


## 3. Probabilistic View

- **Regularization** tương đương với việc đặt prior lên hệ số hồi quy trong mô hình Bayesian.
- **Ridge (L2):**
    - Prior Gaussian (Normal):

$$
\beta_j \sim \mathcal{N}(0, \tau^2)
$$
- **LASSO (L1):**
    - Prior Laplacian:

$$
\beta_j \sim \text{Laplacian}(0, b)
$$
- **Ý nghĩa:**
    - Gaussian prior "ép" hệ số về gần 0 nhưng không triệt tiêu hoàn toàn.
    - Laplacian prior dễ tạo ra hệ số đúng bằng 0 (feature selection).


## 4. Ảnh hưởng của Regularization

- **Tăng bias, giảm variance:**
    - Mô hình tổng quát hóa tốt hơn, ít nhạy cảm với nhiễu dữ liệu.
- **LASSO:**
    - Tốt cho feature selection, mô hình dễ diễn giải.
- **Ridge:**
    - Tốt cho dữ liệu có nhiều đặc trưng liên quan, không loại bỏ hoàn toàn đặc trưng.


## 5. Tổng kết công thức và ý nghĩa

- **Trade-off bias-variance:**
    - Regularization giúp cân bằng giữa độ chính xác và khả năng tổng quát hóa.
- **Chọn loại regularization:**
    - LASSO: Khi muốn chọn đặc trưng quan trọng.
    - Ridge: Khi muốn giữ tất cả đặc trưng nhưng kiểm soát độ lớn hệ số.
- **Elastic Net:** Kết hợp L1 và L2 để tận dụng ưu điểm của cả hai.


## 6. Công thức Elastic Net

$$
J(\beta) = \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda_1 \sum_{j=1}^p |\beta_j| + \lambda_2 \sum_{j=1}^p \beta_j^2
$$

## 7. Visual Summary

- **Ridge:** Constraint là hình tròn, hệ số nhỏ nhưng hiếm khi bằng 0.
- **LASSO:** Constraint là hình thoi, nhiều hệ số bằng 0.
- **Probabilistic:** Ridge ~ Gaussian prior, LASSO ~ Laplacian prior.

***

**Kết luận:** Hiểu rõ các góc nhìn phân tích, hình học và xác suất giúp lựa chọn regularization phù hợp, tối ưu hóa mô hình và kiểm soát độ phức tạp hiệu quả.

