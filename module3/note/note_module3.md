
# Cross-Validation Overview

## 1. Khái niệm Cross-Validation

Cross-validation là kỹ thuật đánh giá mô hình bằng cách chia tập dữ liệu thành nhiều cặp tập huấn luyện và kiểm tra. Mỗi lần lặp, mô hình được huấn luyện trên một phần dữ liệu và đánh giá trên phần còn lại, giúp:

- Đánh giá ổn định hơn so với chia một lần.
- Giảm độ sai lệch do phân chia ngẫu nhiên.


## 2. K-Fold Cross-Validation

1. Chia toàn bộ dữ liệu thành $K$ phần (fold) bằng nhau.
2. Lặp $k=1$ đến $K$:
    - Tập huấn luyện: tất cả các fold trừ fold thứ $k$.
    - Tập kiểm tra: fold thứ $k$.
    - Tính sai số $\text{Error}_k$ trên fold kiểm tra.
3. Sai số trung bình:

$$
\text{CV Error} = \frac{1}{K} \sum_{k=1}^{K} \text{Error}_k
$$

### Công thức tính sai số

- **Mean Squared Error (MSE)** cho hồi quy:

$$
\text{MSE}_k = \frac{1}{n_k} \sum_{i=1}^{n_k} (y_i^{(k)} - \hat{y}_i^{(k)})^2
$$
- **Accuracy** cho phân loại:

$$
\text{Accuracy}_k = \frac{\text{Số dự đoán đúng}}{\text{Tổng số mẫu trong fold }k}
$$


### Triển khai với Python

```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression

model = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=1)

# Với hồi quy, dùng scoring='neg_mean_squared_error'
scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
mse_scores = -scores  # chuyển về giá trị dương
cv_mse = mse_scores.mean()
```


## 3. Variants của K-Fold

- **Stratified K-Fold:** Giữ nguyên tỉ lệ nhãn ở mỗi fold, quan trọng với dữ liệu phân loại mất cân bằng.
- **Leave-One-Out Cross-Validation (LOOCV):** Mỗi fold chỉ chứa 1 mẫu làm kiểm tra, đặc biệt hiệu quả với tập dữ liệu rất nhỏ nhưng tốn tài nguyên.
- **ShuffleSplit:** Chia nhiều lần ngẫu nhiên với tỉ lệ train/test cố định.
- **Stratified ShuffleSplit:** Kết hợp stratification và chia ngẫu nhiên nhiều lần.


## 4. Mối quan hệ giữa Model Complexity và CV Error

- **Underfitting (bias cao):** Mô hình đơn giản, sai số huấn luyện và CV đều cao.
- **Overfitting (variance cao):** Mô hình phức tạp, sai số huấn luyện rất thấp nhưng sai số CV cao.
- **Sweet Spot:** Điểm độ phức tạp tối ưu khi sai số CV đạt cực tiểu.


### Biểu đồ minh họa

```
Error
  ^
  |          .         
  |        .   .       <-- CV Error
  |      .       .  
  |    .           . 
  |  .               .
  | . Training Error  .
  +----------------------> Model Complexity
```


## 5. Lựa chọn K

- $K=5$ hoặc $K=10$ là phổ biến.
- Lớn hơn giảm bias nhưng tăng variance và chi phí tính toán.


## 6. Best Practices

- **Shuffle dữ liệu** trước khi chia.
- Dùng **Stratified K-Fold** với bài toán phân loại.
- Khi dữ liệu nhỏ, xem xét **LOOCV**.
- Sử dụng cross-validation để **tuning hyperparameters**.
- Luôn báo cáo **mean** và **standard deviation** của sai số trên các fold:

$$
\text{CV Mean} \pm \text{CV Std}
$$

***

**Kết luận:** Cross-validation cung cấp ước lượng hiệu năng mô hình ổn định và chính xác hơn so với chia train-test đơn lẻ, đồng thời giúp lựa chọn độ phức tạp và điều chỉnh siêu tham số để tránh underfitting và overfitting.

