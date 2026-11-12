# Product Review Sentiment Analyst

**Phân tích cảm xúc đánh giá sản phẩm thương mại điện tử (Shopee)** — dự án học máy phân loại nội dung đánh giá thành hai nhóm **positive** và **negative**.

> **Trạng thái:** Đã hoàn thành thực nghiệm ban đầu cho Milestone 1 (EDA, kiểm tra nhãn, tiền xử lý và mô hình baseline). Các kết quả bên dưới là trên tập **Validation**, chưa phải kết quả đánh giá cuối cùng trên Test.

## 1. Mục tiêu

- Khám phá đặc điểm dữ liệu đánh giá sản phẩm bằng EDA.
- Kiểm tra chất lượng nhãn và xây dựng chiến lược tiền xử lý văn bản tiếng Việt.
- Huấn luyện mô hình baseline **TF-IDF + Multinomial Naive Bayes**.
- Tiếp tục so sánh các thuật toán học máy trong các milestone tiếp theo.

## 2. Dữ liệu

- **Nguồn tham khảo:** [Shopee Vietnamese Product Reviews Sentiment — Kaggle](https://www.kaggle.com/datasets/dduongdev/shopee-vietnamese-product-reviews-sentiment/data).
- **Định dạng:** JSON Lines (`.jsonl`).
- **Các trường:** `id`, `review`, `rating`, `label`.
- **Dữ liệu gốc:** 9.599 đánh giá; 5.965 `negative` và 3.634 `positive`.
- **Dữ liệu thực nghiệm:** 9.595 đánh giá sau khi loại 4 bản ghi trùng dư theo văn bản đã chuẩn hóa.

Đặt dữ liệu vào:

```text
data/raw/shopee_reviews_dataset.jsonl
```

## 3. Kết quả Milestone 1

### Khám phá dữ liệu (EDA)

- Không phát hiện giá trị thiếu, review rỗng hoặc review trùng chính xác trong dữ liệu gốc ở bước khảo sát ban đầu.
- Phân tích phân bố nhãn, số sao, độ dài review, từ đơn và bigram phổ biến.
- Khảo sát các hiện tượng văn bản như khoảng trắng thừa, từ phủ định viết tắt (`ko`, `k`, `khum`) và dấu câu lặp.
- Kiểm tra thủ công **156 review**: 92 `negative`, 53 `positive`, 11 `neutral` (nhãn `neutral` chỉ dùng để đánh giá chất lượng dữ liệu, **không** phải lớp dự đoán của mô hình). Mức khớp trên 145 trường hợp có nhãn nhị phân rõ ràng là **93,79%**; đây là thống kê của mẫu được kiểm tra, không đại diện mặc nhiên cho toàn bộ dataset.

### Tiền xử lý

- Chuẩn hóa Unicode NFC và chữ thường.
- Loại URL, chuẩn hóa khoảng trắng.
- Chuẩn hóa có kiểm soát một số cách viết phủ định (`ko`, `k`, `khum` → `không`).
- Giữ lại các từ phủ định; không xóa stopwords hàng loạt.
- Kiểm tra trùng lặp theo văn bản đã chuẩn hóa **trước khi chia dữ liệu**.

### Baseline

| Thành phần | Thiết lập |
|---|---|
| Biểu diễn đặc trưng | TF-IDF, unigram + bigram |
| Thuật toán | Multinomial Naive Bayes |
| Chia dữ liệu | Stratified Train / Validation / Test ≈ 70% / 15% / 15% |
| Train | 6.715 review |
| Validation | 1.440 review |
| Test | 1.440 review (giữ riêng cho đánh giá cuối) |

**Chỉ số trên Validation sau khi xử lý trùng:**

| Metric | Giá trị |
|---|---:|
| Accuracy | **93,33%** |
| Macro Precision | 93,18% |
| Macro Recall | 92,59% |
| Macro F1-score | **92,87%** |

Kiểm tra trùng chính xác sau chuẩn hóa cho thấy **0** nội dung trùng giữa Train–Validation, Train–Test và Validation–Test. Kiểm tra này không bảo đảm không có review gần trùng về ngữ nghĩa.

## 4. Cấu trúc thư mục

```text
Product Review Sentiment Analyst/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── data/
│   └── raw/
│       └── shopee_reviews_dataset.jsonl  # tải riêng
├── notebooks/
│   ├── 01_EDA_Preprocessing.ipynb
│   └── 02_Baseline_Model.ipynb
├── results/
│   ├── baseline_metrics.csv
│   ├── data_splits.csv
│   └── label_audit_reviewed.csv        # dữ liệu rà soát nội bộ
└── reports/
    └── Bao_cao_Milestone_1_Shopee_Sentiment.docx
```

> Cấu trúc trên là cấu trúc **đề xuất**. Một số file chỉ xuất hiện sau khi chạy notebook hoặc khi bạn sao chép báo cáo vào thư mục tương ứng.

## 5. Cài đặt và chạy

**Yêu cầu:** Python **3.12**, Git và Jupyter Notebook.

Trong PowerShell tại thư mục dự án:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name product-review-sentiment --display-name "Python (Product Review Sentiment)"
```

Sau đó:

1. Tải dataset từ đường dẫn Kaggle ở mục 2 và đặt đúng tên file trong `data/raw/`.
2. Mở `notebooks/01_EDA_Preprocessing.ipynb`, chọn kernel `Python (Product Review Sentiment)` và chạy các cell.
3. Mở `notebooks/02_Baseline_Model.ipynb`, chạy lần lượt các cell để chia dữ liệu, huấn luyện và đánh giá trên Validation.
4. Kiểm tra kết quả được xuất trong `results/`.

**Lưu ý:** Notebook hiện sử dụng đường dẫn tương đối như `../data/raw/...`; nên chạy với thư mục làm việc là `notebooks/`. Không dùng tập Test để lựa chọn tham số mô hình.

## 6. Kế hoạch tiếp theo

- So sánh Multinomial Naive Bayes, Logistic Regression và Random Forest.
- Đánh giá bằng Stratified K-Fold Cross-Validation.
- Tìm kiếm siêu tham số bằng GridSearchCV hoặc RandomizedSearchCV.
- Phân tích lỗi, độ ổn định và chất lượng nhãn.
- Đánh giá cuối cùng trên Test, lưu mô hình tốt nhất và bổ sung chương trình dự đoán.

## 7. Giấy phép

Mã nguồn của repository được phát hành theo **Apache License 2.0** (xem [`LICENSE`](LICENSE)), với điều kiện các thành viên đóng góp có quyền cấp phép phần mã nguồn tương ứng. Giấy phép này **không mặc nhiên áp dụng** cho dữ liệu Shopee hoặc tài liệu của bên thứ ba.

## 8. Tái lập thực nghiệm

- `random_state=42` được sử dụng trong các bước chia dữ liệu đã mô tả.
- TF-IDF được `fit` trên Train thông qua Scikit-learn `Pipeline`.
- Danh sách phân chia ID được lưu tại `results/data_splits.csv` khi chạy notebook.
- Các chỉ số Baseline được lưu tại `results/baseline_metrics.csv`.
