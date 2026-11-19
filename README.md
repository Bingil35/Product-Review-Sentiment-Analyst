# Product Review Sentiment Analyst

**Phân tích cảm xúc đánh giá sản phẩm thương mại điện tử (Shopee)** — dự án học máy phân loại nội dung đánh giá tiếng Việt thành hai nhãn **`positive`** (tích cực) và **`negative`** (tiêu cực).

**Trạng thái:** Đã hoàn thành EDA, tiền xử lý, baseline, so sánh ba thuật toán, tối ưu siêu tham số, đánh giá cuối cùng trên Test và giao diện demo Streamlit (dự đoán từng đánh giá hoặc hàng loạt từ CSV).

## 1. Mục tiêu và phương pháp

- Khảo sát dữ liệu đánh giá sản phẩm và chất lượng nhãn.
- Xây dựng quy trình tiền xử lý văn bản tiếng Việt, hạn chế rò rỉ dữ liệu (data leakage).
- Biểu diễn văn bản bằng **TF-IDF** (unigram và bigram).
- So sánh **Multinomial Naive Bayes**, **Logistic Regression** và **Random Forest**.
- Đánh giá bằng Stratified K-Fold Cross-Validation, tìm kiếm siêu tham số và các chỉ số Accuracy, Macro Precision, Macro Recall, Macro F1, ROC-AUC.
- Đóng gói mô hình tốt nhất thành `models/best_model.joblib`, cung cấp `predict.py` và giao diện Streamlit.

## 2. Dữ liệu

- **Nguồn tham khảo:** [Shopee Vietnamese Product Reviews Sentiment — Kaggle](https://www.kaggle.com/datasets/dduongdev/shopee-vietnamese-product-reviews-sentiment/data).
- **Định dạng:** JSON Lines (`.jsonl`), gồm các trường `id`, `review`, `rating`, `label`.
- **Dữ liệu gốc:** 9.599 đánh giá; 5.965 `negative` và 3.634 `positive`.
- **Dữ liệu thực nghiệm:** 9.595 đánh giá sau khi loại 4 bản ghi trùng theo văn bản đã chuẩn hóa.

Tải dữ liệu từ nguồn trên và đặt tại:

```text
data/raw/shopee_reviews_dataset.jsonl
```

**Lưu ý về bản quyền:** Dữ liệu Shopee/Kaggle không tự động thuộc phạm vi Apache License 2.0 của mã nguồn. Chỉ chia sẻ hoặc đưa dữ liệu vào repository nếu giấy phép nguồn cho phép.

## 3. Khám phá dữ liệu và tiền xử lý

- Dữ liệu gốc không có review rỗng, giá trị thiếu hoặc bản ghi review trùng chính xác theo kiểm tra ban đầu.
- Khảo sát phân bố nhãn, số sao, độ dài đánh giá, từ đơn, bigram và cách viết không chuẩn.
- Rà soát thủ công 156 review: 92 `negative`, 53 `positive`, 11 `neutral`. Nhãn `neutral` chỉ dùng trong kiểm tra chất lượng nhãn, **không phải** nhãn đầu ra của mô hình. Mức khớp trên 145 trường hợp nhị phân rõ ràng là **93,79%**; mẫu kiểm tra có chủ đích nên không đại diện mặc nhiên cho toàn bộ dữ liệu.
- Chuẩn hóa Unicode NFC, chuyển chữ thường, loại URL và chuẩn hóa khoảng trắng.
- Chuẩn hóa một số phủ định viết tắt (`ko`, `k`, `khum` → `không`); giữ lại từ phủ định, không xóa stopwords hàng loạt.
- Loại trùng theo văn bản chuẩn hóa **trước khi chia dữ liệu**; không phát hiện nội dung trùng chính xác sau chuẩn hóa giữa các tập Train, Validation và Test. Điều này không loại trừ các đánh giá gần trùng về ngữ nghĩa.

### Phân chia dữ liệu

| Tập | Số review | Vai trò |
|---|---:|---|
| Train | 6.715 | Huấn luyện và tối ưu trong giai đoạn phát triển |
| Validation | 1.440 | So sánh, chọn mô hình |
| Test | 1.440 | Đánh giá cuối cùng, giữ độc lập đến khi chọn mô hình |
| **Tổng** | **9.595** | Chia phân tầng theo nhãn, `random_state=42` |

ID và nhãn chia tập được lưu trong `results/data_splits.csv`. Trong file này, tập Validation có thể được ghi là `validation`; notebook so sánh mô hình chuẩn hóa tên thành `val` khi đọc.

## 4. Kết quả thực nghiệm

### Baseline — Multinomial Naive Bayes

Kết quả trên **Validation** của notebook baseline:

| Chỉ số | Giá trị |
|---|---:|
| Accuracy | 93,33% |
| Macro Precision | 93,18% |
| Macro Recall | 92,59% |
| Macro F1-score | **92,87%** |

### So sánh mô hình trên Validation

Các kết quả sau thuộc giai đoạn so sánh/tối ưu mô hình, khác với baseline ban đầu:

| Mô hình | Accuracy | Macro F1 | ROC-AUC |
|---|---:|---:|---:|
| **Logistic Regression** | **94,79%** | **94,45%** | **98,62%** |
| Multinomial Naive Bayes | 93,13% | 92,67% | 97,94% |
| Random Forest | 91,32% | 90,78% | 96,98% |

**Mô hình được chọn:** Logistic Regression + TF-IDF, dựa trên **Macro F1-score trên Validation**. Việc lựa chọn không dựa trên Test.

### Kết quả cuối cùng trên Test

Mô hình đã chọn được huấn luyện lại trên **Train + Validation (8.155 review)** và đánh giá một lần trên **Test độc lập (1.440 review)**.

| Chỉ số | Kết quả Test |
|---|---:|
| **Accuracy** | **95,42%** |
| Macro Precision | 95,42% |
| Macro Recall | 94,81% |
| **Macro F1-score** | **95,10%** |
| ROC-AUC | 98,77% |

Siêu tham số của mô hình cuối cùng:

```python
{
    "classifier__C": 8.53864875617199,
    "classifier__class_weight": None,
    "tfidf__min_df": 5,
    "tfidf__ngram_range": (1, 2),
}
```

Kết quả Test tốt hơn Validation khoảng **0,65 điểm phần trăm Macro F1**, cho thấy mô hình không bị suy giảm rõ rệt trên tập kiểm thử này; không đủ để khẳng định tuyệt đối rằng không có overfitting. Xem thêm confusion matrix, ROC và phân tích lỗi trong `results/milestone2/final/` nếu các file này đã được tạo.

## 5. Cấu trúc dự án

```text
Product Review Sentiment Analyst/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── app.py                         # Giao diện Streamlit
├── predict.py                     # Dự đoán từ dòng lệnh
├── data/
│   └── raw/
│       └── shopee_reviews_dataset.jsonl  # Tải riêng từ Kaggle
├── models/
│   └── best_model.joblib          # Sinh sau notebook 04
├── src/
│   ├── __init__.py
│   └── text_preprocessing.py      # Hàm tiền xử lý dùng khi nạp mô hình
├── notebooks/
│   ├── 01_EDA_Preprocessing.ipynb
│   ├── 02_Baseline_Model.ipynb
│   ├── 03_Model_Comparison.ipynb
│   └── 04_Final_Evaluation.ipynb
├── results/
│   ├── baseline_metrics.csv
│   ├── data_splits.csv
│   ├── label_audit_reviewed.csv
│   └── milestone2/
│       ├── validation_comparison.csv
│       ├── hyperparameter_search_results.csv
│       └── final/
│           ├── test_metrics.csv
│           ├── final_model_metadata.json
│           ├── test_confusion_matrix.png
│           ├── test_roc_curve.png
│           └── test_error_summary.csv
└── reports/
    └── Bao_cao_Milestone_1_Shopee_Sentiment.docx
```

Cây thư mục minh họa các đầu ra dự kiến. Một số file chỉ xuất hiện sau khi chạy notebook, tải dữ liệu hoặc sao chép báo cáo.

## 6. Cài đặt môi trường (Windows / PowerShell)

**Yêu cầu:** Python **3.12**, Git và kết nối mạng khi cài thư viện. Mở PowerShell tại thư mục gốc dự án:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name product-review-sentiment --display-name "Python (Product Review Sentiment)"
```

Nếu PowerShell không cho phép kích hoạt môi trường, có thể dùng trực tiếp `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` hoặc chọn interpreter `.venv` trong VS Code.

**Quan trọng:** File `.joblib` được tạo bằng scikit-learn và joblib; nên sử dụng cùng phiên bản thư viện với môi trường huấn luyện khi nạp mô hình. Nếu gặp cảnh báo không tương thích, kiểm tra phiên bản môi trường đã dùng để xuất model; không tự ý bỏ qua cảnh báo.

## 7. Chạy các notebook theo thứ tự

1. Tải dataset vào `data/raw/shopee_reviews_dataset.jsonl`.
2. Chạy `notebooks/01_EDA_Preprocessing.ipynb` — EDA và kiểm tra dữ liệu.
3. Chạy `notebooks/02_Baseline_Model.ipynb` — tạo `results/data_splits.csv`, huấn luyện baseline.
4. Chạy `notebooks/03_Model_Comparison.ipynb` — so sánh ba mô hình và tìm siêu tham số.
5. Chạy `notebooks/04_Final_Evaluation.ipynb` — chọn mô hình từ kết quả Validation, huấn luyện lại, đánh giá Test và xuất `models/best_model.joblib`.

Chọn kernel `Python (Product Review Sentiment)` trong VS Code/Jupyter. Các notebook dùng đường dẫn tương đối; nên mở/chạy với thư mục làm việc là `notebooks/` nếu notebook yêu cầu.

**Không chạy lại bước chọn mô hình dựa trên kết quả Test.** Nếu muốn thay đổi mô hình/siêu tham số, cần thiết kế một quy trình đánh giá mới phù hợp thay vì liên tục thử trên Test hiện tại.

## 8. Demo Streamlit

Sau khi có `models/best_model.joblib` và `src/text_preprocessing.py`, chạy từ **thư mục gốc dự án**:

```powershell
python -m streamlit run app.py
```

Mở **http://localhost:8501**. Ứng dụng hỗ trợ:

- **Phân tích một đánh giá:** nhập review tiếng Việt, xem nhãn `positive`/`negative` và xác suất của từng lớp.
- **Phân tích hàng loạt (CSV):** tải lên file `.csv` có cột **`review`**, xem kết quả và tải CSV dự đoán. Giao diện hiện giới hạn tối đa **10.000 review/lần**.

Ví dụ CSV:

```csv
review
"Hàng chất lượng, đáng tiền"
"Sản phẩm bị lỗi, rất thất vọng"
"Không tệ như mình nghĩ"
```

Xác suất hiển thị là ước lượng của mô hình, **không phải mức chắc chắn tuyệt đối**. Mô hình chỉ có hai nhãn; nhận xét trung lập hoặc pha trộn ý kiến vẫn bị phân về một trong hai lớp.

## 9. Dự đoán bằng dòng lệnh

Chạy từ thư mục gốc:

```powershell
python predict.py "Hàng đẹp, giao nhanh, rất hài lòng"
python predict.py "Sản phẩm bị lỗi, không đáng tiền"
```

Lệnh sử dụng Pipeline đã huấn luyện; không cần chạy lại notebook để dự đoán nếu đã có model và môi trường phù hợp.

## 10. Tái lập thực nghiệm và giới hạn

- Chia dữ liệu có phân tầng và dùng `random_state=42` trong các bước đã ghi nhận.
- TF-IDF được `fit` trong scikit-learn `Pipeline`, tránh học đặc trưng từ Validation/Test trong giai đoạn phát triển.
- Sử dụng Stratified K-Fold và tìm kiếm siêu tham số trên dữ liệu huấn luyện.
- Lựa chọn mô hình theo Validation Macro F1; Test được giữ riêng cho đánh giá cuối cùng.
- Các chỉ số được báo cáo cho bộ dữ liệu Shopee sử dụng trong dự án, không mặc nhiên phản ánh hiệu năng trên mọi nền tảng, miền sản phẩm hay kiểu ngôn ngữ.
- Các trường hợp mỉa mai, đánh giá đa ý, tiếng lóng mới và nhận xét trung lập có thể gây nhầm lẫn.

## 11. Giấy phép

Mã nguồn của repository được phát hành theo **Apache License 2.0** (xem [`LICENSE`](LICENSE)), với điều kiện người đóng góp có quyền cấp phép mã nguồn tương ứng. Giấy phép này **không mặc nhiên áp dụng** cho dữ liệu Shopee, mô hình bên thứ ba hoặc tài liệu có bản quyền khác.
