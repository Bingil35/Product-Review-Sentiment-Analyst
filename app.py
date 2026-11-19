"""Streamlit demo for the trained Vietnamese product review sentiment Pipeline.

Run from the project root: streamlit run app.py
"""
from __future__ import annotations

import io
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Import must be available when joblib reconstructs the sklearn Pipeline.
from src.text_preprocessing import preprocess_text  # noqa: F401

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
LABEL_NAMES = {"positive": "Tích cực", "negative": "Tiêu cực"}

st.set_page_config(
    page_title="Phân tích cảm xúc | Product Review Sentiment Analyst",
    page_icon="",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.block-container {max-width: 900px; padding-top: 2.2rem; padding-bottom: 3rem;}
.hero {padding: 1.5rem 1.6rem; border-radius: 20px; background: linear-gradient(115deg,#132e47,#285b75); color: white; margin-bottom: 1.3rem;}
.hero h1 {color: white; font-size: 2rem; line-height: 1.2; margin: .3rem 0 .6rem;}
.hero p {color: #e2f0f6; margin: 0;}
.eyebrow {font-size: .8rem; font-weight: 700; letter-spacing: .12em; color: #bfe3f2;}
.result {padding: 1.2rem 1.4rem; border: 1px solid #c9d7e0; border-radius: 15px; margin-top: 1rem;}
.result-title {font-size: 1.25rem; font-weight: 750; margin-bottom: .4rem;}
.note {font-size: .88rem; opacity: .78;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<div class="eyebrow">VIETNAMESE E-COMMERCE</div>
<h1>Phân tích cảm xúc đánh giá sản phẩm</h1>
<p>Nhập nhận xét bằng tiếng Việt để dự đoán cảm xúc tích cực hoặc tiêu cực bằng mô hình đã huấn luyện.</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Đang tải mô hình Machine Learning...")
def load_model(path: str, modified_ns: int):
    # modified_ns invalidates cache if the local model file is replaced.
    # joblib/pickle is unsafe for untrusted files: only load your own model.
    model = joblib.load(path)
    if not hasattr(model, "predict"):
        raise TypeError("Tệp mô hình không hỗ trợ predict().")
    return model


def classify(model, reviews: list[str]) -> tuple[list[str], list[dict[str, float]]]:
    labels = [str(x).lower() for x in model.predict(reviews)]
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(reviews)
        classes = [str(c).lower() for c in model.classes_]
        probabilities = [
            {name: float(prob) for name, prob in zip(classes, row)}
            for row in scores
        ]
    else:
        probabilities = [{} for _ in reviews]
    return labels, probabilities


if not MODEL_PATH.is_file():
    st.error("Không tìm thấy `models/best_model.joblib`.")
    st.info("Hãy chạy thành công notebook `04_Final_Evaluation.ipynb` và đặt `app.py` ở thư mục gốc của dự án.")
    st.stop()

try:
    model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
except Exception as exc:
    st.error("Không thể tải mô hình. Kiểm tra phiên bản thư viện và module `src/text_preprocessing.py`.")
    st.exception(exc)
    st.stop()

st.caption("Mô hình: Logistic Regression · TF-IDF · Tiếng Việt · Phân loại nhị phân")
tab_one, tab_many = st.tabs(["Phân tích một đánh giá", "Phân tích hàng loạt (CSV)"])

with tab_one:
    with st.form("single_prediction", clear_on_submit=False):
        review = st.text_area(
            "Nội dung đánh giá",
            placeholder="Ví dụ: Giao hàng nhanh, sản phẩm tốt hơn mong đợi!",
            height=135,
            max_chars=5000,
        )
        submitted = st.form_submit_button("Phân tích cảm xúc", type="primary", use_container_width=True)

    if submitted:
        if not review.strip():
            st.warning("Vui lòng nhập nội dung đánh giá trước khi phân tích.")
        elif not preprocess_text(review):
            st.warning("Đánh giá không còn nội dung hợp lệ sau khi chuẩn hóa.")
        else:
            try:
                labels, probabilities = classify(model, [review])
                label, probs = labels[0], probabilities[0]
                if label == "positive":
                    st.success("**Kết quả: Tích cực (Positive)**")
                elif label == "negative":
                    st.error("**Kết quả: Tiêu cực (Negative)**")
                else:
                    st.info(f"Nhãn mô hình trả về: `{label}`")

                if probs:
                    st.markdown("**Xác suất dự đoán**")
                    for key in ("positive", "negative"):
                        if key in probs:
                            st.write(f"{LABEL_NAMES[key]} — **{probs[key]:.2%}**")
                            st.progress(min(max(probs[key], 0.0), 1.0))
                    st.caption("Xác suất là ước lượng của mô hình, không phải mức độ chắc chắn tuyệt đối.")
                else:
                    st.caption("Mô hình hiện tại không hỗ trợ predict_proba().")
            except Exception as exc:
                st.error("Không thể dự đoán đánh giá này.")
                st.exception(exc)

with tab_many:
    st.write("Tải lên tệp CSV UTF-8 có cột **`review`**")
    uploaded = st.file_uploader("Chọn tệp CSV", type=["csv"], key="review_csv")
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded, encoding="utf-8-sig")
            if "review" not in df.columns:
                st.error("CSV phải chứa cột tên chính xác là `review`.")
            elif len(df) == 0:
                st.warning("CSV không có dòng dữ liệu.")
            elif len(df) > 10000:
                st.warning("Để tránh quá tải, mỗi lần chỉ xử lý tối đa 10.000 dòng.")
            else:
                valid = df["review"].fillna("").astype(str).map(lambda x: bool(preprocess_text(x)))
                result = df.copy()
                result["predicted_label"] = ""
                result["prob_positive"] = float("nan")
                result["prob_negative"] = float("nan")
                if valid.any():
                    texts = df.loc[valid, "review"].astype(str).tolist()
                    labels, probs = classify(model, texts)
                    result.loc[valid, "predicted_label"] = labels
                    for key in ("positive", "negative"):
                        if any(key in p for p in probs):
                            result.loc[valid, f"prob_{key}"] = [p.get(key, float("nan")) for p in probs]
                st.success(f"Đã phân tích {int(valid.sum()):,}/{len(df):,} đánh giá hợp lệ.")
                st.dataframe(result.head(100), use_container_width=True, hide_index=True)
                csv_bytes = result.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "Tải kết quả CSV",
                    data=csv_bytes,
                    file_name="sentiment_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        except Exception as exc:
            st.error("Không đọc hoặc phân tích được CSV. Kiểm tra định dạng và dữ liệu.")
            st.exception(exc)

st.divider()
st.caption("Dự án học máy · Kết quả chỉ mang tính dự đoán tự động và có thể sai với câu mỉa mai, đánh giá trung tính hoặc ngữ cảnh phức tạp.")
