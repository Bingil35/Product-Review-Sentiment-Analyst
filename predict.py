"""
Usage:
    python predict.py "Giao hàng nhanh, sản phẩm rất tốt"
    python predict.py --text "Sản phẩm lỗi, không hài lòng"
"""
import argparse
from pathlib import Path
import joblib

# Required for unpickling the preprocessing callable in the sklearn Pipeline.
from src.text_preprocessing import preprocess_text  # noqa: F401

DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "best_model.joblib"


def predict_sentiment(text: str, model_path=DEFAULT_MODEL) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Nội dung đánh giá không được để trống.")
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Chưa tìm thấy mô hình: {model_path}. Hãy chạy notebook 04 trước.")
    # Only load model artifacts created by this project; joblib is unsafe for untrusted files.
    model = joblib.load(model_path)
    label = str(model.predict([text])[0])
    result = {"label": label}
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        result["probabilities"] = {
            str(name): round(float(value), 6)
            for name, value in zip(model.classes_, probabilities)
        }
    return result


def main():
    parser = argparse.ArgumentParser(description="Phân loại cảm xúc đánh giá sản phẩm Shopee")
    parser.add_argument("text", nargs="?", help="Nội dung đánh giá")
    parser.add_argument("--text", dest="text_option", help="Nội dung đánh giá (tùy chọn)")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Đường dẫn best_model.joblib")
    args = parser.parse_args()
    text = args.text_option or args.text
    if text is None:
        text = input("Nhập đánh giá: ")
    result = predict_sentiment(text, args.model)
    print("Cảm xúc:", "Tích cực" if result["label"] == "positive" else "Tiêu cực")
    print("Nhãn:", result["label"])
    if "probabilities" in result:
        for label, probability in result["probabilities"].items():
            print(f"  P({label}) = {probability:.2%}")


if __name__ == "__main__":
    main()
