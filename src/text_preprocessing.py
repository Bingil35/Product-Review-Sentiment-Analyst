"""Stable, importable text normalization used by training and inference."""
import re
import unicodedata

NEGATION_MAP = {"ko": "không", "k": "không", "khum": "không"}
NEGATION_PATTERN = re.compile(r"\b(?:ko|k|khum)\b", flags=re.IGNORECASE)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")


def preprocess_text(text):
    """Normalize Vietnamese reviews consistently with Milestone 1."""
    if text is None:
        return ""
    try:
        import pandas as pd
        if pd.isna(text):
            return ""
    except (ImportError, TypeError, ValueError):
        pass
    text = unicodedata.normalize("NFC", str(text)).lower()
    text = URL_PATTERN.sub(" ", text)
    text = NEGATION_PATTERN.sub(lambda m: NEGATION_MAP[m.group(0)], text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()
