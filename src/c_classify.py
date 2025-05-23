import os
import pandas as pd
from transformers import pipeline


LABELS = [
    "billing issue",
    "shipping problem",
    "product defect",
    "customer service",
    "other"
]


class ZeroShotTextClassifier:
    def __init__(self, model_name="facebook/bart-large-mnli", device=0, labels=None):
        self.labels = labels if labels else LABELS
        self.classifier = pipeline("zero-shot-classification", model=model_name, device=device)

    def classify_text(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return "unknown"
        result = self.classifier(text, candidate_labels=self.labels)
        return result["labels"][0]

    def classify_dataframe(self, df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
        df["category"] = df[text_col].apply(self.classify_text)
        return df[[text_col, "category"]]


def main(input_path: str, output_path: str, limit: int = 100):
    df = pd.read_parquet(input_path).head(limit)

    classifier = ZeroShotTextClassifier()
    df_classified = classifier.classify_dataframe(df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_classified.to_parquet(output_path, index=False)


if __name__ == "__main__":
    INPUT_PATH = "data/processed/curated/"
    OUTPUT_PATH = "data/processed/classified/classified.parquet"

    main(INPUT_PATH, OUTPUT_PATH)
