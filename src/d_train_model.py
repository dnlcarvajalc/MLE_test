# src/train_model.py

import pandas as pd
from datasets import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)


class TextClassifierTrainer:
    def __init__(self,
                 input_path: str,
                 output_path: str,
                 model_name: str = "bert-base-uncased"):
        self.input_path = input_path
        self.output_path = output_path
        self.model_name = model_name
        self.df = pd.read_parquet(self.input_path)
        self.label2id = {label: i for i, label in enumerate(self.df["category"].unique())}
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id,
        )

    def tokenize_function(self, example):
        return self.tokenizer(
            example["clean_text"],
            truncation=True,
            padding="max_length"
        )

    def prepare_dataset(self):
        dataset = Dataset.from_pandas(self.df)
        dataset = dataset.map(self.tokenize_function, batched=True)
        dataset = dataset.map(lambda x: {"label": self.label2id[x["category"]]})
        return dataset

    def train(self, output_dir: str = "./models", batch_size: int = 8, epochs: int = 3):
        tokenized_dataset = self.prepare_dataset()
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )
        trainer.train()
        self.model.save_pretrained(self.output_path)
        self.tokenizer.save_pretrained(self.output_path)


if __name__ == "__main__":
    INPUT_PATH = "data/processed/classified/classified.parquet"
    OUTPUT_PATH = "./output"
    trainer = TextClassifierTrainer(INPUT_PATH, OUTPUT_PATH)
    trainer.train()

