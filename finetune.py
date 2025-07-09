import argparse
import json
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, set_seed)

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MODEL_DIR = "./model"
LABEL2ID = {"negative": 0, "positive": 1}
ID2LABEL = {0: "negative", 1: "positive"}

class SentimentDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(item["text"], truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        label = LABEL2ID[item["label"]]
        return {"input_ids": encoding["input_ids"].squeeze(), "attention_mask": encoding["attention_mask"].squeeze(), "labels": torch.tensor(label)}

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", required=True, help="Path to data.jsonl")
    parser.add_argument("-epochs", type=int, default=3)
    parser.add_argument("-lr", type=float, default=3e-5)
    args = parser.parse_args()

    set_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data = load_data(args.data)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = SentimentDataset(data, tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=8,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=1000,
        save_total_limit=1,
        evaluation_strategy="no",
        weight_decay=0.01,
        fp16=False,
        seed=42,
        disable_tqdm=False,
        report_to=[],
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Model saved to {MODEL_DIR}")

if __name__ == "__main__":
    main() 