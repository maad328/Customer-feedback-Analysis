import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

OUTPUT_DIR = "./bert_sentiment"

# -------------------------------
# -------------------------------
df = pd.read_csv("/kaggle/input/cleaned1/sentiment_analysis_cleaned1.csv")
df = df.dropna(subset=["text", "sentiment"])
print(f"Dataset shape: {df.shape}")

label_map = {"positive": 1, "negative": 0}
df["label"] = df["sentiment"].map(label_map)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

print(f"Training samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")

# -------------------------------
# -------------------------------
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def tokenize_batch(texts):
    return tokenizer(texts, padding=False, truncation=True, max_length=128)

train_encodings = tokenize_batch(train_texts)
val_encodings = tokenize_batch(val_texts)

# -------------------------------
# -------------------------------
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = SentimentDataset(train_encodings, train_labels)
val_dataset = SentimentDataset(val_encodings, val_labels)

# -------------------------------
# -------------------------------
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# -------------------------------
# -------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=10,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    report_to=[],
    save_total_limit=2,  # Keep only best + last
)

# -------------------------------
# -------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# -------------------------------
# -------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# -------------------------------
# 8️⃣ Train (NO MANUAL LOOP!)
# -------------------------------
print("Starting training...")
trainer.train()

# Best model is automatically loaded
print("\n=== Final Evaluation on Best Model ===")
predictions = trainer.predict(val_dataset)
preds = np.argmax(predictions.predictions, axis=-1)

print("\nClassification Report:\n", 
      classification_report(val_labels, preds, target_names=["negative", "positive"]))
print("\nConfusion Matrix:\n", confusion_matrix(val_labels, preds))

# Save final best model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)