import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Load data
df = pd.read_csv("../data/news.csv")
df = df.dropna()

# Ensure correct format
df["label"] = df["label"].astype(int)

dataset = Dataset.from_pandas(df)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def preprocess(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    tokens["labels"] = example["label"]   # CRITICAL FIX
    return tokens

dataset = dataset.map(preprocess)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

training_args = TrainingArguments(
    output_dir="./bert_model",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_dir="./logs",
    save_strategy="epoch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()

# FORCE SAVE
trainer.save_model("bert_model")
tokenizer.save_pretrained("bert_model")

print("BERT properly saved")