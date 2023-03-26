import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from dataset import Dataset


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average='macro')
    precision = precision_score(y_true=labels, y_pred=pred, average='macro')
    f1 = f1_score(y_true=labels, y_pred=pred, average='macro')

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


train_data = pd.read_csv("../data/train.csv")
val_data = pd.read_csv("../data/test.csv")

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=9)

label_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8}

X_train, y_train = list(train_data["content"]), [label_mapping[x] for x in train_data["label"]]
X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512)

X_val, y_val = list(val_data["content"]), [label_mapping[x] for x in val_data["label"]]
X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512)

train_dataset = Dataset(X_train_tokenized, y_train)
val_dataset = Dataset(X_val_tokenized, y_val)

wandb.init(project="Greenatom", name="bertforseqclass")

args = TrainingArguments(
    output_dir="bertforseqclass_trained_models",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    # eval_steps=500,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=30,
    seed=0,
    load_best_model_at_end=True,
    report_to=["wandb"]
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model('BertForSeqClass_trained_models')
trainer.save_state()

wandb.finish()
