import torch
import pandas as pd
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import wandb
from sklearn.metrics import f1_score
from classifier import BertClassifier
from dataset import Dataset


def train(model, train_data, val_data, learning_rate, epochs, batch_size=2, model_name='bert_finetuned'):
    train_ds, val_ds = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    run = wandb.init(project="Greenatom", name=model_name)

    for epoch_num in range(epochs):

        total_loss_train = 0
        train_preds = []
        train_labels = []

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()

            train_preds.extend((output.argmax(dim=1).tolist()))
            train_labels.extend(train_label.tolist())

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        f1_score_train = f1_score(train_labels, train_preds, average='macro')
        total_loss_val = 0

        with torch.no_grad():

            val_preds = []
            val_labels = []

            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                val_preds.extend((output.argmax(dim=1).tolist()))
                val_labels.extend(val_label.tolist())

            f1_score_val = f1_score(val_labels, val_preds, average='macro')

        torch.save(model.state_dict(), f"../trained_models/bert_base_uncased_{model_name}_{epoch_num}epochs.pt")

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f}\
                | Train F1: {f1_score_train: .3f}\
                | Val Loss: {total_loss_val / len(val_data): .3f}\
                | Val F1: {f1_score_val: .3f}')

        wandb.log({
            'epoch': epoch_num + 1,
            'train_loss': total_loss_train / len(train_data),
            'train_f1': f1_score_train,
            'val_loss': total_loss_val / len(val_data),
            'val_f1': f1_score_val
        })

    wandb.finish()


train(model=BertClassifier(),
      train_data=pd.read_csv("../data/train.csv"),
      val_data=pd.read_csv("../data/test.csv"),
      learning_rate=1e-5,
      epochs=30,
      batch_size=4,
      model_name='3linesigm')
