import torch
import pandas as pd
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from classifier import BertClassifier
from dataset import Dataset


def train(model, train_data, val_data, learning_rate, epochs, batch_size=4):
    train_ds, val_ds = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')


bc = BertClassifier()
df_train = pd.read_csv("../data/train.csv")
df_val = pd.read_csv("../data/test.csv")
lr = 1e-6
num_of_epochs = 5

train(bc, df_train.iloc[0:1000], df_val.iloc[0:1000], lr, num_of_epochs)
