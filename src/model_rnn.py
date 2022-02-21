import torch.nn.functional as F
import argparse
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
import torch

from torch import cuda, nn
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import AdamW, BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import time

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

parser = argparse.ArgumentParser()
parser.add_argument('--max_len', type=int, default=256)
parser.add_argument('--batch', type=int, default=16)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--test_size', type=float, default=.2)
parser.add_argument('--model', type=str, default='RNN')
args = parser.parse_args()

transformers.logging.set_verbosity_error()

MAX_LEN = args.max_len
BATCH = args.batch
EPOCHS = args.epoch
TEST_SIZE = args.test_size
LEARNING_RATE = 1e-05


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, maxlen):
        self.data = data
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sent1 = str(self.data['text'].iloc[index])
        sent2 = str(self.data['text_source'].iloc[index])
        target = self.data['label'].iloc[index]

        encoding = self.tokenizer.encode_plus(
            sent1, sent2,
            truncation=True,
            add_special_tokens=True,
            max_length=self.maxlen,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'],
            'targets': target
        }


def create_dataloader(df, tokenizer, max_len, batch_size):
    ds = CustomDataset(
        data=df,
        tokenizer=tokenizer,
        maxlen=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size
    )


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

df = pd.read_csv("./dataset/tweets.csv")
df_train, df_test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_SEED)
df_val, df_test = train_test_split(df_test, test_size=.5, random_state=RANDOM_SEED)

train = create_dataloader(df_train, tokenizer, MAX_LEN, batch_size=BATCH)
val = create_dataloader(df_val, tokenizer, MAX_LEN, batch_size=BATCH)
test = create_dataloader(df_test, tokenizer, MAX_LEN, batch_size=BATCH)

device = 'cuda' if cuda.is_available() else 'cpu'


class Recurrent(torch.nn.Module):

    def __init__(self, model, n_classes):
        super(Recurrent, self).__init__()

        if model == 'RNN':
            self.rnn = torch.nn.RNN(256, 100, bidirectional=True, num_layers=1)
        elif model == 'LSTM':
            self.rnn = torch.nn.LSTM(
                256, 100, bidirectional=True, num_layers=1)
        elif model == 'GRU':
            self.rnn = torch.nn.GRU(256, 100, bidirectional=True, num_layers=1)

        self.dropout = nn.Dropout(.3)
        self.fc = torch.nn.Linear(100 * 2, n_classes)

    def forward(self, ids):
        x = self.rnn(ids)[0]
        x = self.fc(self.dropout(x))
        return torch.sigmoid(x)


model = Recurrent(model=args.model, n_classes=2)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)

total_steps = len(train) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

def train_epoch(model, data_loader, optimizer, scheduler, device):
    model = model.train()

    for d in tqdm(data_loader):
        ids = d["input_ids"].to(device, dtype=torch.float)
        targets = d["targets"].to(device)

        outputs = model(ids).squeeze(1)

        loss = nn.CrossEntropyLoss()(outputs, targets)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

def eval_model(model, data_loader, device):
    model = model.eval()

    val_acc = []
    losses = []

    with torch.no_grad():
        for d in tqdm(data_loader):
            ids = d["input_ids"].to(device, dtype=torch.float)
            targets = d["targets"].to(device)

            outputs = model(ids).squeeze(1)
            _, preds = torch.max(outputs, dim=1)

            loss = nn.CrossEntropyLoss()(outputs, targets)
            losses.append(loss.item())

            acc = (preds == targets).cpu().numpy().mean()
            val_acc.append(acc)

    return np.mean(val_acc), np.mean(losses)

def metric(model, data_loader, device):
    model = model.eval()
    targets = []
    outputs = []
    with torch.no_grad():
        for d in tqdm(data_loader):
            ids = d["input_ids"].to(device, dtype=torch.float)
            target = d["targets"].to(device)

            output = model(ids).squeeze(1)
            _, preds = torch.max(output, dim=1)

            targets.extend(target.cpu().detach().numpy().tolist())
            outputs.extend(preds.cpu().detach().numpy().tolist())

    accuracy = metrics.accuracy_score(targets, outputs)
    f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
    f1_score_macro = metrics.f1_score(targets, outputs, average='macro')

    return accuracy, f1_score_macro, f1_score_micro


if __name__ == '__main__':

    print()
    print(args.model, '\n')
    print('Train Phase', '\n')

    start = time.time()

    val_losses = []

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_epoch(model, train, optimizer, scheduler, device)
        val_acc, val_loss = eval_model(model, val, device)

        val_losses.append(val_loss)

        # Early Stopping in patience 5
        if (epoch % 5 == 0) & (epoch != 0):
            if val_losses[epoch - 5] < val_losses[epoch]:
                print('\n early stopping')
                break

        print(f"Val Loss: {val_loss:^10.6f} | Val Acc: {val_acc:^9.2f}")
        print()

    sec = time.time() - start

    print(f"Train time : {sec}", '\n')

    print('Test Phase', '\n')

    test_acc, test_f1_macro, test_f1_micro = metric(model, test, device)

    print(f"Accuracy Score = {test_acc}")
    print(f"F1 Score (Micro) = {test_f1_micro}")
    print(f"F1 Score (Macro) = {test_f1_macro}")