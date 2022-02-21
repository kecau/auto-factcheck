import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import time
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import AdamW, BertForSequenceClassification, get_linear_schedule_with_warmup, BertTokenizer

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

parser = argparse.ArgumentParser()
parser.add_argument('--max_len', type=int, default=256)
parser.add_argument('--batch', type=int, default=16)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--test_size', type=float, default=.2)
args = parser.parse_args()

transformers.logging.set_verbosity_error()

MAX_LEN = args.max_len
BATCH = args.batch
EPOCHS = args.epoch
TEST_SIZE = args.test_size
LEARNING_RATE = 1e-05

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, maxlen):
        self.data = data
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sent1 = str(self.data['text'].iloc[index]).lower()
        sent2 = str(self.data['text_source'].iloc[index]).lower()
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
            'input_ids': encoding['input_ids'].squeeze(1),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target)
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


df = pd.read_csv("D:/Dropbox/coding/auto-factcheck/dataset/tweets.csv")
df_train, df_test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_SEED)
df_val, df_test = train_test_split(df_test, test_size=.5, random_state=RANDOM_SEED)

train = create_dataloader(df_train, tokenizer, MAX_LEN, batch_size=BATCH)
val = create_dataloader(df_val, tokenizer, MAX_LEN, batch_size=BATCH)
test = create_dataloader(df_test, tokenizer, MAX_LEN, batch_size=BATCH)



model = model.to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)

total_steps = len(train) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

def train_epoch(model, data_loader, device, scheduler, optimizer):
    model = model.train()

    for d in tqdm(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        model.zero_grad()
        output = model(
            input_ids=input_ids.squeeze(1),
            labels=targets
        )
        loss = output.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()


def eval_model(model, data_loader, device):
    model = model.eval()

    val_acc = []
    losses = []

    with torch.no_grad():
        for d in tqdm(data_loader):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
                input_ids=input_ids.squeeze(1),
                labels=targets
            )
            _, preds = torch.max(outputs.logits, dim=1)

            loss = outputs.loss
            losses.append(loss.item())

            acc = (preds == targets).cpu().numpy().mean()
            val_acc.append(acc)

    return np.mean(val_acc), np.mean(losses)


def metric(model, data_loader, device):
    model.eval()
    targets = []
    outputs = []
    with torch.no_grad():
        for d in tqdm(data_loader):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            target = d["targets"].to(device)

            output = model(
                input_ids=input_ids.squeeze(1),
                labels=target
            )
            _, preds = torch.max(output.logits, dim=1)

            targets.extend(target.cpu().detach().numpy().tolist())
            outputs.extend(preds.cpu().detach().numpy().tolist())

    accuracy = metrics.accuracy_score(targets, outputs)
    f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
    f1_score_macro = metrics.f1_score(targets, outputs, average='macro')

    return accuracy, f1_score_macro, f1_score_micro


if __name__ == '__main__':
    print()
    print('Train Phase', '\n')

    start = time.time()

    val_losses = []

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_epoch(model, train, device, scheduler, optimizer)
        val_acc, val_loss = eval_model(model, val, device)

        val_losses.append(val_loss)

        # Early Stopping in patience 5
        if (epoch % 5 == 0) & (epoch > 5):
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