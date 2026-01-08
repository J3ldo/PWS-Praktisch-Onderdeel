import os
import re
import random
import json
import time
from codecarbon import EmissionsTracker
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
import shutil

# Setup
nltk.download("punkt")
nltk.download("punkt_tab")

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tracker = EmissionsTracker(project_name="training_test")

# Geheugensteuntje
'''
dataset = {
    "scount": len(spam),
    "hcount": len(ham),
    "spam": spam,
    "ham": ham,

    "avg_spam": [
        {"avg": statistics.mean(words_spam), "median": statistics.median(words_spam)},
        {"avg": statistics.mean(len(i) for i in spam), "median": statistics.median([len(i) for i in spam])}
    ],  # index 0 is het aantal woorden, index 1 het aantal tekens
    "avg_ham": [
        {"avg": statistics.mean(words_ham), "median": statistics.median(words_ham)},
        {"avg": statistics.mean(len(i) for i in ham), "median": statistics.median([len(i) for i in ham])}
    ],

    "sword_count": [common_spam, common_suj_spam],
    "hword_count": [common_ham, common_suj_ham],
    "sminush_count": [common_sum, common_suj_sum],

}
'''
with open("data/all.json", "r") as f:
    data = json.load(f)

# Tekst verwerken voor de AI
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def tokenize_email(subject, body):
    text = clean_text(subject + " " + body)
    return word_tokenize(text)   # Punkt tokenizer


def build_vocab(tokenized_emails, min_freq=2):
    counter = Counter()
    for tokens in tokenized_emails:
        counter.update(tokens)

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)

    return vocab

def encode(tokens, vocab):
    return [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens]


#Dataset voor training instellen
class SpamDataset(Dataset):
    def __init__(self, tokenized_emails, labels, vocab, max_len=300):
        self.emails = tokenized_emails
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.emails)

    def __getitem__(self, idx):
        tokens = self.emails[idx]
        encoded = encode(tokens, self.vocab)

        # Padden
        encoded = encoded[:self.max_len]
        encoded += [0] * (self.max_len - len(encoded))

        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float)
        )


# Neurale network model (LSTM)
class SpamClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        logits = self.fc(hidden[-1])
        return self.sigmoid(logits)


# Training
def train(model, dataloader, epochs=5, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0

        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            preds = model(x).squeeze()
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")


# Voorspellen van spam of ham
def predict_spam(model, subject, body, vocab, max_len=300, threshold=0.5):
    model.eval()
    tokens = tokenize_email(subject, body)
    encoded = encode(tokens, vocab)
    encoded = encoded[:max_len]
    encoded += [0] * (max_len - len(encoded))

    x = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob = model(x).item()

    return prob, prob >= threshold


if __name__ == "__main__":
    with open("data/vocab.json", "r") as f:
        all_info = json.load(f)
    vocab = all_info['vocab']
    tokenized_emails = all_info['tokenized_emails']
    labels = all_info['labels']


    # Maakt het vocabulair voor het netwerk aan.
    tracker.start()

    # Dataset & DataLoader
    dataset = SpamDataset(tokenized_emails, labels, vocab)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Model
    model = SpamClassifier(len(vocab)).to(DEVICE)

    # Begin met het trainen van het model
    print("Training gestart.")
    t1 = time.time()
    train(model, dataloader, epochs=1)

    t2 = time.time()
    print(f"Training voltooid. Dt: {round(t2-t1, 4)}")
    with open("results/dt.txt", "a" if os.path.exists("./results/dt.txt") else "w") as f:
        f.write(f"Dt Training\n"
                f"Dt: {t2-t1}\n\n")
    tracker.stop()
    print(tracker.final_emissions_data.energy_consumed*1000)
    while not os.path.exists("./emissions.csv"): time.sleep(0.5)
    os.rename("./emissions.csv", "emissions_AITRAINING.csv")
    shutil.move("./emissions_AITRAINING.csv", "./results/emissions_AITRAINING.csv")

    #torch.save({
    #    "model": model.state_dict(),
    #    "vocab": vocab
    #}, "spamdetector.pt")

    '''
    # Laad het getrainde model en stel alle waardes in.
    state_dict = torch.load("spamdetector.pt")
    vocab = state_dict['vocab']
    model = SpamClassifier(len(vocab)).to(DEVICE)
    model.load_state_dict(state_dict['model'])
    model.eval()
    '''


    # Test prediction
    '''false_negatives = 0
    for mail in data['spam']:
        subject = mail[0]
        body = mail[1]

        prob, is_spam = predict_spam(model, subject, body, vocab)
        if not is_spam: false_negatives += 1

        #print("\nPrediction:")
        print("Spam probability:", round(prob, 4))
        #print("Is spam:", is_spam)
    print(f"{false_negatives}/{len(data['spam'])}\n"
          f"{false_negatives/len(data['spam'])*100}%")'''
