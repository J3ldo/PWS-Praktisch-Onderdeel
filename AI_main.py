# spam_detector.py
import os.path
import re
import random
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from codecarbon import EmissionsTracker
import shutil


# Setup
nltk.download("punkt")
nltk.download("punkt_tab")

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN = False

tracker = EmissionsTracker(project_name="AI_model")
tracker.start()
tracker.start_task("setup")

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

with open("data/validation.json", "r") as f:
    validation = json.load(f)

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

        # Pad / truncate
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
            logits = model(x).squeeze()
            loss = criterion(logits, y)

            # Na het trainen van een model zonder aangepaste waarden bleek dat ham vaak false positives had (90+%) dus reken ik een foute ham keuze harder aan.
            # Voor aangepaste loss: false negatives: 9.1%, false positives: 90.4%.
            # Na aangepaste loss: 0.4% false negatives, 0.0% false positives.
            loss = torch.where(
                y == 0,
                loss * 5.0,
                loss * 1.0
            ).mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")


# Voorspellen spam of ham
def predict_spam(model, subject, body, vocab, max_len=300, threshold=0.9):
    model.eval()
    tokens = tokenize_email(subject, body)
    encoded = encode(tokens, vocab)
    encoded = encoded[:max_len]
    encoded += [0] * (max_len - len(encoded))

    x = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob = model(x).item()

    return prob, prob >= threshold

def test(spam, ham, model, vocab):
    false_negatives = 0
    false_positives = 0
    print("Testen gestart.")
    print("Spam begonnen.")
    for mail in spam:
        subject = mail[0]
        body = mail[1]

        prob, is_spam = predict_spam(model, subject, body, vocab)
        if not is_spam: false_negatives += 1

        #print("Spam probability:", round(prob, 4))
    print("Ham begonnen.")
    for mail in ham:
        subject = mail[0]
        body = mail[1]

        prob, is_spam = predict_spam(model, subject, body, vocab)
        if round(prob): false_positives += 1

        #print("Ham probability:", round(1 - prob, 4))
    print("Testen afgerond.")
    return false_negatives, false_positives


if __name__ == "__main__":
    if os.path.exists("data/vocab.json"):
        with open("data/vocab.json", "r") as f:
            all_info = json.load(f)
        vocab = all_info['vocab']
        tokenized_emails = all_info['tokenized_emails']
        labels = all_info['labels']
    else:
        all_mails = data['spam']+data['ham']
        indecies = list(range(len(all_mails)))

        # Hussel de data zodat de training objectief blijft.
        zipped1 = list(zip(all_mails, indecies))
        random.shuffle(zipped1)
        all_mails, indecies = zip(*zipped1)

        subjects, bodies, labels = [], [], []
        for mail in all_mails:
            subjects.append(mail[0])
            bodies.append(mail[1])
            labels.append(int(mail[2]))

        # Tokenize de emails
        tokenized_emails = [
            tokenize_email(subj, body)
            for subj, body in zip(subjects, bodies)
        ]

        # Maakt het vocabulair voor het netwerk aan.
        vocab = build_vocab(tokenized_emails)
        with open("data/vocab.json", "w") as f:
            json.dump({"vocab": vocab,
                       "tokenized_emails": tokenized_emails,
                       "labels": labels
                       }, f)
    # Dataset & DataLoader
    dataset = SpamDataset(tokenized_emails, labels, vocab)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Model  
    model = SpamClassifier(len(vocab)).to(DEVICE)

    # Begin met het trainen van het model
    if TRAIN:
        t1 = time.time()
        tracker.stop_task("setup")
        tracker.start_task("training")
        train(model, dataloader, epochs=10)
        try: model.compile()
        except: pass
        torch.save({
            "model": model.state_dict(),
            "vocab": vocab
        }, "data/spamdetector.pt")
        tracker.stop_task("training")
        t2 = time.time()
        print(f"AI DT: {t2-t1}")
    else:
        model.load_state_dict(torch.load("data/spamdetector.pt")['model'])
        tracker.stop_task("setup")
    tracker.start_task("testing")

    # Test prediction
    dt1 = time.time()
    dt1_1 = time.time()
    false_negatives, false_positives = test(data['spam'], data['ham'], model, vocab)
    dt1_2 = time.time()
    tracker.stop_task("testing")
    print(f"{false_negatives}/{len(data['spam'])}\n"
          f"{false_negatives/len(data['spam'])*100}%")
    print(f"{false_positives}/{len(data['ham'])}\n"
          f"{false_positives/len(data['ham'])*100}%")

    tracker.start_task("validation")
    print("Validation gestart.")

    dt2_1 = time.time()
    false_negatives, false_positives = test(validation['spam'], validation['ham'], model, vocab)
    dt2_2 = time.time()
    dt2 = time.time()
    print(f"{false_negatives}/{len(data['spam'])}\n"
          f"{false_negatives/len(data['spam'])*100}%")
    print(f"{false_positives}/{len(data['ham'])}\n"
          f"{false_positives/len(data['ham'])*100}%")
    print(f"\nDt1: {dt1_2-dt1_1}\n"
          f"Dt2: {dt2_2-dt2_1}\n"
          f"Dt totaal: {dt2-dt1}\n")
    with open("results/dt.txt", "a" if os.path.exists("./results/dt.txt") else "w") as f:
        f.write(f"Dt AI MAIN\n"
                f"Dt1: {dt1_2-dt1_1}\n"
                f"Dt2: {dt2_2-dt2_1}\n"
                f"Dt totaal: {dt2-dt1}\n\n")

    tracker.stop_task("validation")
    tracker.stop()
    while not os.path.exists("./emissions.csv"): time.sleep(0.5)
    os.rename("./emissions.csv", "emissions_AIMAIN.csv")
    shutil.move("./emissions_AIMAIN.csv", "./results/emissions_AIMAIN.csv")

