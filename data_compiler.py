from codecarbon import EmissionsTracker
import csv
import re
from collections import Counter
import json
import statistics

tracker = EmissionsTracker()
tracker.start()
csv.field_size_limit(2147483647)


## Notatie binnen csv:
# Enron (29767): onderwerp - inhoud - spam (0 is nee, 1 is ja)
# Assasin (5809): afzender - ontvanger - datum - onderwerp - inhoud - spam (0 is nee, 1 is ja) - bevat urls (0-1)
# Ling (2859): onderwerp - inhoud - spam (0 is nee, 1 is ja)
# TREC (2054): afzender - ontvanger - datum - onderwerp - inhoud - spam (0 is nee, 1 is ja) - bevat urls (0-1)
# enron data: ID - onderwerp - inhoud - Spam of ham - datum
#
# Validation (2000): inhoud, spam (0 is nee, 1 is ja)
spam = []
ham = []

spam_words = []
spam_suj_words = []
ham_words = []
ham_suj_words = []

words_spam = []
words_ham= []

def compile_training():
    print("Bestanden uitlezen")
    with open("training/Ling.csv", "r", encoding="utf8") as f:
        data = csv.reader(f, delimiter=",")
        for enum, i in enumerate(data):
            if i[2] == '1': spam.append(i)
            if i[2] == '0': ham.append(i)

    with open("training/Assassin.csv", "r", encoding="utf8") as f:
        data = csv.reader(f, delimiter=",")
        for enum, i in enumerate(data):
            if i[5] == '1': spam.append([i[3], i[4], i[5]])
            if i[5] == '0': ham.append([i[3], i[4], i[5]])

    with open("training/TREC-06.csv", "r", encoding="utf8") as f:
        data = csv.reader(f, delimiter=",")
        for enum, i in enumerate(data):
            if len(i) <= 5: continue
            if i[5] == '1': spam.append([i[3], i[4], i[5]])
            if i[5] == '0': ham.append([i[3], i[4], i[5]])

    print("Maildata verzamelen")
    for i in spam:
        x = [i.lower() for i in re.findall(r'\b\S+\b', i[1])]
        y = [i.lower() for i in re.findall(r'\b\S+\b', i[0])]
        spam_words.extend(x)
        spam_suj_words.extend(y)
        words_spam.append(len(x)+len(y))

    for i in ham:
        x = [i.lower() for i in re.findall(r'\b\S+\b', i[1])]
        y = [i.lower() for i in re.findall(r'\b\S+\b', i[0])]
        ham_words.extend(x)
        ham_suj_words.extend(y)
        words_ham.append(len(x)+len(y))

    spam_ratio = len(spam)/len(ham)  # Voor het onderwerp gaan wij er van uit dat alle mails even lang zijn.
    spam_word_ratio = sum(words_spam)/sum(words_ham)  # Voor de inhoud kunnen spam mails korter of langer zijn dan ham mails, dus vergelijken wij de woorden.

    print("Woordfrequentie berekenen")
    c_spam = Counter(spam_words)
    c_ham = Counter(ham_words)
    c_suj_spam = Counter(spam_suj_words)
    c_suj_ham = Counter(ham_suj_words)

    common_spam = {i[0]: i[1] for i in c_spam.most_common()}
    common_ham = {i[0]: i[1] for i in c_ham.most_common()}
    common_suj_spam = {i[0]: i[1] for i in c_suj_spam.most_common()}
    common_suj_ham = {i[0]: i[1] for i in c_suj_ham.most_common()}

    common_sum = {}
    common_suj_sum = {}
    for key in common_spam:
        common_sum[key] = common_spam[key] - round(common_ham.get(key, 0)*spam_word_ratio)
    for key in common_ham:
        common_sum[key] = common_spam.get(key, 0) - round(common_ham.get(key, 0)*spam_word_ratio)


    for key in common_suj_spam:
        common_suj_sum[key] = common_suj_spam[key] - round(common_suj_ham.get(key, 0)*spam_ratio)
    for key in common_suj_ham:
        common_suj_sum[key] = common_suj_spam.get(key, 0) - round(common_suj_ham.get(key, 0)*spam_ratio)



    common_sum = dict(sorted(common_sum.items(), key=lambda item: item[1], reverse=True))
    common_suj_sum = dict(sorted(common_suj_sum.items(), key=lambda item: item[1], reverse=True))

    print("Data naar JSON verwerken")
    dataset = {
        "scount": len(spam),
        "hcount": len(ham),
        "spam": spam,
        "ham": ham,

        "avg_spam": [
            {"avg": statistics.mean(words_spam), "median": statistics.median(words_spam)},
            {"avg": statistics.mean(
                [len(re.findall(r'\b\S+\b', i[0])) for i in spam]
            ), "median": statistics.median(
                [len(re.findall(r'\b\S+\b', i[0])) for i in spam]
            )}
        ],  # index 0 is het aantal woorden, index 1 het aantal tekens
        "avg_ham": [
            {"avg": statistics.mean(words_ham), "median": statistics.median(words_ham)},
            {"avg": statistics.mean(
                [len(re.findall(r'\b\S+\b', i[0])) for i in ham]
            ), "median": statistics.median(
                [len(re.findall(r'\b\S+\b', i[0])) for i in ham]
            )}
        ],

        "sword_count": [common_spam, common_suj_spam],
        "hword_count": [common_ham, common_suj_ham],
        "sminush_count": [common_sum, common_suj_sum],

    }
    with open("data/all.json", "w") as f:
        #dataset = json.load(f)
        json.dump(dataset, f, indent=4)

    dataset.pop("spam")
    dataset.pop("ham")
    dataset.pop("sword_count")
    dataset.pop("hword_count")
    with open("data/all_lite.json", "w") as f:
        json.dump(dataset, f, indent=4)

def compile_validation():
    print("Begonnen met compileren validation data")
    validation = {
        "spam": [],
        "ham": []
    }
    with open("validation/Phishing_validation_emails.csv", "r") as f:
        data = csv.reader(f, delimiter=",")
        for enum, i in enumerate(data):
            if enum == 0:
                continue
            if i[1] == "Phishing Email":
                validation['spam'].append(["", i[0], 1])
            else:
                validation['ham'].append(["", i[0], 0])
    with open("training/enron_spam_data.csv", "r", encoding='utf8') as f:
        # Enron (29767): onderwerp - inhoud - spam (0 is nee, 1 is ja)
        data = csv.reader(f, delimiter=",")
        for enum, i in enumerate(data):
            if enum == 0:
                continue
            if i[3] == 'spam':
                validation['spam'].append([i[1], i[2], 1])
            else:
                validation['ham'].append([i[1], i[2], 0])

    with open("data/validation.json", "w") as f:
        json.dump(validation, f)

    print("Klaar met compileren validation data")

if __name__ == '__main__':
    print("Begonnen met compileren")
    compile_training()
    compile_validation()
    print("Klaar met compileren")

    tracker.stop()
    print(tracker.final_emissions_data)
    print(tracker.final_emissions_data.energy_consumed*1000, "Wh")
