import os
import time
from collections import Counter
from codecarbon import EmissionsTracker
import math
import json
import random
import re
import shutil

SPAM_MARGIN = 0.55

tracker = EmissionsTracker(project_name="naive_bayes_classifier")
tracker.start()

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

def calculate_prob(word_freq, total_words):
    word_probs = {}
    for word, freq in word_freq.items():
        word_probs[word] = (freq + 1) / (total_words + len(word_freq))
    return word_probs

def calculate_email_prob(email, spam_probs, ham_probs, spam_word_freq, ham_word_freq, spam_count, ham_count):
    spam_likelihood = math.log(spam_count / (spam_count + ham_count))
    ham_likelihood = math.log(ham_count / (spam_count + ham_count))

    mail_words = [i.lower() for i in re.findall(r'\b\S+\b', email)]

    ham_freq_sum = sum(ham_word_freq.values())
    spam_freq_sum = sum(spam_word_freq.values())
    for word in mail_words:
        if word in spam_probs:
            spam_likelihood += math.log(spam_probs[word])
        else:
            spam_likelihood += math.log(1 / (spam_freq_sum + len(spam_word_freq)))

        if word in ham_probs:
            ham_likelihood += math.log(ham_probs[word])
        else:
            ham_likelihood += math.log(1 / (ham_freq_sum + len(ham_word_freq)))

    if len(mail_words) != 0:
        spam_likelihood /= len(mail_words)
        ham_likelihood /= len(mail_words)
    #print(spam_likelihood-ham_likelihood)
    return spam_likelihood > ham_likelihood + SPAM_MARGIN


spam_probs = calculate_prob(data["sword_count"][0], sum(data["sword_count"][0].values(), data["scount"]))
ham_probs = calculate_prob(data["hword_count"][0], sum(data["hword_count"][0].values(), data["hcount"]))

def test_run():
    false_positives = 0
    false_negatives = 0
    positive_list = []
    negative_list = []

    for mail in data['ham']:
        if calculate_email_prob(mail[1], spam_probs, ham_probs, data['sword_count'][0], data["hword_count"][0], data["scount"], data["hcount"]):
            false_positives += 1
            positive_list.extend([i.lower() for i in re.findall(r'\b\S+\b', mail[1])])
    for mail in data['spam']:
        if not calculate_email_prob(mail[1], spam_probs, ham_probs, data['sword_count'][0], data["hword_count"][0], data["scount"], data["hcount"]):
            false_negatives += 1
            negative_list.extend([i.lower() for i in re.findall(r'\b\S+\b', mail[1])])
    return false_positives, false_negatives, positive_list, negative_list

def validation_run():
    false_positives = 0
    false_negatives = 0
    positive_list = []
    negative_list = []

    for mail in validation['ham']:
        if calculate_email_prob(mail[1], spam_probs, ham_probs, data['sword_count'][0], data["hword_count"][0], data["scount"], data["hcount"]):
            false_positives += 1
            positive_list.extend([i.lower() for i in re.findall(r'\b\S+\b', mail[1])])
    for mail in validation['spam']:
        if not calculate_email_prob(mail[1], spam_probs, ham_probs, data['sword_count'][0], data["hword_count"][0], data["scount"], data["hcount"]):
            false_negatives += 1
            negative_list.extend([i.lower() for i in re.findall(r'\b\S+\b', mail[1])])
    return false_positives, false_negatives, positive_list, negative_list

if __name__ == '__main__':
    t1 = time.time()

    false_positives, false_negatives, positive_list, negative_list = test_run()
    print(f"Total ham: {data['hcount']} - Total spam: {data['scount']}\n"
          f"False positives: {false_positives}/{data['hcount']} - {false_positives/data['hcount']*100}\n"
          f"False negatives: {false_negatives}/{data['scount']} - {false_negatives/data['scount']*100}")

    false_positives, false_negatives, positive_list, negative_list = validation_run()
    print(f"Total ham: {len(validation['spam'])} - Total spam: {len(validation['ham'])}\n"
          f"False positives: {false_positives}/{len(validation['ham'])} - {false_positives/len(validation['ham'])*100}\n"
          f"False negatives: {false_negatives}/{len(validation['spam'])} - {false_negatives/len(validation['spam'])*100}")

    tracker.stop()
    t2 = time.time()

    print(f"Klaar. Dt: {round(t2-t1, 4)}")
    with open("./results/dt.txt", "a" if os.path.exists("./results/dt.txt") else "w") as f:
        f.write(f"Dt NAIVE BAYES (MET VALDIDATION)\n"
                f"Dt: {t2-t1}\n\n")
    while not os.path.exists("./emissions.csv"): time.sleep(0.5)
    os.rename("./emissions.csv", "./emissions_naive.csv")
    shutil.move("./emissions_naive.csv", "./results/emissions_naive.csv")

