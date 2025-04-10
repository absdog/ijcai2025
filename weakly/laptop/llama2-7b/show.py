import json 
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score
import copy
import random

print("llama2-7b, laptop 14")

def generation2sentiment(sentimen2id, generation):
  sentiment = "neutral"
  for s in ["weakly positive"]:
    if s in generation:
      sentiment = s 
  if sentiment != "neutral":
    return sentiment
  else:
    words = generation.split()
    flag = False
    for w in words:
      for key in sentiment2id.keys():
        if key in w.lower(): # "weak positive" positive
          sentiment = key
          flag = True
          break
      if flag == True:
        break
    if flag == False:
      sentiment = "neutral"
    return sentiment

def process_jump(zero_res, five_star_res):
    processed_five_star_res = []
    for th_item, fi_item in zip(zero_res, five_star_res):
        copy_fi_item = copy.deepcopy(fi_item)
        if th_item["pred"] == "positive" and fi_item["pred"] == "neutral":
            copy_fi_item["pred"] = "weakly positive"
        elif th_item["pred"] == "neutral" and  fi_item["pred"] == "positive":
            copy_fi_item["pred"] = "weakly positive"
        elif th_item["pred"] == "negative" and fi_item["pred"] == "neutral":
            copy_fi_item["pred"] == "weakly negative"
        elif th_item["pred"] == "neutral" and fi_item["pred"] == "negative":
            copy_fi_item["pred"] == "weakly negative"
        else:
            pass
        processed_five_star_res.append(copy_fi_item)
    return processed_five_star_res

zero_shot = json.load(open("./3-star/results/zero-shot-results.json", "r", encoding="utf-8"))
zero_rep = zero_shot["report"]
zero_res = zero_shot["results"]

five_star_res = json.load(open("./5-star/results/zero-shot-results.json", "r", encoding="utf-8"))["results"]
del_five_star_res = json.load(open("./5-star-del/results/zero-shot-results.json", "r", encoding="utf-8"))["results"]


# 处理跳的

five_star_res = process_jump(zero_res, five_star_res)
del_five_star_res = process_jump(zero_res, del_five_star_res)

# print("zero-shot")
# print(zero_rep)


# ---------------- count-based -------------------
mapping = {
    "positive": "positive",
    "neutral": "neutral",
    "negative": "negative",
    "weakly positive": None,
    "weakly negative": None
}

pos2other = [0, 0, 0]
neg2other = [0, 0, 0]
sentiment_list = ["positive", "neutral", "negative"]
sentiment2id = {s: i for i, s in enumerate(sentiment_list)}

for d in five_star_res:
    if d["pred"] == "weakly positive":
        if d["sentiment"] == "positive":
            pos2other[0] += 1
        elif d["sentiment"] == "neutral":
            pos2other[1] += 1
        else:
            pos2other[2] += 1
    elif d["pred"] == "weakly negative":
        if d["sentiment"] == "positive":
            neg2other[0] += 1
        elif d["sentiment"] == "neutral":
            neg2other[1] += 1
        else:
            neg2other[2] += 1

mapping["weakly positive"] = sentiment_list[np.argmax(pos2other)]
mapping["weakly negative"] = sentiment_list[np.argmax(neg2other)]

labels, preds = [], []
for d in five_star_res:
    labels.append(sentiment2id[d["sentiment"]])
    preds.append(sentiment2id[mapping[d["pred"]]])

# five_rep = classification_report(labels, preds, digits=4)
five_f1 = round(f1_score(labels, preds, average="weighted"), 4) * 100
five_acc = round(accuracy_score(labels, preds), 4) * 100
print("count-based:")
print("acc:", five_acc)
print("f1:", five_f1)

# ---------------- random-based -------------------
def random_mapping():
    mapping = {
        "positive": "positive",
        "neutral": "neutral",
        "negative": "negative",
        "weakly positive": "positive" if random.random() < 0.5 else "neutral",
        "weakly negative": "negative" if random.random() < 0.5 else "neutral"
    }
    labels, preds = [], []
    for d in five_star_res:
        labels.append(sentiment2id[d["sentiment"]])
        preds.append(sentiment2id[mapping[d["pred"]]])

    five_f1 = round(f1_score(labels, preds, average="weighted"), 4) * 100
    five_acc = round(accuracy_score(labels, preds), 4) * 100
    return five_acc, five_f1
random_five_acc, random_five_f1 = [], []
for i in range(10):
    acc, f1 = random_mapping()
    random_five_acc.append(acc)
    random_five_f1.append(f1)
mean_five_acc = round(np.array(random_five_acc).mean(), 2)
std_five_acc = round(np.array(random_five_acc).std(), 2)
mean_five_f1 = round(np.array(random_five_f1).mean(), 2)
std_five_f1 = round(np.array(random_five_f1).std(), 2)
print("random-based:")
print(f"acc: {mean_five_acc}(+/- {std_five_acc})")
print(f"f1: {mean_five_f1}(+/- {std_five_f1})")

# ---------------- prob-based -------------------

# ---------------- random-based -------------------
def random_mapping():
    mapping = {
        "positive": "positive",
        "neutral": "neutral",
        "negative": "negative",
        "weakly positive": "positive" if random.random() < pos2other[0]/(pos2other[0]+pos2other[1]) else "neutral",
        "weakly negative": "negative" if random.random() < neg2other[2]/(neg2other[1]+neg2other[2]) else "neutral",
    }
    labels, preds = [], []
    for d in five_star_res:
        labels.append(sentiment2id[d["sentiment"]])
        preds.append(sentiment2id[mapping[d["pred"]]])

    five_f1 = round(f1_score(labels, preds, average="weighted"), 4) * 100
    five_acc = round(accuracy_score(labels, preds), 4) * 100
    return five_acc, five_f1
random_five_acc, random_five_f1 = [], []
for i in range(10):
    acc, f1 = random_mapping()
    random_five_acc.append(acc)
    random_five_f1.append(f1)
mean_five_acc = round(np.array(random_five_acc).mean(), 2)
std_five_acc = round(np.array(random_five_acc).std(), 2)
mean_five_f1 = round(np.array(random_five_f1).mean(), 2)
std_five_f1 = round(np.array(random_five_f1).std(), 2)
print("prob-based:")
print(f"acc: {mean_five_acc}(+/- {std_five_acc})")
print(f"f1: {mean_five_f1}(+/- {std_five_f1})")

# mapping = {
#     "positive": "positive",
#     "neutral": "neutral",
#     "negative": "negative",
#     "weakly positive": "positive" if random.random() < pos2other[0]/(pos2other[0]+pos2other[1]) else "neutral",
#     "weakly negative": "negative" if random.random() < neg2other[2]/(neg2other[1]+neg2other[2]) else "neutral",
# }
# labels, preds = [], []
# for d in five_star_res:
#     labels.append(sentiment2id[d["sentiment"]])
#     preds.append(sentiment2id[mapping[d["pred"]]])

# five_rep = classification_report(labels, preds, digits=4)
# print("prob-based five star:")
# print(five_rep)

