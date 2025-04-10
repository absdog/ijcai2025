import json 
import numpy as np
from sklearn.metrics import classification_report
import copy

def generation2sentiment(sentimen2id, generation):
  sentiment = "neutral"
  for s in ["slightly positive"]:
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
            copy_fi_item["pred"] = "slightly positive"
        elif th_item["pred"] == "neutral" and  fi_item["pred"] == "positive":
            copy_fi_item["pred"] = "slightly positive"
        elif th_item["pred"] == "negative" and fi_item["pred"] == "neutral":
            copy_fi_item["pred"] == "slightly negative"
        elif th_item["pred"] == "neutral" and fi_item["pred"] == "negative":
            copy_fi_item["pred"] == "slightly negative"
        else:
            pass
        processed_five_star_res.append(copy_fi_item)
    return processed_five_star_res

zero_shot = json.load(open("./3-star/results/zero-shot-results.json", "r", encoding="utf-8"))
zero_rep = zero_shot["report"]
zero_res = zero_shot["results"]

five_star_res = json.load(open("./5-star/results/zero-shot-results.json", "r", encoding="utf-8"))["results"]
# del_five_star_res = json.load(open("./5-star-del/results/zero-shot-results.json", "r", encoding="utf-8"))["results"]


# 处理跳的
five_star_res = process_jump(zero_res, five_star_res)
# del_five_star_res = process_jump(zero_res, del_five_star_res)

print("zero-shot")
print(zero_rep)

mapping = {
    "positive": "positive",
    "neutral": "neutral",
    "negative": "negative",
    "slightly positive": None,
    "slightly negative": None
}

pos2other = [0, 0, 0]
neg2other = [0, 0, 0]
sentiment_list = ["positive", "neutral", "negative"]
sentiment2id = {s: i for i, s in enumerate(sentiment_list)}

for d in five_star_res:
    if d["pred"] == "slightly positive":
        if d["sentiment"] == "positive":
            pos2other[0] += 1
        elif d["sentiment"] == "neutral":
            pos2other[1] += 1
        else:
            pos2other[2] += 1
    elif d["pred"] == "slightly negative":
        if d["sentiment"] == "positive":
            neg2other[0] += 1
        elif d["sentiment"] == "neutral":
            neg2other[1] += 1
        else:
            neg2other[2] += 1

mapping["slightly positive"] = sentiment_list[np.argmax(pos2other)]
mapping["slightly negative"] = sentiment_list[np.argmax(neg2other)]

labels, preds = [], []
for d in five_star_res:
    labels.append(sentiment2id[d["sentiment"]])
    preds.append(sentiment2id[mapping[d["pred"]]])

five_rep = classification_report(labels, preds, digits=4)
print("five star:")
print(five_rep)

# # 如何删标签
# del_gls = {
#     "slightly positive": None,
#     "slightly negative": None
# }
# for gl in del_gls.keys():
#     should, not_should = 0, 0
#     preference_sentiment = mapping[gl]
#     for th_item, fi_item in zip(zero_res, five_star_res):
#         if fi_item["pred"] == gl and th_item["pred"] != preference_sentiment:
#             if th_item["pred"] == th_item["sentiment"]: # 不应该进来的样本
#                 not_should += 1
#             else:
#                 should += 1
#     if should > not_should:
#         del_gls[gl] = True 
#     else:
#         del_gls[gl] = False

# print("del:")
# print(del_gls)

# # del
# del_preds, del_labels = [], []
# err = 0
# for d in del_five_star_res:
#     try:
#         del_labels.append(sentiment2id[d["sentiment"]])
#         pred_sentiment = generation2sentiment(sentiment2id, d["generation"])
#         del_preds.append(sentiment2id[mapping[d["pred"]]])
#     except:
#         err += 1

# del_five_rep = classification_report(del_labels, del_preds, digits=4)
# print("del five star:")
# print(del_five_rep)
