import json 
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score
import copy
import random
import torch

def set_seed(seed): # 设置随机种子
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  torch.backends.cudnn.deterministic = True

print("blip2, twitter-2015")

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
            # import pdb; pdb.set_trace()
            copy_fi_item["pred"] = "weakly positive"
        elif th_item["pred"] == "neutral" and  fi_item["pred"] == "positive":
            # import pdb; pdb.set_trace()
            copy_fi_item["pred"] = "weakly positive"
        elif th_item["pred"] == "negative" and fi_item["pred"] == "neutral":
            copy_fi_item["pred"] = "weakly negative"
        elif th_item["pred"] == "neutral" and fi_item["pred"] == "negative":
            copy_fi_item["pred"] = "weakly negative"
        else:
            pass
        processed_five_star_res.append(copy_fi_item)
    return processed_five_star_res

set_seed(202501)

zero_shot = json.load(open("./3-star/results/zero-shot-results.json", "r", encoding="utf-8"))
zero_rep = zero_shot["report"]
zero_res = zero_shot["results"]

five_star_res = json.load(open("./5-star/results/zero-shot-results.json", "r", encoding="utf-8"))["results"]
# del_five_star_res = json.load(open("./5-star-del/results/zero-shot-results.json", "r", encoding="utf-8"))["results"]


few_five_star_res = []
few_zero_res = []
for zero_item, five_item in zip(zero_res, five_star_res):
    if random.random() <= 0.1:
        few_zero_res.append(zero_item)
        few_five_star_res.append(five_item)

# 处理跳的
five_star_res = process_jump(few_zero_res, few_five_star_res)
# del_five_star_res = process_jump(zero_res, del_five_star_res)

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

for idx, d in enumerate(five_star_res):
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

# # ---------------- random-based -------------------
# def random_mapping():
#     mapping = {
#         "positive": "positive",
#         "neutral": "neutral",
#         "negative": "negative",
#         "weakly positive": "positive" if random.random() < 0.5 else "neutral",
#         "weakly negative": "negative" if random.random() < 0.5 else "neutral"
#     }
#     labels, preds = [], []
#     for d in five_star_res:
#         labels.append(sentiment2id[d["sentiment"]])
#         preds.append(sentiment2id[mapping[d["pred"]]])

#     five_f1 = round(f1_score(labels, preds, average="weighted"), 4) * 100
#     five_acc = round(accuracy_score(labels, preds), 4) * 100
#     return five_acc, five_f1
# random_five_acc, random_five_f1 = [], []
# for i in range(10):
#     acc, f1 = random_mapping()
#     random_five_acc.append(acc)
#     random_five_f1.append(f1)
# mean_five_acc = round(np.array(random_five_acc).mean(), 2)
# std_five_acc = round(np.array(random_five_acc).std(), 2)
# mean_five_f1 = round(np.array(random_five_f1).mean(), 2)
# std_five_f1 = round(np.array(random_five_f1).std(), 2)
# print("random-based:")
# print(f"acc: {mean_five_acc}(+/- {std_five_acc})")
# print(f"f1: {mean_five_f1}(+/- {std_five_f1})")

# # ---------------- prob-based -------------------

# # ---------------- random-based -------------------
# def random_mapping():
#     mapping = {
#         "positive": "positive",
#         "neutral": "neutral",
#         "negative": "negative",
#         "weakly positive": "positive" if random.random() < pos2other[0]/(pos2other[0]+pos2other[1]) else "neutral",
#         "weakly negative": "negative" if random.random() < neg2other[2]/(neg2other[1]+neg2other[2]) else "neutral",
#     }
#     labels, preds = [], []
#     for d in five_star_res:
#         labels.append(sentiment2id[d["sentiment"]])
#         preds.append(sentiment2id[mapping[d["pred"]]])

#     five_f1 = round(f1_score(labels, preds, average="weighted"), 4) * 100
#     five_acc = round(accuracy_score(labels, preds), 4) * 100
#     return five_acc, five_f1
# random_five_acc, random_five_f1 = [], []
# for i in range(10):
#     acc, f1 = random_mapping()
#     random_five_acc.append(acc)
#     random_five_f1.append(f1)
# mean_five_acc = round(np.array(random_five_acc).mean(), 2)
# std_five_acc = round(np.array(random_five_acc).std(), 2)
# mean_five_f1 = round(np.array(random_five_f1).mean(), 2)
# std_five_f1 = round(np.array(random_five_f1).std(), 2)
# print("prob-based:")
# print(f"acc: {mean_five_acc}(+/- {std_five_acc})")
# print(f"f1: {mean_five_f1}(+/- {std_five_f1})")

# # mapping = {
# #     "positive": "positive",
# #     "neutral": "neutral",
# #     "negative": "negative",
# #     "weakly positive": "positive" if random.random() < pos2other[0]/(pos2other[0]+pos2other[1]) else "neutral",
# #     "weakly negative": "negative" if random.random() < neg2other[2]/(neg2other[1]+neg2other[2]) else "neutral",
# # }
# # labels, preds = [], []
# # for d in five_star_res:
# #     labels.append(sentiment2id[d["sentiment"]])
# #     preds.append(sentiment2id[mapping[d["pred"]]])

# # five_rep = classification_report(labels, preds, digits=4)
# # print("prob-based five star:")
# # print(five_rep)


# clear_ids, amb_ids = [], []
# for idx, d in enumerate(five_star_res):
#     if d["pred"] == "weakly positive" or d["pred"] == "weakly negative":
#         amb_ids.append(idx)
#     else:
#         clear_ids.append(idx)

# # zero-shot clear acc
# labels, preds = [], []
# for idx in clear_ids:
#     labels.append(sentiment2id[zero_res[idx]["sentiment"]])
#     preds.append(sentiment2id[mapping[zero_res[idx]["pred"]]])
# zc_acc = round(accuracy_score(labels, preds)*100, 2)
# print("zero-shot clear acc:", zc_acc)
# # zero-shot amb acc
# labels, preds = [], []
# for idx in amb_ids:
#     labels.append(sentiment2id[zero_res[idx]["sentiment"]])
#     preds.append(sentiment2id[mapping[zero_res[idx]["pred"]]])
# za_acc = round(accuracy_score(labels, preds)*100, 2)
# print("zero-shot amb acc:", za_acc)
# # fine-grained clear acc
# labels, preds = [], []
# for idx in clear_ids:
#     labels.append(sentiment2id[five_star_res[idx]["sentiment"]])
#     preds.append(sentiment2id[mapping[five_star_res[idx]["pred"]]])
# fc_acc = round(accuracy_score(labels, preds)*100, 2)
# print("fine-grained clear acc:", fc_acc)
# # fine-grained amb acc
# labels, preds = [], []
# for idx in amb_ids:
#     labels.append(sentiment2id[five_star_res[idx]["sentiment"]])
#     preds.append(sentiment2id[mapping[five_star_res[idx]["pred"]]])
# fa_acc = round(accuracy_score(labels, preds)*100, 2)
# print("fine-grained clear acc:", fa_acc)


# import matplotlib.pyplot as plt
# import numpy as np

# # 数据
# x_labels = ['Zero-Shot', 'Ours']  # x轴标签
# groups = ['Non-Amb Set', 'Amb Set']  # 柱子组
# data = {
#     'Zero-Shot': [zc_acc, za_acc],  # a 对应两根柱子的高度
#     'Ours': [fc_acc, fa_acc]   # b 对应两根柱子的高度
# }

# # 设置柱子宽度和位置
# x = np.arange(len(x_labels))/5  # x轴位置
# group_width = 0.2  # 组宽度
# bar_width = group_width / (len(groups) + 2)  # 每组柱子宽度

# colors = ['#489BB3', '#4862AC']  # 深蓝色与天蓝色

# # 绘制柱状图
# fig, ax = plt.subplots(figsize=(8, 6))
# # ax.set_xlim(len(x_labels))  # 调整范围，使间距变紧
# for i, group in enumerate(groups):
#     heights = [data[label][i] for label in x_labels]  # 每组数据
#     # bar_positions = x + i * bar_width - bar_width / 2
#     bar_positions = (x + (i - (len(groups) - 1) / 2) * (bar_width + 0.03))  # 调整间距
#     bars = ax.bar(bar_positions, heights, bar_width, label=group, edgecolor='black', linewidth=2, color=colors[i])  # 中心对齐，并添加黑色边框
#     # 在柱子上标出数值
#     for bar, height in zip(bars, heights):
#         ax.text(bar.get_x() + bar.get_width() / 2, height + 0.1, f'{height}%', ha='center', va='bottom', fontsize=18)

# # 添加标签和标题
# # ax.set_xlabel('Flan-T5-XXL, Twitter', fontsize=12)
# ax.set_ylabel('Accuracy', fontsize=20)
# ax.set_title('Blip2, Twitter-2015', fontsize=24)
# ax.set_xticks(x)
# ax.set_xticklabels(x_labels, fontsize=20)
# ax.legend(fontsize=18)

# # 显示图表
# plt.tight_layout()
# plt.savefig('blip2_twitter-2015.png', dpi=1000)
