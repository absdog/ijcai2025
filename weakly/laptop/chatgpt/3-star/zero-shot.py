import os
import sys
import random
import json
import torch
import copy
import argparse
import transformers
transformers.logging.set_verbosity_error()
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
from utils import get_logger, set_seed, load_dataset, construct_prompt, generation2sentiment
from sklearn.metrics import classification_report
import requests

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="../../../../../data/laptop", type=str, help="dir of data")
parser.add_argument("--dataset", default="laptop", type=str, help="The name of dataset.")
parser.add_argument("--result_path", default="../second/results/sec-results.json", type=str)
parser.add_argument("--model_name", default="../../../../../models/flan-t5-large")
parser.add_argument("--max_new_tokens", default=100, type=int)
parser.add_argument("--log_name", type=str, default="t15-ft-llava-7b")
parser.add_argument("--save_dir", default="./results", type=str, help="save model at save_path")
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument("--model_seed", default=20240, type=int)
parser.add_argument("--prefix", default="train", type=str)
args = parser.parse_args()

# args.se = {
#   "positive": "the strongest positive sentiment",
#   # "posneu": "the uncertain sentiment between positive and neutral",
#   "neutral": "indicates that the sentiment of the given aspect is clearly neutral, without any positive or negative sentiments mixed in",
#   # "negneu": "the uncertain sentiment between negative and neutral",
#   "negative": "the strongest negative sentiment",
#   "unknown": "can not determine the sentiment for the aspect based on the given information"
# }

# import pdb; pdb.set_trace()
set_seed(args.model_seed)
# init processor
def model(messages):
  response = requests.post(
    url="https://api.openai-sb.com/v1/chat/completions",
    headers={
      "Authorization": "Bearer sb-21f368782abb514eb442711707a406d943831e9c1d530d7a",
      "Content-Type": "application/json"
    },
    data=json.dumps({
      "model": "gpt-3.5-turbo",
      "stream": False,
      "messages": messages
    })
  )
  return response

args.save_path = f"{args.save_dir}/"
try:
  os.mkdir(args.save_path)
except:
  pass

logger = get_logger(args)
logger.info(args)

args.logger = logger
args.id2sentiment = ["positive", "neutral", "negative"]
args.sentiment2id = {sent: idx for idx, sent in enumerate(args.id2sentiment)}

test_set = load_dataset(args, prefix=args.prefix)

pendding_set = test_set
logger.info(f"zero-shot, pendding set: {len(pendding_set)}")

targets = []
preds, labels = [], []
num_notin = 0

pendding_set = []
with open("./gsm8k_test.jsonl", "r", encoding="utf-8") as fp:
  pendding_set = [json.loads(l.strip("\n")) for l in fp.readlines()]

# import pdb; pdb.set_trace()

for data in tqdm(pendding_set[:200]):

  target = copy.deepcopy(data)
  # prompt = construct_prompt(args, data["tweet"], data["aspect"], args.id2sentiment)
  prompt = f"“{data['answer']}”\n请写出宏观的计算步骤，不要显示具体计算细节，用一段英文表述"
  response = model([{
    "role": "user",
    "content": prompt
  }])
  try:
    response = json.loads(response.text)
    generation = response["choices"][0]["message"]["content"]
  except:
    print("error")
    continue
  # import pdb; pdb.set_trace()
  # pred = generation2sentiment(args, generation)

  target.update({
    "cot": generation
  })
  targets.append(target) 

  # preds.append(args.sentiment2id[pred])
  # labels.append(args.sentiment2id[target["sentiment"]])

# report = classification_report(labels, preds, digits=4)
# print(report)
json.dump(targets, open(f"./gsm8k_test.json", "w", encoding="utf-8"), ensure_ascii=False)
  






