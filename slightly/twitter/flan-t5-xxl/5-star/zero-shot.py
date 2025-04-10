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

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="../../../../../data/twitter14", type=str, help="dir of data")
parser.add_argument("--dataset", default="twitter14", type=str, help="The name of dataset.")
parser.add_argument("--result_path", default="../second/results/sec-results.json", type=str)
parser.add_argument("--model_name", default="../../../../../models/flan-t5-xxl")
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
processor = T5Tokenizer.from_pretrained(args.model_name)
model = T5ForConditionalGeneration.from_pretrained(
  args.model_name, 
  # torch_dtype=torch.bfloat16
).to(args.device)

args.save_path = f"{args.save_dir}/"
try:
  os.mkdir(args.save_path)
except:
  pass

logger = get_logger(args)
logger.info(args)

args.processor = processor 
args.logger = logger
args.id2sentiment = ["positive", "slightly positive", "neutral", "slightly negative", "negative"]
args.sentiment2id = {sent: idx for idx, sent in enumerate(args.id2sentiment)}

test_set = load_dataset(args, prefix=args.prefix)

pendding_set = test_set
logger.info(f"zero-shot, pendding set: {len(pendding_set)}")

targets = []
# preds, labels = [], []
num_notin = 0
for data in tqdm(pendding_set):

  target = copy.deepcopy(data)
  prompt = construct_prompt(args, data["tweet"], data["aspect"], args.id2sentiment)
  inputs = args.processor(
    text=prompt,
    # max_length=args.max_input_len,
    padding=True,
    return_tensors="pt",
    # truncation=True,
  ).to(args.device)
  generation_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
  # generate_outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=0.9, output_scores=True, return_dict_in_generate=True)
  # scores = torch.cat(generation_ids, dim=0) # generate length * vocab size
  # uncertainty = (1.0 / torch.max(scores, dim=1).values).mean().item()
  # generate_ids = torch.argmax(scores, dim=1)
  generation = processor.batch_decode(generation_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
  # generation = generation.split("ASSISTANT:")[-1].lower()
  pred = generation2sentiment(args, generation)

  target.update({
    "prompt": prompt,
    "generation": generation,
    "pred": pred
  })
  targets.append(target) 

# report = classification_report(labels, preds, digits=4)
# print(report)
json.dump({"report": None, "results": targets}, open(f"{args.save_path}/zero-shot-results.json", "w", encoding="utf-8"), ensure_ascii=False)
  






