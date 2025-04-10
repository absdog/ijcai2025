import os
import sys
import random
import json
import torch
import copy
import argparse
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
transformers.logging.set_verbosity_error()
import pandas as pd
from tqdm import tqdm
from utils import get_logger, set_seed, load_dataset, construct_prompt, generation2sentiment
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="../../../../../data/laptop", type=str, help="dir of data")
parser.add_argument("--dataset", default="laptop", type=str, help="The name of dataset.")
parser.add_argument("--result_path", default="../second/results/sec-results.json", type=str)
parser.add_argument("--model_name", default="../../../../../models/glm-4-9b-chat")
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
# processor = T5Tokenizer.from_pretrained(args.model_name)
# model = transformers.pipeline(
#   "text-generation",
#   model=args.model_name,
#   tokenizer=args.model_name,
#   model_kwargs={"torch_dtype": torch.bfloat16},
#   device_map={"": args.device},
# )
# terminators = [
#   model.tokenizer.eos_token_id,
#   model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
# ]

tokenizer = AutoTokenizer.from_pretrained(f"{args.model_name}", trust_remote_code=True)
args.tokenizer = tokenizer

model = AutoModelForCausalLM.from_pretrained(
  f"{args.model_name}",
  torch_dtype=torch.bfloat16,
  low_cpu_mem_usage=True,
  trust_remote_code=True
).to(args.device).eval()

args.save_path = f"{args.save_dir}/"
try:
  os.mkdir(args.save_path)
except:
  pass

logger = get_logger(args)
logger.info(args)

# args.processor = processor 
args.logger = logger
# args.id2sentiment = ["positive", "weakly positive", "neutral", "weakly negative", "negative"]
args.id2sentiment = ["positive", "weakly positive", "neutral", "negative"]
args.sentiment2id = {sent: idx for idx, sent in enumerate(args.id2sentiment)}

test_set = load_dataset(args, prefix=args.prefix)

pendding_set = test_set
logger.info(f"zero-shot, pendding set: {len(pendding_set)}")

targets = []
preds, labels = [], []
num_notin = 0
for data in tqdm(pendding_set):

  target = copy.deepcopy(data)
  prompt = [
    {"role": "user", "content": ""}
  ]
  prompt[0]["content"] = construct_prompt(args, data["tweet"], data["aspect"], args.id2sentiment)
  inputs = tokenizer.apply_chat_template(
    prompt, 
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
    return_dict=True
  ).to(args.device)
  outputs = model.generate(
    **inputs,
    max_length=inputs['input_ids'].shape[1] + args.max_new_tokens,
    do_sample=False,
  )
  generation = tokenizer.decode(outputs[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
  # import pdb; pdb.set_trace()
  # generation = output[0]["generated_text"][-1]["content"]
  # import pdb; pdb.set_trace()
  # print(output[0]["generated_text"][-1]["content"].split("\n"))
  # generation = generation.split("ASSISTANT:")[-1].lower()
  pred = generation2sentiment(args, generation)

  target.update({
    "generation": generation,
    "pred": pred
  })
  targets.append(target) 

#   preds.append(args.sentiment2id[pred])
#   labels.append(args.sentiment2id[target["sentiment"]])

# report = classification_report(labels, preds, digits=4)
# print(report)
json.dump({"report": None, "results": targets}, open(f"{args.save_path}/zero-shot-results.json", "w", encoding="utf-8"), ensure_ascii=False)
  






