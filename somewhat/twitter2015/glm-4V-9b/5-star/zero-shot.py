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
# from transformers import AutoProcessor, get_linear_schedule_with_warmup,LlavaForConditionalGeneration, GenerationConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from utils import get_logger, set_seed, load_dataset, construct_prompt, generation2sentiment
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="../../../../../data/twitter", type=str, help="dir of data")
parser.add_argument("--dataset", default="twitter2015", type=str, help="The name of dataset.")
parser.add_argument("--model_name", default="../../../../../models/glm-4v-9b")
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
# processor = AutoProcessor.from_pretrained(args.model_name)
# model = LlavaForConditionalGeneration.from_pretrained(
#   args.model_name, 
#   torch_dtype=torch.float16,
#   # torch_dtype=torch.bfloat16
# ).to(args.device)

tokenizer = AutoTokenizer.from_pretrained(f"{args.model_name}", trust_remote_code=True)
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
args.tokenizer = tokenizer 
args.logger = logger
args.id2sentiment = ["positive", "somewhat positive", "neutral", "somewhat negative", "negative"]
args.sentiment2id = {sent: idx for idx, sent in enumerate(args.id2sentiment)}

test_set = load_dataset(args, prefix=args.prefix)

pendding_set = test_set
logger.info(f"zero-shot, pendding set: {len(pendding_set)}")

targets = []
preds, labels = [], []
num_notin = 0
for data in tqdm(pendding_set):

  target = copy.deepcopy(data)
  prompt, images = construct_prompt(args, data["tweet"], data["aspect"], data["image_id"], args.id2sentiment)
  inputs = tokenizer.apply_chat_template(
    [{"role": "user", "image": images[0], "content": prompt}],
    add_generation_prompt=True, tokenize=True, return_tensors="pt",
    return_dict=True
  ).to(args.device)
  outputs = model.generate(**inputs, max_length=inputs['input_ids'].shape[1] + args.max_new_tokens, do_sample=False)
  generation_ids = outputs[0, inputs['input_ids'].shape[1]:]
  # generate_outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=0.9, output_scores=True, return_dict_in_generate=True)
  # scores = torch.cat(generation_ids, dim=0) # generate length * vocab size
  # uncertainty = (1.0 / torch.max(scores, dim=1).values).mean().item()
  # generate_ids = torch.argmax(scores, dim=1)
  generation = tokenizer.decode(generation_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
  generation = generation.lower()
  # import pdb; pdb.set_trace()
  pred = generation2sentiment(args, generation)

  target.update({
    "generation": generation,
    "pred": pred
  })
  targets.append(target) 

  # preds.append(args.sentiment2id[pred])
  # labels.append(args.sentiment2id[target["sentiment"]])

# report = classification_report(labels, preds, digits=4)
# print(report)
json.dump({"report": None, "results": targets}, open(f"{args.save_path}/zero-shot-results.json", "w", encoding="utf-8"), ensure_ascii=False)
  






