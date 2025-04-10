import os
import sys
import json
import logging
import torch 
import numpy as np
import pandas as pd
from PIL import Image
import base64

def get_logger(args):
  # remove previous handlers
  root = logging.root
  for h in root.handlers[:]:
    root.removeHandler(h)
    h.close()

  if args.save_path != None:
    model_name = args.model_name.split("/")[-1]
    dataset_name = args.dataset
    os.makedirs(args.save_path, exist_ok=True)
    logging.basicConfig(
      format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
      datefmt="%m/%d/%Y %H:%M:%S",
      level=logging.INFO,
      handlers=[
        logging.FileHandler(f"{args.save_path}/log.txt", mode="a", delay=False),
        logging.StreamHandler(sys.stdout),
      ],
    )
  else:
    logging.basicConfig(
      format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
      datefmt="%m/%d/%Y %H:%M:%S",
      level=logging.INFO,
      handlers=[logging.StreamHandler(sys.stdout)],
    )
  logger = logging.getLogger('Multimodal ABSA')
  logger.setLevel(10)
  return logger

def load_dataset(args, prefix="train"):
  dataset = json.load(open(f"{args.data_dir}/{prefix}.json", "r", encoding="utf-8"))
  new_dataset = []
  for d in dataset:
    tweet = " ".join(d["token"])
    for a in d["aspects"]:
      new_dataset.append({
        "sample_id": None,
        "tweet": tweet,
        "aspect": " ".join(a["term"]),
        "sentiment": a["polarity"]
      })
  return new_dataset

def construct_prompt(args, tweet, aspect, sentiment_list):
  str_sentiment_list = str(sentiment_list).replace("'", "")
  prompt = None
  prompt = f"Given a set of sentiment labels {str_sentiment_list}, your task is to determine the sentiment polarity associated with a specific aspect in the provided text.\nText is \'{tweet}\'\nAspect is \'{aspect}\'\nPlease determine which sentiment of {str_sentiment_list} corresponds to \'{aspect}\' based on the context."
  return prompt

def generation2sentiment(args, generation):
  sentiment = "neutral"
  for s in ["somewhat positive", "somewhat negative"]:
    if s in generation:
      sentiment = s 
  if sentiment != "neutral":
    return sentiment
  else:
    words = generation.split()
    flag = False
    for w in words:
      for key in args.sentiment2id.keys():
        if key in w.lower(): # "weak positive" positive
          sentiment = key
          flag = True
          break
      if flag == True:
        break
    if flag == False:
      sentiment = "neutral"
    return sentiment




def set_seed(seed): # 设置随机种子
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  torch.backends.cudnn.deterministic = True