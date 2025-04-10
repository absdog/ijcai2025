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
  id2sentiment = ["negative", "neutral", "positive"]
  df = pd.read_csv(f"{args.data_dir}/{args.dataset}/{prefix}.tsv", sep="\t")
  if prefix == "test":
    df = df.rename({"index": "sentiments", "#1 ImageID": "image_ids", "#2 String": "tweets", "#2 String.1": "aspects"}, axis=1)
  else:
    df = df.rename({"#1 Label": "sentiments", "#2 ImageID": "image_ids", "#3 String": "tweets", "#3 String.1": "aspects"}, axis=1).drop(["index"], axis=1)
  dataset = []
  for idx, (tweet, aspect, image_id, sentiment) in enumerate(zip(df.tweets.to_list(), df.aspects.to_list(), df.image_ids.to_list(), df.sentiments.to_list())):
    dataset.append({
      "sample_id": idx,
      "tweet": tweet.replace("$T$", aspect),
      "aspect": aspect,
      "image_id": image_id,
      "sentiment": id2sentiment[sentiment],
      "prefix": prefix
    })
  return dataset

def construct_prompt(args, tweet, aspect, image_id, sentiment_list):
  str_sentiment_list = str(sentiment_list).replace("'", "")
  images = []
  images_dir = f"{args.data_dir}/{args.dataset}_images"
  prompt = None
  # prompt = f"USER: Given a set of sentiment labels {str_sentiment_list}, your task is to determine the sentiment polarity associated with a specific aspect in the provided text. An image, supplied as additional context, will assist in inferring the sentiment towards the aspect.\nImage is <image>\nText is \'{tweet}\'\nAspect is \'{aspect}\'\nPlease determine which sentiment of {str_sentiment_list} corresponds to \'{aspect}\' based on the context.\nASSISTANT:"
  prompt = f"[INST]Given a set of sentiment labels {str_sentiment_list}, your task is to determine the sentiment polarity associated with a specific aspect in the provided text. An image, supplied as additional context, will assist in inferring the sentiment towards the aspect.\nImage is <image>\nText is \'{tweet}\'\nAspect is \'{aspect}\'\nPlease determine which sentiment of {str_sentiment_list} corresponds to \'{aspect}\' based on the context.ASSISTANT:[/INST]"
  images.append(Image.open(f"{images_dir}/{image_id}").convert('RGB'))
  return prompt, images

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