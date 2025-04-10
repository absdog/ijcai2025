import transformers
transformers.logging.set_verbosity_error()
from transformers import (
    BertModel,
    BertTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)
import logging
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from collections import defaultdict
from torch import nn, optim
import json
import random
import os
import argparse

from transformers import AutoModel, AutoTokenizer, AutoProcessor 
from data_utils import TwitterDataset, create_data_loader, train_, eval_model
from model import SentimentClassifier 
from transformers import AutoProcessor, LlavaForConditionalGeneration
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../FITE-DE/data', type=str, help="dir of data")
parser.add_argument('--dataset_name', default='twitter2017', type=str, help="The name of dataset.")
parser.add_argument('--caption_path', default='../FITE-DE/captions/twitter2017_images.json', type=str, help="path for the captions file.")
parser.add_argument('--LLMs_gen', default='../MALSC_LLMs_gen/llava_twitter17', type=str, help="name for the LLMs gen file.")
parser.add_argument('--llava_name', default='./llava-1.5-7b-hf')
parser.add_argument('--max_len_input', default=200, type=int, help="max len for the input text")
parser.add_argument('--max_len_label', default=500, type=int, help="max len for the label text")
parser.add_argument('--r', default=8, type=int, help="lora r")
parser.add_argument('--lora_alpha', default=8, type=int, help="lora lora_alpha")
parser.add_argument('--num_epochs', default=100, type=int, help="Training epochs")
parser.add_argument('--batch_size', default=1, type=int, help="batch size")
parser.add_argument('--inter', default=30, type=int, help="inter for printing training logs")
parser.add_argument('--lr', default=1e-6, type=float, help="learning rate")
parser.add_argument('--warmup_rate', default=0.01, type=float)
parser.add_argument('--save_dir', default='saved_models', type=str, help="save model at save_path")
args = parser.parse_args()

def get_logger(output_dir=None):
    # remove previous handlers
    root = logging.root
    for h in root.handlers[:]:
        root.removeHandler(h)
        h.close()

    if output_dir != None:
        os.makedirs(output_dir, exist_ok=True)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[
                logging.FileHandler("{}/log.txt".format(output_dir), mode="a", delay=False),
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

logger = get_logger(args.save_dir)

# Load and massage the dataframes.
test_tsv = args.data_dir+'/'+args.dataset_name+'/test.tsv'
train_tsv = args.data_dir+'/'+args.dataset_name+'/train.tsv'
dev_tsv = args.data_dir+'/'+args.dataset_name+'/dev.tsv'
test_df = pd.read_csv(test_tsv, sep="\t")
train_df = pd.read_csv(train_tsv, sep="\t")
val_df = pd.read_csv(dev_tsv, sep="\t")

images_dir = args.data_dir+'/'+args.dataset_name+'_images'
print(images_dir)

test_df = test_df.rename(
    {
        "index": "sentiment",
        "#1 ImageID": "image_id",
        "#2 String": "tweet_content",
        "#2 String.1": "target",
    },
    axis=1
)
train_df = train_df.rename(
    {
        "#1 Label": "sentiment",
        "#2 ImageID": "image_id",
        "#3 String": "tweet_content",
        "#3 String.1": "target",
    },
    axis=1
).drop(["index"], axis=1)

val_df = val_df.rename(
    {
        "#1 Label": "sentiment",
        "#2 ImageID": "image_id",
        "#3 String": "tweet_content",
        "#3 String.1": "target",
    },
    axis=1
).drop(["index"], axis=1)

# Load the image captions.
with open(args.caption_path, "r") as f:
    image_captions = json.load(f)

# Instantiate the processor.
processor = AutoProcessor.from_pretrained(args.llava_name)
args.processor = processor 

print(test_df.image_id.to_numpy())

test_gen = args.LLMs_gen+'/llava_34b-v1.6_reason_by_label_'+args.dataset_name+'_test_result_lst.json'
train_gen = args.LLMs_gen+'/llava_34b-v1.6_reason_by_label_'+args.dataset_name+'_train_result_lst.json'
val_gen = args.LLMs_gen+'/llava_34b-v1.6_reason_by_label_'+args.dataset_name+'_val_result_lst.json'

with open(test_gen, 'r') as f:
    test_gen_results = json.load(f)

with open(train_gen, 'r') as f:
    train_gen_results = json.load(f)

with open(val_gen, 'r') as f:
    val_gen_results = json.load(f)


train_data_loader = create_data_loader(
    train_df, processor, args.max_len_input, args.max_len_label, args.batch_size, image_captions, train_gen_results, images_dir
)
val_data_loader = create_data_loader(
    val_df, processor, args.max_len_input, args.max_len_label, args.batch_size*8, image_captions, val_gen_results, images_dir
)
test_data_loader = create_data_loader(
    test_df, processor, args.max_len_input, args.max_len_label, args.batch_size*8, image_captions, test_gen_results, images_dir
)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

model = SentimentClassifier(args).cuda()
config = LoraConfig(
    target_modules=["q_proj","v_proj"],
    bias="none",
    r=args.r,
    lora_alpha=args.lora_alpha
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

model.save_pretrained(args.save_dir+'/best_model_lora')
