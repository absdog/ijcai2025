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
from accelerate import Accelerator

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../../../data/twitter', type=str, help="dir of data")
parser.add_argument('--dataset_name', default='twitter2015', type=str, help="The name of dataset.")
parser.add_argument('--caption_path', default='../../../data/twitter/captions/twitter2017_images.json', type=str, help="path for the captions file.")
parser.add_argument('--LLMs_gen', default='../../../data/twitter/llava_twitter17', type=str, help="name for the LLMs gen file.")
parser.add_argument('--llava_name', default='../../../models/llava-1.5-7b-hf')
parser.add_argument('--max_len_input', default=200, type=int, help="max len for the input text")
parser.add_argument('--max_len_label', default=500, type=int, help="max len for the label text")
parser.add_argument('--r', default=8, type=int, help="lora r")
parser.add_argument('--lora_alpha', default=8, type=int, help="lora lora_alpha")
parser.add_argument('--num_epochs', default=100, type=int, help="Training epochs")
parser.add_argument('--batch_size', default=4, type=int, help="batch size")
parser.add_argument('--inter', default=30, type=int, help="inter for printing training logs")
parser.add_argument('--lr', default=2e-6, type=float, help="learning rate")
parser.add_argument('--warmup_rate', default=0.01, type=float)
parser.add_argument('--save_dir', default='saved_models', type=str, help="save model at save_path")
parser.add_argument("--device", default="cuda:0", type=str)
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
# import pdb; pdb.set_trace()
with open(args.caption_path, "r") as f:
    image_captions = json.load(f)

# Instantiate the processor.
processor = AutoProcessor.from_pretrained(args.llava_name)
args.processor = processor 

print(test_df.image_id.to_numpy())

# llava_34b-v1.6_reason_by_label_twitter2017_test_result_lst

# test_gen = args.LLMs_gen+'/llava_34b-v1.6_reason_by_label_twitter2017_test_result_lst.json'
# train_gen = args.LLMs_gen+'/llava_34b-v1.6_reason_by_label_twitter2017_train_result_lst.json'
# val_gen = args.LLMs_gen+'/llava_34b-v1.6_reason_by_label_twitter2017_val_result_lst.json'

# train_df = train_df.drop_duplicates(subset=["image_id"])
# test_df = test_df.drop_duplicates(subset=["image_id"])
# val_df = val_df.drop_duplicates(subset=["image_id"])

# with open(test_gen, 'r') as f:
#     test_gen_results = json.load(f)
#     # test_gen_results = [test_gen_results[i-1] for i in test_df.index]

# with open(train_gen, 'r') as f:
#     train_gen_results = json.load(f)
#     # train_gen_results = [train_gen_results[i-1] for i in train_df.index]

# with open(val_gen, 'r') as f:
#     val_gen_results = json.load(f)
    # val_gen_results = [val_gen_results[i-1] for i in val_df.index]


train_data_loader = create_data_loader(
    train_df, processor, args.max_len_input, args.max_len_label, args.batch_size, image_captions, None, images_dir
)
val_data_loader = create_data_loader(
    val_df, processor, args.max_len_input, args.max_len_label, args.batch_size*2, image_captions, None, images_dir
)
test_data_loader = create_data_loader(
    test_df, processor, args.max_len_input, args.max_len_label, args.batch_size*2, image_captions, None, images_dir
)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

model = SentimentClassifier(args).to(args.device)
# import pdb; pdb.set_trace()

config = LoraConfig(
    target_modules=["q_proj", "v_proj"],
    bias="none",
    r=args.r,
    lora_alpha=args.lora_alpha
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

# Configure the optimizer and scheduler.
optimizer = AdamW(model.parameters(), lr=args.lr)
total_steps = len(train_data_loader) * args.num_epochs 
print("total steps:", total_steps)
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=args.warmup_rate*total_steps, 
    num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss().cuda()

# import pdb; pdb.set_trace()
# from accelerate import Accelerator
# accelerator = Accelerator(mixed_precision="fp16")
# model, optimizer, train_data_loader, test_data_loader, val_data_loader, scheduler = accelerator.prepare(
#     model, optimizer, train_data_loader, test_data_loader, val_data_loader, scheduler
# )
# import pdb; pdb.set_trace()
train_(
    args, 
    model, 
    train_data_loader, 
    val_data_loader, 
    test_data_loader, 
    loss_fn, 
    optimizer, 
    scheduler, 
    logger,
    # accelerator
)
