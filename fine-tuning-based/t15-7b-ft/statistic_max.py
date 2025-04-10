import json
import os
import pandas as pd
from transformers import AutoProcessor, LlavaForConditionalGeneration
import numpy as np
import argparse

id2label={0:'negative', 1:'neutral', 2:'positive'}
label2id={'negative':0, 'neutral':1, 'positive':2}

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../FITE-DE/data', type=str, help="dir of data")
parser.add_argument('--dataset_name', default='twitter2015', type=str, help="The name of dataset.")
parser.add_argument('--LLMs_gen', default='../MALSC_LLMs_gen/llava_twitter15', type=str, help="name for the LLMs gen file.")
parser.add_argument('--llava_name', default='./llava-1.5-7b-hf')
args = parser.parse_args()

processor = AutoProcessor.from_pretrained(args.llava_name)

# Load and massage the dataframes.
test_tsv = args.data_dir+'/'+args.dataset_name+'/test.tsv'
train_tsv = args.data_dir+'/'+args.dataset_name+'/train.tsv'
dev_tsv = args.data_dir+'/'+args.dataset_name+'/dev.tsv'
test_df = pd.read_csv(test_tsv, sep="\t")
train_df = pd.read_csv(train_tsv, sep="\t")
val_df = pd.read_csv(dev_tsv, sep="\t")

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

test_gen = args.LLMs_gen+'/llava_34b-v1.6_reason_by_label_'+args.dataset_name+'_test_result_lst.json'
train_gen = args.LLMs_gen+'/llava_34b-v1.6_reason_by_label_'+args.dataset_name+'_train_result_lst.json'
val_gen = args.LLMs_gen+'/llava_34b-v1.6_reason_by_label_'+args.dataset_name+'_val_result_lst.json'

with open(test_gen, 'r') as f:
    test_gen_results = json.load(f)

with open(train_gen, 'r') as f:
    train_gen_results = json.load(f)

with open(val_gen, 'r') as f:
    val_gen_results = json.load(f)

test_data = test_df.to_dict('records')
train_data = train_df.to_dict('records')
val_data = val_df.to_dict('records')

data_all = test_data + train_data + val_data
results_all = test_gen_results + train_gen_results + val_gen_results

max_ti = []

i = 0
for r,d in zip(results_all, data_all):
    sentiment_target = d['target']
    label = id2label[d['sentiment']]
    tweet = d['tweet_content'].replace('$T$', sentiment_target)
    response = r['response']
    response = response + ' Thus, the sentiment of '+sentiment_target+' is '+label+'.'
    prompt=f"USER: <image>\nGiven a set of sentiment labels [positive, neutral, negative], your task is to determine the sentiment polarity associated with a specific aspect in the provided text. An image, supplied as additional context, will assist in inferring the sentiment towards the aspect. \n\n### Input\n- Text: {tweet} \n- Aspect: {sentiment_target}\n\n### Output\nFormat your answer using the following structured steps:\n1. **Text Analysis:** Describe the sentiment conveyed in the text.\n2. **Image Interpretation:** Discuss any sentiment indicators present in the image.\n3. **Conclusion:** Provide a summary that integrates the insights from the text and the image.\n\nASSISTANT:"
    inputs = prompt + response 

    encoding = processor(
        text=inputs,
        return_tensors="pt",
    )
    max_ti.append(len(encoding['input_ids'][0]))
    i += 1
    print(i)

# 将列表转换为NumPy数组
arr_ti = np.array(max_ti)

# 计算平均值
mean_ti = np.mean(arr_ti)

# 计算标准差
std_ti = np.std(arr_ti)

# 打印结果
print("max_ti：")
print("平均值：", mean_ti)
print("标准差：", std_ti)
