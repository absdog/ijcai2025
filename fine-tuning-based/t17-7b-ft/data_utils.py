import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report, accuracy_score
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu
import copy
from tqdm import tqdm
import json
from torch.cuda.amp import GradScaler, autocast

id2label={0:'negative', 1:'neutral', 2:'positive'}
label2id={'negative':0, 'neutral':1, 'positive':2}

# Construct the dataset.
class TwitterDataset(Dataset):
    def __init__(
        self,
        tweets: np.array,
        labels: np.array,
        sentiment_targets: np.array,
        image_ids: np.array,
        image_captions,
        images_dir,
        processor,
        max_len_input: int,
        max_len_label: int,
        gen_results
    ):
        """
        Downstream code expects reviews and targets to be NumPy arrays.
        """
        self.tweets = tweets
        self.labels = labels
        self.processor = processor
        self.sentiment_targets = sentiment_targets
        self.image_captions = image_captions
        self.image_ids = image_ids
        self.images_dir = images_dir 
        self.max_len_input = max_len_input
        self.max_len_label = max_len_label
        self.gen_results = gen_results

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, item):
        # import pdb; pdb.set_trace()
        IGNORE_ID = -100
        tweet = str(self.tweets[item])
        sentiment_target = self.sentiment_targets[item]
        # remove the $T$
        tweet = tweet.replace('$T$', sentiment_target)

        try:
            caption = self.image_captions[self.image_ids[item]]
        except KeyError:  # A couple of the images have no content.
            caption = "I dont know the caption of this image"
        # import pdb; pdb.set_trace()
        response = f"The sentiment of {sentiment_target} is {id2label[self.labels[item]]}."
        # response = response + ' Thus, the sentiment of '+sentiment_target+' is '+id2label[self.labels[item]]+'.'

        prompt=f"USER: <image>\nGiven a set of sentiment labels [positive, neutral, negative], your task is to determine the sentiment polarity associated with a specific aspect in the provided text. An image, supplied as additional context, will assist in inferring the sentiment towards the aspect. \n\n### Input\n- Text: {tweet} \n- Aspect: {sentiment_target}\n\nASSISTANT:"

        label = prompt + ' ' + response
        encoding_label = self.processor(
            text=label,
            max_length=self.max_len_label,
            padding="max_length",
            return_tensors="pt",
            truncation=True
        )

        # set ignore index
        label_ids = encoding_label['input_ids'].flatten()
        train_ids = copy.deepcopy(label_ids)
        train_attention_mask = encoding_label['attention_mask'].flatten()
        padding_len = len(train_attention_mask)-train_attention_mask.sum().item()
        prompt_len = len(self.processor.tokenizer.encode(prompt))
        label_ids[:padding_len+prompt_len] = IGNORE_ID

        # image
        image_path = self.images_dir+'/'+self.image_ids[item]
        image = Image.open(image_path).convert('RGB')

        encoding_input = self.processor(
            text=prompt,
            images=image,
            max_length=self.max_len_input,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
        
        # for test
        test_ids = encoding_input["input_ids"].flatten() 
        test_attention_mask = encoding_input["attention_mask"].flatten() 
        pixel_values = encoding_input['pixel_values']

        return {
            "train_ids": train_ids,
            "train_attention_mask": train_attention_mask,
            "label_ids": label_ids,
            "label": label,
            "test_ids": test_ids,
            "test_attention_mask": test_attention_mask,
            "pixel_values": pixel_values.squeeze(0),
        }


# Construct the data loaders.
def create_data_loader(df, processor, max_len_input, max_len_label, batch_size, image_captions, gen_results, images_dir):
    ds = TwitterDataset(
        tweets=df.tweet_content.to_numpy(),
        labels=df.sentiment.to_numpy(),
        sentiment_targets=df.target.to_numpy(),
        image_ids=df.image_id.to_numpy(),
        image_captions=image_captions,
        images_dir=images_dir,
        processor=processor,
        max_len_input=max_len_input,
        max_len_label=max_len_label,
        gen_results=gen_results
    )
    return DataLoader(ds, batch_size=batch_size)

def batch_bleu(references, candidates):
    # Initialize BLEU score
    total_bleu_score = 0.0

    # Iterate over the batch data
    for ref, cand in zip(references, candidates):
        # Compute BLEU score for each reference-candidate pair
        bleu_score = sentence_bleu([ref], cand)
        total_bleu_score += bleu_score

    # Average BLEU score over the batch
    average_bleu_score = total_bleu_score / len(references)

    return average_bleu_score

'''
    "train_ids": train_ids,
    "train_attention_mask": train_attention_mask,
    "label_ids": label_ids,
    "label": label,
    "test_ids": test_ids,
    "test_attention_mask": test_attention_mask,
    "pixel_values": pixel_values.squeeze(0),
'''

def train_(args, model, train_dataloader, eval_dataloader, test_dataloader, loss_fn, optimizer, scheduler, logger):
    steps = 0
    best_f1 = 0
    # scaler = GradScaler()
    for e in range(args.num_epochs+1):
        # report = eval_model(args, model, test_dataloader)
        # print(report)
        # print(f"epoch: [{e}] acc: {acc} f1: {f1}")
        # logger.info(f"epoch: [{e}] acc: {acc} f1: {f1}")
        model = model.train()
        losses = []
        for d in train_dataloader:
            train_ids = d['train_ids'].to(args.device)
            train_attention_mask = d['train_attention_mask'].to(args.device)
            label_ids = d['label_ids'].to(args.device)
            pixel_values = d['pixel_values'].to(args.device)
            # with autocast():
            output = model(
                input_ids=train_ids, 
                attention_mask=train_attention_mask, 
                pixel_values=pixel_values, 
                labels=label_ids
            )
            loss = output.loss  
            # if str(loss.item()) == "nan":
            #     logger.info(f"epoch: [{e}] loss: {loss.item()}, continue ...")
            #     optimizer.zero_grad()
            #     continue
            losses.append(loss.item())
            # import pdb; pdb.set_trace()
            loss.backward()
            # accelerator.backward(loss)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if steps % args.inter == 0:
                logger.info(f"epoch: [{e}] loss: {np.mean(losses)}")
            
            if steps % (4*args.inter) == 0:
                acc, f1 = eval_model(args, model, test_dataloader, e)
                # report = eval_model(args, model, test_dataloader)
                # print(report)
                # print(f"epoch: [{e}] acc: {acc} f1: {f1}")
                logger.info(f"epoch: [{e}] acc: {acc} f1: {f1}")
                if f1 > best_f1:
                    best_f1 = f1 
                    model.save_pretrained(args.save_dir+'/best_model')
            steps += 1
        # logger.info("saving model")
        # model.save_pretrained(args.save_dir+'/epoch_'+str(e))


def gen2id(gen_list, inp_list):
    # import pdb; pdb.set_trace()
    label2id = {'negative':0, 'neutral':1, 'positive':2}
    _gen_list = [g[len(i):] for g, i in zip(gen_list, inp_list)]
    id_list = []
    for g in _gen_list:
        gs = g.split()
        # rev_gs = gs[::-1]
        flag = 0
        for t in gs:
            # import pdb; pdb.set_trace()
            for key in label2id.keys():
                if key in t.lower():
                    id_list.append(label2id[key])
                    flag = 1
                    break
            if flag == 1:
                break
        if flag == 0:
            id_list.append(1)

    return id_list

'''
    "train_ids": train_ids,
    "train_attention_mask": train_attention_mask,
    "label_ids": label_ids,
    "label": label,
    "test_ids": test_ids,
    "test_attention_mask": test_attention_mask,
    "pixel_values": pixel_values.squeeze(0),
'''

def eval_model(args, model, data_loader, epoch=None):
    print("start test ...")
    model = model.eval()

    input_texts = []
    label_texts = []
    gen_ret = []
    gol_ret = []
    correct_predictions = 0

    with torch.no_grad():
        for d in tqdm(data_loader):
            label = d['label']
            test_ids = d['test_ids'].to(args.device)
            test_attention_mask = d['test_attention_mask'].to(args.device)
            pixel_values = d['pixel_values'].to(args.device)
            preds = model.generate(
                input_ids=test_ids, 
                attention_mask=test_attention_mask, 
                pixel_values=pixel_values
            )
            # import pdb; pdb.set_trace()
            input_texts.extend(args.processor.batch_decode(test_ids, skip_special_tokens=True))
            label_texts.extend(label)
            gen_ret.extend(args.processor.batch_decode(preds, skip_special_tokens=True))
            gol_ret.extend(label)

    gen_id, gol_id = gen2id(gen_ret, input_texts), gen2id(gol_ret, input_texts)
    assert len(gen_id) == len(gol_id) == len(input_texts) == len(label_texts)

    id2label=['negative', 'neutral', 'positive']
    record = []
    for idx, (input_text, pred_text, label_text, p, l) in enumerate(zip(input_texts, gen_ret, label_texts, gen_id, gol_id)):
        record.append({
            "id": idx,
            "input_text": input_text,
            "pred_text": pred_text,
            "pred": id2label[p],
            "label_text": label_text,
            "gt": id2label[l]
        })
    json.dump(record, open(f"{args.save_dir}/predict_record_{epoch}.json", "w", encoding="utf-8"), ensure_ascii=False)

    # report = classification_report(gol_id, gen_id, digits=4)
    acc = accuracy_score(gol_id, gen_id)
    f1 = f1_score(gol_id, gen_id, average="weighted")

    return acc, f1
