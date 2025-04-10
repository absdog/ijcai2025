'''
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-base")

encoding = tokenizer.encode_plus(
            "translate English to German: The house is wonderful.",
            add_special_tokens=True,
            max_length=30,
            return_token_type_ids=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )
labels = encoding["input_ids"]
labels[labels == tokenizer.pad_token_id] = -100
print(labels)
'''

from transformers import AutoTokenizer, T5ForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("../MALSC_LLMs_vitt52caption/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("../MALSC_LLMs_vitt52caption/flan-t5-base")

# training
input_ids = tokenizer("The dog walks in a very large park, so where is the dog", return_tensors="pt").input_ids
labels = tokenizer(' '.join("<extra_id_0> dog <extra_id_1> the <extra_id_2>".split()*50), return_tensors="pt").input_ids
outputs = model(input_ids=input_ids, labels=labels)
loss = outputs.loss
print(loss)
logits = outputs.logits
print(logits.size())

output = model.generate(input_ids=input_ids)
print(tokenizer.batch_decode(output, skip_special_tokens=True))
'''
# inference
input_ids = tokenizer(
    "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
).input_ids  # Batch size 1
outputs = model.generate(input_ids)
print(tokenizer.batch_decode(outputs))
print(tokenizer.batch_decode(input_ids))
# studies have shown that owning a dog is good for you.
'''
