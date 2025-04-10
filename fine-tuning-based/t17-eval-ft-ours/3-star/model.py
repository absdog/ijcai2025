# import transformers
# from transformers import (
#     BertModel,
#     BertTokenizer,
#     AdamW,
#     get_linear_schedule_with_warmup,
#     AutoModel
# )
import torch
import torch.nn.functional as F
import torch.nn as nn

from transformers import LlavaForConditionalGeneration, BitsAndBytesConfig

# Construct and instantiate the classifier.
class SentimentClassifier(nn.Module):
    def __init__(self, args):
        super(SentimentClassifier, self).__init__()

        print("Loading pretrained model.")
        # 4bit quant 
        '''
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.multimodal_model = LlavaForConditionalGeneration.from_pretrained(args.llava_name, torch_dtype=torch.float16, quantization_config=bnb_config)
        '''
        self.multimodal_model = LlavaForConditionalGeneration.from_pretrained(
            args.model_name, 
            torch_dtype=torch.float16,
            # load_in_8bit=True
        )
        self.args = args

    def forward(self, input_ids, attention_mask, pixel_values, labels):
        t_outputs = self.multimodal_model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            pixel_values=pixel_values, 
            labels=labels
        )
        return t_outputs 

    def generate(self, input_ids, attention_mask, pixel_values):
        t_outputs = self.multimodal_model.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            pixel_values=pixel_values, 
            max_new_tokens=300,
            do_sample=False
        )
        return t_outputs
