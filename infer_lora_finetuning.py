# -*- coding: utf-8 -*-
# @Time    : 2023/3/9 15:29
import os
import torch
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from deep_training.nlp.models.chatglm import TransformerChatGlmLMHeadModel, setup_model_profile, ChatGLMConfig,ChatGLMForConditionalGeneration
from deep_training.nlp.models.lora import LoraArguments, LoraModel
from transformers import HfArgumentParser

from data_utils import train_info_args, NN_DataHelper,get_deepspeed_config
from tokenization_chatglm import ChatGLMTokenizer


class MyTransformer(TransformerChatGlmLMHeadModel, with_pl=True):
    def __init__(self, *args, **kwargs):
        lora_args: LoraArguments = kwargs.pop('lora_args')
        super(MyTransformer, self).__init__(*args, **kwargs)
        self.lora_args = lora_args
        if lora_args.with_lora:
            model = LoraModel(self.backbone, lora_args)
            print('*' * 30,'lora info')
            model.print_trainable_parameters()
            self.set_model(model, copy_attr=False)



if __name__ == '__main__':
    train_info_args['seed'] = None
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, LoraArguments))
    model_args, training_args, data_args, _ = parser.parse_dict(train_info_args)

    setup_model_profile()

    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer: ChatGLMTokenizer
    tokenizer, _, _, _ = dataHelper.load_tokenizer_and_config(
        tokenizer_class_name=ChatGLMTokenizer, config_class_name=ChatGLMConfig)


    config = ChatGLMConfig.from_pretrained('./best_ckpt')
    config.initializer_weight = False

    lora_args = LoraArguments.from_pretrained('./best_ckpt')

    assert lora_args.inference_mode == True

    model = MyTransformer(config=config, model_args=model_args, training_args=training_args,lora_args=lora_args)
    # 加载lora权重
    model.backbone.from_pretrained(model.backbone.model, pretrained_model_name_or_path = './best_ckpt', lora_config = lora_args)

    base_model: ChatGLMForConditionalGeneration = model.backbone.model.model
    # 按需修改
    base_model.half().to(torch.device('cuda:0'))
    base_model = base_model.eval()

    response, history = base_model.chat(tokenizer, "写一个诗歌，关于冬天", history=[],max_length=1024)
    print('写一个诗歌，关于冬天',' ',response)

    response, history = base_model.chat(tokenizer, "晚上睡不着应该怎么办", history=[],max_length=1024)
    print('晚上睡不着应该怎么办',' ',response)



