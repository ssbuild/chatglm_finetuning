# -*- coding: utf-8 -*-
# @Time    : 2023/3/9 15:29
import os
import torch
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from deep_training.nlp.models.chatglm import TransformerChatGlmLMHeadModel, setup_model_profile, ChatGLMConfig,ChatGLMForConditionalGeneration
from transformers import HfArgumentParser

from data_utils import train_info_args, NN_DataHelper
from tokenization_chatglm import ChatGLMTokenizer


class MyTransformer(TransformerChatGlmLMHeadModel, with_pl=True):
    def __init__(self, *args, **kwargs):
        super(MyTransformer, self).__init__(*args, **kwargs)



if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    model_args, training_args, data_args = parser.parse_dict(train_info_args)


    setup_model_profile()


    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer: ChatGLMTokenizer
    tokenizer, config, label2id, id2label = dataHelper.load_tokenizer_and_config(
        tokenizer_class_name=ChatGLMTokenizer, config_class_name=ChatGLMConfig)


    #修改配置
    def modify_config(config):
        # 小参数
        config.inference = True
        config.max_sequence_length = 1024


    #加载新训练权重
    train_weight = './best_ckpt/best.pt'
    if os.path.exists(train_weight):
        config = ChatGLMConfig.from_pretrained('./best_ckpt')
        modify_config(config)
        model = MyTransformer.load_from_checkpoint(train_weight, config=config,
                                                   model_args=model_args,
                                                   training_args=training_args)
    else:
        modify_config(config)
        config.num_layers = 14 # 官方默认32层 ， 使用小层推理
        model = MyTransformer(config=config, model_args=model_args, training_args=training_args)

    model.eval()
    model.half()
    model.to(torch.device('cuda:0'))

    # prompts = [
    #     "晚上睡不着应该怎么办",
    # ]
    # o = tokenizer.batch_encode_plus(prompts, truncation=True,
    #                                 max_length=256,
    #                                 return_attention_mask=False,
    #                                 return_token_type_ids=False)
    # input_ids = o['input_ids']
    # # 预测
    # with torch.inference_mode():
    #     model_: ChatGLMForConditionalGeneration = model.backbone.model
    #     results = model_.generate(input_ids, max_length=256, bos_token_id=config.decoder_start_token_id,
    #                               pad_token_id=config.pad_token_id,
    #                               eos_token_id=config.eos_token_id)
    #
    #     for result in results:
    #         print(result)
    #         print("\n==================================\n")

    model_: ChatGLMForConditionalGeneration = model.backbone.model
    gen_kwards = {
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    with torch.inference_mode():
        response, history = model_.chat(tokenizer, "你好", history=[],max_length=1024,**gen_kwards)
        print('你好',' ',response)

        response, history = model_.chat(tokenizer, "晚上睡不着应该怎么办", history=history,max_length=1024,**gen_kwards)
        print('晚上睡不着应该怎么办',' ',response)