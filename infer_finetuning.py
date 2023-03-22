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
        super(MyTransformer, self).__init__(*args, **kwargs)




if __name__ == '__main__':
    train_info_args['seed'] = None
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, LoraArguments))
    model_args, training_args, data_args, _ = parser.parse_dict(train_info_args)

    setup_model_profile()

    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer: ChatGLMTokenizer
    tokenizer, _, _, _ = dataHelper.load_tokenizer_and_config(
        tokenizer_class_name=ChatGLMTokenizer, config_class_name=ChatGLMConfig)

    ###################### 注意 选最新权重
    #选择最新的权重 ， 根据时间排序 选最新的
    config = ChatGLMConfig.from_pretrained('./best_ckpt')

    if get_deepspeed_config() is None:
        train_weight = './best_ckpt/last-v3.ckpt'
        assert os.path.exists(train_weight)
        model = MyTransformer.load_from_checkpoint(train_weight, config=config,
                                                   model_args=model_args,
                                                   training_args=training_args,
                                                   strict=False)
    else:
        #deepspeed权重 两张方式加载权重
        # 1.权重转换
        # from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
        # convert_zero_checkpoint_to_fp32_state_dict('./best_ckpt/last.ckpt/checkpoint','./best_ckpt/last.ckpt')
        # assert os.path.exists('./best_ckpt/last.ckpt')
        # model = MyTransformer(config=config, model_args=model_args, training_args=training_args)
        # model.load_state_dict(state_dict=torch.load('./best_ckpt/last.ckpt'), strict=False)

        #2. 直接加载权重
        train_weight = './best_ckpt/last.ckpt/checkpoint/mp_rank_00_model_states.pt'
        assert os.path.exists(train_weight)
        model = MyTransformer(config=config, model_args=model_args, training_args=training_args)
        model.load_state_dict(state_dict= torch.load(train_weight), strict=False)




    model.eval()

    base_model: ChatGLMForConditionalGeneration = model.backbone.model
    # 按需修改，目前只支持 4/8 bit 量化
    base_model.half().quantize(4).to(torch.device('cuda:0'))

    with torch.inference_mode():
        response, history = base_model.chat(tokenizer, "写一个诗歌，关于冬天", history=[],max_length=1024)
        print('写一个诗歌，关于冬天',' ',response)

        response, history = base_model.chat(tokenizer, "晚上睡不着应该怎么办", history=[],max_length=1024)
        print('晚上睡不着应该怎么办',' ',response)



