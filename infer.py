# -*- coding: utf-8 -*-
# @Time    : 2023/3/9 15:29
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from deep_training.nlp.models.chatglm import setup_model_profile, ChatGLMConfig, ChatGLMForConditionalGeneration
from deep_training.nlp.models.lora import LoraArguments
from transformers import HfArgumentParser

from data_utils import train_info_args, NN_DataHelper
from models import MyTransformer
from tokenization_chatglm import ChatGLMTokenizer

if __name__ == '__main__':
    train_info_args['seed'] = None
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, LoraArguments))
    model_args, training_args, data_args, _ = parser.parse_dict(train_info_args)

    setup_model_profile()

    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer: ChatGLMTokenizer
    tokenizer, config, _,_ = dataHelper.load_tokenizer_and_config(
        tokenizer_class_name=ChatGLMTokenizer, config_class_name=ChatGLMConfig)

    # 官方28层
    config.precision = None
    config.num_layers = 28
    config.initializer_weight = False
    config.eos_token_id = 150005
    pl_model = MyTransformer(config=config, model_args=model_args, training_args=training_args)


    model = pl_model.get_glm_model()
    # 按需修改，目前只支持 4/8 bit 量化
    model.half().quantize(4).cuda()
    model = model.eval()

    # 注意 长度不等于2048 会影响效果
    response, history = model.chat(tokenizer, "你好", history=[],max_length=2048)
    print('你好',' ',response)

    response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history,max_length=2048)
    print('晚上睡不着应该怎么办',' ',response)

    # response, history = base_model.chat(tokenizer, "写一个诗歌，关于冬天", history=[],max_length=30)
    # print('写一个诗歌，关于冬天',' ',response)

