# -*- coding: utf-8 -*-
# @Time    : 2023/3/9 15:29
import os
import re
from collections import OrderedDict

import torch
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from deep_training.nlp.models.chatglm import setup_model_profile, ChatGLMConfig
from deep_training.nlp.models.lora import LoraArguments
from transformers import HfArgumentParser

from data_utils import train_info_args, NN_DataHelper, get_deepspeed_config
from models import MyTransformer,ChatGLMTokenizer

deep_config = get_deepspeed_config()


if __name__ == '__main__':
    train_info_args['seed'] = None
    train_info_args['model_name_or_path'] = None

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
    config.initializer_weight = False


    if deep_config is None:
        train_weight = './best_ckpt/last-v3.ckpt'
        assert os.path.exists(train_weight)
        pl_model = MyTransformer.load_from_checkpoint(train_weight, config=config,model_args=model_args,
                                                   training_args=training_args,strict=False)
    else:

        #建议直接使用转换脚本命令 支持 deepspeed stage 0,1,2,3， 生成 ./best_ckpt/last.ckpt/best.pt 权重文件
        # cd best_ckpt/last.ckpt
        # python zero_to_fp32.py . best.pt
        train_weight = './best_ckpt/last.ckpt/best.pt'

        #deepspeed stage 0,1,2 不必须执行上面命令
        #train_weight = './best_ckpt/last.ckpt/checkpoint/mp_rank_00_model_states.pt'

        assert os.path.exists(train_weight)
        weights_dict = torch.load(train_weight)
        weights_dict_new = OrderedDict()
        for k,v in (weights_dict['module'] if 'module' in weights_dict else weights_dict).items():
            weights_dict_new[re.sub(r'_forward_module\.', '', k)] = v
        pl_model = MyTransformer(config=config, model_args=model_args, training_args=training_args)
        pl_model.load_state_dict(state_dict= weights_dict_new, strict=False)

    model = pl_model.get_glm_model()

    if not model.quantized:
        # 按需修改，目前只支持 4/8 bit 量化 ， 可以保存量化模型
        model.half().quantize(4).cuda()
    else:
        #已经量化，已经保存微调后的量化模型可以 直接加载
        model.half().cuda()
    model = model.eval()


    #注意 长度不等于2048 会影响效果
    response, history = model.chat(tokenizer, "写一个诗歌，关于冬天", history=[],max_length=2048,
                                        eos_token_id=config.eos_token_id,
                                        do_sample=True, top_p=0.7, temperature=0.95,)
    print('写一个诗歌，关于冬天',' ',response)

    response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=[],max_length=2048,
                                        eos_token_id=config.eos_token_id,
                                        do_sample=True, top_p=0.7, temperature=0.95,)
    print('晚上睡不着应该怎么办',' ',response)
