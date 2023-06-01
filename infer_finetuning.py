# -*- coding: utf-8 -*-
# @Time    : 2023/3/9 15:29
import torch
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from transformers import HfArgumentParser
from data_utils import train_info_args, NN_DataHelper, get_deepspeed_config
from models import MyTransformer,ChatGLMTokenizer,setup_model_profile, ChatGLMConfig,LoraArguments

deep_config = get_deepspeed_config()


if __name__ == '__main__':
    train_info_args['seed'] = None
    train_info_args['model_name_or_path'] = None

    parser = HfArgumentParser((ModelArguments,  DataArguments))
    model_args,data_args= parser.parse_dict(train_info_args,allow_extra_keys=True)

    setup_model_profile()

    dataHelper = NN_DataHelper(model_args, None, data_args)
    tokenizer: ChatGLMTokenizer
    tokenizer, _, _, _ = dataHelper.load_tokenizer_and_config(
        tokenizer_class_name=ChatGLMTokenizer, config_class_name=ChatGLMConfig)

    ###################### 注意 选最新权重
    #选择最新的权重 ， 根据时间排序 选最新的
    config = ChatGLMConfig.from_pretrained('./best_ckpt')
    config.initializer_weight = False
    pl_model = MyTransformer(config=config, model_args=model_args)
    if deep_config is None:
        train_weight = './best_ckpt/last-v3.ckpt'
    else:
        #使用转换脚本命令 生成 ./best_ckpt/last.ckpt/best.pt 权重文件
        # cd best_ckpt/last.ckpt
        # python zero_to_fp32.py . best.pt
        train_weight = './best_ckpt/last/best.pt'

    #加载微调权重
    pl_model.load_sft_weight(train_weight,strict=True)

    model = pl_model.get_llm_model()
    #保存hf权重
    #config.save_pretrained('convert/')

    # 保存sft p-tuning-v2 权重
    #  pl_model.save_sft_weight('convert/pytorch_model_sft_ptv2.bin')

    #保存sft权重
    # pl_model.save_sft_weight('convert/pytorch_model_sft.bin')



    if not model.quantized:
        # 按需修改，目前只支持 4/8 bit 量化 ， 可以保存量化模型
        model.half().quantize(4).cuda()
    else:
        #已经量化，已经保存微调后的量化模型可以 直接加载
        model.half().cuda()
    model = model.eval()

    text_list = [
        "写一个诗歌，关于冬天",
        "晚上睡不着应该怎么办",
    ]
    for input in text_list:
        response, history = model.chat(tokenizer, input, history=[],max_length=2048,
                                            eos_token_id=config.eos_token_id,
                                            do_sample=True, top_p=0.7, temperature=0.95,)
        print("input",input)
        print("response", response)

