# -*- coding: utf-8 -*-
# @Time    : 2023/3/9 15:29
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import torch
from deep_training.data_helper import ModelArguments
from transformers import HfArgumentParser
from data_utils import config_args, NN_DataHelper, get_deepspeed_config
from deep_training.zoo.model_zoo.glm4v.llm_model import MyTransformer,ChatGLM4Tokenizer,setup_model_profile, ChatGLMConfig,PetlArguments

deep_config = get_deepspeed_config()


if __name__ == '__main__':
    config_args['seed'] = None
    config_args['model_name_or_path'] = None

    config_args['seed'] = None
    parser = HfArgumentParser((ModelArguments,))
    (model_args,) = parser.parse_dict(config_args, allow_extra_keys=True)

    setup_model_profile()

    dataHelper = NN_DataHelper(model_args)
    tokenizer: ChatGLM4Tokenizer
    tokenizer, _, _, _ = dataHelper.load_tokenizer_and_config(
        tokenizer_class_name=ChatGLM4Tokenizer, config_class_name=ChatGLMConfig)

    ###################### 注意 选最新权重
    #选择最新的权重 ， 根据时间排序 选最新的
    config = ChatGLMConfig.from_pretrained('../scripts/best_ckpt')
    config.initializer_weight = False
    pl_model = MyTransformer(config=config, model_args=model_args, torch_dtype=torch.float16,)
    if deep_config is None:
        train_weight = '../scripts/best_ckpt/last-v3.ckpt'
    else:
        #使用转换脚本命令 生成 ./best_ckpt/last/best.pt 权重文件
        # cd best_ckpt/last
        # python zero_to_fp32.py . best.pt
        train_weight = '../scripts/best_ckpt/last/best.pt'

    #加载微调权重
    pl_model.load_sft_weight(train_weight,strict=False)

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
        ("图中的狗是什么品种？", "../assets/demo.jpeg"),
        ("这张图片的背景里有什么内容？", "../assets/ghost.jpg"),
    ]
    for (input, image_path) in text_list:
        response, history = model.chat(tokenizer,image_path, input, history=[],max_length=2048,
                                            eos_token_id=config.eos_token_id,
                                            do_sample=True, top_p=0.7, temperature=0.95,)
        print("input",input)
        print("response", response)

