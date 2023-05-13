# -*- coding: utf-8 -*-
# @Time    : 2023/3/9 15:29
import os
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from transformers import HfArgumentParser
from data_utils import train_info_args, NN_DataHelper
from models import MyTransformer,ChatGLMTokenizer,load_in_8bit,setup_model_profile, ChatGLMConfig,LoraArguments


if __name__ == '__main__':
    train_info_args['seed'] = None
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, LoraArguments))
    model_args, training_args, data_args, _ = parser.parse_dict(train_info_args)

    setup_model_profile()

    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer: ChatGLMTokenizer
    tokenizer, _, _, _ = dataHelper.load_tokenizer_and_config(
        tokenizer_class_name=ChatGLMTokenizer, config_class_name=ChatGLMConfig)


    ckpt_dir = './best_ckpt'
    config = ChatGLMConfig.from_pretrained(ckpt_dir)
    config.initializer_weight = False
    lora_args = LoraArguments.from_pretrained(ckpt_dir)
    assert lora_args.inference_mode == True and config.pre_seq_len is None
    pl_model = MyTransformer(config=config, model_args=model_args, training_args=training_args,lora_args=lora_args,
                             load_in_8bit=load_in_8bit,device_map="auto")
    # 加载lora权重
    pl_model.backbone.from_pretrained(pl_model.backbone.model, pretrained_model_name_or_path = ckpt_dir, lora_config = lora_args)
    if load_in_8bit:
        pl_model.eval().cuda()
    else:
        pl_model.eval().half().cuda()

    enable_merge_weight = False
    if enable_merge_weight:

        # 合并lora 权重 保存
        pl_model.save_pretrained_merge_lora(os.path.join(ckpt_dir,'pytorch_model_merge.bin'))
    else:
        model = pl_model.get_glm_model()

        response, history = model.chat(tokenizer, "写一个诗歌，关于冬天", history=[],max_length=2048,
                                            eos_token_id=config.eos_token_id,
                                            do_sample=True, top_p=0.7, temperature=0.95,)
        print('写一个诗歌，关于冬天',' ',response)

        response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=[],max_length=2048,
                                            eos_token_id=config.eos_token_id,
                                            do_sample=True, top_p=0.7, temperature=0.95,)
        print('晚上睡不着应该怎么办',' ',response)



