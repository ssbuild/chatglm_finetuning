# -*- coding: utf-8 -*-
# @Time    : 2023/3/9 15:29
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from deep_training.nlp.models.chatglm import setup_model_profile, ChatGLMConfig
from deep_training.nlp.models.lora import LoraArguments
from transformers import HfArgumentParser

from data_utils import train_info_args, NN_DataHelper
from models import MyTransformer,ChatGLMTokenizer


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

    assert lora_args.inference_mode == True and config.pre_seq_len is None

    pl_model = MyTransformer(config=config, model_args=model_args, training_args=training_args,lora_args=lora_args)
    # 加载lora权重
    pl_model.backbone.from_pretrained(pl_model.backbone.model, pretrained_model_name_or_path = './best_ckpt', lora_config = lora_args)

    model = pl_model.get_glm_model()
    # 按需修改
    model.half().cuda()
    model = model.eval()

    response, history = model.chat(tokenizer, "写一个诗歌，关于冬天", history=[],max_length=2048,
                                        eos_token_id=config.eos_token_id,
                                        do_sample=True, top_p=0.7, temperature=0.95,)
    print('写一个诗歌，关于冬天',' ',response)

    response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=[],max_length=2048,
                                        eos_token_id=config.eos_token_id,
                                        do_sample=True, top_p=0.7, temperature=0.95,)
    print('晚上睡不着应该怎么办',' ',response)



