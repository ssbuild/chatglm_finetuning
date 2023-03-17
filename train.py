# -*- coding: utf-8 -*-
import copy
import json
import os
from typing import Any
import numpy as np
import torch
from deep_training.data_helper import ModelArguments, DataArguments, TrainingArguments
from deep_training.nlp.models.chatglm import TransformerChatGlmLMHeadModel, ChatGLMConfig, setup_model_profile
from deep_training.utils.trainer import SimpleModelCheckpoint,ModelCheckpoint,Callback
from pytorch_lightning  import Trainer
from pytorch_lightning.strategies import DeepSpeedStrategy
from transformers import HfArgumentParser

from data_utils import NN_DataHelper, train_info_args, preprocess, postprocess,get_deepspeed_config
from tokenization_chatglm import ChatGLMTokenizer


class MyTransformer(TransformerChatGlmLMHeadModel, with_pl=True):
    def __init__(self, *args, **kwargs):
        super(MyTransformer, self).__init__(*args, **kwargs)




if __name__ == '__main__':


    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    model_args, training_args, data_args = parser.parse_dict(train_info_args)

    # 并行
    setup_model_profile()
    # 保存最小loss模型
    checkpoint_callback = ModelCheckpoint('./best_ckpt',monitor='loss',
                                          save_weights_only=False,
                                          save_last=True,
                                          save_top_k=1,
                                          # every_n_train_steps=1000,
                                          every_n_epochs=1)

    deepspeed_config = get_deepspeed_config()
    strategy = 'ddp' if torch.cuda.device_count() > 1 else None
    if deepspeed_config is not None and len(deepspeed_config):
        strategy = DeepSpeedStrategy(config=deepspeed_config)

    trainer = Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=training_args.max_epochs,
        max_steps=training_args.max_steps,
        accelerator="gpu",replace_sampler_ddp=False,
        devices=data_args.devices,
        enable_progress_bar=True,
        default_root_dir=data_args.output_dir,
        gradient_clip_val=training_args.max_grad_norm,
        accumulate_grad_batches=training_args.gradient_accumulation_steps,
        num_sanity_val_steps=0,
        strategy=strategy
        # precision=16,#半精度
    )

    dataHelper = NN_DataHelper(model_args, training_args, data_args)

    tokenizer, config, label2id, id2label = dataHelper.load_tokenizer_and_config(tokenizer_class_name=ChatGLMTokenizer,
                                                                                 config_class_name=ChatGLMConfig)
    ChatGLMConfig.save_pretrained('best_ckpt')

    config.precision = 16


    # 缓存数据集
    if data_args.do_train:
        dataHelper.make_dataset_with_args(data_args.train_file,mixed_data=False,shuffle=True,mode='train')
    if data_args.do_eval:
        dataHelper.make_dataset_with_args(data_args.eval_file, mode='eval')
    if data_args.do_test:
        dataHelper.make_dataset_with_args(data_args.test_file,mode='test')


    model = MyTransformer(config=config, model_args=model_args, training_args=training_args)

    ckpt_path = './best_ckpt/best.pt'
    if not data_args.convert_onnx:
        # if os.path.exists(ckpt_path):
        #     # 加载权重继续训练
        #     model = MyTransformer.load_from_checkpoint(ckpt_path, config=config,
        #                                                model_args=model_args,
        #                                                training_args=training_args)

        train_datasets = dataHelper.load_random_sampler(dataHelper.train_files,
                                                        with_load_memory=True,
                                                        collate_fn=dataHelper.collate_fn,
                                                        batch_size=training_args.train_batch_size,
                                                        shuffle=True,infinite=True,num_processes=trainer.world_size,process_index=trainer.global_rank)

        if train_datasets is not None:
            trainer.fit(model, train_dataloaders=train_datasets)

    else:
        # 加载权重
        model = MyTransformer.load_from_checkpoint(ckpt_path, config=config,
                                                       model_args=model_args,
                                                       training_args=training_args)
        input_sample = (
            ("input_ids", torch.ones(size=(1, 128), dtype=torch.int32)),
        )
        input_names = ("input_ids",)
        output_names = ("pred_ids",)
        dynamic_axes = None or {"input_ids": [0, 1],
                                "pred_ids": [0, 1]}
        model.convert_to_onnx('./best_ckpt/best.onnx',
                              input_sample=input_sample,
                              input_names=input_names,
                              output_names=output_names,
                              dynamic_axes=dynamic_axes)