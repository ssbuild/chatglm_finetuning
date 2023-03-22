# -*- coding: utf-8 -*-
import logging
import os
from typing import Dict, Any, Optional

import torch
from deep_training.data_helper import ModelArguments, DataArguments, TrainingArguments
from deep_training.nlp.models.chatglm import TransformerChatGlmLMHeadModel, ChatGLMConfig, setup_model_profile, \
    ChatGLMForConditionalGeneration
from deep_training.nlp.models.lora import LoraArguments, LoraModel
from deep_training.utils.trainer import ModelCheckpoint, SimpleModelCheckpoint


from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DeepSpeedStrategy
from transformers import HfArgumentParser

from data_utils import NN_DataHelper, train_info_args, get_deepspeed_config, preprocess, postprocess
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




class MySimpleModelCheckpoint(SimpleModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super(MySimpleModelCheckpoint, self).__init__(*args, **kwargs)
        lora_args: LoraArguments = self.external_kwargs['lora_args']
        if lora_args.with_lora:
            self.weight_file = './best_ckpt'
            self.last_weight_file = './last_ckpt'

    def load_model_from_ckpt(self):
        model_args = self.external_kwargs['model_args']
        training_args = self.external_kwargs['training_args']
        lora_args = LoraArguments.from_pretrained(self.last_weight_file)
        pl_module = MyTransformer(lora_args=lora_args,
                              config=config,
                              model_args=model_args,
                              training_args=training_args)


        pl_module.backbone.from_pretrained(pl_module.backbone.model,self.last_weight_file)
        return pl_module


    def on_save_model(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:

        lora_args : LoraArguments =  self.external_kwargs['lora_args']
        # 保存权重
        if not lora_args.with_lora:
            super(MySimpleModelCheckpoint, self).on_save_model(trainer, pl_module)
        else:
            monitor_candidates = self._monitor_candidates(trainer)
            monitor_candidates.update(self.on_get_metric(trainer, pl_module))
            val = monitor_candidates.get(self.monitor, None)

            #保存loss最小权重
            if self.update_best(val):
                logging.info('epoch {} ,step {} , save best {}, {}\n'.format(monitor_candidates['epoch'],
                                                                             monitor_candidates['step'],
                                                                             self.best[self.monitor],
                                                                             self.weight_file))
                pl_module.backbone.save_pretrained(self.weight_file)
            #保存最新权重
            pl_module.backbone.save_pretrained(self.last_weight_file)
            # 从最新权重加载模型
            pl_module = self.load_model_from_ckpt()



            
if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, LoraArguments))
    model_args, training_args, data_args, lora_args = parser.parse_dict(train_info_args)

    # 并行
    setup_model_profile()
    deepspeed_config = get_deepspeed_config()

    # 保存最小loss模型
    if lora_args.with_lora:
        assert deepspeed_config is None,ValueError('lora mode does not support deepspeed')
        checkpoint_callback = MySimpleModelCheckpoint(monitor="loss",
                              every_n_epochs = 1,
                              every_n_train_steps=2000 // training_args.gradient_accumulation_steps,
                              #模型参数
                              model_args=model_args,
                              training_args=training_args,
                              lora_args=lora_args,)
    else:
        checkpoint_callback = ModelCheckpoint('./best_ckpt', monitor='loss',
                                              save_weights_only=False,
                                              save_last=True,
                                              save_top_k=1,
                                              # every_n_train_steps=1000,
                                              every_n_epochs=1)


    strategy = 'ddp' if torch.cuda.device_count() > 1 else None
    if deepspeed_config is not None and len(deepspeed_config):
        strategy = DeepSpeedStrategy(config=deepspeed_config,)



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

    tokenizer, config, _,_ = dataHelper.load_tokenizer_and_config(tokenizer_class_name=ChatGLMTokenizer,config_class_name=ChatGLMConfig)

    # 额外参数
    checkpoint_callback.tokenizer = tokenizer
    checkpoint_callback.data_args = data_args

    config.save_pretrained('best_ckpt')

    # 缓存数据集
    if data_args.do_train:
        dataHelper.make_dataset_with_args(data_args.train_file,mixed_data=False,shuffle=True,mode='train')
    if data_args.do_eval:
        dataHelper.make_dataset_with_args(data_args.eval_file, mode='eval')
    if data_args.do_test:
        dataHelper.make_dataset_with_args(data_args.test_file,mode='test')


    model = MyTransformer(config=config, model_args=model_args, training_args=training_args,lora_args=lora_args)

    ckpt_path = './best_ckpt/best.pt'
    if not data_args.convert_onnx:
        # if os.path.exists(ckpt_path):
        #     if not lora_args.with_lora:
        #         # 加载权重继续训练
        #         model = MyTransformer.load_from_checkpoint(ckpt_path, config=config,model_args=model_args,training_args=training_args,lora_args=lora_args)
        #     else:
        #         # 加载lora权重 继续训练  0.0.20版本支持lora 继续训练
        #         model.backbone.from_pretrained(model.backbone.model, ckpt_path)

        #deepspeed 保证整批次
        def dataset_loader_filter_fn(dataset):
            host_num = 1
            limit_count = len(dataset)
            limit_count = int(limit_count // (data_args.devices * training_args.train_batch_size * host_num)) * (data_args.devices * training_args.train_batch_size * host_num)
            return dataset.limit(int(limit_count))

        with_record_iterable_dataset = False
        train_datasets = dataHelper.load_random_sampler(dataHelper.train_files,
                                                        with_load_memory=True,
                                                        collate_fn=dataHelper.collate_fn,
                                                        batch_size=training_args.train_batch_size,
                                                        drop_last=True,#多卡建议扔掉
                                                        shuffle=True,infinite=True,
                                                        num_processes=trainer.world_size,
                                                        process_index=trainer.global_rank,
                                                        with_record_iterable_dataset=with_record_iterable_dataset,
                                                        dataset_loader_filter_fn=dataset_loader_filter_fn if not with_record_iterable_dataset else None)

        if train_datasets is not None:
            trainer.fit(model, train_dataloaders=train_datasets)

    else:
        if not lora_args.with_lora:
            # 加载权重
            model = MyTransformer.load_from_checkpoint(ckpt_path, config=config,
                                                       model_args=model_args,
                                                       training_args=training_args,
                                                       lora_args=lora_args)
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
        else:
            # 加载权重
            lora_args = LoraArguments.from_pretrained('./best_ckpt')
            pl_module = MyTransformer(lora_args=lora_args,
                                      config=config,
                                      model_args=model_args,
                                      training_args=training_args)
            # 二次加载权重
            pl_module.backbone.from_pretrained(pl_module.backbone.model, './best_ckpt')

            model_: ChatGLMForConditionalGeneration
            model_ = pl_module.backbone.model.model