# -*- coding: utf-8 -*-
import logging

import torch
from deep_training.data_helper import ModelArguments, DataArguments, TrainingArguments
from deep_training.nlp.models.chatglm import ChatGLMConfig, setup_model_profile
from deep_training.nlp.models.lora.v2 import LoraArguments
from deep_training.utils.trainer import ModelCheckpoint, SimpleModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.strategies import DeepSpeedStrategy
from transformers import HfArgumentParser

from data_utils import NN_DataHelper, train_info_args, get_deepspeed_config
from models import MyTransformer,ChatGLMTokenizer


class MySimpleModelCheckpoint(SimpleModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super(MySimpleModelCheckpoint, self).__init__(*args, **kwargs)
        lora_args: LoraArguments = self.external_kwargs['lora_args']
        if lora_args:
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
        if lora_args is None:
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
            # # 从最新权重加载模型
            # pl_module = self.load_model_from_ckpt()



            
if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, LoraArguments))
    model_args, training_args, data_args, lora_args = parser.parse_dict(train_info_args)
    lora_args = lora_args.config
    #
    setup_model_profile()
    deepspeed_config = get_deepspeed_config()

    # 保存最小loss模型
    if lora_args:
        assert deepspeed_config is None,ValueError('lora mode does not support deepspeed')
        checkpoint_callback = MySimpleModelCheckpoint(
                              # monitor="loss",
                              every_n_epochs = 1,
                              every_n_train_steps=2000 // training_args.gradient_accumulation_steps,
                              #模型参数
                              model_args=model_args,
                              training_args=training_args,
                              lora_args=lora_args,)
    else:
        checkpoint_callback = ModelCheckpoint('./best_ckpt',
                                              # monitor='loss',
                                              save_weights_only=False,
                                              save_last=True,
                                              save_top_k=1,
                                              # every_n_train_steps=1000,
                                              every_n_epochs=1)


    strategy = 'ddp' if torch.cuda.device_count() > 1 else 'auto'
    if deepspeed_config is not None and len(deepspeed_config):
        strategy = DeepSpeedStrategy(config=deepspeed_config,)



    trainer = Trainer(
        callbacks=[checkpoint_callback,LearningRateMonitor(logging_interval='step')],
        max_epochs=training_args.max_epochs,
        max_steps=training_args.max_steps,
        accelerator="gpu",
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
    config.eos_token_id = 130005

    if config.pre_seq_len is not None and lora_args is not None:
        raise ValueError('with lora and ptuning v2 cannot open at the same time')

    if config.pre_seq_len is not None:
        if config.quantization_bit:
            raise Exception('量化模型不支持微调训练')

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


    pl_model = MyTransformer(config=config, model_args=model_args, training_args=training_args,lora_args=lora_args)



    ckpt_path = './best_ckpt/best.pt'
    if not data_args.convert_onnx:
        #  只恢复权重 ， 不恢复步数和优化器 ，
        #  如果想恢复步数， 修改 trainer.fit(pl_model, train_dataloaders=train_datasets，ckpt=ckpt_path)  注lora 当前不支持恢复步数。
        # if os.path.exists(ckpt_path):
        #     if not lora_args.with_lora:
        #         # 加载权重继续训练
        #         pl_model = MyTransformer.load_from_checkpoint(ckpt_path, config=config,model_args=model_args,training_args=training_args,lora_args=lora_args,strict=False)
        #     else:
        #         # 加载lora权重 继续训练  0.0.20版本支持lora 继续训练
        #         pl_model.backbone.from_pretrained(pl_model.backbone.model, pretrained_model_name_or_path=ckpt_path,lora_config=lora_args,strict=False)

        def dataset_loader_filter_fn(dataset):
            print('*' * 30,'total',len(dataset))
            return dataset
        train_datasets = dataHelper.load_distributed_random_sampler(
            dataHelper.train_files,
            with_load_memory=True,
            collate_fn=dataHelper.collate_fn,
            batch_size=training_args.train_batch_size,
            drop_last=True,#多卡建议扔掉
            num_processes=trainer.world_size, process_index=trainer.global_rank,
            dataset_loader_filter_fn=dataset_loader_filter_fn
        )

        if train_datasets is not None:
            trainer.fit(pl_model, train_dataloaders=train_datasets)

    else:
        if lora_args is not None:
            # 加载权重
            pl_model = MyTransformer.load_from_checkpoint(ckpt_path, config=config,
                                                       model_args=model_args,
                                                       training_args=training_args,
                                                       lora_args=lora_args,strict=False)
            # input_sample = (
            #     ("input_ids", torch.ones(size=(1, 128), dtype=torch.int32)),
            #     ("attention_mask", torch.ones(size=(1, 1,128,128), dtype=torch.int32)),
            #     ("position_ids", torch.ones(size=(1, 2, 128), dtype=torch.int32)),
            # )
            # input_names = ("input_ids",'attention_mask','position_ids')
            # output_names = ("pred_ids",)
            # dynamic_axes = None or {"input_ids": [0, 1],
            #                         "attention_mask": [0, 0,1,1],
            #                         "position_ids": [0, 0,1],
            #                         "pred_ids": [0, 1]}
            # pl_module.convert_to_onnx('./best_ckpt/best.onnx',
            #                       input_sample=input_sample,
            #                       input_names=input_names,
            #                       output_names=output_names,
            #                       dynamic_axes=dynamic_axes)

            model = pl_model.get_glm_model()
            #保存huggingface model
            model.save_pretrained('huggingface_model',max_shard_size='10GB')
        else:
            # 加载权重
            lora_args = LoraArguments.from_pretrained('./best_ckpt')
            pl_module = MyTransformer(lora_args=lora_args,
                                      config=config,
                                      model_args=model_args,
                                      training_args=training_args)
            # 二次加载权重
            pl_module.backbone.from_pretrained(pl_module.backbone.model, pretrained_model_name_or_path='./best_ckpt',lora_config=lora_args)

            model = pl_model.get_glm_model()