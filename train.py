# -*- coding: utf-8 -*-
import logging
import os
import torch
from deep_training.data_helper import ModelArguments, DataArguments, TrainingArguments
from deep_training.trainer.pl.modelcheckpoint import ModelCheckpointEx
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.strategies import DeepSpeedStrategy
from transformers import HfArgumentParser
from data_utils import NN_DataHelper, train_info_args, get_deepspeed_config,global_args
from aigc_zoo.model_zoo.chatglm.llm_model import MyTransformer, ChatGLMTokenizer,EffiArguments,ChatGLMConfig, setup_model_profile

            
if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, EffiArguments))
    model_args, training_args, data_args, lora_args = parser.parse_dict(train_info_args)
    lora_args = lora_args.config

    setup_model_profile()

    output_weight_dir = './best_ckpt'

    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    config_kwargs = {"pre_seq_len": global_args["pre_seq_len"],
                     "prefix_projection": global_args["pre_seq_len"]}
    if global_args["num_layers"] > 0:
        config_kwargs["num_layers"] = global_args["num_layers"]
    tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config(tokenizer_class_name=ChatGLMTokenizer,
                                                                   config_class_name=ChatGLMConfig,config_kwargs=config_kwargs)
    assert tokenizer.eos_token_id == 130005


    if config.quantization_bit != 0 and lora_args is not None:
        raise AssertionError("quantization lora not support")

    if config.pre_seq_len is not None and lora_args is not None:
        raise ValueError('with lora and ptuning v2 cannot open at the same time')

    dataHelper.make_dataset_all()

    deepspeed_config = get_deepspeed_config()
    strategy = 'ddp' if torch.cuda.device_count() > 1 else 'auto'
    if deepspeed_config is not None and len(deepspeed_config):
        strategy = DeepSpeedStrategy(config=deepspeed_config, )

    is_bf16_supported = torch.cuda.is_bf16_supported()
    checkpoint_callback = ModelCheckpointEx(
        # monitor='loss',
        dirpath=output_weight_dir,
        save_weights_only=True,
        save_last=True,
        save_top_k=1,
        # every_n_train_steps=2000 // training_args.gradient_accumulation_steps,
        every_n_epochs=1,
        lora_args=lora_args,
        # monitor="loss"，mode = "min", save_top_k = 10 按loss存储10个模型
        # monitor="step"，mode = "max", save_top_k = 10 按步存储最后10个模型
    )

    # 精度 根据实际情况做调整
    if config.quantization_bit != 0 and not config.pre_seq_len:
        precision = '32'
    elif is_bf16_supported:
        precision = 'bf16'
    else:
        precision = '16'

    if global_args["quantization_config"] is not None and global_args["quantization_config"].load_in_8bit:
        precision = "32"

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
        strategy=strategy,
        precision= precision , #  #可以自行尝试  "32": "32-true", "16": "16-mixed", "bf16": "bf16-mixed"
    )


    pl_model = MyTransformer(config=config, model_args=model_args, training_args=training_args, lora_args=lora_args,
                             num_layers_freeze=global_args["num_layers_freeze"],#
                             quantization_config=global_args["quantization_config"],
                             device_map={"": trainer.local_rank} if trainer.world_size > 1 else "auto",
                             torch_dtype=torch.float16,
                             new_num_tokens=len(tokenizer),  # 可能扩充词 , 还有一些隐藏token, 如果不需要可自行注释
                             )

    config.save_pretrained(output_weight_dir)

    #恢复权重继续训练
    # pl_model.load_sft_weight('./best_ckpt/best.pt',is_trainable=True)

    if config.pre_seq_len is not None:
        # P-tuning v2
        pl_model.get_llm_model().half()
        pl_model.get_llm_model().transformer.prefix_encoder.float()
    else:
        if config.quantization_bit != 0:
            pl_model = pl_model.half() if not is_bf16_supported else pl_model.bfloat16()
        else:
            # Finetune
            pl_model = pl_model.float() if not is_bf16_supported else pl_model.bfloat16()


    def dataset_loader_filter_fn(dataset):
        print('*' * 30, 'total', len(dataset))
        return dataset


    train_datasets = dataHelper.load_distributed_random_sampler(
        dataHelper.train_files,
        with_load_memory=data_args.data_backend == 'record',
        collate_fn=dataHelper.collate_fn,
        batch_size=training_args.train_batch_size,
        drop_last=True,  # 多卡建议扔掉
        num_processes=trainer.world_size, process_index=trainer.global_rank,
        dataset_loader_filter_fn=dataset_loader_filter_fn,
        num_workers=0
    )

    if train_datasets is not None:
        trainer.fit(pl_model, train_dataloaders=train_datasets)

