# @Time    : 2023/3/28 21:56
# @Author  : tk

import torch
from deep_training.nlp.models.chatglm import ChatGLMForConditionalGeneration, logger
from deep_training.nlp.models.lora import LoraArguments, LoraModel
from deep_training.nlp.models.transformer import TransformerBase

from tokenization_chatglm import ChatGLMTokenizer

load_pretrain_weight_int4 = False

# 加载量化后的权重需调用此方法
def quantize_variable_weight(self, bits: int, quantize_embeddings=False, use_quantization_cache=False, empty_init=False, **kwargs):
    if bits == 0:
        return
    from chatglm_6b_int4.quantization import quantize, QuantizedEmbedding, QuantizedLinear, load_cpu_kernel
    if self.quantized:
        if self.device == torch.device("cpu"):
            logger.info("Already quantized, reloading cpu kernel.")
            load_cpu_kernel(**kwargs)
        else:
            logger.info("Already quantized.")
        return self

    self.quantized = True

    self.config.quantization_bit = bits
    self.config.quantization_embeddings = quantize_embeddings

    self.transformer = quantize(self.transformer, bits, use_quantization_cache=use_quantization_cache,
                                empty_init=empty_init, **kwargs)

    if quantize_embeddings:
        logger.info("Applying quantization to embeddings")
        self.transformer.word_embeddings = QuantizedEmbedding(
            weight_bit_width=bits,
            weight_tensor=self.transformer.word_embeddings.weight.to(self.device),
            num_embeddings=self.transformer.word_embeddings.num_embeddings,
            embedding_dim=self.transformer.word_embeddings.embedding_dim,
            dtype=torch.half,
            device=self.transformer.word_embeddings.weight.device,
        )
        self.lm_head = QuantizedLinear(
            weight_bit_width=bits,
            weight_tensor=self.lm_head.weight.to(self.device),
            bias_tensor=None,
            in_features=self.lm_head.in_features,
            out_features=self.lm_head.out_features,
            bias=False,
            quantized_weight=self.transformer.word_embeddings.weight,
            quantized_weight_scale=self.transformer.word_embeddings.weight_scale,
            dtype=torch.half,
            device=self.lm_head.weight.device,
        )

    return self


class MyChatGLMForConditionalGeneration(ChatGLMForConditionalGeneration):
    def __init__(self,*args,**kwargs):
        super(MyChatGLMForConditionalGeneration, self).__init__(*args)

        self.is_quantize_weight = False
        # 加载int权重 ， 推理模型
        if load_pretrain_weight_int4:
            self.is_quantize_weight = True
            self.quantized = False

            quantization_bit = 4
            if quantization_bit:
                quantize_variable_weight(self,
                                        bits=quantization_bit, # 4 or 8 bit
                                        quantization_embeddings=False,
                                        use_quantization_cache=True,
                                        empty_init=True)


class MyTransformerChatGlmLMHeadModel(TransformerBase):
    def __init__(self, *args,**kwargs):
        super(MyTransformerChatGlmLMHeadModel, self).__init__(*args,**kwargs)
        self.set_model(self.from_pretrained(MyChatGLMForConditionalGeneration, *args, **kwargs))


class MyTransformer(MyTransformerChatGlmLMHeadModel, with_pl=True):
    def __init__(self, *args, **kwargs):
        lora_args: LoraArguments = kwargs.pop('lora_args',None)
        super(MyTransformer, self).__init__(*args, **kwargs)
        self.lora_args = lora_args


        # 非 lora模式冻结示例
        # assert lora_args.with_lora = False
        # need_frozen_list = []
        # M: nn.Module = self.backbone
        # for param in M.named_parameters():
        #     if param[0] in need_frozen_list:
        #         param[1].requires_grad = False

        if lora_args is not None and lora_args.with_lora:
            model = LoraModel(self.backbone, lora_args)
            print('*' * 30,'lora info')
            model.print_trainable_parameters()
            self.set_model(model, copy_attr=False)


    def get_glm_model(self) -> MyChatGLMForConditionalGeneration:
        if self.lora_args is not None and self.lora_args.with_lora:
            return self.backbone.model.model
        return self.backbone.model