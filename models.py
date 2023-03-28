# @Time    : 2023/3/28 21:56
# @Author  : tk

from deep_training.nlp.models.chatglm import TransformerChatGlmLMHeadModel, ChatGLMForConditionalGeneration
from deep_training.nlp.models.lora import LoraArguments, LoraModel
from data_utils import ignore_mismatched_sizes

class MyTransformer(TransformerChatGlmLMHeadModel, with_pl=True):
    def __init__(self, *args, **kwargs):
        lora_args: LoraArguments = kwargs.pop('lora_args',None)
        # int4 模型
        if 'ignore_mismatched_sizes' not in kwargs:
            kwargs['ignore_mismatched_sizes'] = ignore_mismatched_sizes

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


    def get_glm_model(self) -> ChatGLMForConditionalGeneration:
        if self.lora_args is not None and self.lora_args.with_lora:
            return self.backbone.model.model
        return self.backbone.model

