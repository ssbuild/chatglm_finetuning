# @Time    : 2023/3/28 21:56
# @Author  : tk
from transformers import PretrainedConfig

from models.chatglm_model import *

class MyTransformer(MyTransformerChatGlmLMHeadModel, with_pl=True):
    def __init__(self, *args, **kwargs):
        lora_args: LoraArguments = kwargs.pop('lora_args',None)
        super(MyTransformer, self).__init__(*args, **kwargs)
        self.lora_args = lora_args

        if lora_args is not None and lora_args.with_lora:
            model = LoraModel(self.backbone, lora_args)
            print('*' * 30,'lora info')
            model.print_trainable_parameters()
            self.set_model(model, copy_attr=False)
        elif global_num_layers_freeze > 0 and self.config.pre_seq_len is None:  # 非 lora freeze 非 ptuning模式
            M: nn.Module = self.backbone
            for param in M.named_parameters():
                result = re.match(re.compile('.*transformer.layers.(\\d+)'),param[0])
                if result is not None:
                    n_layer = int(result.group(1))
                    if n_layer < global_num_layers_freeze:
                        param[1].requires_grad = False
                        print('freeze layer',param[0])


    def get_llm_model(self) -> MyChatGLMForConditionalGeneration:
        if self.lora_args is not None and self.lora_args.with_lora:
            return self.backbone.model.model
        return self.backbone.model

    def save_pretrained_merge_lora(self,sft_weight_path: str):
        assert not global_load_in_8bit , ValueError('load_in_8bit is not support merge')
        assert os.path.exists(os.path.dirname(sft_weight_path))
        assert self.lora_args is not None and self.lora_args.with_lora
        lora_model : LoraModel = self.backbone
        model: nn.Module = lora_model.merge_and_unload()
        #保存hf权重，可用infer.py推理
        # torch.save(model.model.state_dict(),weight_path_file)
        model.model.save_pretrained(sft_weight_path)
        return model

    def save_pretrained_merge_lora_and_restore(self, sft_weight_path: str):
        assert not global_load_in_8bit, ValueError('load_in_8bit is not support merge')
        assert os.path.exists(os.path.dirname(sft_weight_path))
        assert self.lora_args is not None and self.lora_args.with_lora
        lora_model: LoraModel = self.backbone
        lora_model.merge_adapter()
        # 保存hf权重，可用infer.py推理
        #torch.save(lora_model.model.model.state_dict(), weight_path_file)
        lora_model.model.model.save_pretrained(sft_weight_path)
        lora_model.unmerge_adapter()

    def load_sft_weight(self, sft_weight_path: str, is_trainable=False, strict=False):
        if self.lora_args is not None and self.lora_args.with_lora:
            # 加载lora权重
            self.backbone.from_pretrained(self.backbone.model, pretrained_model_name_or_path=sft_weight_path,
                                          is_trainable=is_trainable)
        else:
            # 加载sft 或者 p-tuning-v2权重
            self.get_llm_model().load_state_dict(torch.load(sft_weight_path), strict=strict)

    def save_sft_weight(self,sft_weight_path, merge_lora_weight=False):
        if self.lora_args is not None and self.lora_args.with_lora:
            if merge_lora_weight:
                # lora 合并权重 转换 hf权重
                self.save_pretrained_merge_lora(sft_weight_path)
            else:
                #只保存 lora 权重
                self.backbone.save_pretrained(sft_weight_path)
        else:
            config: PretrainedConfig = self.model.config
            if config.pre_seq_len is not None and config.pre_seq_len > 0:
                # 保存sft p-tuning-v2 权重
                torch.save(self.get_llm_model().state_dict(), sft_weight_path)
            else:
                #保存hf权重
                config.save_pretrained(sft_weight_path)
                self.get_llm_model().save_pretrained(sft_weight_path)

