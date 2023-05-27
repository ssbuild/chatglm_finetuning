# @Time    : 2023/3/28 21:56
# @Author  : tk
from collections import OrderedDict
from transformers import PretrainedConfig, PreTrainedModel
from models.chatglm_model import *



class SftWeightMinMax:

    def save_pretrained_merge_lora(self,sft_weight_path: str):
        assert os.path.exists(os.path.dirname(sft_weight_path))
        assert self.lora_args is not None and self.lora_args.with_lora
        lora_model : LoraModel = self.backbone
        model: nn.Module = lora_model.merge_and_unload()
        #保存hf权重，可用infer.py推理
        # torch.save(model.model.state_dict(),weight_path_file)
        model.model.save_pretrained(sft_weight_path)
        return model

    def save_pretrained_merge_lora_and_restore(self, sft_weight_path: str):
        assert os.path.exists(os.path.dirname(sft_weight_path))
        assert self.lora_args is not None and self.lora_args.with_lora
        lora_model: LoraModel = self.backbone
        lora_model.merge_adapter()
        # 保存hf权重，可用infer.py推理
        #torch.save(lora_model.model.model.state_dict(), weight_path_file)
        lora_model.model.model.save_pretrained(sft_weight_path)
        lora_model.unmerge_adapter()

    def load_sft_weight(self, sft_weight_path: str, is_trainable=False, strict=False):
        assert os.path.exists(sft_weight_path)
        if self.lora_args is not None and self.lora_args.with_lora:
            # 恢复权重
            self.backbone: LoraModel
            self.backbone.load_adapter(sft_weight_path, adapter_name="default", is_trainable=is_trainable)
        else:
            weight_dict = torch.load(sft_weight_path)
            weights_dict_new = OrderedDict()
            valid_keys = ['module','state_dict']
            for k in valid_keys:
                if k in weight_dict:
                    weight_dict = weight_dict[k]
                    break
            for k, v in weight_dict.items():
                k = re.sub(r'_forward_module\.', '', k)
                rm_key = '_TransformerLightningModule__backbone'
                if k.startswith(rm_key):
                    base_model_prefix = self.backbone.base_model_prefix
                    k = re.sub(r'{}.{}.'.format(rm_key, base_model_prefix), '', k)
                weights_dict_new[k] = v
            # 加载sft 或者 p-tuning-v2权重
            self.get_llm_model().load_state_dict(weights_dict_new, strict=strict)

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

class MyTransformer(MyTransformerChatGlmLMHeadModel,SftWeightMinMax, with_pl=True):
    def __init__(self, *args, **kwargs):
        lora_args: LoraArguments = kwargs.pop('lora_args',None)
        num_layers_freeze = kwargs.pop('num_layers_freeze',-1)
        super(MyTransformer, self).__init__(*args, **kwargs)
        self.lora_args = lora_args


        if lora_args is not None and lora_args.with_lora:
            self.backbone.enable_input_require_grads()
            model = LoraModel(self.backbone, lora_args)
            print('*' * 30,'lora info')
            model.print_trainable_parameters()
            self.set_model(model, copy_attr=False)

            # for name, module in model.named_modules():
            #     if isinstance(module, LoraLayer):
            #         module = module.to(torch.bfloat16)
            #     if 'norm' in name:
            #         module = module.to(torch.float32)
            #     if 'lm_head' in name or 'embed_tokens' in name:
            #         if hasattr(module, 'weight'):
            #             if module.weight.dtype == torch.float32:
            #                 module = module.to(torch.bfloat16)

        elif num_layers_freeze > 0 and self.config.pre_seq_len is None:  # 非 lora freeze 非 ptuning模式
            M: nn.Module = self.backbone
            for param in M.named_parameters():
                result = re.match(re.compile('.*transformer.layers.(\\d+)'),param[0])
                if result is not None:
                    n_layer = int(result.group(1))
                    if n_layer < num_layers_freeze:
                        param[1].requires_grad = False
                        print('freeze layer',param[0])

    def get_model_lr(self, model=None, lr=None):
        # for n, p in self.named_parameters():
        #     print(n, p.requires_grad)
        lr = lr if lr is not None else self.config.task_specific_params['learning_rate']
        if self.lora_args is not None and self.lora_args.with_lora:
            return [(self.backbone, lr)]
        return super(MyTransformer, self).get_model_lr(model, lr)

    def get_llm_model(self) -> MyChatGLMForConditionalGeneration:
        if self.lora_args is not None and self.lora_args.with_lora:
            return self.backbone.model.model
        return self.backbone.model

