# @Time    : 2023/3/28 21:56
# @Author  : tk

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


    def get_glm_model(self) -> MyChatGLMForConditionalGeneration:
        if self.lora_args is not None and self.lora_args.with_lora:
            return self.backbone.model.model
        return self.backbone.model

    def save_hf_pretrained(self,save_directory):
        model = self.get_glm_model()
        model.config.save_pretrained(save_directory)
        model.save_pretrained(save_directory)

    def save_pretrained_merge_lora(self,weight_path_file: str):
        assert not load_in_8bit , ValueError('load_in_8bit is not support merge')
        assert os.path.exists(os.path.dirname(weight_path_file))
        assert self.lora_args is not None and self.lora_args.with_lora
        lora_model : LoraModel = self.backbone
        model: nn.Module = lora_model.merge_and_unload()
        #保存hf权重，可用infer.py推理
        torch.save(model.model.state_dict(),weight_path_file)
        return model

    def save_pretrained_merge_lora_and_restore(self, weight_path_file: str):
        assert not load_in_8bit, ValueError('load_in_8bit is not support merge')
        assert os.path.exists(os.path.dirname(weight_path_file))
        assert self.lora_args is not None and self.lora_args.with_lora
        lora_model: LoraModel = self.backbone
        lora_model.merge_adapter()
        # 保存hf权重，可用infer.py推理
        torch.save(lora_model.model.model.state_dict(), weight_path_file)
        lora_model.unmerge_adapter()
