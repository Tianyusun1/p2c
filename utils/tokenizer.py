# /home/610-sty/STY_T2i_v4.4/utils/tokenizer.py
from transformers import AutoTokenizer


class ChineseTokenizer:
    # ✅ 修改点：将默认路径指向 Taiyi-Stable-Diffusion 的 tokenizer 目录
    def __init__(self, model_path='/home/610-sty/huggingface/Taiyi-Stable-Diffusion-1B-Chinese-v0.1/tokenizer'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def __call__(self, texts, **kwargs):
        return self.tokenizer(
            texts,
            return_tensors=kwargs.get("return_tensors", "pt"),
            padding=kwargs.get("padding", True),
            truncation=kwargs.get("truncation", True),
            max_length=kwargs.get("max_length", 77),  # CLIP default
        )