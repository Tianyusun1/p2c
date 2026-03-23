from transformers import AutoTokenizer


class ChineseTokenizer:
    def __init__(self, model_path='/home/610-sty/huggingface/Taiyi-CLIP-Roberta-large-326M-Chinese'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def __call__(self, texts, **kwargs):
        return self.tokenizer(
            texts,
            return_tensors=kwargs.get("return_tensors", "pt"),
            padding=kwargs.get("padding", True),
            truncation=kwargs.get("truncation", True),
            max_length=kwargs.get("max_length", 77),  # CLIP default
        )
