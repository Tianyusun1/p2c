import io
import braceexpand
import json
import random
import webdataset as wds
from PIL import Image
from torchvision import transforms

# 只需要这两个
USED_KEYS = {"jpg": "instance_image", "json": "poem_data"}

prompt_list = ["{}"]

def expand_urls(urls):
    if isinstance(urls, str):
        urllist = urls.split("::")
        result = []
        for url in urllist:
            result.extend(braceexpand.braceexpand(url))
        return result
    else:
        return list(urls)

def expand_text(text):
    pat = random.choice(prompt_list)
    return pat.format(text)

def custom_decoder(key, data):
    if key.endswith("jpg"):
        with io.BytesIO(data) as stream:
            img = Image.open(stream)
            img.load()
            return img.convert("RGB")
    elif key.endswith("json"):
        return json.loads(data.decode("utf-8"))
    else:
        return None

def verify_keys(samples, required_keys, handler=wds.handlers.reraise_exception):
    for sample in samples:
        try:
            for key in required_keys:
                assert key in sample, f"Sample missing {key}, found keys: {sample.keys()}"
            yield {key: sample[key] for key in required_keys}
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break

key_verifier = wds.filters.pipelinefilter(verify_keys)

class ImageEmbeddingDataset(wds.DataPipeline, wds.compat.FluidInterface):
    def __init__(
            self,
            urls,
            tokenizer=None,
            extra_keys=[],
            size=512,
            handler=wds.handlers.reraise_exception,
            resample=False,
            shuffle_shards=True,
            center_crop=False,
            drop_phrases=False,           # ← 新增
            phrase_dropout_rate=0.3       # ← 新增
    ):
        super().__init__()
        keys = list(USED_KEYS.keys()) + extra_keys
        self.size = size
        self.drop_phrases = drop_phrases
        self.phrase_dropout_rate = phrase_dropout_rate


        if resample:
            assert not shuffle_shards, "Cannot both resample and shuffle"
            self.append(wds.ResampledShards(urls))
        else:
            self.append(wds.SimpleShardList(urls))
            if shuffle_shards:
                self.append(wds.filters.shuffle(1000))

        self.append(wds.tarfile_to_samples(handler=handler))
        self.append(wds.decode(custom_decoder, handler=handler))
        self.append(key_verifier(required_keys=keys, handler=handler))
        self.append(wds.map(self.preproc))

    def preproc(self, sample):
        # 图像处理
        image = sample["jpg"]
        transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        image = transform(image)

        # 文本处理
        poem_data = sample["json"]
        poem = poem_data.get("poem", "")
        nouns = poem_data.get("nouns", [])

        # === 新增关键词 Dropout 逻辑 ===
        if self.drop_phrases and random.random() < self.phrase_dropout_rate:
            nouns = []

        return {
            "instance_image": image,
            "instance_prompt": poem,
            "phrases": nouns
        }
