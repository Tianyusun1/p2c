import os
import time
import torch
import argparse
import pandas as pd
from PIL import Image
from torchvision.utils import save_image

# --- 导入所有 checkpoint 中涉及的类 ---
from utils.tokenizer import ChineseTokenizer
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import Model  # ← 新增这一行

# --- 注册所有需要的全局类为“安全” ---
torch.serialization.add_safe_globals([
    ChineseTokenizer,
    BertTokenizerFast,
    Tokenizer,
    Model  # ← 新增这一项
])

# --- 之后再导入可能触发反序列化的模块 ---
from pytorch_lightning.utilities.cloud_io import load as pl_load
from models.model import STYText2ImageModel

def count_parameters(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_model_from_checkpoint(checkpoint_path, map_location="cuda"):
    """
    从检查点文件加载完整的 STYText2ImageModel 模型。
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = pl_load(checkpoint_path, map_location=map_location)
    hparams = checkpoint["hyper_parameters"]
    print(f"Loaded hyper-parameters: {hparams}")

    tokenizer = ChineseTokenizer()
    model = STYText2ImageModel(
        tokenizer=tokenizer,
        learning_rate=hparams.get("learning_rate", 5e-5),
        freeze_unet_epochs=hparams.get("freeze_unet_epochs", 0),
        use_learnable_extractor=hparams.get("use_learnable_extractor", False)
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to(map_location)
    print("✅ Model loaded successfully!")

    # 打印参数量
    total_params, trainable_params = count_parameters(model)
    print(f"📊 Total parameters: {total_params:,}")
    print(f"📊 Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    return model


def generate_image(model, poem, output_path="generated_image.png", num_images=1):
    """
    使用加载的模型生成图像，并返回推理时间（秒）。
    """
    device = next(model.parameters()).device
    print(f"Generating image for: '{poem}'")

    poems = [poem]
    with torch.no_grad():
        text_embeddings = model.text_embedding(poems, is_inference=True)
        text_embeddings["raw_text"] = poems

        unet_to_use = model.ema.ema_model if model.ema_active else model.diffusion_model.unet
        unet_to_use.eval()
        original_unet = model.diffusion_model.unet
        model.diffusion_model.unet = unet_to_use

        # 🔥 开始计时
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()

        try:
            generated_images = model.diffusion_model.generate_images(
                text_embeddings=text_embeddings,
                num_images=num_images
            )
        finally:
            model.diffusion_model.unet = original_unet

        # 🔥 结束计时
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()
        inference_time = end_time - start_time
        print(f"⏱️ Inference time: {inference_time:.3f} seconds")

    # 保存图像
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    save_image(generated_images, output_path, nrow=2, padding=2, normalize=False)
    print(f"🎨 Generated image saved to: {output_path}")

    if num_images == 1:
        single_img = generated_images[0].cpu()
        single_img = (single_img.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
        pil_img = Image.fromarray(single_img)
        pil_img.save(output_path.replace(".png", "_single.png"))

    return inference_time


def main():
    parser = argparse.ArgumentParser(description="Inference script for STY Text-to-Image Model")
    parser.add_argument("--checkpoint", type=str,
                        default="/home/610-sty/Poe2CLP/STY_T2i_v4.5/checkpoints/sty-epochepoch=99.ckpt",
                        help="Path to the trained model checkpoint (.ckpt file)")
    parser.add_argument("--input_excel", type=str, default="/home/610-sty/data/test.xlsx",
                        help="Path to the input Excel file containing poems")
    parser.add_argument("--output_dir", type=str, default="outputs/inference_new",
                        help="Directory to save the generated images")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on (e.g., 'cuda' or 'cpu')")
    args = parser.parse_args()

    # 1. 加载模型（自动打印参数量）
    model = load_model_from_checkpoint(args.checkpoint, map_location=args.device)

    # 2. 读取 Excel
    print(f"Reading poems from: {args.input_excel}")
    try:
        df = pd.read_excel(args.input_excel)
        if 'poem' not in df.columns:
            raise ValueError("The Excel file must contain a column named 'poem'.")
        poems = df['poem'].dropna().tolist()
        print(f"Found {len(poems)} poems in the Excel file.")
    except Exception as e:
        print(f"❌ Error reading Excel file: {e}")
        return

    # 3. 批量生成 + 统计平均时间
    total_inference_time = 0.0
    valid_count = 0

    for idx, poem in enumerate(poems, start=1):
        safe_poem = "".join(c for c in poem if c.isalnum() or c in " _-").strip()
        output_path = os.path.join(args.output_dir, f"poem_{idx:03d}_{safe_poem[:30]}.png")
        try:
            t = generate_image(model=model, poem=poem, output_path=output_path, num_images=1)
            total_inference_time += t
            valid_count += 1
        except Exception as e:
            print(f"❌ Failed to generate image for poem {idx}: '{poem}'. Error: {e}")

    # 4. 输出平均推理时间
    if valid_count > 0:
        avg_time = total_inference_time / valid_count
        print(f"\n✅ Average inference time per image: {avg_time:.3f} seconds (over {valid_count} samples)")


if __name__ == "__main__":
    main()