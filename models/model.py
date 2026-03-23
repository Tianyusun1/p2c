# /home/610-sty/compare/poe2clp/models/model.py
import torch
from torch import nn
import torchvision
from pytorch_lightning import LightningModule
from ema_pytorch import EMA
import os

from models.embedding import EnhancedChineseTextEmbedding
from models.diffusion import DiffusionModel

class STYText2ImageModel(LightningModule):
    def __init__(self, tokenizer, learning_rate=5e-6, freeze_unet_epochs=0, use_learnable_extractor=True):
        super().__init__()
        self.save_hyperparameters(ignore=['tokenizer'])
        self.tokenizer = tokenizer

        # 1. 增强文本嵌入模块
        self.text_embedding = EnhancedChineseTextEmbedding(
            tokenizer=tokenizer,
            use_learnable_extractor=use_learnable_extractor
        )

        # 冻结 RoBERTa 主干
        for param in self.text_embedding.roberta.parameters():
            param.requires_grad = False

        # 2. 扩散模型配置
        taiyi_sd_path = "/home/610-sty/huggingface/Taiyi-Stable-Diffusion-1B-Chinese-v0.1"
        diffusion_config = {
            "pretrained_vae_path": f"{taiyi_sd_path}/vae",
            "pretrained_unet_path": f"{taiyi_sd_path}/unet",
            "pretrained_scheduler_path": f"{taiyi_sd_path}/scheduler",
            "prediction_type": "epsilon",
            "perceptual_loss_weight": 0.05
        }
        self.diffusion_model = DiffusionModel(config=diffusion_config)

        # 3. 初始化 EMA
        self.ema = EMA(
            self.diffusion_model.unet,
            beta=0.9999,
            update_after_step=100,
            update_every=10
        )
        self.print_model_param_stats()

    def print_model_param_stats(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"====== 可训练参数量：{trainable_params:,} ======")

    def training_step(self, batch, batch_idx):
        images = batch["instance_image"]
        poems = batch["instance_prompt"]
        phrases = batch["phrases"]

        # 提取特征
        text_embeddings = self.text_embedding(poems, phrases, is_inference=False)
        text_embeddings["raw_text"] = poems

        # 执行训练步
        total_loss = self.diffusion_model.train_step(images, text_embeddings)

        if torch.isnan(total_loss):
            return sum(p.sum() * 0 for p in self.parameters() if p.requires_grad)

        self.log("train_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=images.shape[0])
        return total_loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # 更新 EMA 权重
        if hasattr(self, "ema"):
            self.ema.update()

    def on_train_epoch_end(self):
        """Epoch 结束采样"""
        if not hasattr(self.trainer.datamodule, 'train_dataloader'): return
        try:
            sample_batch = next(iter(self.trainer.datamodule.train_dataloader()))
        except: return

        num_images = min(4, sample_batch["instance_image"].shape[0])
        sample_poems = sample_batch["instance_prompt"][:num_images]

        # 推理模式提取
        text_embeddings = self.text_embedding(sample_poems, is_inference=True)
        
        # 切换到 EMA 模型进行推理
        self.ema.ema_model.eval()
        original_unet = self.diffusion_model.unet
        self.diffusion_model.unet = self.ema.ema_model

        try:
            with torch.no_grad():
                generated_images = self.diffusion_model.generate_images(text_embeddings=text_embeddings, num_images=num_images)
        finally:
            self.diffusion_model.unet = original_unet

        # 保存图片
        grid = torchvision.utils.make_grid(generated_images, nrow=num_images)
        os.makedirs("outputs", exist_ok=True)
        torchvision.utils.save_image(grid, f"outputs/epoch_{self.current_epoch}.png")

    def configure_optimizers(self):
        # 1. 收集 LoRA 参数
        lora_params = []
        for name, module in self.diffusion_model.unet.named_modules():
            if "processor" in name:
                lora_params.extend([p for p in module.parameters() if p.requires_grad])
        
        # 2. 收集自定义层参数
        # ✅ 使用 getattr 防御 AttributeError
        custom_trainable_modules = [
            self.text_embedding,
            getattr(self.diffusion_model, 'local_proj', None),
            getattr(self.diffusion_model, 'text_proj', None),
            getattr(self.diffusion_model, 'image_proj', None),
            getattr(self.diffusion_model.unet, 'fused_text_proj', None),
            getattr(self.diffusion_model.unet, 'local_to_fused', None),
            getattr(self.diffusion_model.unet, 'context_fusion', None),
        ]
        
        custom_params = []
        for m in custom_trainable_modules:
            if m is not None:
                custom_params.extend([p for p in m.parameters() if p.requires_grad])
        
        # 确保没有重复参数
        all_ids = set()
        final_custom_params = []
        for p in custom_params:
            if id(p) not in all_ids:
                all_ids.add(id(p))
                final_custom_params.append(p)

        optimizer = torch.optim.AdamW([
            {"params": lora_params, "lr": self.hparams.learning_rate * 5, "weight_decay": 1e-2},
            {"params": final_custom_params, "lr": self.hparams.learning_rate, "weight_decay": 1e-2}
        ])

        return optimizer