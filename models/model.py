import torch
from torch import nn
import torchvision
from pytorch_lightning import LightningModule
from ema_pytorch import EMA
import os

# 确保这些自定义模块的路径在你的项目中是正确的
from models.embedding import EnhancedChineseTextEmbedding
from models.diffusion import DiffusionModel

class STYText2ImageModel(LightningModule):
    def __init__(self, tokenizer, learning_rate=5e-6, freeze_unet_epochs=0, use_learnable_extractor=True):
        super().__init__()
        self.save_hyperparameters(ignore=['tokenizer'])
        self.freeze_unet_epochs = freeze_unet_epochs
        self.tokenizer = tokenizer

        # 1. 增强文本嵌入模块
        self.text_embedding = EnhancedChineseTextEmbedding(
            tokenizer=tokenizer,
            use_learnable_extractor=use_learnable_extractor
        )

        # 冻结 Taiyi Text Encoder (RoBERTa) 主干，只训练新增的特征提取层
        for param in self.text_embedding.roberta.parameters():
            param.requires_grad = False
        print("Frozen Taiyi Text Encoder (RoBERTa) parameters.")

        # 2. 扩散模型底座配置
        taiyi_sd_path = "/home/610-sty/huggingface/Taiyi-Stable-Diffusion-1B-Chinese-v0.1"
        
        diffusion_config = {
            "pretrained_model_name_or_path": taiyi_sd_path,
            "pretrained_vae_path": f"{taiyi_sd_path}/vae",
            "pretrained_unet_path": f"{taiyi_sd_path}/unet",
            "pretrained_scheduler_path": f"{taiyi_sd_path}/scheduler",
            "prediction_type": "epsilon",
            "noise_scheduler_type": "ddim",
            "perceptual_loss_weight": 0.05
        }

        self.diffusion_model = DiffusionModel(config=diffusion_config)

        # 冻结 VAE decoder 
        for param in self.diffusion_model.vae.parameters():
            param.requires_grad = False
        print("Frozen VAE parameters.")

        # 3. 初始化 EMA (指数移动平均) 用于模型平滑
        self.ema = EMA(
            self.diffusion_model.unet,
            beta=0.9999,
            update_after_step=100,
            update_every=10
        )
        self.ema_active = True
        
        self.print_model_param_stats()

    def print_model_param_stats(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        percent = 100 * trainable_params / total_params
        print("====== 模型参数统计 ======")
        print(f"总参数量：{total_params:,}")
        print(f"可训练参数量：{trainable_params:,}")
        print(f"可训练比例：{percent:.2f}%")
        print("=========================")

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """每个 batch 结束后更新 EMA 权重"""
        if self.ema_active:
            self.ema.update()

    def training_step(self, batch, batch_idx):
            images = batch["instance_image"]
            poems = batch["instance_prompt"]
            phrases = batch["phrases"]

            text_embeddings = self.text_embedding(poems, phrases, is_inference=False)
            text_embeddings["raw_text"] = poems

            total_loss = self.diffusion_model.train_step(images, text_embeddings)

            # ✅ 修复 AMP 兼容的 NaN 处理逻辑
            if torch.isnan(total_loss):
                print(f"[Warning] NaN loss at batch {batch_idx}. Skipping optimization safely.")
                # 创建一个与计算图挂钩的 0 损失，防止 backward() 报错
                # 我们利用一个必定存在的参数乘以 0
                dummy_loss = sum(p.sum() * 0 for p in self.parameters() if p.requires_grad)
                return dummy_loss

            self.log("train_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=images.shape[0])
            return total_loss

    def on_train_epoch_end(self):
        """每个 Epoch 结束进行可视化采样"""
        # 避免在 WebDataset 上使用 next(iter(loader)) 导致死锁，建议通过 trainer 内部获取
        if not hasattr(self.trainer.datamodule, 'train_dataloader'):
            return
            
        train_loader = self.trainer.datamodule.train_dataloader()
        try:
            # 仅取一个 batch 里的前几张图
            sample_batch = next(iter(train_loader))
        except Exception:
            return

        num_images = min(4, sample_batch["instance_image"].shape[0])
        sample_poems = sample_batch["instance_prompt"][:num_images]
        real_images = sample_batch["instance_image"][:num_images]

        # 推理模式提取特征
        text_embeddings = self.text_embedding(sample_poems, is_inference=True)
        text_embeddings["raw_text"] = sample_poems

        # 保存短语注意力图（用于科研分析）
        if "phrase_attention" in text_embeddings:
            self.save_phrase_attention_maps(text_embeddings, sample_poems, "outputs/phrase_attention_maps", self.current_epoch)

        # 使用 EMA 模型进行推理生成
        self.ema.ema_model.eval()
        original_unet = self.diffusion_model.unet
        self.diffusion_model.unet = self.ema.ema_model

        try:
            with torch.no_grad():
                generated_images = self.diffusion_model.generate_images(text_embeddings=text_embeddings, num_images=num_images)
        finally:
            self.diffusion_model.unet = original_unet # 恢复原始 UNet 用于训练

        # 图像后处理与保存
        real_images = (real_images.to(generated_images.device).clamp(-1, 1) + 1) / 2
        comparison = torch.cat([real_images, generated_images], dim=0)
        grid = torchvision.utils.make_grid(comparison, nrow=num_images)
        
        save_dir = "outputs"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"comparison_epoch_{self.current_epoch:03d}.png")
        torchvision.utils.save_image(grid, save_path)

    def save_phrase_attention_maps(self, text_embeddings, poems, save_dir, epoch):
        """保存短语级注意力分配的可视化图"""
        import matplotlib.pyplot as plt
        os.makedirs(save_dir, exist_ok=True)

        phrase_attention = text_embeddings.get("phrase_attention")
        phrase_spans = text_embeddings.get("phrase_spans")
        phrase_scores = text_embeddings.get("phrase_scores")

        if phrase_attention is None:
            return

        for i, (poem, attn, spans, scores) in enumerate(zip(poems, phrase_attention, phrase_spans, phrase_scores)):
            # 简单处理 Token 转文字逻辑
            phrase_texts = []
            for start, end in spans:
                phrase_texts.append(poem[start:end] if start != end else "PAD")

            plt.figure(figsize=(max(8, len(phrase_texts) * 1.5), 4))
            attn_np = attn.detach().cpu().numpy()
            plt.bar(range(len(phrase_texts)), attn_np, color='skyblue')
            plt.xticks(range(len(phrase_texts)), phrase_texts, rotation=45)
            plt.title(f"Phrase Attention - Epoch {epoch}")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"epoch_{epoch}_sample_{i}.png"))
            plt.close()

    def configure_optimizers(self):
        # 1. 收集 LoRA 参数 (在 Attention Processors 中)
        lora_params = []
        for name, module in self.diffusion_model.unet.named_modules():
            if "processor" in name:
                lora_params.extend([p for p in module.parameters() if p.requires_grad])
        
        # 2. 收集自定义结构参数（融合模块、投影层等）
        custom_trainable_modules = [
            self.text_embedding,
            self.diffusion_model.local_proj,
            self.diffusion_model.text_proj,
            self.diffusion_model.image_proj,
            getattr(self.diffusion_model.unet, 'fused_text_proj', None),
            getattr(self.diffusion_model.unet, 'local_to_fused', None),
            getattr(self.diffusion_model.unet, 'context_fusion', None),
        ]
        
        custom_params = []
        for module in custom_trainable_modules:
            if module is not None:
                custom_params.extend([p for p in module.parameters() if p.requires_grad])
        
        print(f"LoRA 参数数量: {len(lora_params)}")
        print(f"自定义适配层参数数量: {len(custom_params)}")

        # 3. 分层设置学习率：LoRA 可以稍高，自定义融合层建议稍低以保稳
        optimizer = torch.optim.AdamW([
            {"params": lora_params, "lr": 5e-5, "weight_decay": 1e-2},
            {"params": custom_params, "lr": 1e-5, "weight_decay": 1e-2}
        ])

        # 4. 学习率调度器：根据验证/训练 Loss 自动减半
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss_epoch",
                "strict": True,
                "name": "ReduceLROnPlateau",
            }
        }