# /home/610-sty/compare/poe2clp/models/model.py
import torch
from torch import nn
from pytorch_lightning import LightningModule
from models.embedding import EnhancedChineseTextEmbedding
from models.diffusion import DiffusionModel
from ema_pytorch import EMA
import os
import torchvision

class STYText2ImageModel(LightningModule):
    def __init__(self, tokenizer, learning_rate=5e-6, freeze_unet_epochs=0, use_learnable_extractor=True):
        super().__init__()
        self.save_hyperparameters()
        self.freeze_unet_epochs = freeze_unet_epochs
        self.tokenizer = tokenizer

        self.text_embedding = EnhancedChineseTextEmbedding(
            tokenizer,
            use_learnable_extractor=use_learnable_extractor
        )

        # 冻结 Taiyi Text Encoder (RoBERTa) 主干
        for param in self.text_embedding.roberta.parameters():
            param.requires_grad = False
        print("Frozen Taiyi Text Encoder (RoBERTa) parameters.")

        # ✅ 修改点：统一底座路径为 Taiyi-Stable-Diffusion
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

        # 冻结 VAE decoder (由 DiffusionModel 内部处理，这里再次确认)
        for param in self.diffusion_model.vae.parameters():
            param.requires_grad = False
        print("Frozen VAE parameters.")

        # 初始化 EMA
        self.ema = EMA(
            self.diffusion_model.unet,
            beta=0.9999,
            update_after_step=100,
            update_every=10
        )
        self.ema_active = True

        self.loss_history = []
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

    def on_before_zero_grad(self, optimizer):
        if self.ema_active:
            self.ema.update()

    def training_step(self, batch, batch_idx):
        images = batch["instance_image"]
        poems = batch["instance_prompt"]
        phrases = batch["phrases"]

        text_embeddings = self.text_embedding(poems, phrases, is_inference=False)
        text_embeddings["raw_text"] = poems

        total_loss = self.diffusion_model.train_step(images, text_embeddings)

        self.clip_gradients(self.trainer.optimizers[0], gradient_clip_val=1.0)

        # ✅ 显式传入 batch_size 以消除警告
        self.log("train_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=images.shape[0])
        
        return total_loss

    def save_phrase_attention_maps(self, text_embeddings, poems, save_dir, epoch):
        """保存短语级注意力图"""
        import matplotlib.pyplot as plt
        import os

        os.makedirs(save_dir, exist_ok=True)

        phrase_attention = text_embeddings.get("phrase_attention")
        phrase_spans = text_embeddings.get("phrase_spans")
        phrase_scores = text_embeddings.get("phrase_scores")

        if phrase_attention is None or phrase_spans is None:
            return

        for i, (poem, attn, spans, scores) in enumerate(zip(poems, phrase_attention, phrase_spans, phrase_scores)):
            poem_tokens = self.tokenizer(poem, return_tensors="pt", padding=False, truncation=True, max_length=77)
            input_ids = poem_tokens['input_ids'][0]
            if hasattr(self.tokenizer, 'tokenizer') and hasattr(self.tokenizer.tokenizer, 'convert_ids_to_tokens'):
                tokens = self.tokenizer.tokenizer.convert_ids_to_tokens(input_ids)
            else:
                tokens = [f"ID_{id_}" for id_ in input_ids.tolist()]

            phrase_texts = []
            for start, end in spans:
                if start == 0 and end == 0:
                    phrase_texts.append("<PAD>")
                else:
                    phrase_texts.append("".join(tokens[start:end+1]))

            plt.figure(figsize=(max(8, len(phrase_texts) * 1.5), 4))
            attn_np = attn.detach().cpu().numpy()
            scores_np = scores.detach().cpu().numpy()
            bars = plt.bar(range(len(phrase_texts)), attn_np, color='skyblue')
            plt.xticks(range(len(phrase_texts)), phrase_texts, rotation=45, ha='right')
            plt.ylim(0, 1.0)

            for bar, weight, score in zip(bars, attn_np, scores_np):
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{weight:.3f}\n({score:.2f})',
                         ha='center', va='bottom', fontsize=8)

            plt.tight_layout()
            save_path = os.path.join(save_dir, f"phrase_attention_epoch_{epoch:03d}_sample_{i+1:02d}.png")
            plt.savefig(save_path)
            plt.close()

    def on_train_epoch_end(self):
        train_loader = self.trainer.datamodule.train_dataloader()
        try:
            sample_batch = next(iter(train_loader))
        except StopIteration:
            return

        num_images = min(4, sample_batch["instance_image"].shape[0])
        sample_poems = sample_batch["instance_prompt"][:num_images]
        real_images = sample_batch["instance_image"][:num_images]

        text_embeddings = self.text_embedding(sample_poems, is_inference=True)
        text_embeddings["raw_text"] = sample_poems

        if "phrase_attention" in text_embeddings:
            self.save_phrase_attention_maps(text_embeddings, sample_poems, "outputs/phrase_attention_maps", self.current_epoch)

        self.ema.ema_model.eval()
        original_unet = self.diffusion_model.unet
        self.diffusion_model.unet = self.ema.ema_model

        try:
            generated_images = self.diffusion_model.generate_images(text_embeddings=text_embeddings, num_images=num_images)
        finally:
            self.diffusion_model.unet = original_unet

        real_images = (real_images.to(generated_images.device).clamp(-1, 1) + 1) / 2
        comparison = torch.cat([real_images, generated_images], dim=0)
        grid = torchvision.utils.make_grid(comparison, nrow=num_images)
        save_dir = "outputs"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"comparison_epoch_{self.current_epoch:03d}.png")
        torchvision.utils.save_image(grid, save_path)

    def configure_optimizers(self):
        # 1. 自动获取已解冻的 LoRA 参数
        lora_params = []
        for name, module in self.diffusion_model.unet.named_modules():
            if "processor" in name:
                lora_params.extend([p for p in module.parameters() if p.requires_grad])
        
        # 2. 获取已解冻的自定义适配层参数
        custom_trainable_modules = [
            self.text_embedding,
            self.diffusion_model.local_proj,
            self.diffusion_model.text_proj,
            self.diffusion_model.image_proj,
            self.diffusion_model.unet.fused_text_proj,
            self.diffusion_model.unet.local_to_fused,
            self.diffusion_model.unet.context_fusion, # 确保包含融合模块
        ]
        
        custom_params = []
        for module in custom_trainable_modules:
            if module is not None:
                custom_params.extend([p for p in module.parameters() if p.requires_grad])
        
        print(f"LoRA 参数数量: {len(lora_params)}")
        print(f"自定义适配层参数数量: {len(custom_params)}")

        # 3. 设置学习率
        lora_lr = 5e-5
        custom_lr = 1e-5
        
        optimizer = torch.optim.AdamW([
            {"params": lora_params, "lr": lora_lr, "weight_decay": 0.01},
            {"params": custom_params, "lr": custom_lr, "weight_decay": 0.01}
        ])

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