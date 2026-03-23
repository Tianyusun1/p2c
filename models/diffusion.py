# /home/610-sty/compare/poe2clp/models/diffusion.py
import torch
import torch.nn.functional as F
from torch import nn
from diffusers import AutoencoderKL, DDPMScheduler
from torchvision.utils import save_image
import lpips
from models.unet_custom import CustomUNet2DConditionModel

from transformers import BertTokenizer, AutoModel

class DiffusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 1. 加载预训练模块
        self.vae = AutoencoderKL.from_pretrained(config["pretrained_vae_path"])
        
        # ✅ 核心优化：强制指定 cross_attention_dim=768 以匹配 Taiyi-SD
        self.unet = CustomUNet2DConditionModel.from_pretrained(
            config["pretrained_unet_path"],
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=False,
            cross_attention_dim=768
        )
        
        # ✅ 关键修改：减少可训练参数
        # 首先冻结 UNet 的所有主干权重
        self.unet.requires_grad_(False)
        
        # 仅解冻自定义适配层 (Adapter Layers)
        self.unet.fused_text_proj.requires_grad_(True)
        self.unet.local_to_fused.requires_grad_(True)
        self.unet.context_fusion.requires_grad_(True)
        
        # 显式解冻 LoRA 参数 (位于各层 processor 中)
        for name, module in self.unet.named_modules():
            if "processor" in name:
                for param in module.parameters():
                    param.requires_grad = True

        self.noise_scheduler = DDPMScheduler.from_pretrained(config["pretrained_scheduler_path"])
        self.prediction_type = config.get("prediction_type", "epsilon")
        self.perceptual_loss_weight = config.get("perceptual_loss_weight", 0.05)

        # 冻结 VAE 参数
        self.vae.requires_grad_(False)

        # LPIPS 感知损失
        self.lpips_loss = lpips.LPIPS(net='alex')
        self.lpips_loss.eval()
        self.lpips_loss.requires_grad_(False)

        # 局部特征映射层 (解冻)
        self.local_proj = nn.Linear(128, 768)
        self.local_proj.requires_grad_(True)

        # 自动迁移 noise_scheduler 参数到 GPU
        scheduler_device = next(self.unet.parameters()).device
        for name, value in vars(self.noise_scheduler).items():
            if isinstance(value, torch.Tensor):
                setattr(self.noise_scheduler, name, value.to(scheduler_device))

        # 使用 Taiyi-SD 的 Text Encoder 计算 CLIP Loss
        taiyi_sd_path = "/home/610-sty/huggingface/Taiyi-Stable-Diffusion-1B-Chinese-v0.1"
        self.clip_text_encoder = AutoModel.from_pretrained(f"{taiyi_sd_path}/text_encoder").to(self.unet.device)
        self.clip_tokenizer = BertTokenizer.from_pretrained(f"{taiyi_sd_path}/tokenizer")

        self.clip_text_encoder.eval()
        self.clip_text_encoder.requires_grad_(False)

        # 添加投影层 (维度 768)
        self.text_proj = nn.Linear(768, 768)
        self.image_proj = nn.Linear(3, 768)
        self.tv_loss_weight = 0.001

    @staticmethod
    def tv_loss(img):
        # ✅ 强制转为 float32 防止 fp16 下溢出崩溃
        img = img.float()
        batch_size, c, h, w = img.size()
        h_diff = torch.sum(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
        w_diff = torch.sum(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
        return (h_diff + w_diff) / (c * h * w)

    def train_step(self, images, text_embeddings):
        # 修正图像范围
        if images.min() < -1.5 or images.max() > 1.5:
            images = torch.clamp(images, -1.0, 1.0)

        images = images.to(dtype=torch.float32)
        device = self.unet.device

        global_embed = text_embeddings['global_embed'].to(device)
        local_embeds = text_embeddings['local_embeds'].to(device)

        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * 0.18215

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=latents.device
        ).long()

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # UNet 前向传播
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            global_embed=global_embed,
            local_embeds=local_embeds
        ).sample

        # 基础 MSE Loss
        if self.prediction_type == "v_prediction":
            alpha_t = self.noise_scheduler.alphas_cumprod[timesteps].sqrt().view(-1, 1, 1, 1)
            sigma_t = (1 - self.noise_scheduler.alphas_cumprod[timesteps]).sqrt().view(-1, 1, 1, 1)
            v_target = alpha_t * noise - sigma_t * latents
            mse_loss = F.mse_loss(noise_pred, v_target)
        else:
            mse_loss = F.mse_loss(noise_pred, noise)

        # 辅助 Loss 计算 (VAE 解码安全保护)
        with torch.no_grad():
            # ✅ 修复：必须除以 0.18215 并强制转 FP32 防止 Core Dump
            latents_for_decode = (latents / 0.18215).float()
            original_vae_dtype = next(self.vae.parameters()).dtype
            self.vae.to(torch.float32)
            pred_images = self.vae.decode(latents_for_decode).sample
            self.vae.to(original_vae_dtype)
            gt_images = images

        # ✅ 感知损失 (强制 float32)
        perceptual_loss = self.lpips_loss(pred_images.float().clamp(-1, 1), gt_images.float().clamp(-1, 1)).mean()

        # CLIP loss
        raw_texts = text_embeddings.get('raw_text', ["a photo"] * pred_images.shape[0])
        text_inputs = self.clip_tokenizer(
            raw_texts, padding=True, truncation=True, max_length=77, return_tensors="pt"
        ).to(device)
        text_outputs = self.clip_text_encoder(**text_inputs)
        text_features = self.text_proj(text_outputs.last_hidden_state.mean(dim=1))
        
        image_features = self.image_proj(pred_images.mean(dim=[2, 3]))
        
        # ✅ 归一化安全保护 (加 eps 防止除 0)
        text_features = F.normalize(text_features.float(), p=2, dim=-1, eps=1e-8)
        image_features = F.normalize(image_features.float(), p=2, dim=-1, eps=1e-8)

        clip_sim = (image_features * text_features).sum(dim=1).mean()
        clip_loss = 1 - clip_sim
        tv_loss_value = self.tv_loss(pred_images)

        # 合并 Total Loss
        total_loss = (
            mse_loss +
            0.05 * perceptual_loss + 
            0.1 * clip_loss +       
            0.001 * tv_loss_value   
        )

        return total_loss

    def validation_step(self, images, text_embeddings):
        return self.train_step(images, text_embeddings)

    @torch.no_grad()
    def generate_images(self, text_embeddings, save_path=None, num_images=1, image_hint=None):
        self.unet.eval()
        device = self.unet.device

        global_embed = text_embeddings['global_embed'].to(device)
        local_embeds = text_embeddings['local_embeds'].to(device)

        latents = torch.randn((num_images, 4, 64, 64), device=device) * 0.18215

        for t in reversed(range(self.noise_scheduler.config.num_train_timesteps)):
            timesteps = torch.tensor(t, dtype=torch.long, device=device)
            noise_pred = self.unet(latents, timesteps, global_embed=global_embed, local_embeds=local_embeds).sample
            latents = self.noise_scheduler.step(model_output=noise_pred, timestep=timesteps, sample=latents).prev_sample

        # 推理时 FP32 保护
        latents = (latents / 0.18215).float()
        original_vae_dtype = next(self.vae.parameters()).dtype
        self.vae.to(torch.float32)
        images = self.vae.decode(latents).sample
        self.vae.to(original_vae_dtype)

        images = (images.clamp(-1, 1) + 1) / 2
        if save_path:
            save_image(images, save_path, nrow=2)

        return images