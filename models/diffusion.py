# /home/610-sty/compare/poe2clp/models/diffusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler
import lpips
from models.unet_custom import CustomUNet2DConditionModel 

class DiffusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 1. 加载预训练模块
        self.vae = AutoencoderKL.from_pretrained(config["pretrained_vae_path"])
        
        # 强制指定 768 维加载，解决 Shape Mismatch 问题
        self.unet = CustomUNet2DConditionModel.from_pretrained(
            config["pretrained_unet_path"],
            cross_attention_dim=768,
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=False
        )
        
        # 2. 冻结与解冻逻辑
        self.unet.requires_grad_(False)
        trainable_names = ["fused_text_proj", "local_to_fused", "context_fusion", "processor"]
        for name, module in self.unet.named_modules():
            if any(key in name for key in trainable_names):
                for param in module.parameters():
                    param.requires_grad = True

        self.noise_scheduler = DDPMScheduler.from_pretrained(config["pretrained_scheduler_path"])
        self.prediction_type = config.get("prediction_type", "epsilon")

        self.vae.requires_grad_(False)
        self.vae.eval()

        # LPIPS 保持 eval
        self.lpips_loss = lpips.LPIPS(net='alex').eval()
        self.lpips_loss.requires_grad_(False)

        self.local_proj = nn.Linear(128, 768)
        nn.init.xavier_uniform_(self.local_proj.weight)

        self.text_proj = nn.Linear(768, 768)
        self.image_proj = nn.Linear(3, 768) 
        with torch.no_grad():
            nn.init.normal_(self.text_proj.weight, std=0.01)
            nn.init.normal_(self.image_proj.weight, std=0.01)
        
        self.perceptual_loss_weight = config.get("perceptual_loss_weight", 0.01)
        self.clip_loss_weight = 0.0 
        self.tv_loss_weight = 0.001

    def train_step(self, images, text_embeddings):
        images = images.to(dtype=torch.float32).clamp(-1.0, 1.0)
        device = images.device
        
        global_embed = text_embeddings['global_embed'].to(device)
        local_embeds = text_embeddings['local_embeds'].to(device)
        local_mask = text_embeddings.get('local_mask').to(device)

        # 1. VAE 编码
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample() * 0.18215

        # 2. 加噪
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # 3. UNet 预测
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            global_embed=global_embed,
            local_embeds=local_embeds,
            local_mask=local_mask
        ).sample

        # 数值防御
        noise_pred = torch.nan_to_num(noise_pred, nan=0.0).clamp(-15.0, 15.0)

        # 4. 损失计算
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device)
        sqrt_alpha_prod = alphas_cumprod[timesteps].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]).sqrt().view(-1, 1, 1, 1)

        if self.prediction_type == "v_prediction":
            target = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * latents
            loss_main = F.huber_loss(noise_pred.float(), target.float(), delta=1.0)
            # 推导预测的 x0 用于感知损失
            pred_x0 = sqrt_alpha_prod * noisy_latents - sqrt_one_minus_alpha_prod * noise_pred
        else:
            loss_main = F.huber_loss(noise_pred.float(), noise.float(), delta=1.0)
            # 推导预测的 x0: (xt - sqrt(1-alpha)*eps) / sqrt(alpha)
            pred_x0 = (noisy_latents - sqrt_one_minus_alpha_prod * noise_pred) / sqrt_alpha_prod

        # 5. 改进的感知损失 (对 UNet 有梯度回传)
        perceptual_loss = torch.tensor(0.0, device=device)
        if self.perceptual_loss_weight > 0:
            # 解码 UNet 预测的图像
            pred_images = self.vae.decode(pred_x0 / 0.18215).sample
            perceptual_loss = self.lpips_loss(pred_images.clamp(-1, 1), images).mean()

        total_loss = loss_main + self.perceptual_loss_weight * perceptual_loss
        
        if torch.isnan(total_loss):
            return sum(p.sum() * 0 for p in self.parameters() if p.requires_grad)

        return total_loss

    @torch.no_grad()
    def generate_images(self, text_embeddings, num_images=1):
        self.unet.eval()
        device = self.unet.device
        
        global_embed = text_embeddings['global_embed'].to(device)
        local_embeds = text_embeddings['local_embeds'].to(device)
        local_mask = text_embeddings.get('local_mask').to(device)
        
        latents = torch.randn((num_images, 4, 64, 64), device=device)
        
        # 采样循环
        for t in reversed(range(len(self.noise_scheduler.alphas_cumprod))):
            timesteps = torch.tensor([t], device=device).expand(num_images)
            noise_pred = self.unet(
                latents, 
                timesteps, 
                global_embed=global_embed, 
                local_embeds=local_embeds, 
                local_mask=local_mask
            ).sample
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample
            
        # ✅ 防御黑屏：强制转 FP32 进行解码
        latents = latents.to(torch.float32)
        self.vae.to(torch.float32) 
        
        images = self.vae.decode(latents / 0.18215).sample
        return (images.clamp(-1, 1) + 1) / 2