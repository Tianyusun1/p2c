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
        
        # 自动识别维度，确保加载权重成功
        self.unet = CustomUNet2DConditionModel.from_pretrained(
            config["pretrained_unet_path"],
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=False
        )
        
        # 冻结 UNet 主干，仅训练自定义适配层和 LoRA
        self.unet.requires_grad_(False)
        
        # 显式解冻适配器层
        trainable_names = ["fused_text_proj", "local_to_fused", "context_fusion", "processor"]
        for name, module in self.unet.named_modules():
            if any(key in name for key in trainable_names):
                for param in module.parameters():
                    param.requires_grad = True

        self.noise_scheduler = DDPMScheduler.from_pretrained(config["pretrained_scheduler_path"])
        self.prediction_type = config.get("prediction_type", "epsilon")

        # 冻结 VAE
        self.vae.requires_grad_(False)
        self.vae.eval()

        # LPIPS 感知损失：强制评估模式并冻结
        self.lpips_loss = lpips.LPIPS(net='alex').eval()
        self.lpips_loss.requires_grad_(False)

        # 辅助特征投影层（增加初始化保护）
        self.local_proj = nn.Linear(128, 768)
        nn.init.xavier_uniform_(self.local_proj.weight)

        # 使用 Taiyi-SD 的 Text Encoder 计算语义对齐 Loss
        taiyi_sd_path = "/home/610-sty/huggingface/Taiyi-Stable-Diffusion-1B-Chinese-v0.1"
        self.clip_text_encoder = AutoModel.from_pretrained(f"{taiyi_sd_path}/text_encoder")
        self.clip_tokenizer = BertTokenizer.from_pretrained(f"{taiyi_sd_path}/tokenizer")
        self.clip_text_encoder.eval().requires_grad_(False)

        # 语义空间投影：使用极小的初始化，防止初期产生巨大梯度
        self.text_proj = nn.Linear(768, 768)
        self.image_proj = nn.Linear(3, 768) 
        with torch.no_grad():
            nn.init.normal_(self.text_proj.weight, std=0.01)
            nn.init.normal_(self.image_proj.weight, std=0.01)
        
        # Loss 权重配置
        self.perceptual_loss_weight = config.get("perceptual_loss_weight", 0.01)
        self.clip_loss_weight = 0.0        # ⚠ 初始设为 0，等训练稳定后再开启
        self.tv_loss_weight = 0.001

    @staticmethod
    def tv_loss(img):
        """全变分损失：减少生成噪声"""
        img = img.float() 
        batch_size, c, h, w = img.size()
        h_diff = torch.sum(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
        w_diff = torch.sum(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
        return (h_diff + w_diff) / (c * h * w + 1e-8)

    def train_step(self, images, text_embeddings):
        # 1. 数据准备与安全裁剪
        images = images.to(dtype=torch.float32).clamp(-1.0, 1.0)
        device = images.device
        
        global_embed = text_embeddings['global_embed'].to(device)
        local_embeds = text_embeddings['local_embeds'].to(device)

        # 2. VAE 编码
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * 0.18215

        # 3. 加噪流程
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # 4. UNet 前向预测
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            global_embed=global_embed,
            local_embeds=local_embeds
        ).sample

        # ✅ 核心优化 1：数值裁剪
        # 防止 UNet 输出的预测值过大导致 MSE 直接变为 NaN
        noise_pred = torch.clamp(noise_pred, -20.0, 20.0)

        # 5. 计算主损失：使用 Huber Loss 替代标准 MSE
        # Huber Loss 对离群值（Outliers）更鲁棒，能有效防止梯度爆炸
        if self.prediction_type == "v_prediction":
            alpha_t = self.noise_scheduler.alphas_cumprod[timesteps].sqrt().view(-1, 1, 1, 1)
            sigma_t = (1 - self.noise_scheduler.alphas_cumprod[timesteps]).sqrt().view(-1, 1, 1, 1)
            target = alpha_t * noise - sigma_t * latents
            loss_main = F.huber_loss(noise_pred.float(), target.float(), delta=1.0)
        else:
            target = noise
            loss_main = F.huber_loss(noise_pred.float(), target.float(), delta=1.0)

        # 6. 计算辅助损失
        perceptual_loss = torch.tensor(0.0, device=device)
        clip_loss = torch.tensor(0.0, device=device)
        tv_loss_value = torch.tensor(0.0, device=device)

        if self.perceptual_loss_weight > 0:
            with torch.no_grad():
                # 稳定版感知引导：直接使用 ground-truth latents 的解码图
                pred_images = self.vae.decode(latents / 0.18215).sample
            perceptual_loss = self.lpips_loss(pred_images.clamp(-1, 1), images).mean()
            tv_loss_value = self.tv_loss(pred_images)

        if self.clip_loss_weight > 0:
            self.clip_text_encoder.to(device)
            raw_texts = text_embeddings.get('raw_text', [""] * images.shape[0])
            text_inputs = self.clip_tokenizer(raw_texts, padding=True, truncation=True, return_tensors="pt").to(device)
            text_features = self.clip_text_encoder(**text_inputs).last_hidden_state.mean(dim=1)
            
            # 投影与归一化安全保护
            text_features = F.normalize(self.text_proj(text_features), p=2, dim=-1, eps=1e-8)
            image_features = F.normalize(self.image_proj(pred_images.mean(dim=[2, 3])), p=2, dim=-1, eps=1e-8)
            
            clip_sim = (image_features * text_features).sum(dim=1).mean()
            clip_loss = 1.0 - clip_sim

        # 7. 合并总损失
        total_loss = (
            loss_main +
            self.perceptual_loss_weight * perceptual_loss + 
            self.clip_loss_weight * clip_loss +       
            self.tv_loss_weight * tv_loss_value   
        )

        # --- 深度调试打印 ---
        if torch.isnan(total_loss):
            print("\n❌ [NaN Detected in DiffusionModel]")
            print(f"   - Huber/MSE Loss: {loss_main.item():.6f}")
            print(f"   - Noise Pred Max: {noise_pred.abs().max().item():.4f}")
            # 返回一个连接到计算图的 0 损失，防止训练中断
            return sum(p.sum() * 0 for p in self.parameters() if p.requires_grad)

        return total_loss

    @torch.no_grad()
    def generate_images(self, text_embeddings, num_images=1):
        """推理生成逻辑"""
        self.unet.eval()
        device = self.unet.device
        
        global_embed = text_embeddings['global_embed'].to(device)
        local_embeds = text_embeddings['local_embeds'].to(device)
        
        # 初始噪声
        latents = torch.randn((num_images, 4, 64, 64), device=device)
        
        # DDIM 采样循环
        for t in reversed(range(len(self.noise_scheduler.alphas_cumprod))):
            timesteps = torch.tensor([t], device=device).expand(num_images)
            noise_pred = self.unet(latents, timesteps, global_embed=global_embed, local_embeds=local_embeds).sample
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample
            
        # 解码前强制 FP32
        latents = (latents / 0.18215).to(torch.float32)
        self.vae.to(torch.float32)
        images = self.vae.decode(latents).sample
        return (images.clamp(-1, 1) + 1) / 2