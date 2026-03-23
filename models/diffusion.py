# /home/610-sty/STY_T2i_v4.4/models/diffusion.py
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

        # 加载预训练模块
        self.vae = AutoencoderKL.from_pretrained(config["pretrained_vae_path"])
        self.unet = CustomUNet2DConditionModel.from_pretrained(
            config["pretrained_unet_path"],
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=False
        )
        self.noise_scheduler = DDPMScheduler.from_pretrained(config["pretrained_scheduler_path"])
        self.prediction_type = config.get("prediction_type", "epsilon")
        # 调整感知损失权重
        self.perceptual_loss_weight = config.get("perceptual_loss_weight", 0.1)  # 降低到 0.1
        # 👆

        # 冻结 VAE 参数
        for param in self.vae.parameters():
            param.requires_grad = False

        # LPIPS 感知损失
        self.lpips_loss = lpips.LPIPS(net='alex')
        self.lpips_loss.eval()
        for p in self.lpips_loss.parameters():
            p.requires_grad = False

        # local_proj 保留在这里
        self.local_proj = nn.Linear(128, 768)

        # 自动迁移 noise_scheduler 所有 tensor 参数到 GPU
        scheduler_device = next(self.unet.parameters()).device
        for name, value in vars(self.noise_scheduler).items():
            if isinstance(value, torch.Tensor):
                setattr(self.noise_scheduler, name, value.to(scheduler_device))

        # 使用 Taiyi-CLIP 模型
        clip_model_path = "/home/610-sty/huggingface/Taiyi-CLIP-Roberta-large-326M-Chinese"
        self.clip_text_encoder = AutoModel.from_pretrained(clip_model_path).to(self.unet.device)
        self.clip_tokenizer = BertTokenizer.from_pretrained(clip_model_path)

        self.clip_text_encoder.eval()
        for p in self.clip_text_encoder.parameters():
            p.requires_grad = False

        # 添加投影层
        self.text_proj = nn.Linear(1024, 768)
        self.image_proj = nn.Linear(3, 768)

        # TV Loss - 进一步降低
        self.tv_loss_weight = 0.001  # ✅ 从 0.0001 提高到 0.001

    @staticmethod
    def tv_loss(img):
        batch_size, c, h, w = img.size()
        h_diff = torch.sum(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
        w_diff = torch.sum(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
        return (h_diff + w_diff) / (c * h * w)

    def train_step(self, images, text_embeddings):
        # 修正图像范围 - 确保在 [-1, 1] 范围内
        if images.min() < -1.5 or images.max() > 1.5:
            # print(f"Warning: Image range seems incorrect [{images.min():.3f}, {images.max():.3f}], clamping to [-1, 1]")
            images = torch.clamp(images, -1.0, 1.0)
        
        # 减少调试信息
        # print(f"Images shape: {images.shape}")
        # print(f"Images range: [{images.min():.3f}, {images.max():.3f}]")
        # print(f"Images mean/std: {images.mean():.3f}/{images.std():.3f}")

        images = images.to(dtype=torch.float32)
        device = self.unet.device

        global_embed = text_embeddings['global_embed'].to(device)
        local_embeds = text_embeddings['local_embeds'].to(device)
        # print(f"Global embed shape: {global_embed.shape}")
        # print(f"Local embeds shape: {local_embeds.shape}")

        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * 0.18215
        # print(f"Latents shape: {latents.shape}")

        noise = torch.randn_like(latents)
        # print(f"Noise shape: {noise.shape}")

        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=latents.device
        ).long()
        # print(f"Timesteps: {timesteps[:3]}...")

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        # print(f"Noisy latents shape: {noisy_latents.shape}")

        # 调用 unet.forward
        # print("Calling UNet forward...")
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            global_embed=global_embed,
            local_embeds=local_embeds
        ).sample
        # print(f"Noise pred shape: {noise_pred.shape}")

        # ======= Loss 计算 =======
        if self.prediction_type == "v_prediction":
            alpha_t = self.noise_scheduler.alphas_cumprod[timesteps].sqrt().view(-1, 1, 1, 1)
            sigma_t = (1 - self.noise_scheduler.alphas_cumprod[timesteps]).sqrt().view(-1, 1, 1, 1)
            v_target = alpha_t * noise - sigma_t * latents
            mse_loss = F.mse_loss(noise_pred, v_target)
        else:
            mse_loss = F.mse_loss(noise_pred, noise)
        # print(f"MSE loss: {mse_loss.item():.6f}")

        with torch.no_grad():
            pred_images = self.vae.decode(latents).sample
            gt_images = images

        # 感知损失
        perceptual_loss = self.lpips_loss(pred_images.clamp(-1, 1), gt_images.clamp(-1, 1)).mean()
        # print(f"Perceptual loss: {perceptual_loss.item():.6f}")

        # CLIP loss
        raw_texts = text_embeddings.get('raw_text', ["a photo"] * pred_images.shape[0])
        text_inputs = self.clip_tokenizer(
            raw_texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(device)
        text_outputs = self.clip_text_encoder(**text_inputs)
        text_features = text_outputs.last_hidden_state.mean(dim=1)
        text_features = self.text_proj(text_features)
        text_features = F.normalize(text_features, p=2, dim=-1)
        # print(f"Text features shape: {text_features.shape}")

        image_features = pred_images.mean(dim=[2, 3])
        image_features = self.image_proj(image_features)
        image_features = F.normalize(image_features, p=2, dim=-1)
        # print(f"Image features shape: {image_features.shape}")

        clip_sim = (image_features * text_features).sum(dim=1).mean()
        clip_loss = 1 - clip_sim
        # print(f"CLIP loss: {clip_loss.item():.6f}")

        tv_loss_value = self.tv_loss(pred_images)
        # print(f"TV loss: {tv_loss_value.item():.6f}")

        # 调整损失权重（关键修改点）
        total_loss = (
            mse_loss +
            0.1 * perceptual_loss +     # ✅ 降低感知损失权重从 0.1 到 0.05
            0.3 * clip_loss +            # ✅ 显著降低 CLIP 权重从 0.3 到 0.1
            0.01 * tv_loss_value        # ✅ 提高 TV loss 从 0.0001 到 0.001
        )
        # print(f"Total loss: {total_loss.item():.6f}")

        return total_loss

    def validation_step(self, images, text_embeddings):
        return self.train_step(images, text_embeddings)

    @torch.no_grad()
    def generate_images(self, text_embeddings, save_path=None, num_images=1, image_hint=None):
        """
        生成图像
        """
        # 减少调试信息
        # print(f"Num images: {num_images}")
        # print(f"Image hint provided: {image_hint is not None}")
        
        self.unet.eval()
        device = self.unet.device

        global_embed = text_embeddings['global_embed'].to(device)
        local_embeds = text_embeddings['local_embeds'].to(device)
        # print(f"Global embed shape (gen): {global_embed.shape}")
        # print(f"Local embeds shape (gen): {local_embeds.shape}")

        latents = torch.randn(
            (num_images, self.unet.config.in_channels, 64, 64),
            device=device
        ) * 0.18215
        # print(f"Initial latents shape: {latents.shape}")

        for t in reversed(range(self.noise_scheduler.config.num_train_timesteps)):
            timesteps = torch.tensor(t, dtype=torch.long, device=device)
            self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device)

            noise_pred = self.unet(
                latents,
                timesteps,
                global_embed=global_embed,
                local_embeds=local_embeds
            ).sample

            latents = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=timesteps,
                sample=latents
            ).prev_sample

        latents = latents / 0.18215
        images = self.vae.decode(latents).sample
        images = (images.clamp(-1, 1) + 1) / 2
        # print(f"Final generated images shape: {images.shape}")

        if save_path:
            save_image(images, save_path, nrow=2)

        return images