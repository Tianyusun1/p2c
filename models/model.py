# /home/610-sty/STY_T2i_v4.4/models/model.py
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

        # 冻结 Taiyi-CLIP 主干
        for param in self.text_embedding.roberta.parameters():
            param.requires_grad = False
        print("Frozen Taiyi-CLIP RoBERTa parameters.")

        diffusion_config = {
            "pretrained_model_name_or_path": "/home/610-sty/huggingface/stable-diffusion-v1-5",
            "pretrained_vae_path": "/home/610-sty/huggingface/stable-diffusion-v1-5/vae",
            "pretrained_unet_path": "/home/610-sty/huggingface/stable-diffusion-v1-5/unet",
            "pretrained_scheduler_path": "/home/610-sty/huggingface/stable-diffusion-v1-5/scheduler",
            "prediction_type": "epsilon",
            "noise_scheduler_type": "ddim",
            "perceptual_loss_weight": 0.1
        }

        self.diffusion_model = DiffusionModel(config=diffusion_config)

        # 冻结 VAE decoder
        for param in self.diffusion_model.vae.decoder.parameters():
            param.requires_grad = False
        print("Frozen VAE decoder parameters.")

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

        # 减少调试信息
        # print(f"\n=== EPOCH {self.current_epoch}, BATCH {batch_idx} ===")
        # print(f"Batch size: {len(poems)}")
        # print(f"Images shape: {images.shape}")
        # print(f"Sample text: {poems[0] if poems else 'N/A'}")

        text_embeddings = self.text_embedding(poems, phrases, is_inference=False)
        text_embeddings["raw_text"] = poems

        total_loss = self.diffusion_model.train_step(images, text_embeddings)

        self.clip_gradients(self.trainer.optimizers[0], gradient_clip_val=1.0)

        self.log("train_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
        
        # print(f"Batch loss: {total_loss.item():.6f}")
        # print(f"=== END BATCH {batch_idx} ===\n")
        
        return total_loss

    # 👇 新增方法: 保存短语级 Attention Maps
    def save_phrase_attention_maps(self, text_embeddings, poems, save_dir, epoch):
        """
        保存短语级注意力图
        """
        import matplotlib.pyplot as plt
        import os

        os.makedirs(save_dir, exist_ok=True)

        phrase_attention = text_embeddings.get("phrase_attention")  # (B, K)
        phrase_spans = text_embeddings.get("phrase_spans")          # List[List[Tuple]]
        phrase_scores = text_embeddings.get("phrase_scores")        # (B, K)

        if phrase_attention is None or phrase_spans is None:
            print("Warning: Missing phrase_attention or phrase_spans. Skipping.")
            return

        for i, (poem, attn, spans, scores) in enumerate(zip(poems, phrase_attention, phrase_spans, phrase_scores)):
            # 获取 tokens
            poem_tokens = self.tokenizer(
                poem,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=77
            )
            input_ids = poem_tokens['input_ids'][0]
            if hasattr(self.tokenizer, 'tokenizer') and hasattr(self.tokenizer.tokenizer, 'convert_ids_to_tokens'):
                tokens = self.tokenizer.tokenizer.convert_ids_to_tokens(input_ids)
            else:
                id_list = input_ids.tolist()
                tokens = [f"ID_{id_}" for id_ in id_list]

            # 构造短语文本
            phrase_texts = []
            for start, end in spans:
                if start == 0 and end == 0:  # padding
                    phrase_texts.append("<PAD>")
                else:
                    phrase_texts.append("".join(tokens[start:end+1]))

            # 绘图
            plt.figure(figsize=(max(8, len(phrase_texts) * 1.5), 4))
            attn_np = attn.detach().cpu().numpy()
            scores_np = scores.detach().cpu().numpy()
            bars = plt.bar(range(len(phrase_texts)), attn_np, color='skyblue')
            plt.xlabel('Phrases')
            plt.ylabel('Attention Weight')
            plt.title(f'Phrase Attention - Sample {i+1} - Epoch {epoch:03d}')
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
            print(f"Saved phrase attention for sample {i+1} to {save_path}")

    def on_train_epoch_end(self):
        # print(f"\n=== EPOCH {self.current_epoch} END ===")
        train_loader = self.trainer.datamodule.train_dataloader()
        try:
            sample_batch = next(iter(train_loader))
        except StopIteration:
            print("警告：训练数据加载器为空，跳过验证图像生成。")
            return

        num_images = min(4, sample_batch["instance_image"].shape[0])
        sample_poems = sample_batch["instance_prompt"][:num_images]
        real_images = sample_batch["instance_image"][:num_images]

        # print(f"Generating validation images for {num_images} samples...")
        # print(f"Sample texts: {sample_poems}")

        # 使用 is_inference=True 获取 phrase_attention
        text_embeddings = self.text_embedding(sample_poems, is_inference=True)
        text_embeddings["raw_text"] = sample_poems

        # 保存短语注意力图
        if "phrase_attention" in text_embeddings:
            self.save_phrase_attention_maps(
                text_embeddings=text_embeddings,
                poems=sample_poems,
                save_dir="outputs/phrase_attention_maps",
                epoch=self.current_epoch
            )

        # 使用 EMA 权重生成图像
        self.ema.ema_model.eval()
        original_unet = self.diffusion_model.unet
        self.diffusion_model.unet = self.ema.ema_model

        try:
            # print("Generating images with EMA model...")
            generated_images = self.diffusion_model.generate_images(
                text_embeddings=text_embeddings,
                num_images=num_images
            )
            # print(f"Generated images shape: {generated_images.shape}")
        finally:
            self.diffusion_model.unet = original_unet

        real_images = real_images.to(generated_images.device)
        real_images = (real_images.clamp(-1, 1) + 1) / 2

        comparison = torch.cat([real_images, generated_images], dim=0)
        grid = torchvision.utils.make_grid(comparison, nrow=num_images)
        save_dir = "outputs"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"comparison_epoch_{self.current_epoch:03d}.png")
        torchvision.utils.save_image(grid, save_path)
        print(f"已保存真图+生成对比: {save_path}")

        with open(os.path.join(save_dir, f"comparison_epoch_{self.current_epoch:03d}.txt"), "w", encoding="utf-8") as f:
            for i, poem in enumerate(sample_poems):
                f.write(f"{i+1}: {poem}\n")
        
        # print(f"=== EPOCH {self.current_epoch} VALIDATION COMPLETE ===\n")

    def configure_optimizers(self):
        # 1. 获取 LoRA 参数
        lora_params = []
        lora_param_names = []
        for name, module in self.diffusion_model.unet.named_modules():
            if hasattr(module, "processor") and hasattr(module.processor, 'to_q_lora'):
                if module.processor.to_q_lora is not None:
                    lora_params.extend(list(module.processor.to_q_lora.parameters()))
                    lora_param_names.extend([f"{name}.to_q_lora.{i}" for i in range(len(list(module.processor.to_q_lora.parameters())))])
                if module.processor.to_k_lora is not None:
                    lora_params.extend(list(module.processor.to_k_lora.parameters()))
                    lora_param_names.extend([f"{name}.to_k_lora.{i}" for i in range(len(list(module.processor.to_k_lora.parameters())))])
                if module.processor.to_v_lora is not None:
                    lora_params.extend(list(module.processor.to_v_lora.parameters()))
                    lora_param_names.extend([f"{name}.to_v_lora.{i}" for i in range(len(list(module.processor.to_v_lora.parameters())))])
                if module.processor.to_out_lora is not None:
                    lora_params.extend(list(module.processor.to_out_lora.parameters()))
                    lora_param_names.extend([f"{name}.to_out_lora.{i}" for i in range(len(list(module.processor.to_out_lora.parameters())))])
        
        print(f"Found {len(lora_params)} LoRA parameters")

        # 2. 获取自定义模块的可训练参数
        # ✅ 方案一：禁用交叉引导模块（用于测试）
        custom_trainable_modules = [
            self.text_embedding,
            self.diffusion_model.local_proj,
            self.diffusion_model.text_proj,
            self.diffusion_model.image_proj,
            self.diffusion_model.unet.fused_text_proj,
            self.diffusion_model.unet.local_to_fused,
            # self.diffusion_model.unet.image_encoder,      # ❌ 暂时禁用
            # self.diffusion_model.unet.cross_guidance,     # ❌ 暂时禁用
        ]
        
        # ✅ 方案二：启用交叉引导模块（优化版本）
        # custom_trainable_modules = [
        #     self.text_embedding,
        #     self.diffusion_model.local_proj,
        #     self.diffusion_model.text_proj,
        #     self.diffusion_model.image_proj,
        #     self.diffusion_model.unet.fused_text_proj,
        #     self.diffusion_model.unet.local_to_fused,
        #     self.diffusion_model.unet.image_encoder,      # ✅ 启用
        #     self.diffusion_model.unet.cross_guidance,     # ✅ 启用
        # ]
        
        custom_params = []
        custom_param_names = []
        for module in custom_trainable_modules:
            if module is not None:
                params = [p for p in module.parameters() if p.requires_grad]
                custom_params.extend(params)
                # 获取参数名称用于调试
                for name, param in module.named_parameters():
                    if param.requires_grad:
                        custom_param_names.append(f"{module.__class__.__name__}.{name}")
        
        print(f"Found {len(custom_params)} custom parameters")

        # 3. 合并参数
        trainable_params = lora_params + custom_params

        print(f"LoRA 参数数量: {len(lora_params)}")
        print(f"自定义可训练参数数量: {len(custom_params)}")
        print(f"总优化参数数量: {len(trainable_params)}")

        if not trainable_params:
            raise ValueError("没有找到需要优化的参数！请检查模型配置。")

        # 4. 配置优化器 - 调整学习率（关键修改点）
        # LoRA 参数使用较低学习率
        lora_lr = 5e-5  # ✅ 从 1e-4 降到 5e-5
        # 新模块参数使用适中学习率
        custom_lr = 1e-5  # ✅ 从 1e-4 降到 1e-5 以保持稳定
        
        print(f"LoRA learning rate: {lora_lr}")
        print(f"Custom modules learning rate: {custom_lr}")

        optimizer = torch.optim.AdamW([
            {"params": lora_params, "lr": lora_lr, "weight_decay": 0.01},
            {"params": custom_params, "lr": custom_lr, "weight_decay": 0.01}
        ])

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            min_lr=1e-7
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