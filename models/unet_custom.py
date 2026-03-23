# /home/610-sty/STY_T2i_v4.4/models/unet_custom.py
import torch
from torch import nn
from diffusers import UNet2DConditionModel
# 确保导入的是修改后的处理器
from models.attention import LocalInjectionAttnProcessor # 确保路径正确

class LightweightViT(nn.Module):
    """轻量级图像 patch 编码器"""
    def __init__(self, in_channels=3, patch_size=8, embed_dim=768):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

    def forward(self, x):  # x: (B, 3, H, W)
        x = self.patch_embed(x)  # (B, 768, H', W')
        x = x.flatten(2).permute(0, 2, 1)  # (B, N, 768)
        x = self.norm(x)
        return x  # (B, N, 768)

class CrossGuidanceBlock(nn.Module):
    """文本特征 ← 图像特征 的 cross-attention"""
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        # 初始化注意力权重
        self._init_weights()

    def _init_weights(self):
        # MultiheadAttention 的权重会在内部初始化，但我们确保 dropout 正确设置
        pass

    def forward(self, text_features, image_features):
        # text_features: (B, L_text, D)
        # image_features: (B, L_image, D)
        print(f"[CrossGuidance] Text features: {text_features.shape}, Image features: {image_features.shape}")
        
        attn_out, attn_weights = self.cross_attn(
            query=text_features,
            key=image_features,
            value=image_features
        )
        print(f"[CrossGuidance] Attention output: {attn_out.shape}")
        
        result = self.norm(text_features + self.dropout(attn_out))
        print(f"[CrossGuidance] Final output: {result.shape}")
        return result

class ContextFusionModule(nn.Module):
    """基于注意力的全局-局部上下文融合模块"""
    def __init__(self, embed_dim=768):
        super().__init__()
        # 全局向量关注局部上下文
        self.global_cross_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True, dropout=0.1)
        # 局部向量关注全局上下文
        self.local_cross_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True, dropout=0.1)
        # 融合门控机制
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        # 层归一化
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, global_embed, local_embeds):
        # global_embed: (B, 1, 768)
        # local_embeds: (B, K, 768)
        
        # 1. 全局向量通过局部上下文增强
        global_enhanced, _ = self.global_cross_attn(
            query=global_embed, 
            key=local_embeds, 
            value=local_embeds
        )
        global_enhanced = self.norm(global_embed + global_enhanced)
        
        # 2. 局部向量通过全局上下文增强
        global_broadcast = global_embed.expand(-1, local_embeds.size(1), -1)
        local_enhanced, _ = self.local_cross_attn(
            query=local_embeds,
            key=global_broadcast,
            value=global_broadcast
        )
        local_enhanced = self.norm(local_embeds + local_enhanced)
        
        # 3. 门控融合机制
        global_info = global_enhanced.expand(-1, local_embeds.size(1), -1)
        gate = self.fusion_gate(torch.cat([global_info, local_enhanced], dim=-1))
        fused_local = gate * global_info + (1 - gate) * local_enhanced
        
        # 4. 返回融合后的上下文 [全局增强向量, 融合后的局部向量]
        return torch.cat([global_enhanced, fused_local], dim=1)

class CustomUNet2DConditionModel(UNet2DConditionModel):
    def __init__(self, *args, **kwargs):
        # 首先调用父类初始化，加载预训练的 UNet 结构和权重
        super().__init__(*args, **kwargs)

        # 为融合后的 global+local embed 添加映射层
        target_cross_attn_dim = getattr(self.config, 'cross_attention_dim', 768)
        self.fused_text_proj = nn.Linear(768, target_cross_attn_dim)
        self.local_to_fused = nn.Linear(128, 768) # local_embeds 128 -> 768 for fusion
        
        # 初始化投影层权重
        nn.init.xavier_uniform_(self.fused_text_proj.weight)
        nn.init.zeros_(self.fused_text_proj.bias)
        nn.init.xavier_uniform_(self.local_to_fused.weight)
        nn.init.zeros_(self.local_to_fused.bias)

        # 新增智能融合模块
        self.context_fusion = ContextFusionModule(embed_dim=768)

        # 新增模块：图像编码器和交叉引导模块（暂时注释掉）
        # ✅ 暂时关闭这些模块以避免引入噪声
        # self.image_encoder = LightweightViT(in_channels=3, patch_size=8, embed_dim=768)
        # self.cross_guidance = CrossGuidanceBlock(embed_dim=768, num_heads=8)

        # --- 修正后的 Attention Processor 设置逻辑 ---
        processor_dict = {}
        attn_processor_names = list(self.attn_processors.keys())
        print(f"[UNet Init] Found {len(attn_processor_names)} attention processors")
        
        for name in attn_processor_names:
            is_cross_attn = "attn2" in name
            print(f"[UNet Init] Processing {name} (cross-attn: {is_cross_attn})")

            module_name = name[:-len(".processor")] if name.endswith(".processor") else name
            try:
                attn_module = self.get_submodule(module_name)
                cross_attention_dim = getattr(attn_module, 'cross_attention_dim', None)
                print(f"[UNet Init] Module {module_name}: cross_attention_dim = {cross_attention_dim}")
            except Exception as e:
                print(f"Warning: Could not get module {module_name} or its cross_attention_dim: {e}")
                cross_attention_dim = None

            processor = LocalInjectionAttnProcessor(
                local_dim=768,
                rank=4,
                cross_attention_dim=cross_attention_dim,
                save_attn=False
            )

            try:
                processor._init_lora(attn_module, cross_attention_dim)
                print(f"[UNet Init] Successfully initialized LoRA for {name}")
            except Exception as e:
                print(f"Warning: Failed to initialize LoRA for {name}: {e}")

            processor_dict[name] = processor

        self.set_attn_processor(processor_dict)
        print(f"[UNet Init] Successfully set {len(processor_dict)} attention processors")

        # 强制允许输出 attentions (如果需要)
        self.config.output_attentions = True

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states=None,
        global_embed=None,          # (B, 768)
        local_embeds=None,          # (B, K, 128)
        **kwargs
    ):
        """
        global_embed: (B, 768) -> 全局文本表示
        local_embeds: (B, K, 128) -> 局部关键词表示
        """
        print(f"[UNet Forward] Sample: {sample.shape}, Timestep: {timestep}")
        print(f"[UNet Forward] Global embed: {global_embed.shape if global_embed is not None else None}")
        print(f"[UNet Forward] Local embeds: {local_embeds.shape if local_embeds is not None else None}")

        fused_context = None
        if global_embed is not None:
            B, D_global = global_embed.shape
            # 1. 处理 global_embed: (B, 768) -> (B, 1, 768)
            global_part = global_embed.unsqueeze(1) # (B, 1, 768)
            print(f"[UNet Forward] Global part: {global_part.shape}")

            # 2. 处理 local_embeds: (B, K, 128) -> (B, K, 768)
            local_part = None
            if local_embeds is not None:
                local_part = self.local_to_fused(local_embeds) # (B, K, 768)
                print(f"[UNet Forward] Local part: {local_part.shape}")

            # 3. 使用智能融合模块替代简单拼接
            if local_part is not None:
                # 使用基于注意力的智能融合
                fused_context = self.context_fusion(global_part, local_part)
                print(f"[UNet Forward] Fused context (smart fusion): {fused_context.shape}")
            else:
                fused_context = global_part # (B, 1, 768)
                print(f"[UNet Forward] Fused context (global only): {fused_context.shape}")

            # 4. 投影到 UNet 的 cross_attention_dim
            fused_context = self.fused_text_proj(fused_context) # (B, 1+K, target_dim)
            print(f"[UNet Forward] Final context: {fused_context.shape}")

        # 过滤掉 diffusers 不支持的参数
        kwargs.pop("output_attentions", None)
        kwargs.pop("cross_attention_kwargs", None)

        print(f"[UNet Forward] Calling parent forward with encoder_hidden_states: {fused_context.shape if fused_context is not None else None}")
        
        result = super().forward(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=fused_context,
            **kwargs
        )
        
        print(f"[UNet Forward] Parent forward result sample: {result.sample.shape}")
        return result