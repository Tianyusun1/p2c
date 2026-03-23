# /home/610-sty/compare/poe2clp/models/unet_custom.py
import torch
from torch import nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
from models.attention import LocalInjectionAttnProcessor 

# --- 1. 定义辅助模块 ---

class ContextFusionModule(nn.Module):
    """
    基于注意力的全局-局部上下文融合模块 (极高稳定性版)
    """
    def __init__(self, embed_dim=768):
        super().__init__()
        # 交叉注意力层
        self.global_cross_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True, dropout=0.1)
        self.local_cross_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True, dropout=0.1)
        
        # 多重归一化层，确保数值始终在安全范围内
        self.norm_in_g = nn.LayerNorm(embed_dim)
        self.norm_in_l = nn.LayerNorm(embed_dim)
        self.norm_final = nn.LayerNorm(embed_dim)

        # 融合门控
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, global_embed, local_embeds):
        # 1. 输入防御性裁剪与归一化
        g_in = self.norm_in_g(global_embed.clamp(-50, 50))
        l_in = self.norm_in_l(local_embeds.clamp(-50, 50))
        
        # 2. 互注意增强
        # 全局向量关注局部短语
        g_enhanced, _ = self.global_cross_attn(query=g_in, key=l_in, value=l_in)
        g_res = self.norm_final(g_in + self.dropout(g_enhanced))
        
        # 局部短语关注全局信息
        l_enhanced, _ = self.local_cross_attn(query=l_in, key=g_in, value=g_in)
        l_res = self.norm_final(l_in + self.dropout(l_enhanced))
        
        # 3. 门控融合
        g_info = g_res.expand(-1, l_res.size(1), -1)
        # 拼接特征并计算权重
        gate_input = torch.cat([g_info, l_res], dim=-1)
        gate = self.fusion_gate(gate_input)
        
        # 采用 lerp 风格融合，数值更稳定
        fused_local = gate * g_info + (1.0 - gate) * l_res
        
        # 4. 最终拼接：[1个全局增强, K个局部融合]
        out = torch.cat([g_res, fused_local], dim=1)
        return self.norm_final(out).clamp(-30, 30)

# --- 2. 定义核心 Custom UNet 类 ---

class CustomUNet2DConditionModel(UNet2DConditionModel):
    def __init__(self, *args, **kwargs):
        # 首先调用父类初始化加载底座权重
        super().__init__(*args, **kwargs)

        # 自动探测底座维度
        self.target_dim = getattr(self.config, 'cross_attention_dim', 768)
        print(f"[UNet Init] Stable Diffusion target cross_attention_dim: {self.target_dim}")

        # 1. 投影层：将融合后的 768 维特征映射到实际底座维度 (如 1280)
        self.fused_text_proj = nn.Linear(768, self.target_dim)
        self.local_to_fused = nn.Linear(128, 768)
        self.context_fusion = ContextFusionModule(embed_dim=768)

        # 2. 增强初始化保护
        with torch.no_grad():
            if self.target_dim == 768:
                nn.init.eye_(self.fused_text_proj.weight)
            else:
                # 针对 1280 维使用极小的分布，防止初始输出过大
                nn.init.trunc_normal_(self.fused_text_proj.weight, std=0.01)
            nn.init.zeros_(self.fused_text_proj.bias)
            
            nn.init.xavier_uniform_(self.local_to_fused.weight)
            nn.init.zeros_(self.local_to_fused.bias)

        # 3. 注入 Attention 处理器 (LoRA)
        self._setup_custom_processors()
        
        self.config.output_attentions = True

    def _setup_custom_processors(self):
        processor_dict = {}
        # 遍历所有注意力层
        for name in self.attn_processors.keys():
            is_cross_attn = "attn2" in name
            module_name = name[:-len(".processor")] if name.endswith(".processor") else name
            
            try:
                attn_module = self.get_submodule(module_name)
                c_dim = getattr(attn_module, 'cross_attention_dim', None)
            except:
                c_dim = None

            # 创建自定义注入处理器
            processor = LocalInjectionAttnProcessor(
                local_dim=768, 
                rank=4,
                cross_attention_dim=c_dim
            )

            if is_cross_attn:
                processor._init_lora(attn_module, c_dim)
            
            processor_dict[name] = processor

        self.set_attn_processor(processor_dict)

    def forward(self, sample, timestep, encoder_hidden_states=None, global_embed=None, local_embeds=None, **kwargs):
        """
        拦截输入并执行全局-局部融合
        """
        # 如果没有外部传入 embed，保持原样（兼容原生接口）
        fused_context = encoder_hidden_states

        if global_embed is not None:
            # 1. 预处理
            g_part = global_embed.unsqueeze(1) # (B, 1, 768)
            l_part = self.local_to_fused(local_embeds) if local_embeds is not None else None
            
            # 2. 融合逻辑
            if l_part is not None:
                # 融合结果包含 1+K 个 token
                fused = self.context_fusion(g_part, l_part)
            else:
                fused = g_part

            # 3. 最终投影并检查 NaN
            fused_context = self.fused_text_proj(fused)
            
            if torch.isnan(fused_context).any():
                # 若在此处发现 NaN，将其替换为 0 并警告
                fused_context = torch.where(torch.isnan(fused_context), torch.zeros_like(fused_context), fused_context)

        # 移除层不支持的冗余参数
        kwargs.pop("output_attentions", None)
        kwargs.pop("cross_attention_kwargs", None)

        return super().forward(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=fused_context,
            **kwargs
        )