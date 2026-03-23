# /home/610-sty/compare/poe2clp/models/unet_custom.py
import torch
from torch import nn
from diffusers import UNet2DConditionModel
from models.attention import LocalInjectionAttnProcessor 

class ContextFusionModule(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.global_cross_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True, dropout=0.1)
        self.local_cross_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True, dropout=0.1)
        self.norm_in_g = nn.LayerNorm(embed_dim)
        self.norm_in_l = nn.LayerNorm(embed_dim)
        self.norm_final = nn.LayerNorm(embed_dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Sigmoid()
        )

    def forward(self, global_embed, local_embeds, local_mask=None):
        g_in = self.norm_in_g(global_embed.nan_to_num().clamp(-50, 50))
        l_in = self.norm_in_l(local_embeds.nan_to_num().clamp(-50, 50))
        
        # ✅ 防御 NaN：如果 local_mask 全为空，跳过 Attention
        if local_mask is not None and not local_mask.any():
            g_enhanced = torch.zeros_like(g_in)
        else:
            g_enhanced, _ = self.global_cross_attn(
                query=g_in, key=l_in, value=l_in, 
                key_padding_mask=~local_mask if local_mask is not None else None
            )
        
        g_res = self.norm_final(g_in + g_enhanced)
        l_res = self.norm_final(l_in + self.local_cross_attn(query=l_in, key=g_in, value=g_in)[0])
        
        gate = self.fusion_gate(torch.cat([g_res.expand(-1, l_res.size(1), -1), l_res], dim=-1))
        fused_local = gate * g_res.expand(-1, l_res.size(1), -1) + (1.0 - gate) * l_res
        
        return self.norm_final(torch.cat([g_res, fused_local], dim=1)).clamp(-30, 30)

class CustomUNet2DConditionModel(UNet2DConditionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_dim = getattr(self.config, 'cross_attention_dim', 768)
        self.fused_text_proj = nn.Linear(768, self.target_dim)
        self.local_to_fused = nn.Linear(128, 768)
        self.context_fusion = ContextFusionModule(embed_dim=768)
        self._setup_custom_processors()

    def forward(self, sample, timestep, encoder_hidden_states=None, global_embed=None, local_embeds=None, local_mask=None, **kwargs):
        fused_context = encoder_hidden_states
        if global_embed is not None:
            fused = self.context_fusion(global_embed.unsqueeze(1), self.local_to_fused(local_embeds), local_mask=local_mask)
            fused_context = self.fused_text_proj(fused).nan_to_num()

        return super().forward(sample=sample, timestep=timestep, encoder_hidden_states=fused_context, **kwargs)

    def _setup_custom_processors(self):
        processor_dict = {}
        for name in self.attn_processors.keys():
            attn_module = self.get_submodule(name[:-len(".processor")] if name.endswith(".processor") else name)
            c_dim = getattr(attn_module, 'cross_attention_dim', None)
            processor = LocalInjectionAttnProcessor(local_dim=768, rank=4, cross_attention_dim=c_dim)
            if "attn2" in name: processor._init_lora(attn_module, c_dim)
            processor_dict[name] = processor
        self.set_attn_processor(processor_dict)