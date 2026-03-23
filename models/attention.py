# /home/610-sty/STY_T2i_v4.4/models/attention.py
import torch
import torch.nn as nn
from diffusers.models.attention_processor import AttnProcessor

# 修改为包含LoRA和局部注入的处理器
class LocalInjectionAttnProcessor(nn.Module): # 继承 nn.Module
    def __init__(self, local_dim: int = 768, save_attn: bool = False, rank: int = 4, cross_attention_dim: int = None):
        super().__init__() # 调用 nn.Module 的 __init__
        self.local_dim = local_dim
        self.save_attn = save_attn
        self.latest_attention_map = None
        self.rank = rank
        self.cross_attention_dim = cross_attention_dim # 保存 cross_attention_dim

        # LoRA 参数 (使用 register_module 或直接赋值为属性，使其成为子模块)
        # 初始化为 None，稍后在 _init_lora 中创建
        self.to_q_lora = None
        self.to_k_lora = None
        self.to_v_lora = None
        self.to_out_lora = None

    def _init_lora(self, attn_module, cross_attention_dim):
        """初始化 LoRA 参数，并将其注册为子模块"""
        in_features = attn_module.to_q.in_features
        out_features = attn_module.to_q.out_features
        hidden_size = attn_module.to_k.in_features

        # LoRA 矩阵 ✅ 添加显式初始化
        self.to_q_lora = nn.Sequential(
            nn.Linear(in_features, self.rank, bias=False),
            nn.Linear(self.rank, out_features, bias=False)
        )
        # Key 的输入维度取决于 cross_attention_dim (cross-attn) 或 hidden_size (self-attn)
        k_in_features = cross_attention_dim if cross_attention_dim is not None else hidden_size
        self.to_k_lora = nn.Sequential(
            nn.Linear(k_in_features, self.rank, bias=False),
            nn.Linear(self.rank, out_features, bias=False)
        )
        self.to_v_lora = nn.Sequential(
            nn.Linear(k_in_features, self.rank, bias=False),
            nn.Linear(self.rank, out_features, bias=False)
        )
        self.to_out_lora = nn.Sequential(
            nn.Linear(out_features, self.rank, bias=False),
            nn.Linear(self.rank, out_features, bias=False)
        )
        
        # ✅ 显式初始化 LoRA 参数
        self._init_lora_weights()

    def _init_lora_weights(self):
        """初始化 LoRA 权重，使用较小的随机值"""
        if self.to_q_lora is not None:
            nn.init.xavier_uniform_(self.to_q_lora[0].weight)
            nn.init.zeros_(self.to_q_lora[1].weight)  # 输出层初始化为0，避免初始干扰
            
        if self.to_k_lora is not None:
            nn.init.xavier_uniform_(self.to_k_lora[0].weight)
            nn.init.zeros_(self.to_k_lora[1].weight)
            
        if self.to_v_lora is not None:
            nn.init.xavier_uniform_(self.to_v_lora[0].weight)
            nn.init.zeros_(self.to_v_lora[1].weight)
            
        if self.to_out_lora is not None:
            nn.init.xavier_uniform_(self.to_out_lora[0].weight)
            nn.init.zeros_(self.to_out_lora[1].weight)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **kwargs
    ):
        # 注意：local_context 不再通过 kwargs 传递，而是融合在 encoder_hidden_states 中
        # 但保留此逻辑以兼容性（如果仍被调用）
        local_context = kwargs.pop("local_context", None)

        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        device = hidden_states.device
        query = attn.to_q(hidden_states)
        # 应用 LoRA to query
        if self.to_q_lora is not None:
            query_lora = self.to_q_lora(hidden_states)
            query += query_lora
        query = attn.head_to_batch_dim(query)

        # 处理 Key 和 Value
        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
            # 应用 LoRA to key and value
            if self.to_k_lora is not None:
                key_lora = self.to_k_lora(encoder_hidden_states)
                key += key_lora
            if self.to_v_lora is not None:
                value_lora = self.to_v_lora(encoder_hidden_states)
                value += value_lora
        else:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        if self.save_attn and not self.training:
            self.latest_attention_map = attention_probs.detach().cpu()

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # 应用 to_out 和 LoRA to_out
        residual = hidden_states
        hidden_states = attn.to_out[0](hidden_states)
        if self.to_out_lora is not None:
            hidden_states_lora = self.to_out_lora(residual)
            hidden_states += hidden_states_lora
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states