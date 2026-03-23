# /home/610-sty/STY_T2i_v4.4/models/embedding.py
import torch
import torch.nn as nn
from transformers import AutoModel
import jieba.analyse

class PhraseAttentionExtractor(nn.Module):
    """
    改进的关键词提取器：支持短语级提取 + 注意力机制
    """
    def __init__(self, encoder: nn.Module, max_phrases: int, phrase_dim: int, max_phrase_len: int = 5):
        super().__init__() # 修正语法错误
        self.encoder = encoder
        self.max_phrases = max_phrases
        self.phrase_dim = phrase_dim
        self.max_phrase_len = max_phrase_len
        hidden_size = encoder.config.hidden_size

        # 跨度评分网络：融合起始、结束、平均、差值特征
        self.span_scorer = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.LayerNorm(hidden_size), 
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )

        # 短语特征投影
        self.proj = nn.Linear(hidden_size, phrase_dim)

        # 注意力门控机制
        self.attention_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

        # 最终输出归一化
        self.out_norm = nn.LayerNorm(phrase_dim)

    def forward(self, input_ids, attention_mask):
        B, L = input_ids.shape
        device = input_ids.device

        # 获取编码器输出
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, L, D)
        mask = attention_mask.bool()  # (B, L)

        batch_phrase_embs = []
        batch_phrase_masks = []
        batch_phrase_scores = []
        batch_phrase_spans = []

        # 逐样本处理
        for b in range(B):
            valid_spans = []
            
            for i in range(L):
                if not mask[b, i]: continue
                for j in range(i, min(L, i + self.max_phrase_len)):
                    if not mask[b, j]: break

                    start_emb = hidden[b, i]
                    end_emb = hidden[b, j]
                    span_emb = hidden[b, i:j+1]
                    mean_emb = span_emb.mean(dim=0)
                    diff_emb = end_emb - start_emb

                    feature = torch.cat([start_emb, end_emb, mean_emb, diff_emb], dim=-1)
                    score = self.span_scorer(feature) # (1)

                    valid_spans.append((score, i, j, mean_emb))

            # 修正点 1：确保排序键是标量，避免 Tensor 比较报错
            valid_spans.sort(key=lambda x: x[0].item(), reverse=True)
            
            topk = min(self.max_phrases, len(valid_spans))
            topk_list = valid_spans[:topk]
            
            topk_features = [s[3] for s in topk_list]
            topk_indices = [(s[1], s[2]) for s in topk_list]
            topk_scores = [s[0].reshape(1) for s in topk_list] # 统一形状

            # 填充补齐
            while len(topk_features) < self.max_phrases:
                topk_features.append(torch.zeros(hidden.size(-1), device=device))
                topk_indices.append((0, 0))
                topk_scores.append(torch.tensor([-10.0], device=device)) # 低分填充

            batch_phrase_embs.append(torch.stack(topk_features))
            # 修正点 2：显式拼接分数，确保维度为 (max_phrases)
            batch_phrase_scores.append(torch.cat(topk_scores, dim=0))
            batch_phrase_masks.append(torch.tensor([1]*topk + [0]*(self.max_phrases - topk), dtype=torch.bool, device=device))
            batch_phrase_spans.append(topk_indices)

        # 堆叠 Batch
        phrase_embs = torch.stack(batch_phrase_embs)  # (B, K, D)
        phrase_scores = torch.stack(batch_phrase_scores)  # (B, K)
        phrase_masks = torch.stack(batch_phrase_masks)  # (B, K)

        # 注意力权重计算
        gate_logits = self.attention_gate(phrase_embs).squeeze(-1)
        gate_logits = gate_logits.masked_fill(~phrase_masks, -1e9)
        phrase_attention = torch.softmax(gate_logits, dim=-1)

        # 投影与归一化
        phrase_embeds = self.out_norm(self.proj(phrase_embs))

        return {
            "phrase_embeds": phrase_embeds,
            "phrase_masks": phrase_masks,
            "phrase_attention": phrase_attention,
            "phrase_scores": phrase_scores,
            "phrase_spans": batch_phrase_spans
        }

class EnhancedChineseTextEmbedding(nn.Module):
    def __init__(self, tokenizer, embed_dim=768, phrase_dim=128, max_phrases=5, use_learnable_extractor=True):
        super().__init__()
        self.tokenizer = tokenizer
        
        taiyi_sd_path = '/home/610-sty/huggingface/Taiyi-Stable-Diffusion-1B-Chinese-v0.1'
        self.text_encoder = AutoModel.from_pretrained(f"{taiyi_sd_path}/text_encoder")
        self.roberta = self.text_encoder.base_model
        
        self.max_phrases = max_phrases
        self.use_learnable_extractor = use_learnable_extractor

        self.global_proj = nn.Linear(self.text_encoder.config.hidden_size, embed_dim)
        self.phrase_proj = nn.Linear(self.text_encoder.config.hidden_size, phrase_dim)

        # ✅ 增加输出归一化层，杜绝向量量级过大导致 NaN
        self.out_norm_global = nn.LayerNorm(embed_dim)
        self.out_norm_local = nn.LayerNorm(phrase_dim)

        if self.use_learnable_extractor:
            self.keyword_extractor = PhraseAttentionExtractor(
                encoder=self.roberta,
                max_phrases=self.max_phrases,
                phrase_dim=phrase_dim
            )

        self.register_buffer('phrase_pad', torch.zeros(phrase_dim))

    def extract_keywords_static(self, poem, top_k=5):
        return jieba.analyse.extract_tags(poem, topK=top_k)

    def forward(self, poems, phrases=None, is_inference=False):
        device = next(self.parameters()).device
        
        poem_tokens = self.tokenizer(
            poems, return_tensors="pt", padding=True, truncation=True, max_length=77
        ).to(device)
        
        poem_output = self.roberta(**poem_tokens).last_hidden_state
        # 全局特征：取 [CLS] 并归一化
        global_embed = self.out_norm_global(self.global_proj(poem_output[:, 0]))

        if (is_inference or phrases is None) and self.use_learnable_extractor:
            phrase_data = self.keyword_extractor(poem_tokens.input_ids, poem_tokens.attention_mask)
            return {
                "global_embed": global_embed,
                "local_embeds": phrase_data["phrase_embeds"],
                "local_mask": phrase_data["phrase_masks"], # ✅ 这个 mask 会传给 UNet 解决报错
                "raw_text": poems,
                "phrase_attention": phrase_data["phrase_attention"],
                "phrase_scores": phrase_data["phrase_scores"],
                "phrase_spans": phrase_data["phrase_spans"]
            }

        # 静态提取或训练阶段
        if is_inference:
            all_phrases = [self.extract_keywords_static(p, top_k=self.max_phrases) for p in poems]
        else:
            all_phrases = [p[:self.max_phrases] if p else [] for p in phrases]
            
        local_embeds, local_mask = self.encode_phrases(all_phrases, device)
        
        # ✅ 对短语也进行归一化
        local_embeds = self.out_norm_local(local_embeds)

        return {
            "global_embed": global_embed,
            "local_embeds": local_embeds,
            "local_mask": local_mask,
            "raw_text": poems
        }

    def encode_phrases(self, all_phrases, device):
        batch_local_embeds = []
        batch_local_mask = []

        for phrase_list in all_phrases:
            if not phrase_list:
                batch_local_embeds.append(self.phrase_pad.unsqueeze(0).expand(self.max_phrases, -1))
                batch_local_mask.append(torch.zeros(self.max_phrases, dtype=torch.bool, device=device))
                continue

            phrase_tokens = self.tokenizer(
                phrase_list, return_tensors="pt", padding=True, truncation=True, max_length=10
            ).to(device)
            
            phrase_out = self.roberta(**phrase_tokens).last_hidden_state[:, 0]
            phrase_embed = self.phrase_proj(phrase_out)

            # 填充逻辑
            if phrase_embed.size(0) < self.max_phrases:
                pad_len = self.max_phrases - phrase_embed.size(0)
                pad = self.phrase_pad.unsqueeze(0).expand(pad_len, -1)
                phrase_embed = torch.cat([phrase_embed, pad], dim=0)
                mask = torch.tensor([True] * len(phrase_list) + [False] * pad_len, device=device)
            else:
                phrase_embed = phrase_embed[:self.max_phrases]
                mask = torch.tensor([True] * self.max_phrases, device=device)

            batch_local_embeds.append(phrase_embed)
            batch_local_mask.append(mask)

        return torch.stack(batch_local_embeds), torch.stack(batch_local_mask)