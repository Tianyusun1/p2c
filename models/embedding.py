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
        super().__init__()
        self.encoder = encoder
        self.max_phrases = max_phrases
        self.phrase_dim = phrase_dim
        self.max_phrase_len = max_phrase_len
        hidden_size = encoder.config.hidden_size

        # 跨度评分网络：融合起始、结束、平均、差值特征
        self.span_scorer = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )

        # 短语特征投影
        self.proj = nn.Linear(hidden_size, phrase_dim)

        # 注意力门控机制（用于融合多个短语）
        self.attention_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

        # 可选：增加位置编码（增强位置感知）
        self.position_emb = nn.Embedding(max_phrase_len, hidden_size)

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
        batch_phrase_spans = []  # 用于调试/可视化

        # 逐样本处理
        for b in range(B):
            valid_spans = []
            span_features = []

            # 枚举所有合法短语跨度
            for i in range(L):
                if not mask[b, i]: continue
                for j in range(i, min(L, i + self.max_phrase_len)):
                    if not mask[b, j]: break

                    start_emb = hidden[b, i]  # (D,)
                    end_emb = hidden[b, j]    # (D,)
                    span_emb = hidden[b, i:j+1]  # (len, D)
                    mean_emb = span_emb.mean(dim=0)  # (D,)
                    diff_emb = end_emb - start_emb  # (D,)

                    # 拼接特征
                    feature = torch.cat([start_emb, end_emb, mean_emb, diff_emb], dim=-1)  # (4*D,)
                    score = self.span_scorer(feature).squeeze()  # scalar

                    valid_spans.append((score, i, j))
                    span_features.append(mean_emb)  # 使用平均池化作为短语表示

            # 按得分排序，取 top-k
            valid_spans.sort(key=lambda x: x[0], reverse=True)
            topk = min(self.max_phrases, len(valid_spans))
            topk_spans = valid_spans[:topk]
            topk_features = [span_features[valid_spans.index(s)] for s in topk_spans]

            # 填充至 max_phrases
            while len(topk_features) < self.max_phrases:
                topk_features.append(torch.zeros_like(span_features[0]))
            while len(topk_spans) < self.max_phrases:
                topk_spans.append((torch.tensor(-float('inf')).to(device), 0, 0))

            # 堆叠
            phrase_embs = torch.stack(topk_features)  # (K, D)
            phrase_scores = torch.stack([s for s, _, _ in topk_spans])  # (K,)
            phrase_mask = torch.tensor([1]*topk + [0]*(self.max_phrases - topk), dtype=torch.bool, device=device)

            batch_phrase_embs.append(phrase_embs)
            batch_phrase_scores.append(phrase_scores)
            batch_phrase_masks.append(phrase_mask)
            batch_phrase_spans.append([(i, j) for _, i, j in topk_spans])

        # 堆叠 batch
        phrase_embs = torch.stack(batch_phrase_embs)  # (B, K, D)
        phrase_scores = torch.stack(batch_phrase_scores)  # (B, K)
        phrase_masks = torch.stack(batch_phrase_masks)  # (B, K)

        # 注意力权重计算（门控机制）
        gate_logits = self.attention_gate(phrase_embs).squeeze(-1)  # (B, K)
        gate_logits = gate_logits.masked_fill(~phrase_masks, float('-inf'))
        phrase_attention = torch.softmax(gate_logits, dim=-1)  # (B, K)

        # 投影到 phrase_dim
        phrase_embeds = self.proj(phrase_embs)  # (B, K, phrase_dim)

        return {
            "phrase_embeds": phrase_embeds,
            "phrase_masks": phrase_masks,
            "phrase_attention": phrase_attention,
            "phrase_scores": phrase_scores,
            "phrase_spans": batch_phrase_spans  # 用于调试
        }

class EnhancedChineseTextEmbedding(nn.Module):
    def __init__(self, tokenizer, embed_dim=768, phrase_dim=128, max_phrases=5, use_learnable_extractor=True):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_encoder = AutoModel.from_pretrained(
            '/home/610-sty/huggingface/Taiyi-CLIP-Roberta-large-326M-Chinese'
        )
        self.roberta = self.text_encoder.base_model
        self.max_phrases = max_phrases
        self.use_learnable_extractor = use_learnable_extractor

        self.global_proj = nn.Linear(self.text_encoder.config.hidden_size, embed_dim)
        self.phrase_proj = nn.Linear(self.text_encoder.config.hidden_size, phrase_dim)

        if self.use_learnable_extractor:
            self.keyword_extractor = PhraseAttentionExtractor(
                encoder=self.roberta,
                max_phrases=self.max_phrases,
                phrase_dim=phrase_dim,
                max_phrase_len=5  # 可调
            )

        self.register_buffer('phrase_pad', torch.zeros(phrase_dim))

    def extract_keywords_static(self, poem, top_k=5):
        return jieba.analyse.extract_tags(poem, topK=top_k)

    def forward(self, poems, phrases=None, is_inference=False):
        device = next(self.parameters()).device
        batch_size = len(poems)

        poem_tokens = self.tokenizer(
            poems,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(device)
        poem_output = self.roberta(**poem_tokens).last_hidden_state  # (B, L, D)
        global_embed = self.global_proj(poem_output[:, 0])  # [CLS] -> (B, 768)

        if is_inference and self.use_learnable_extractor:
            phrase_data = self.keyword_extractor(
                input_ids=poem_tokens.input_ids,
                attention_mask=poem_tokens.attention_mask
            )
            return {
                "global_embed": global_embed,
                "local_embeds": phrase_data["phrase_embeds"],
                "local_mask": phrase_data["phrase_masks"],
                "raw_text": poems,
                "phrase_attention": phrase_data["phrase_attention"],
                "phrase_scores": phrase_data["phrase_scores"],
                "phrase_spans": phrase_data["phrase_spans"]
            }
        elif is_inference and not self.use_learnable_extractor:
            all_phrases = [self.extract_keywords_static(p, top_k=self.max_phrases) for p in poems]
            local_embeds, local_mask = self.encode_phrases(all_phrases, device)
        else:
            all_phrases = [p[:self.max_phrases] if p else [] for p in phrases]
            local_embeds, local_mask = self.encode_phrases(all_phrases, device)

        return {
            "global_embed": global_embed,
            "local_embeds": local_embeds,
            "local_mask": local_mask,
            "raw_text": poems
        }

    def encode_phrases(self, all_phrases, device):
        local_embeds = []
        local_mask = []

        for phrase_list in all_phrases:
            phrase_len = len(phrase_list)
            if phrase_len == 0:
                phrase_embed = self.phrase_pad.unsqueeze(0).expand(self.max_phrases, -1)
                local_embeds.append(phrase_embed)
                local_mask.append([0] * self.max_phrases)
                continue

            phrase_tokens = self.tokenizer(
                phrase_list,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=10
            ).to(device)
            phrase_output = self.roberta(**phrase_tokens).last_hidden_state[:, 0]  # (K, D)
            phrase_embed = self.phrase_proj(phrase_output)

            if phrase_embed.size(0) < self.max_phrases:
                pad_len = self.max_phrases - phrase_embed.size(0)
                pad = self.phrase_pad.unsqueeze(0).expand(pad_len, phrase_embed.shape[-1])
                phrase_embed = torch.cat([phrase_embed, pad], dim=0)
                mask = [1] * phrase_len + [0] * pad_len
            else:
                phrase_embed = phrase_embed[:self.max_phrases]
                mask = [1] * self.max_phrases

            local_embeds.append(phrase_embed)
            local_mask.append(mask)

        local_embeds = torch.stack(local_embeds)  # (B, K, 128)
        local_mask = torch.tensor(local_mask, device=device, dtype=torch.bool)  # (B, K)
        return local_embeds, local_mask