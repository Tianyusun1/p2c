# utils/clip_utils.py
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel

class ClipSimilarity(torch.nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(model_path)
        self.processor = CLIPProcessor.from_pretrained(model_path)

        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.clip_model.eval()

    @torch.no_grad()
    def forward(self, images, texts):
        """
        images: (B, 3, H, W) tensor scaled to [0, 1]
        texts: list of str
        """
        device = images.device
        inputs = self.processor(
            text=texts, images=images, return_tensors="pt", padding=True
        ).to(device)

        outputs = self.clip_model(**inputs)
        image_embeds = outputs.image_embeds  # (B, D)
        text_embeds = outputs.text_embeds    # (B, D)

        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        # cosine distance loss
        loss = 1 - (image_embeds * text_embeds).sum(dim=-1)  # (B,)
        return loss.mean()
