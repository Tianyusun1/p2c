from torch.utils.data._utils.collate import default_collate
import torch
from torchvision import transforms

def custom_collate(batch):
    # 提取各个字段
    images = [item["instance_image"] for item in batch]
    prompts = [item["instance_prompt"] for item in batch]
    phrases = [
        item["phrases"] if isinstance(item["phrases"], list) else []
        for item in batch
    ]

    # 图像预处理（确保是 tensor 并归一化）
    transform = transforms.Compose([
        transforms.Normalize([0.5], [0.5])  # ✅ 不再需要 ToTensor()
    ])

    # 注意：images 已经是 tensor，直接堆叠即可
    image_tensor = default_collate(images)

    # 应用归一化
    image_tensor = transform(image_tensor)

    # 构建最终 batch 输出
    collated_batch = {
        "instance_image": image_tensor,
        "instance_prompt": prompts,
        "phrases": phrases
    }

    return collated_batch