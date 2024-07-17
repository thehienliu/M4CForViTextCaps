# Multimodal Multi-Copy Mesh (M4C) Module

A custom implementation of the M4C module designed specifically for the Vietnamese Image Captioning Task. For a detailed example, please refer to the accompanying notebook.
![m4cimage](https://github.com/user-attachments/assets/d38649a9-0dca-4de5-9f0d-942e95195eae)
## Installation

```bash
# Clone the repository
git clone https://github.com/thehienliu/M4CForViTextCaps.git

# Navigate to the project directory
cd M4CForViTextCaps
```

## Usage
Here's an example of how to use the M4C module in your project:

```python
from model.m4c import M4C
import torch

# Initialize the model
phobert_model.embeddings.word_embeddings.requires_grad = False
fixed_ans_emb = phobert_model.embeddings.word_embeddings.weight

model = M4C(
    obj_in_dim=2048,
    ocr_in_dim=812,
    hidden_size=768,
    n_heads=12,
    d_k=64,
    n_layers=4,
    vocab_size=tokenizer.vocab_size + 1,
    fixed_ans_emb=fixed_ans_emb
).to(device)

# Example input format
samples = {
    'id': batch_id,
    'obj_boxes': obj_boxes_tensor,
    'obj_features': obj_features_tensor,
    'ocr_boxes': ocr_boxes_tensor,
    'ocr_token_embeddings': ocr_token_embeddings_tensor,
    'ocr_rec_features': ocr_rec_features_tensor,
    'ocr_det_features': ocr_det_features_tensor,
    'join_attn_mask': join_attn_mask,
    'labels': labels_,
    'texts': texts_,
    'raw_captions': raw_captions
}

# Forward pass
outputs = model(samples, device="cuda")
```
