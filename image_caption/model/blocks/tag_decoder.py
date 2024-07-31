from torch import nn

from image_caption.model.layers.fusion_layer import FusionLayer


class TagDecoder(nn.Module):
    """
    take tag_embedding and vision embedding as input, and output tag logits
    it consists of Nx fusion layers and one linear layer at the end
    """
    def __init__(self, tag_embedding, d_model, ffn_hidden, n_head, n_layers, drop_prob):
        self.tag_emb = nn.Parameter(tag_embedding)
        pass

    def forward(self, vision_embedding):
        pass


if __name__ == "__main__":
    import torch
    vision_emb = torch.rand((1, 196, 768))
    tag_emb = torch.rand((1, 607, 768))
    decoder = TagDecoder(tag_emb, d_model=768, ffn_hidden=2048, n_head=8, n_layers=6, drop_prob=0.1)
    output = decoder(vision_emb)
    print(f"Output shape: {output.shape}")  # 1 * 607 * 1
