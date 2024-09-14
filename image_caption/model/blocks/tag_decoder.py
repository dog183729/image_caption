from torch import nn
import torch
import sys
sys.path.append("C:/Users/Chris/Desktop/直通硅谷/project/image_caption/")
from image_caption.model.layers.fusion_layer import FusionLayer


class TagDecoder(nn.Module):
    """
    take tag_embedding and vision embedding as input, and output tag logits
    it consists of Nx fusion layers and one linear layer at the end
    """
    def __init__(self, tag_embedding, d_model, ffn_hidden, n_head, n_layers, drop_prob):
        super(TagDecoder, self).__init__()
        self.tag_emb = nn.Parameter(tag_embedding)
        self.fusion_layers = [
            FusionLayer(
                d_model=d_model, ffn_hidden=ffn_hidden, n_head=n_head, drop_prob=drop_prob
            ) for _ in range(n_layers)
        ]
        self.linear = nn.Linear(d_model, 1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, vision_embedding):
        tag_embedding = self.tag_emb.unsqueeze(0).repeat(vision_embedding.shape[0], 1, 1).to(self.device)  # (N, num_tags, d_model)

        for layer in self.fusion_layers:
            tag_embedding = layer(vision_embedding, tag_embedding)  # (N, num_tags, d_model)
        tag_logits = self.linear(tag_embedding)  # (N, num_tags, 1)
        return tag_logits[:, :, 0]  # (N, num_tags)


if __name__ == "__main__":
    import torch
    vision_emb = torch.rand((3, 196, 768))
    tag_emb = torch.rand((607, 768))
    decoder = TagDecoder(tag_emb, d_model=768, ffn_hidden=2048, n_head=8, n_layers=6, drop_prob=0.1)
    tag_logits = decoder(vision_emb)
    print(f"Output shape: {tag_logits.shape}")  # 1 * 607
