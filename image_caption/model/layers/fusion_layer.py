from torch import nn
from torch.nn import LayerNorm, MultiheadAttention
import torch
import sys
sys.path.append("C:/Users/Chris/Desktop/直通硅谷/project/image_caption/")
from image_caption.model.layers.feed_forward_layer import FeedForwardLayer


class FusionLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(FusionLayer, self).__init__()

        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head, batch_first=True)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = FeedForwardLayer(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, vision_emb, tag_emb):
        """
        :param vision_emb: vision embedding with shape (batch_size, video_seq_len, d_model)
        :param tag_emb: tag embedding with shape (batch_size, num_tags, d_model)
        :return: fused embedding with shape (batch_size, num_tags, d_model)
        """
        # 1. compute vision - tag cross attention
        x, _ = self.cross_attention(query=tag_emb, key=vision_emb, value=vision_emb)  # b, seq_len, d_model
        x = self.dropout1(x)
        x = self.norm1(x + tag_emb)  # add & norm

        # 2. feed forward network
        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x


if __name__ == "__main__":
    import torch
    vision_emb = torch.rand((1, 196, 768))
    tag_emb = torch.rand((1, 607, 768))
    layer = FusionLayer(d_model=768, ffn_hidden=2048, n_head=8, drop_prob=0.1)
    output = layer(vision_emb, tag_emb)
    print(f"Output shape: {output.shape}")
