from torch import nn


class FeedForwardLayer(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(FeedForwardLayer, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


if __name__ == "__main__":
    import torch
    x = torch.randn((1, 128, 768))  # batch_size * seq_len * d_model
    layer = FeedForwardLayer(d_model=768, hidden=2048)
    output = layer(x)
    print(f"Output shape: {output.shape}")
