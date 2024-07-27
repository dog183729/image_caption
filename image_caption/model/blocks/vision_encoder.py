import torch
from torch import nn
from transformers import SiglipVisionModel, GPT2Tokenizer


class VisionEncoder(nn.Module):
    def __init__(self, d_model):
        super(VisionEncoder, self).__init__()
        self.vision_encoder = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
        self.projection = nn.Linear(self.vision_encoder.config.hidden_size, d_model)

    def forward(self, pixel_values):
        """
        :param pixel_values: batch_size * 3 * 224 * 224
        :return:
        """
        # Vision encoder forward pass
        vision_output = self.vision_encoder(pixel_values=pixel_values)  # batch_size * 196 * 768
        vision_embedding = self.projection(vision_output.last_hidden_state)  # batch_size * 196 * 768
        return vision_embedding


if __name__ == "__main__":
    model = VisionEncoder(d_model=768)

    dummy_pixel_values = torch.rand(1, 3, 224, 224)
    output = model(pixel_values=dummy_pixel_values)
    print("Output shape:", output.shape)
