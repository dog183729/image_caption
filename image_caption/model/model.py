import torch
from torch import nn
from transformers import SiglipVisionModel, GPT2LMHeadModel, GPT2Tokenizer

from image_caption.model.blocks.vision_encoder import VisionEncoder
from image_caption.model.blocks.caption_decoder import CaptionDecoder

class VisionLanguageModel(nn.Module):
    def __init__(self):
        super(VisionLanguageModel, self).__init__()
        self.vision_encoder = VisionEncoder(d_model=768)
        self.caption_decoder = CaptionDecoder()

    def forward(self, input_ids, attention_mask, pixel_values, labels):
        """
        :param input_ids: batch_size * text_seq_len
        :param attention_mask: batch_size * text_seq_len
        :param pixel_values: batch_size * 3 * 224 * 224
        :param labels: batch_size * text_seq_len
        :return:
        """
        # Vision encoder forward pass
        vision_embedding = self.vision_encoder(pixel_values)

        # Caption decoder forward path
        caption_loss = self.caption_decoder(vision_embedding, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return caption_loss


if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    tokens = tokenizer("A caption for the image.", return_tensors="pt")
    input_ids = tokens.input_ids
    attention_mask = tokens.attention_mask

    # TODO: use `input_ids` as labels since shifting is implicitly handled by Huggingface.
    labels = torch.cat([input_ids[:, 1:], torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long).to(input_ids.device)], dim=1)

    model = VisionLanguageModel()

    dummy_pixel_values = torch.rand(1, 3, 224, 224)
    loss = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=dummy_pixel_values, labels=labels)
    print("Loss:", loss.item())