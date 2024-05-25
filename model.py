import torch
from torch import nn
from transformers import SiglipVisionModel, GPT2LMHeadModel, GPT2Tokenizer
class VisionLanguageModel(nn.Module):
    def __init__(self):
        super(VisionLanguageModel, self).__init__()
        self.vision_encoder = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
        self.text_decoder = GPT2LMHeadModel.from_pretrained("distilgpt2")
        self.projection = nn.Linear(self.vision_encoder.config.hidden_size, self.text_decoder.config.n_embd)

    def forward(self, input_ids, attention_mask, pixel_values, labels):
        vision_output = self.vision_encoder(pixel_values=pixel_values)
        vision_embedding = self.projection(vision_output.last_hidden_state)

        decoder_input_embeddings = self.text_decoder.transformer.wte(input_ids)
        vision_embedding = vision_embedding.unsqueeze(1).expand(-1, decoder_input_embeddings.size(1), -1)
        combined_embeddings = decoder_input_embeddings + vision_embedding

        text_output = self.text_decoder(inputs_embeds=combined_embeddings, labels=labels, return_dict=True)
        return text_output.loss
    dummy_pixel_values = torch.rand(1, 3, 224, 224)

tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
tokens = tokenizer("A caption for the image.", return_tensors="pt")
input_ids = tokens.input_ids
attention_mask = tokens.attention_mask

labels = torch.cat([input_ids[:, 1:], torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long).to(input_ids.device)], dim=1)
model = VisionLanguageModel()

loss = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=dummy_pixel_values, labels=labels)
print("Loss:", loss.item())