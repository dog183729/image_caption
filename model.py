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
        """
        :param input_ids: batch_size * text_seq_len
        :param attention_mask: batch_size * text_seq_len
        :param pixel_values: batch_size * 3 * 224 * 224
        :param labels: batch_size * text_seq_len
        :return:
        """
        vision_output = self.vision_encoder(pixel_values=pixel_values)  # batch_size * 196 * 768
        vision_embedding = self.projection(vision_output.last_hidden_state)  # batch_size * 196 * 768

        # TODO: add comments to record the output shape of each step.
        decoder_input_embeddings = self.text_decoder.transformer.wte(input_ids)  # batch_size * text_seq_len * 768
        # vision_embedding = vision_embedding.expand(-1, decoder_input_embeddings.size(1), -1)
        # combined_embeddings = decoder_input_embeddings + vision_embedding
        combined_embedding = torch.concat([vision_embedding, decoder_input_embeddings], dim=1)  # batch_size * (196 + text_seq_len) * 768

        # TODO: append -100 to `label` on the left to make it have the shape of (batch_size, 196 + text_seq_len)
        # TODO: append 1 to `attention_mask` on the left to make it have the shape of (batch_size, 196 + text_seq_len)

        text_output = self.text_decoder(inputs_embeds=combined_embedding, attention_mask=attention_mask, labels=labels, return_dict=True)
        return text_output.loss


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
