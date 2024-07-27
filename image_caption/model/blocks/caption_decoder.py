import torch
from torch import nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class CaptionDecoder(nn.Module):
    def __init__(self):
        super(CaptionDecoder, self).__init__()
        self.text_decoder = GPT2LMHeadModel.from_pretrained("distilgpt2")

    def forward(self, vision_embedding, input_ids, attention_mask, labels):
        """
        :param vision_embedding: batch_size * video_seq_len * 768
        :param input_ids: batch_size * text_seq_len
        :param attention_mask: batch_size * text_seq_len
        :param labels: batch_size * text_seq_len
        :return:
        """
        # Text decoder input embeddings
        decoder_input_embeddings = self.text_decoder.transformer.wte(input_ids)  # batch_size * text_seq_len * 768

        # Combine vision embeddings and text input embeddings
        combined_embedding = torch.cat([vision_embedding, decoder_input_embeddings],
                                       dim=1)  # batch_size * (196 + text_seq_len) * 768

        # TODO: append -100 to `label` on the left to make it have the shape of (batch_size, 196 + text_seq_len)
        # Append -100 to labels on the left to match the combined embedding shape
        extended_labels = torch.cat(
            [torch.full((labels.size(0), vision_embedding.size(1)), -100).to(labels.device), labels],
            dim=1)  # batch_size * (196 + text_seq_len)

        # TODO: append 1 to `attention_mask` on the left to make it have the shape of (batch_size, 196 + text_seq_len)
        # Append 1 to attention mask on the left to match the combined embedding shape
        extended_attention_mask = torch.cat(
            [torch.ones(attention_mask.size(0), vision_embedding.size(1)).to(attention_mask.device), attention_mask],
            dim=1)  # batch_size * (196 + text_seq_len)

        # Text decoder forward pass
        text_output = self.text_decoder(inputs_embeds=combined_embedding, attention_mask=extended_attention_mask,
                                        labels=extended_labels, return_dict=True)
        return text_output.loss


if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    tokens = tokenizer("A caption for the image.", return_tensors="pt")
    input_ids = tokens.input_ids
    attention_mask = tokens.attention_mask

    # TODO: use `input_ids` as labels since shifting is implicitly handled by Huggingface.
    labels = torch.cat(
        [input_ids[:, 1:], torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long).to(input_ids.device)], dim=1)

    model = CaptionDecoder()

    dummy_vision_embedding = torch.rand(1, 196, 768)
    loss = model(vision_embedding=dummy_vision_embedding, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    print("Loss:", loss.item())