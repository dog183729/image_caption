import torch
from torch import nn
from transformers import GPT2Tokenizer
import sys
sys.path.append("C:/Users/Chris/Desktop/直通硅谷/project/image_caption/")
from image_caption.model.blocks.vision_encoder import VisionEncoder
from image_caption.model.blocks.tag_decoder import TagDecoder
from image_caption.model.blocks.caption_decoder import CaptionDecoder
from image_caption.data.dataloader import get_tokenizer

class VisionLanguageModel(nn.Module):
    def __init__(self, tag_embedding, ids_to_tags, d_model, ffn_hidden, n_head, n_layers, drop_prob):
        super(VisionLanguageModel, self).__init__()
        self.vision_encoder = VisionEncoder(d_model=768)
        self.ids_to_tags = ids_to_tags
        self.tag_decoder = TagDecoder(tag_embedding, d_model, ffn_hidden, n_head, n_layers, drop_prob)
        self.caption_decoder = CaptionDecoder()
        self.text_tokenizer = get_tokenizer()

    def postprocess(self, tag_logits, threshs):
        probs = torch.sigmoid(tag_logits)  # N * num_tags
        batch_size = probs.shape[0]

        tag_lists = []
        for i in range(batch_size):
            tag_ids = torch.where(tag_logits[i] >= threshs)[0]
            tag_names = [self.ids_to_tags[idx.item()] for idx in tag_ids]
            tag_lists.append(tag_names)

        return tag_lists

    def forward(self, input_ids, attention_mask, pixel_values, labels, threshs=0.6):
        """
        :param input_ids: batch_size * text_seq_len
        :param attention_mask: batch_size * text_seq_len
        :param pixel_values: batch_size * 3 * 224 * 224
        :param labels: batch_size * text_seq_len
        :return:
        """
        # Vision encoder forward pass
        vision_embedding = self.vision_encoder(pixel_values)

        # Tag decoder forward path
        tag_logits = self.tag_decoder(vision_embedding)  # N * num_tags

        # Tag postprocess (tag_logits, thresholds) -> positive tag names
        tag_names = self.postprocess(tag_logits, threshs)

        # Caption decoder forward path
        caption_loss = self.caption_decoder(vision_embedding, tag_names, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return tag_logits, caption_loss

    def generate(self, pixel_values, threshs=0.6):
        """
        :param pixel_values: batch_size * 3 * 224 * 224
        :return: list of captions
        """
        # Vision encoder forward pass
        vision_embedding = self.vision_encoder(pixel_values)

        # Tag decoder forward path
        tag_logits = self.tag_decoder(vision_embedding)  # N * num_tags

        # Tag postprocess (tag_logits, thresholds) -> positive tag names
        tag_names = self.postprocess(tag_logits, threshs)

        # Generate tag embedding
        tag_names = [",".join(tag_list) for tag_list in tag_names]  # ["tag1,tag2,tag3", "tag4,tag5", ...]
        tag_inputs = self.text_tokenizer(tag_names, padding="longest", return_tensors="pt", max_length=384,
                                         truncation=True)

        tag_input_ids, tag_attention_mask = tag_inputs.input_ids.long().to(
            vision_embedding.device), tag_inputs.attention_mask.to(vision_embedding.device)
        tag_input_embeddings = self.caption_decoder.text_decoder.transformer.wte(tag_input_ids)  # batch_size * tag_seq_len * 768

        # Create inputs fot caption decoder
        combined_embedding = torch.cat([vision_embedding, tag_input_embeddings],
                                       dim=1)  # batch_size * (196 + tag_seq_len) * 768
        extended_attention_mask = torch.cat(
            [torch.ones(tag_attention_mask.size(0), vision_embedding.size(1)).to(tag_attention_mask.device), tag_attention_mask],
            dim=1)  # batch_size * (196 + tag_seq_len)

        # Call caption generation
        outputs = self.caption_decoder.text_decoder.generate(
            inputs_embeds=combined_embedding, attention_mask=extended_attention_mask, max_length=1024
        )
        captions = self.text_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return tag_logits, captions


if __name__ == "__main__":
    from image_caption.data.dataset import COCOCaptionDataset

    # init dataset
    ann_path = "captions_train2017_with_tags_updated.json"
    images_dir = "C:/Users/Chris/Desktop/直通硅谷/project/image_caption/images/train2017"
    dataset = COCOCaptionDataset(ann_path=ann_path, images_dir=images_dir)

    # init caption inputs
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    tokens = tokenizer("A caption for the image.", return_tensors="pt")
    input_ids = tokens.input_ids
    attention_mask = tokens.attention_mask

    # TODO: use `input_ids` as labels since shifting is implicitly handled by Huggingface.
    labels = torch.cat([input_ids[:, 1:], torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long).to(input_ids.device)], dim=1)

    model = VisionLanguageModel(
        tag_embedding=dataset.all_tags_embedding, ids_to_tags=dataset.ids_to_tags, d_model=768, ffn_hidden=2048, n_head=8, n_layers=6, drop_prob=0.1
    )

    dummy_pixel_values = torch.rand(1, 3, 224, 224)
    tag_logits, caption_loss = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=dummy_pixel_values, labels=labels)
    print("Caption loss:", caption_loss.item())

    captions = model.generate(pixel_values=dummy_pixel_values)
    print(f"Captions: {captions}")