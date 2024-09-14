import torch
from torch import nn
from transformers import GPT2Tokenizer
import sys
sys.path.append("C:/Users/Chris/Desktop/直通硅谷/project/image_caption/")
from image_caption.model.blocks.vision_encoder import VisionEncoder
from image_caption.model.blocks.tag_decoder import TagDecoder
from image_caption.model.blocks.caption_decoder import CaptionDecoder

class VisionLanguageModel(nn.Module):
    def __init__(self, tag_embedding, ids_to_tags, d_model, ffn_hidden, n_head, n_layers, drop_prob):
        super(VisionLanguageModel, self).__init__()
        self.vision_encoder = VisionEncoder(d_model=768)
        self.ids_to_tags = ids_to_tags
        self.tag_decoder = TagDecoder(tag_embedding, d_model, ffn_hidden, n_head, n_layers, drop_prob)
        self.caption_decoder = CaptionDecoder()

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
        return caption_loss


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
    loss = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=dummy_pixel_values, labels=labels)
    print("Loss:", loss.item())