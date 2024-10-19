import requests

import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoTokenizer

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    # add padding token
    special_tokens = {"pad_token": "<PAD>"}
    tokenizer.add_special_tokens(special_tokens)
    return tokenizer


class COCOCaptionDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, sampler, num_workers, pin_memory, drop_last):
        super().__init__(
            dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
            num_workers=num_workers, collate_fn=self.collate_fn,
            pin_memory=pin_memory, drop_last=drop_last
        )
        self.num_classes = len(dataset.all_tags)
        self.image_processor = AutoImageProcessor.from_pretrained("google/siglip-base-patch16-224")
        self.tokenizer = get_tokenizer()

    def collate_fn(self, batch):
        """
        :param batch:
        :return:
            pixel_values: N * 3 * height * width
            input_ids: N * seq_len
            attention_mask: N * seq_len
            labels: N * seq_len
        """
        """load images from image_path and convert to tensor"""
        pixel_values = []
        captions = []
        tag_labels = []
        for image_path, caption, tag_ids in batch:
            image = Image.open(image_path).convert("RGB")
            image = self.image_processor(image, return_tensors="pt")
            pixel_values.append(image.pixel_values)
            captions.append(caption + self.tokenizer.eos_token)

            label = torch.zeros((1, self.num_classes))  # 1 * num_classes
            label[:, tag_ids] = 1
            tag_labels.append(label)

        tag_labels = torch.concat(tag_labels, dim=0)  # batch_size * num_classes
        pixel_values = torch.concat(pixel_values, dim=0)
        inputs = self.tokenizer(captions, padding="longest", return_tensors="pt", max_length=384, truncation=True)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return pixel_values, tag_labels, input_ids, attention_mask, labels, captions


def get_dataloader(dataset, batch_size, shuffle, sampler=None, num_workers=0, pin_memory=False, drop_last=False):
    return COCOCaptionDataLoader(dataset, batch_size, shuffle, sampler, num_workers, pin_memory, drop_last)


if __name__ == "__main__":
    from dataset import COCOCaptionDataset
    # ann_path = "C:/Users/Chris/Desktop/直通硅谷/project/image_caption-feat-add-dataloader/annotations/captions_val2017.json"
    # images_dir = "C:/Users/Chris/Desktop/直通硅谷/project/image_caption-feat-add-dataloader/images/val2017"
    ann_path = "captions_train2017_with_tags_updated.json"
    images_dir = "/Users/shuangliu/Downloads/data/coco/images/train2017"
    dataset = COCOCaptionDataset(ann_path=ann_path, images_dir=images_dir)
    dataloader = get_dataloader(dataset, batch_size=4, shuffle=True)
    for batch in dataloader:
        pixel_values, tag_labels, input_ids, attention_mask, labels, captions = batch
        print(f"pixel_values shape: {pixel_values.shape}")
        print(f"tag_labels shape: {tag_labels.shape}")
        print(f"input_ids shape: {input_ids.shape}")
        print(f"attention_mask shape: {attention_mask.shape}")
        print(f"labels shape: {labels.shape}")
        break

    from IPython import embed
    embed()
    assert False
