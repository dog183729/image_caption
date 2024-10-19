
import json

import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import CLIPTextModel, AutoTokenizer

import os

class COCOCaptionDataset(Dataset):
    def __init__(self, ann_path, images_dir):
        # load annotation file
        with open(ann_path, "r") as f:
            self.anns = json.load(f)
        self.all_tags = self.anns["all_tags"]
        self.ids_to_tags = {i: tag for i, tag in enumerate(self.all_tags)}  # {1: tag_name1, 2: tag_name2, ...}

        # initiate tag tokenier and encoder
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.tag_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        self.all_tags_embedding = self.compute_embedding()  # 607 * 768

        self.images_dir = images_dir
        self.ids_to_filenames = {e["id"]: e["file_name"] for e in self.anns["images"]}

    def find_similarity_tags(self, tag, top_n=5):
        """
        Given an arbitrary tag, find the top_N most similar tags from `all_tags`
        :param tag: str representing any tag label
        :return: dictionary of top_N most similar tags from `all_tags` and the corresponding similarity score
        """
        # 1. calculate embedding of tag
        tag_ids = self.tokenizer(tag, padding="longest", return_tensors="pt", max_length=77)
        tag_embedding = self.tag_encoder(**tag_ids).pooler_output  # 1 * 768

        # 2. calculate dot product of tag_embedding and self.all_tags_embedding
        tag_embedding_normed = tag_embedding / tag_embedding.norm(p=2, dim=1, keepdim=True)  # 1 * 768
        all_tag_embedding_normed = self.all_tags_embedding / self.all_tags_embedding.norm(p=2, dim=1, keepdim=True)  # num_tags * 768
        similarities = tag_embedding_normed @ all_tag_embedding_normed.transpose(0, 1)  # 1 * num_tags
        max_ids = similarities.argsort(descending=True)[0, :top_n]
        max_similarities = similarities[0, max_ids]
        print(f"top {top_n} similar tags to {tag}:")
        for i in range(top_n):
            print(f"{self.all_tags[max_ids[i].item()]}: {max_similarities[i].item()}")

    @torch.no_grad()
    def compute_embedding(self):
        print("Computing tag embeddings...")
        tag_embedding = []
        for tag in tqdm(self.all_tags):
            tag_ids = self.tokenizer(tag, padding="longest", return_tensors="pt", max_length=77)
            tag_embedding.append(self.tag_encoder(**tag_ids))

        tag_embedding = torch.concat([t.pooler_output for t in tag_embedding], dim=0)  # (num_tags, 768)
        return tag_embedding.detach()

    def __len__(self):
        """return the dataset size"""
        return len(self.anns["annotations"])

    def __getitem__(self, idx):
        """given an index between [0, len(data)), return the corresponding data"""
        ann = self.anns["annotations"][idx]
        
        # TODO: download COCO dataset to local and convert url to local file path
        filename = self.ids_to_filenames[ann["image_id"]]
        image_path = os.path.join(self.images_dir, filename)
        caption = ann["caption"]

        tag_ids = []
        for tag in ann["tags"]:
            if tag in self.all_tags:
                tag_ids.append(self.all_tags.index(tag))

        return image_path, caption, tag_ids


if __name__ == "__main__":
    import random
    # ann_path = "C:/Users/Chris/Desktop/直通硅谷/project/image_caption-feat-add-dataloader/annotations/captions_val2017.json"
    # images_dir = "C:/Users/Chris/Desktop/直通硅谷/project/image_caption-feat-add-dataloader/images/val2017"
    ann_path = "captions_train2017_with_tags_updated.json"
    images_dir = "/Users/shuangliu/Downloads/data/coco/images/train2017"
    dataset = COCOCaptionDataset(ann_path=ann_path, images_dir=images_dir)
    # image_path, caption, tag_ids = random.choice(dataset)
    # tag_names = [dataset.all_tags[tag_id] for tag_id in tag_ids]
    # print(image_path, caption, tag_ids, tag_names)

    # test similarity
    dataset.find_similarity_tags(tag="furred animal swimming on water", top_n=5)
