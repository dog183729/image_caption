import cv2
import numpy as np
import json
from torch.utils.data import Dataset


class COCOCaptionDataset(Dataset):
    def __init__(self, ann_path):
        # load annotation file
        with open(ann_path, "r") as f:
            self.anns = json.load(f)
        self.ids_to_urls = {e["id"]: e["coco_url"] for e in self.anns["images"]}

    def __len__(self):
        """return the dataset size"""
        return len(self.anns["annotations"])

    def __getitem__(self, idx):
        """given an index between [0, len(data)), return the corresponding data"""
        ann = self.anns["annotations"][idx]
        # TODO: download COCO dataset to local and convert url to local file path
        url = self.ids_to_urls[ann["image_id"]]
        caption = ann["caption"]
        return url, caption


if __name__ == "__main__":
    import random
    dataset = COCOCaptionDataset(ann_path="/Users/shuangliu/Downloads/data/coco/annotations/captions_val2017.json")
    data = random.choice(dataset)
    print(data)
