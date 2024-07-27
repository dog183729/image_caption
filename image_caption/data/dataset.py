import cv2
import numpy as np
import json
from torch.utils.data import Dataset
import os

class COCOCaptionDataset(Dataset):
    def __init__(self, ann_path, images_dir):
        # load annotation file
        with open(ann_path, "r") as f:
            self.anns = json.load(f)
        self.all_tags = self.anns["all_tags"]
        self.images_dir = images_dir
        self.ids_to_filenames = {e["id"]: e["file_name"] for e in self.anns["images"]}
        # print(self.anns["annotations"])
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
            tag_ids.append(self.all_tags.index(tag))

        return image_path, caption, tag_ids


if __name__ == "__main__":
    import random
    # ann_path = "C:/Users/Chris/Desktop/直通硅谷/project/image_caption-feat-add-dataloader/annotations/captions_val2017.json"
    # images_dir = "C:/Users/Chris/Desktop/直通硅谷/project/image_caption-feat-add-dataloader/images/val2017"
    ann_path = "captions_train2017_with_tags.json"
    images_dir = "/Users/shuangliu/Downloads/data/coco/images/train2017"
    dataset = COCOCaptionDataset(ann_path=ann_path, images_dir=images_dir)
    image_path, caption, tag_ids = random.choice(dataset)
    tag_names = [dataset.all_tags[tag_id] for tag_id in tag_ids]
    print(image_path, caption, tag_ids, tag_names)
