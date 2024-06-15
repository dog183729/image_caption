import cv2
import numpy as np
import json
from torch.utils.data import Dataset


class COCOCaptionDataset(Dataset):
    def __init__(self, ann_path, images_dir):
        # load annotation file
        with open(ann_path, "r") as f:
            self.anns = json.load(f)
        self.images_dir = images_dir
        self.ids_to_urls = {e["id"]: e["coco_url"] for e in self.anns["images"]}

    def __len__(self):
        """return the dataset size"""
        return len(self.anns["annotations"])

    def __getitem__(self, idx):
        """given an index between [0, len(data)), return the corresponding data"""
        ann = self.anns["annotations"][idx]
        # TODO: download COCO dataset to local and convert url to local file path
        filename = self.ids_to_filenames[image_id]
        image_path = os.path.join(self.images_dir, filename)
        caption = ann["caption"]
        return image_path, caption


if __name__ == "__main__":
    import random
    dataset = COCOCaptionDataset(ann_path="C:/Users/Chris/Desktop/直通硅谷/project/image_caption-feat-add-dataloader/annotations/captions_val2017.json")
    data = random.choice(dataset)
    print(data)

