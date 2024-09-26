import os

import torch

from image_caption.data.dataset import COCOCaptionDataset
from image_caption.data.dataloader import get_dataloader
from image_caption.model.model import VisionLanguageModel


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ann_path = 'captions_train2017_with_tags_updated.json'
    # images_dir = 'C:/Users/Chris/Desktop/直通硅谷/project/image_caption/images/train2017'
    images_dir = '/Users/shuangliu/Downloads/data/coco/images/train2017'
    checkpoint_path = 'checkpoint.pth'

    # Load dataset and dataloader
    dataset = COCOCaptionDataset(ann_path=ann_path, images_dir=images_dir)
    dataloader = get_dataloader(dataset, batch_size=8, shuffle=False)

    # Initialize model
    model = VisionLanguageModel(
        tag_embedding=dataset.all_tags_embedding,
        ids_to_tags=dataset.ids_to_tags,
        d_model=768,
        ffn_hidden=2048,
        n_head=8,
        n_layers=6,
        drop_prob=0.1
    ).to(device)

    # load trained checkpoint
    if os.path.isfile(checkpoint_path):
        state = torch.load(checkpoint_path)
        model.load_state_dict(state["model_state_dict"])

    for i, vdata in enumerate(dataloader):
        pixel_values, tag_labels, input_ids, attention_mask, labels, gt_captions = vdata
        pixel_values, input_ids, attention_mask, labels = pixel_values.to(device), input_ids.to(device), attention_mask.to(device), labels.to(device)

        tag_logits, captions = model.generate(pixel_values)

        # compute AUC score for each class

        # compute bleu score for each data


if __name__ == "__main__":
    main()
