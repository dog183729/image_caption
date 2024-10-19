import os

import numpy as np
from tqdm import tqdm
import torch

from image_caption.data.dataset import COCOCaptionDataset
from image_caption.data.dataloader import get_dataloader
from image_caption.model.model import VisionLanguageModel

from sklearn.metrics import roc_auc_score
from nltk.translate.bleu_score import sentence_bleu


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ann_path = '/mnt/bn/algo-masp-nas-2/benchmark/coco/annotations/captions_train2017_with_tags_updated.json'
    # images_dir = 'C:/Users/Chris/Desktop/直通硅谷/project/image_caption/images/train2017'
    images_dir = '/mnt/bn/algo-masp-nas-2/benchmark/coco/train2017'
    checkpoint_path = 'checkpoint.pth_epoch2_batch18492.pth'

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
        print(f"Loading pretrained checkpoints from {checkpoint_path}...")
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])

    all_logits = []
    all_captions = []
    all_labels = []
    all_gt_captions = []
    for i, vdata in tqdm(enumerate(dataloader)):
        if i >= 10:
            break
        pixel_values, tag_labels, input_ids, attention_mask, labels, gt_captions = vdata
        pixel_values, input_ids, attention_mask, labels = pixel_values.to(device), input_ids.to(device), attention_mask.to(device), labels.to(device)

        tag_logits, captions = model.generate(pixel_values)
        all_logits.append(tag_logits)
        all_captions += captions
        all_labels.append(tag_labels)
        all_gt_captions += gt_captions

    # compute AUC score for each class
    all_logits = torch.concat(all_logits, dim=0)  # N * 617
    all_probs = all_logits.sigmoid()
    all_labels = torch.concat(all_labels, dim=0)  # N * 617
    all_probs = all_probs.cpu().numpy()
    all_labels = all_labels.cpu().numpy()

    results = {}
    for i, tag_name in tqdm(dataset.ids_to_tags.items()):
        probs = all_probs[:, i]
        labels = all_labels[:, i]
        if labels.max() == 0:
            continue
        positive_num = labels.sum()
        score = roc_auc_score(labels, probs)
        results[tag_name] = {"score": score, "num_samples": int(positive_num)}
    avg_roc_score = sum(v["score"] * v["num_samples"] for v in results.values()) / sum(v["num_samples"] for v in results.values())

    # compute bleu score for each data
    all_bleu_scores = []
    for i, (caption, gt_caption) in enumerate(tqdm(zip(all_captions, all_gt_captions))):
        print(f"\nground truth caption: {gt_caption}")
        print(f"\ngenerated caption: {caption}")
        references = [gt_caption.split()]  # remove eos token
        hypothesis = caption.split()
        score = sentence_bleu(references, hypothesis)
        all_bleu_scores.append(score)
    avg_bleu_score = np.mean(all_bleu_scores)

    print(f"Average ROC-AUC score: {avg_roc_score}")
    print(f"Average BLEU-4 score: {avg_bleu_score}")


if __name__ == "__main__":
    main()
