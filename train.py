# TODO: refer to this documentation to implement training script: https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup
from data.dataset import COCOCaptionDataset
from data.dataloader import get_dataloader
from model import VisionLanguageModel
from datetime import datetime
import os
from tqdm import tqdm


def train_one_epoch(epoch_index, tb_writer, model, training_loader, optimizer, loss_fn, device):
    running_loss = 0.0
    last_loss = 0.0
    count = 0
    for data in tqdm(training_loader):
        pixel_values, input_ids, attention_mask, labels = data
        pixel_values = pixel_values.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels)

        loss = outputs
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        count += 1
        if count % 10 == 0:
            last_loss = running_loss / 10
            print(f'  batch {count + 1} loss: {last_loss}')
            tb_x = epoch_index * len(training_loader) + count + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.0

    return last_loss

def main():
    # Hyperparameters
    batch_size = 1
    num_epochs = 5
    learning_rate = 2e-5
    max_steps = 1000
    warmup_steps = 100

    # Paths
    ann_path = 'C:/Users/Chris/Desktop/直通硅谷/project/image_caption-feat-add-dataloader/annotations/captions_train2017.json'
    images_dir = 'C:/Users/Chris/Desktop/直通硅谷/project/image_caption-feat-add-dataloader/images/train2017'

    # Load dataset and dataloader
    dataset = COCOCaptionDataset(ann_path=ann_path, images_dir=images_dir)
    training_loader = get_dataloader(dataset, batch_size=batch_size, shuffle=True)
    validation_loader = get_dataloader(dataset, batch_size=batch_size, shuffle=False)

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VisionLanguageModel().to(device)

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps)
    loss_fn = torch.nn.CrossEntropyLoss()

    # TensorBoard writer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/vision_language_trainer_{timestamp}')
    epoch_number = 0

    best_vloss = float('inf')

    for epoch in range(num_epochs):
        print(f'EPOCH {epoch_number + 1}:')

        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer, model, training_loader, optimizer, loss_fn, device)

        running_vloss = 0.0
        model.eval()

        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                v_pixel_values, v_input_ids, v_attention_mask, v_labels = vdata
                v_pixel_values = v_pixel_values.to(device)
                v_input_ids = v_input_ids.to(device)
                v_attention_mask = v_attention_mask.to(device)
                v_labels = v_labels.to(device)

                v_outputs = model(input_ids=v_input_ids, attention_mask=v_attention_mask, pixel_values=v_pixel_values, labels=v_labels)
                v_loss = v_outputs.loss
                running_vloss += v_loss.item()

        avg_vloss = running_vloss / (i + 1)
        print(f'LOSS train {avg_loss} valid {avg_vloss}')

        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch_number + 1)
        writer.flush()

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f'model_{timestamp}_{epoch_number}.pth'
            torch.save(model.state_dict(), model_path)

        epoch_number += 1

if __name__ == '__main__':
    main()