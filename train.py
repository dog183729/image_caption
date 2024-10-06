import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from transformers import get_linear_schedule_with_warmup, GPT2Tokenizer
from image_caption.data.dataset import COCOCaptionDataset
from image_caption.data.dataloader import get_dataloader
from datetime import datetime
import os
from tqdm import tqdm
from image_caption.model.blocks.vision_encoder import VisionEncoder
from image_caption.model.blocks.tag_decoder import TagDecoder
from image_caption.model.blocks.caption_decoder import CaptionDecoder
from image_caption.model.model import VisionLanguageModel

def train_one_epoch(epoch_index, tb_writer, model, training_loader, optimizer, device, save_checkpoint_freq, checkpoint_path):
    running_loss = 0.0
    avg_loss = 0.0
    count = 0
    accumulate_steps = 32
    total_batches = len(training_loader)
    optimizer.zero_grad()

    for data in tqdm(training_loader):
        pixel_values, tag_labels, input_ids, attention_mask, labels, _ = data
        pixel_values = pixel_values.to(device)
        tag_labels = tag_labels.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        tag_logits, caption_loss = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels)

        # tag loss
        tag_loss_fn = BCEWithLogitsLoss()
        tag_loss = tag_loss_fn(tag_logits, tag_labels)

        loss = (tag_loss + caption_loss) / accumulate_steps
        # loss = tag_loss / accumulate_steps
        loss.backward()

        running_loss += loss.item()
        count += 1
        avg_loss += loss.item()

        if count % accumulate_steps == 0:
            print(f'  batch {count} loss: {running_loss} tag_loss: {tag_loss}, caption_loss: {caption_loss}')
            optimizer.step()
            optimizer.zero_grad()

            tb_x = epoch_index * len(training_loader) + count + 1
            tb_writer.add_scalar('Loss/train', avg_loss, tb_x)
            running_loss = 0.0

        # Save checkpoint at one-fourth, half, three-fourths, and full epoch
        if count % (total_batches // save_checkpoint_freq) == 0:
            checkpoint_name = f'{checkpoint_path}_epoch{epoch_index + 1}_batch{count}.pth'
            save_checkpoint({
                'epoch': epoch_index + 1,
                'batch': count,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_name)
            print(f"Checkpoint saved at batch {count} of epoch {epoch_index + 1}")

    avg_loss /= count
    return avg_loss

def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model, optimizer, device):
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Loaded checkpoint '{checkpoint_path}' (epoch {epoch})")
        return epoch
    else:
        print(f"No checkpoint found at '{checkpoint_path}', starting from scratch.")
        return 0

def main():
    # Hyperparameters
    batch_size = 1
    num_epochs = 5
    learning_rate = 2e-5
    max_steps = 1000
    warmup_steps = 100
    save_checkpoint_freq = 4  # Save checkpoint every one-fourth of an epoch

    # Paths
    ann_path = 'captions_train2017_with_tags_updated.json'
    # images_dir = 'C:/Users/Chris/Desktop/直通硅谷/project/image_caption/images/train2017'
    images_dir = '/Users/shuangliu/Downloads/data/coco/images/train2017'
    checkpoint_path = 'checkpoint.pth'

    # Load dataset and dataloader
    dataset = COCOCaptionDataset(ann_path=ann_path, images_dir=images_dir)
    training_loader = get_dataloader(dataset, batch_size=batch_size, shuffle=True)
    validation_loader = get_dataloader(dataset, batch_size=batch_size, shuffle=False)

    # Load model components
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")

    # Initialize VisionLanguageModel
    model = VisionLanguageModel(
        tag_embedding=dataset.all_tags_embedding, 
        ids_to_tags=dataset.ids_to_tags, 
        d_model=768, 
        ffn_hidden=2048, 
        n_head=8, 
        n_layers=6, 
        drop_prob=0.1
    ).to(device)

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps)

    # Load from checkpoint if available
    start_epoch = load_checkpoint(checkpoint_path, model, optimizer, device)

    # TensorBoard writer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/vision_language_trainer_{timestamp}')
    
    for epoch in range(start_epoch, num_epochs):
        print(f'EPOCH {epoch + 1}:')

        model.train(True)
        avg_loss = train_one_epoch(epoch, writer, model, training_loader, optimizer, device, save_checkpoint_freq, checkpoint_path)

        running_vloss = 0.0
        model.eval()

        with torch.no_grad():
            for i, vdata in tqdm(enumerate(validation_loader)):
                v_pixel_values, v_tag_labels, v_input_ids, v_attention_mask, v_labels, _ = vdata
                v_pixel_values = v_pixel_values.to(device)
                v_input_ids = v_input_ids.to(device)
                v_attention_mask = v_attention_mask.to(device)
                v_labels = v_labels.to(device)

                v_tag_logits, v_caption_loss = model(input_ids=v_input_ids, attention_mask=v_attention_mask, pixel_values=v_pixel_values, labels=v_labels)
                tag_loss_fn = BCEWithLogitsLoss()
                v_tag_loss = tag_loss_fn(v_tag_logits, v_tag_labels)
                running_vloss += (v_tag_loss.item() + v_caption_loss.item())

        avg_vloss = running_vloss / (i + 1)
        print(f'LOSS train {avg_loss} valid {avg_vloss}')

        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch + 1)
        writer.flush()

        # Save checkpoint at the end of each epoch
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'{checkpoint_path}_epoch{epoch + 1}.pth')

if __name__ == '__main__':
    main()