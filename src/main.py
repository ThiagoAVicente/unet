from model import UNet
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import Subset
import random

import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Main")

def main():
    load_dotenv()

    image_size = int(os.getenv("IMAGE_SIZE", 256))
    in_channels = int(os.getenv("IN_CHANNELS", 3))
    out_channels = int(os.getenv("OUT_CHANNELS", 19))
    num_downs = int(os.getenv("NUM_DOWNS", 4))
    batch_size = int(os.getenv("BATCH_SIZE", 8))
    learning_rate = float(os.getenv("LEARNING_RATE", 0.001))
    epochs = int(os.getenv("EPOCHS", 10))
    data_root = os.getenv("DATA_ROOT", "data")
    save_path = os.getenv("SAVE_PATH", "unknown.pth")
    limit = int(os.getenv("DATASET_LIMIT",100))

    # ensure dirs exist
    os.makedirs(data_root, exist_ok=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    logger.info(f"Configuration: image_size={image_size}, in_channels={in_channels}, "
                    f"out_channels={out_channels}, num_downs={num_downs}")

    # initialize model
    #model = UNet(
    #    image_size = image_size,
    #    in_channels= in_channels,
    #    out_channels= out_channels,
    #    num_downs=num_downs
    #)
    model = UNet.load("models/pets-07.pth")

    # fefine transformations
    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    target_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),
        lambda x: x.squeeze(0) if x.dim() == 3 else x,
        lambda x: x.long()
    ])

    full_dataset = OxfordIIITPet(
        root=data_root,
        split="trainval",
        target_types="segmentation",
        transform=image_transform,
        target_transform=target_transform,
        download=True
    )

    all_indices = list(range(len(full_dataset)))
    random.shuffle(all_indices)  # Randomize the order

    # Take the first N indices from the shuffled list
    indices = all_indices[:min(limit, len(full_dataset))]

    dataset = Subset(full_dataset, indices)

    data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train_model(data_loader,criterion,optimizer,epochs)
    model.save(save_path)

if __name__ == "__main__":
    main()
