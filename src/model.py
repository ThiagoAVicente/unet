
import torch.nn as nn
import torch
from typing import Optional
import logging

logger = logging.getLogger("Model")

class UNet(nn.Module):
    def __init__(self, image_size:int = 256, in_channels:int = 1, out_channels:int = 1, num_downs:int = 1 ) -> None:

        super().__init__()

        if num_downs < 1:
            raise ValueError(
                "Number of downscaling must be greater or equal to 1."
            )
        if image_size % (2 ** num_downs) != 0:
            raise ValueError(
                f"[{image_size}] is not a valid image_size. Must be a multiple of {2 ** num_downs}."
            )

        # save parameters
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_downs = num_downs

        self.epoch:int = 0
        self.loss:int = -1 # No loss because no training was done yet

        def module(a,b):
            return nn.Sequential(
                nn.Conv2d(a,b,3),
                nn.ReLU(),
                nn.Conv2d(b,b,3),
                nn.ReLU(),
            )

        logger.info("Creating encoder layers")
        encoder_blocks =  [module(self.in_channels,64)] # in -> 64
        encoder_blocks += [module(64*i, 64*(i+1)) for i in range(1,num_downs)]
        self.encoder = nn.ModuleList(encoder_blocks)

        logger.info("Creating decoder layers")
        decoder_blocks = []
        for i in range ( num_downs - 1, 0, -1):
            decoder_blocks.append( module(64*(i+1), 64*i))
        decoder_blocks.append(module(64,self.out_channels)) # 64 -> out
        self.decoder = nn.ModuleList(decoder_blocks)

        if len(encoder_blocks) == len(decoder_blocks):
            logging.info(f"Successfully created model with {num_downs} downscaling(s).")

    def save(self, file_name:str = "checkpoint.pth"):
        """Save weights and metadata of the model to a file"""
        data = {
            "epoch": self.epoch,
            "loss": self.loss,
            "params": {
                "image_size": self.image_size,
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "num_downs": self.num_downs,
            },
            "state_dict": self.state_dict(),
        }
        torch.save(data, file_name)
        logger.info(f"Model saved to {file_name}")

    @classmethod
    def load(cls, file_path:str) -> Optional["UNet"]:
        """Load model and metadata from a file"""
        try:
            checkpoint = torch.load(file_path)
            params = checkpoint["params"]

            model = cls(**params)
            model.load_state_dict(checkpoint["state_dict"])

            model.epoch = checkpoint["epoch"]
            model.loss = checkpoint["loss"]

            logger.info(f"Model loaded from {file_path}, epoch {model.epoch}")
            return model

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
            
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """process data in the network"""
        #TODO
        return x
    
    def train(self) -> None:
        pass
    
    