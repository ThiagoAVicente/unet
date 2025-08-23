
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

        # save parameters
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_downs = num_downs

        self.epoch:int = 0
        self.loss:float = .0 # No loss because no training was done yet

        # functions
        self.downscale = nn.MaxPool2d(kernel_size = 2, stride = 2)
        def upscale(inc,outc):
           return  nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(inc, outc, kernel_size=3, padding=1)
        )

        def module(a,b):
            return nn.Sequential(
                nn.Conv2d(a,b,3, padding=1),
                nn.ReLU(),
                nn.Conv2d(b,b,3, padding=1),
                nn.ReLU(),
            )

        logger.info("Creating encoder layers")
        channels =  [self.in_channels, 64] # in -> 64
        for i in range (1, num_downs):
            channels.append(channels[i]*2)

        encoder_blocks = [module( channels[i], channels[i+1]) for i in range(num_downs)]
        self.encoder = nn.ModuleList(encoder_blocks)

        logger.info("Creating bottom layer")
        self.bottom = module(channels[-1],channels[-1]*2) # module

        logger.info("Creating decoder layers")
        decoder_blocks = []
        upscalers = [] 
        
        for i in range ( num_downs, 0, -1):
            inc = channels[i]*2
            outc = channels[i]
            decoder_blocks.append(
                module(inc,outc)
            )
            upscalers.append(upscale(inc,outc))
            
        self.decoder = nn.ModuleList(decoder_blocks)
        self.upscalers = nn.ModuleList(upscalers)

        self.out = nn.Conv2d(64,self.out_channels,kernel_size=1)

        if len(encoder_blocks) == len(decoder_blocks):
            logger.info(f"Successfully created model with {num_downs} downscaling(s).")



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
        """
        process data in the network
        x:torch.Tensor -> 4D tensor with"""

        # validate x sizes
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (batch, channels, height, width), got {x.dim()}D ")

        if x.size(1) != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {x.size(1)}")

        h = x.size(2)
        w = x.size(3)

        if h != w != self.image_size:
            logger.warning(f"Input size {h}x{w} differs from expected {self.image_size}x{self.image_size}")
            return torch.zeros(x.size(0),self.out_channels,self.image_size,self.image_size)

        # propagate data
        current = x
        encoder_outputs = [] # stack to store intermediate outputs

        # encode
        for enc in self.encoder:
            current = enc(current)
            encoder_outputs.append(current)
            current = self.downscale(current)

        # bottom
        current = self.bottom(current)

        for i,dec in enumerate(self.decoder):
            # upscale
            current = self.upscalers[i](current)
            enc_output = encoder_outputs.pop()
            current = torch.cat([current,enc_output],dim=1)

            # doubleconv-
            current = dec(current)
            
            
            


        return self.out(current)

    def train_model(self, train_loader, criterion, optimizer, epochs:int, lr:float = .001) -> None:
        """
        Train the unet model
        train_loader: pytorch dataloader containing input images and target
        criterion: loss function
        optimizer: pytorch optimizer
        epochs: number of cycles trough the training dataset
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        trainer = logger.getChild("Trainer")
        trainer.info("Starting model trainment")
        for epoch in range(epochs):
            self.train()
            running_loss = .0

            for inputs,targets in train_loader:
                inputs,targets = inputs.to(device), targets.to(device)

                # parameter gradients to 0
                optimizer.zero_grad()

                outputs = self(inputs)
                
                loss = criterion(outputs,targets)

                running_loss += loss.item()

                # autograd
                loss.backward()
                optimizer.step()

            self.epoch += 1
            self.loss = running_loss / len(train_loader)

            trainer.info(f"Epoch {epoch} : loss {self.loss}")
        return

    def __str__(self) -> str:
        """Return a string representation of the model structure"""
        structure = []

        # encoder
        structure.append("UNet Architecture:")
        structure.append("\nENCODER BLOCKS:")
        for i, block in enumerate(self.encoder):
            structure.append(f"  Block {i}: {block}")

        # bottom
        structure.append("\nBOTTOM:")
        structure.append(f"  {self.bottom}")

        #decoder
        structure.append("\nDECODER BLOCKS:")
        for i, block in enumerate(self.decoder):
            structure.append(f"  Block {i}: {block}")

        #output
        structure.append("\nOUTPUT:")
        structure.append(f"  {self.out}")

        return '\n'.join(structure)
