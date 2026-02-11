import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fluid_adam import FluidAdam

EPOCHS = 10
LATENT_DIM = 100
IMG_SIZE = 28
BATCH_SIZE = 64
LR = 0.0002

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

def run_gan(optimizer_name="Adam"):
    print(f"--- Training GAN with {optimizer_name} ---")
    
    os.makedirs("./images", exist_ok=True)
    img_shape = (1, IMG_SIZE, IMG_SIZE)
    
    dataloader = DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize([0.5], [0.5])
                       ])),
        batch_size=BATCH_SIZE, shuffle=True
    )

    generator = Generator(LATENT_DIM, img_shape).cuda()
    discriminator = Discriminator(img_shape).cuda()
    adversarial_loss = nn.BCELoss().cuda()

    if optimizer_name == "FluidAdam":
        optimizer_G = FluidAdam(generator.parameters(), lr=LR, betas=(0.5, 0.999), nu=0.7)
        optimizer_D = FluidAdam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999), nu=0.7)
    else:
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

    g_losses = []
    d_losses = []
    
    start_time = time.time()

    for epoch in range(EPOCHS):
        for i, (imgs, _) in enumerate(dataloader):
            imgs = imgs.cuda()
            valid = torch.ones(imgs.size(0), 1).cuda()
            fake = torch.zeros(imgs.size(0), 1).cuda()

            optimizer_G.zero_grad()
            z = torch.randn(imgs.size(0), LATENT_DIM).cuda()
            gen_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
        
        print(f"[Epoch {epoch+1}/{EPOCHS}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())

    end_time = time.time()
    print(f"Total Time: {end_time - start_time:.2f}s")
    
    return g_losses, d_losses

if __name__ == "__main__":
    if torch.cuda.is_available():
        g_adam, d_adam = run_gan("Adam")
        g_fluid, d_fluid = run_gan("FluidAdam")
        
        plt.figure(figsize=(10, 5))
        plt.plot(g_adam, label='G Loss (Adam)', linestyle='--')
        plt.plot(g_fluid, label='G Loss (FluidAdam)')
        plt.title("Generator Loss Comparison")
        plt.legend()
        plt.show()
    else:
        print("CUDA not available. Skipping GAN training.")