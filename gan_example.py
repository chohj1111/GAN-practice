import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision import utils
import torchvision.datasets as dsets

import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import models 


''' 
Functions
'''
def sample_z(batch_size, d_noise=100):
    return torch.rand(batch_size, d_noise, device=device)

def imshow_grid(img):
    img = utils.make_grid(img.cpu().detach())
    img = (img+1)/2
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

def run_epoch(generator, discriminator, g_optimizer, d_optimizer):
    generator.train()
    discriminator.train()

    criterion = nn.BCELoss()

    for img_batch, label_batch in train_data_loader:
        img_batch, label_batch = img_batch.to(device), label_batch.to(device)
        img_batch = img_batch.reshape(-1, 28 * 28)

        p_real = discriminator(img_batch)
        p_fake = discriminator(generator(sample_z(batch_size)))

        # loss_real = (-1) * torch.log(p_real)
        # loss_fake = (-1) * torch.log(1 - p_fake)

        # loss_d = (loss_real + loss_fake).mean()
        loss_d = criterion(p_real, torch.ones(p_real.size()).to(device)) + criterion(
            p_fake, torch.zeros(p_fake.size()).to(device)
        )

        d_optimizer.zero_grad()
        loss_d.backward()
        d_optimizer.step()

        p_fake = discriminator(generator(sample_z(batch_size)))
        loss_g = criterion(p_fake, torch.ones(p_fake.size()).to(device))

        g_optimizer.zero_grad()
        loss_g.backward()
        g_optimizer.step()


def evaluate_model(generator, discriminator):
    p_real, p_fake = 0.0, 0.0

    generator.eval()
    discriminator.eval()

    for img_batch, label_batch in test_data_loader:
        img_batch, label_batch = img_batch.to(device), label_batch.to(device)
        with torch.autograd.no_grad():
            p_real += (torch.sum(discriminator(img_batch.view(-1, 28 * 28))).item())/10000.0
            p_fake += (torch.sum(discriminator(generator(sample_z(batch_size)))).item())/10000.0

    return p_real, p_fake


def init_params(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)
        else:
            nn.init.uniform_(p, 0.1, 0.2)



batch_size = 200
dropout_p = 0.1
d_noise = 100
d_hidden = 256
epoch_num = 200

device = torch.device("cuda:2")


normalize = transforms.Normalize(mean=[0.5, ], std=[0.5, ])
preprocess = transforms.Compose([
                    transforms.ToTensor(),
                    normalize
])

# MNIST dataset
train_data = dsets.MNIST(root="data/", train=True, transform=preprocess, download=True)
test_data = dsets.MNIST(root="data/", train=False, transform=preprocess, download=True)

train_data_loader = DataLoader(train_data, batch_size, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size, shuffle=False)


G = models.G().to(device)
D = models.D().to(device)

init_params(G)
init_params(D)

optimizer_g = optim.Adam(G.parameters(), lr=0.0002)
optimizer_d = optim.Adam(D.parameters(), lr=0.0002)

p_real_trace = []
p_fake_trace = []

print("Train Started")
for epoch in tqdm(range(epoch_num)):
    run_epoch(G, D, optimizer_g, optimizer_d)
    p_real, p_fake = evaluate_model(G, D)

    p_real_trace.append(p_real)
    p_fake_trace.append(p_fake)

    if (epoch + 1) % 50 == 0:
        print("(epoch %i/200) p_real: %f, p_g: %f" % (epoch + 1, p_real, p_fake))
        cv2.imshow_grid(G(sample_z(16)).view(-1, 1, 28, 28))

# plot loss
plt.plot(p_fake_trace, label='D(x_generated)')
plt.plot(p_real_trace, label='D(x_real)')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()

# print test image 
vis_loader = torch.utils.data.DataLoader(test_data, 16, True)
img_vis, label_vis   = next(iter(vis_loader))
imshow_grid(img_vis)

imshow_grid(G(sample_z(16,100)).view(-1, 1, 28, 28))
