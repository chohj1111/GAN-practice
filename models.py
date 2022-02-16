import torch
import torch.nn as nn

class G(nn.Module):
    def __init__(self, d_noise=100, d_hidden=256, dropout_p=0.1):
        super(G, self).__init__()
        self.G = nn.Sequential(
            nn.Linear(d_noise,d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_p), 
            nn.Linear(d_hidden,d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_p), 
            nn.Linear(d_hidden, 28 * 28),
            nn.Tanh()
        )
    def forward(self, z):
        img_fake = self.G(z) 
        return img_fake

class D(nn.Module):
    def __init__(self, d_noise=100, d_hidden=256, dropout_p=0.1):
        super(D, self).__init__()

        self.D = nn.Sequential(
            nn.Linear(28*28, d_hidden),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(d_hidden, d_hidden),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(d_hidden,1),
            nn.Sigmoid()
        )
    def forward(self, img_fake): 
        output = self.D(img_fake)
        return output