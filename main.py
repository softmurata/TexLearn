import argparse
import torch
import torch.nn as nn
import numpy as np
from model import Unet
from torch.utils.data import DataLoader
from dataset import ImageDataset
from torch import optim

parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=3)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--epoch', type=int, default=10)

parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.999)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# create dataset class
train_dataset = ImageDataset()

# build data loader class
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

# loss function
loss_function = nn.MSELoss()

# build model and optimizer
model = Unet(in_channels=3, out_channels=3, mode='linear')
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

# switch train mode
model.train()

for e in range(args.epoch):

    losses = []

    for idx, data in enumerate(train_dataloader):

        samples, labels = data
        print(samples.shape)

        # to cuda device
        samples = samples.to(device)
        labels = labels.to(device)

        output = model(samples)
        print('output shape:', output.shape)
        loss = loss_function(output, labels)
        print('loss:', loss.item())

        losses.append(loss.item())

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = np.mean(losses)
    print('epoch:', e, 'loss:', avg_loss)

