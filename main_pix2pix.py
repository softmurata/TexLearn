import argparse
import os
import torch
import torch.nn as nn
from dataset_pix2pix import ImageDataset
from torch.utils.data import DataLoader
from torch import optim


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='./results/models/')
parser.add_argument('--model_type', type=str, default='light', help='1. light(input size=(128, 128)), 2. middle(input_size=(256, 256)), 3. heavy(input_size=512, 512))')
parser.add_argument('--generator_type', type=str, default='unet',
                    help='unet, resnet18, resnet34, resnet50, resnet101, resnet152')
parser.add_argument('--dataset_dir', type=str, default='./anime_images/')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--l1_lambda', type=float, default=100)
parser.add_argument('--save_freq', type=int, default=5)
args = parser.parse_args()

os.makedirs(args.model_path, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.model_type == 'light':
    image_size = 128
elif args.model_type == 'middle':
    image_size = 256
else:
    image_size = 512

# create train dataset class
train_dataset = ImageDataset(image_size=image_size, dataset_dir=args.dataset_dir, phase='train')

# build data loader class
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)


# case
# 1. line => rgb
# generator: in_channels=1, out_channels=3
# discriminator: in_channels=3, out_channels=1

# build model and optimizer
# import generator module
if args.model_type == 'light':
    from model_pix2pix_light import UnetGenerator, ResnetGenerator, Discriminator
elif args.model_type == 'middle':
    from model_pix2pix_middle import UnetGenerator, ResnetGenerator, Discriminator
else:
    from model_pix2pix import UnetGenerator, ResnetGenerator, Discriminator
    
if args.generator_type == 'unet':
    generator = UnetGenerator(in_channels=1, out_channels=3, mode='nearest')
else:
    generator = ResnetGenerator(in_channels=1, out_channels=3, resnet_type=args.generator_type)
discriminator = Discriminator(in_channels=4, out_channels=1)

# criterion
loss_func = nn.BCEWithLogitsLoss()
l1_loss_func = nn.L1Loss()

optimizer_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

# for patchGAN
batch_size = args.batch_size
ones = torch.ones(batch_size, 1, 8, 8)
zeros = torch.zeros(batch_size, 1, 8, 8)

for e in range(args.epoch):

    for idx, data in enumerate(train_dataloader):
        # shape
        # real_a => mask, real_b => rgb
        real_gray, real_color = data
        batch_len = len(real_color)

        # correspond to device
        real_gray = real_gray.to(device)
        real_color = real_color.to(device)
        
        # print(real_a.shape)
        fake_color = generator(real_gray)
        fake_color_tensor = fake_color.detach()  # g(real) => rgb
        
        
        # Generator
        disc_input = torch.cat([fake_color, real_gray], dim=1)
        out = discriminator(disc_input)
        # BCE loss
        loss_g_bce = loss_func(out, ones[:batch_len])
        # L1 loss
        loss_g_l1 = l1_loss_func(fake_color, real_color)
        loss_g = loss_g_bce + args.l1_lambda * loss_g_l1
        
        # backpropagation for generator
        optimizer_d.zero_grad()
        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()
        
        # Discriminator
        # real part
        disc_real_input = torch.cat([real_color, real_gray], dim=1)
        real_out = discriminator(disc_real_input)
        loss_d_real = loss_func(real_out, ones[:batch_len])
        # fake part
        disc_fake_input = torch.cat([fake_color_tensor, real_gray], dim=1)
        fake_out = discriminator(disc_fake_input)
        loss_d_fake = loss_func(fake_out, zeros[:batch_len])
        
        loss_d = loss_d_real + loss_d_fake
        
        # backpropagation for discriminator
        optimizer_d.zero_grad()
        optimizer_g.zero_grad()
        loss_d.backward()
        optimizer_d.step()
        

        print('===> Epoch[{}]({}/{}): Loss D: {:.4f}  Loss G: {:.4f}'.format(e, idx, len(train_dataloader), loss_d.item(), loss_g.item()))

    # save checkpoint
    if e % args.save_freq == 0:
        # save generator parameters
        generator_state_dict = {'epoch': e, 'model': generator.state_dict(), 'optimizer': optimizer_g.state_dict(),
                                'loss_g': loss_g}
        generator_path = args.model_path + 'checkpoint_generator_%05d.pth.tar' % e
        torch.save(generator_state_dict, generator_path)

        # save discriminator parameters
        discriminator_state_dict = {'epoch': e, 'model': discriminator.state_dict(),
                                    'optimizer': optimizer_d.state_dict(), 'loss_d': loss_d}
        discriminator_path = args.model_path + 'checkpoint_discriminator_%05d.pth.tar' % e
        torch.save(discriminator_state_dict, discriminator_path)





