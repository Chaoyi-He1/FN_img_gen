import argparse
import datetime
import glob
import json
import math
import os
import random
import time
import pickle
from pathlib import Path
import tempfile

import yaml
import torch
import torch.distributed as dist
import numpy as np
from utils import torch_distributed_zero_first
from tensorboardX import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
import torch.multiprocessing
from PIL import Image
from torchvision import transforms

import utils.misc as utils
from encoder.encoder import Enc_models
from train.train_enc import *
from diffusers.models import AutoencoderKL
from diffusers import AutoencoderKLCogVideoX
from torchvision.datasets import ImageFolder

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.multiprocessing.set_sharing_strategy('file_system')


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()
    
    
def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config', default='config/cfg.yaml', type=str,
                        help='path to config file')
    parser.add_argument('opts', help='Modify config options using the command-line', default=None,
                        nargs=argparse.REMAINDER)
    
    # output directory
    parser.add_argument('--resume', default='trained_models/model_', type=str, metavar='PATH',
                        help='path to latest checkpoint (default none)')
    parser.add_argument('--save_dir', default='trained_models/', type=str,
                        help='directory to save checkpoints')
    
    # dataset parameters
    parser.add_argument('--data_path', default='data/train/', type=str,
                        help='path to dataset')
    
    # training parameters
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument('--seed', default=42, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--device', default='cuda', type=str,
                        help='device to use for training / testing')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='number of data loading workers')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='input batch size for training')
    parser.add_argument('--epochs', default=100, type=int,
                        help='number of epochs to train')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='start epoch')
    parser.add_argument('--print_freq', default=200, type=int,
                        help='print frequency')
    parser.add_argument('--save_freq', default=5, type=int,
                        help='save frequency')
    
    # optimizer parameters
    parser.add_argument('--lr', default=0.0001, type=float,
                        help='initial learning rate')
    parser.add_argument('--lrf', default=0.1, type=float,
                        help='learning rate factor')
    parser.add_argument('--clip_max_norm', default=.1, type=float,
                        help='gradient clipping max norm')
    
    # distributed training parameter
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--amp', action='store_false', default=True,
                        help='use automatic mixed precision training')
    
    args = parser.parse_args()
    assert args.config is not None
    return args


def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    utils.init_distributed_mode(args)
    if args.rank in [-1, 0]:
        print(args)
    
    assert args.batch_size % args.world_size == 0, "--batch-size must be divisible by number of GPUs"
    
    if args.device == 'cuda':
        torch.backends.cudnn.benchmark = True
    if args.amp:
        assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."
        assert torch.backends.cudnn.version() >= 7603, "Amp requires cudnn >= 7603"
    
    device = torch.device(args.device)
    # print the gpu model name to check if it is A100
    print(torch.cuda.get_device_name())
    
    # load hyper parameters
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    # Setup experiment
    if args.rank in [-1, 0] and args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    # create tensorboard writer
    writer = None
    if args.rank in [-1, 0]:
        writer = SummaryWriter()
    
    # create model
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.requires_grad_(False)
    vae.eval()
    
    encoder = Enc_models[config['enc_model']](
        input_size=config['image_size'] // 8, 
        num_classes=config['num_classes'],
        num_fourier_terms=config['num_fourier_terms'],
    )
    
    start_epoch = args.start_epoch
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp) if args.amp else None
    
    # load checkpoint
    if args.resume.endswith('.pth'):
        print(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        
        try:
            encoder.load_state_dict(checkpoint['encoder'], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --resume=.../model.pth" % (args.resume, args.config)
            raise KeyError(s) from e
        
        for layer_name, p_model in encoder.named_parameters():
            if not torch.equal(p_model, checkpoint['ssm_model'][layer_name]):
                print(f"Model and checkpoint parameters are not equal: model: {layer_name}")
                
        print("Encoder model loaded correctly")
        start_epoch = checkpoint['epoch'] + 1
        scaler.load_state_dict(checkpoint['scaler'])
        del checkpoint
        print(f"Loaded checkpoint: {args.resume}")
    
    encoder = encoder.to(device)
    
    # move to distributed mode
    encoder = torch.nn.parallel.DistributedDataParallel(encoder, device_ids=[args.gpu])
    encoder_without_ddp = encoder.module
    
    # create optimizer and scheduler and model info
    p_to_optimize, n_p, n_p_o, layers = [], 0, 0, 0
    for p in encoder.module.parameters():
        n_p += p.numel()
        if p.requires_grad:
            n_p_o += p.numel()
            p_to_optimize.append(p)
            layers += 1
    print(f"Total number of parameters: {n_p}, number of parameters to optimize: {n_p_o}, number of layers: {layers}")
    
    args.lr *= max(1., args.world_size * args.batch_size / 64)
    optimizer = torch.optim.AdamW(p_to_optimize, lr=args.lr)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch - 1 if args.resume.endswith('.pth') else start_epoch
    if args.resume.endswith('.pth'):
        scheduler.step()
    
    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, config['image_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=args.rank,
        shuffle=True,
        seed=args.seed
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        pin_memory=True,
        drop_last=True
    )
    
    # start training
    start_time = time.time()
    total_time = start_time
    for epoch in range(start_epoch, args.epochs):
        train_dict = train_one_epoch(
            encoder, dataloader, optimizer, device, epoch, vae, args.clip_max_norm, scaler,
            args.print_freq, args.batch_size, 1000, args.rank, encoder_without_ddp, scheduler,
            args.save_dir, writer
        )
    total_time = time.time() - total_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    
    if args.rank in [-1, 0]:
        writer.close()
        print('Training time {}'.format(total_time_str))
        
    cleanup()

if __name__ == '__main__':
    args = parse_args()
    main(args)