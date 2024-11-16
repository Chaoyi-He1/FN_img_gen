import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.misc import *
from typing import Iterable
from diffusers.models import AutoencoderKL, AutoencoderKLCogVideoX
import torch.amp
from encoder.encoder import Encoder
import utils
import torch.distributed as dist
from tensorboardX import SummaryWriter

def train_one_epoch(
    encoder: Encoder, data_loader: Iterable, optimizer: torch.optim.Optimizer, device: torch.device, 
    epoch: int, vae: AutoencoderKL, max_norm: float = 0.0, scaler: torch.amp.GradScaler = None,
    print_freq: int = 100, batch_size: int = 64, save_freq: int = 1000, rank: int = -1,
    encoder_without_ddp: torch.nn.Module = None, scheduler: torch.optim.lr_scheduler = None,
    save_dir: str = "checkpoints", tb_writer: SummaryWriter = None
):
    encoder.train()
    metric_logger = MetricLogger(delimiter="; ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    train_steps = epoch * len(data_loader) // batch_size
    
    for img, lable in metric_logger.log_every(data_loader, print_freq, header):
        train_steps += 1
        img = img.to(device)
        lable = lable.to(device)
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            with torch.no_grad():
                img = vae.encode(img).latent_dist.sample().mul_(0.18215)
        
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            y = encoder(img, lable)
            
            loss = F.mse_loss(y, img)
        
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if max_norm > 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm)
        
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        if train_steps % save_freq == 0:
            if rank in [-1, 0]:
                utils.save_on_master({
                    'encoder': encoder.state_dict(),
                    'scaler': scaler.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                }, os.path.join(save_dir, f"model_{epoch}_{train_steps:07d}.pth"))
                
            if tb_writer is not None:
                tb_writer.add_scalar("train/loss", metric_logger.global_avg("loss"), train_steps)
                tb_writer.add_scalar("train/lr", metric_logger.global_avg("lr"), train_steps)
            
            scheduler.step()
            dist.barrier()
    
    metric_logger.synchronize_between_processes()
    print("Averaged loss: ", metric_logger.global_avg("loss"))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}