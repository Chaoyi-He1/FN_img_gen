import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.misc import *
from typing import Iterable
from diffusers.models import AutoencoderKL, AutoencoderKLCogVideoX
import torch.amp
from model.encoder import Encoder
from model.decoder import Decoder
from model.FN_block import FN_coefficient, FourierSeries_Reconstruction
from model.loss import total_loss
import utils
import torch.distributed as dist
from tensorboardX import SummaryWriter

def train_one_epoch(
    encoder: Encoder, decoder: Decoder, fn_block: FN_coefficient, loss_fn: total_loss,
    data_loader: Iterable, optimizer: torch.optim.Optimizer, device: torch.device, 
    epoch: int, vae: AutoencoderKL, max_norm: float = 0.0, scaler: torch.amp.GradScaler = None,
    print_freq: int = 100, batch_size: int = 64, save_freq: int = 1000, rank: int = -1,
    encoder_without_ddp: torch.nn.Module = None, decoder_without_ddp: torch.nn.Module = None, FN_coefficient_without_ddp: torch.nn.Module = None,
    scheduler: torch.optim.lr_scheduler = None, save_dir: str = "checkpoints", tb_writer: SummaryWriter = None
):
    encoder.train()
    metric_logger = MetricLogger(delimiter="; ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    train_steps = epoch * len(data_loader) // batch_size
    
    for img, lable in metric_logger.log_every(data_loader, print_freq, header):
        train_steps += 1
        img = img.to(device).half()
        lable = lable.to(device)
        
        if vae is not None:
            with torch.amp.autocast('cuda', enabled=scaler is not None):
                with torch.no_grad():
                    img = vae.encode(img).latent_dist.sample().mul_(0.18215)
        
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            encoded = encoder(img, lable)
            fourier_coefficients = fn_block(img, lable)
            An, Bn = {'Axy': fourier_coefficients['Axy'], 'Ayx': fourier_coefficients['Ayx']}, {'Bxy': fourier_coefficients['Bxy'], 'Byx': fourier_coefficients['Byx']}
            fourier_reconstructed_encoded = FourierSeries_Reconstruction(An, Bn, num_patch=encoder.module.x_embedder.num_patches, hidden_size=encoder.module.hidden_size)
            decoded = decoder(fourier_reconstructed_encoded, lable)
            
            total_loss, enc_loss, fourier_loss, dec_loss = loss_fn(encoded, fourier_coefficients, lable, decoded, img)
        
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        
        if max_norm > 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm)
        
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        metric_logger.update(total_loss=total_loss.item())
        metric_logger.update(enc_loss=enc_loss.item())
        metric_logger.update(fourier_loss=fourier_loss.item())
        metric_logger.update(dec_loss=dec_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        if train_steps % save_freq == 0:
            if rank in [-1, 0]:
                utils.save_on_master({
                    'encoder': encoder_without_ddp.state_dict(),
                    'decoder': decoder_without_ddp.state_dict(),
                    'fn_model': FN_coefficient_without_ddp.state_dict(),
                    'scaler': scaler.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                }, os.path.join(save_dir, f"model_{epoch}_{train_steps:07d}.pth"))
                
            if tb_writer is not None:
                tb_writer.add_scalar("train_loss/total_loss", metric_logger.meters['total_loss'].global_avg, train_steps)
                tb_writer.add_scalar("train_loss/enc_loss", metric_logger.meters['enc_loss'].global_avg, train_steps)
                tb_writer.add_scalar("train_loss/fourier_loss", metric_logger.meters['fourier_loss'].global_avg, train_steps)
                tb_writer.add_scalar("train_loss/dec_loss", metric_logger.meters['dec_loss'].global_avg, train_steps)
                tb_writer.add_scalar("lr", metric_logger.meters['lr'].global_avg, train_steps)
            
            dist.barrier()
    
    metric_logger.synchronize_between_processes()
    print("Averaged loss: ", metric_logger.meters['loss'].global_avg)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}