'''
Use this block to learn the fourier coefficients from the input image wich is used to regress the tokens of the image generated by the encoder.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Attention
import numpy as np


class res_block_with_label_condition(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, label_dim: int = 256, downsample = None):
        super(res_block_with_label_condition, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.label_dim = label_dim
        self.label_emb = nn.Linear(label_dim, out_channels)
    
    def forward(self, x, label):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        label = self.label_emb(label)
        out = out * (1 + F.sigmoid(label / label.max()))
        
        return out
    

class FN_coefficient(nn.Module):
    '''
    This model uses a resnet like architecture with label conditioning to learn the fourier coefficients of the input image.
    '''
    def __init__(self, input_size: int = 32, 
                 in_channels: int = 4, 
                 num_fourier_terms: int = 256, 
                 num_classes: int = 1000):
        super(FN_coefficient, self).__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.num_fourier_terms = num_fourier_terms
        self.num_classes = num_classes
        
        self.label_emb = nn.Embedding(num_classes, 256)
        
        # resnet model
        self.res_block = nn.ModuleList([
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        ])
        in_channels, input_size = 8, input_size
        for _ in range(4):
            self.res_block.append(res_block_with_label_condition(in_channels, in_channels, label_dim=256))
            self.res_block.append(res_block_with_label_condition(in_channels, in_channels * 2, 
                                                                 downsample=nn.Conv2d(in_channels, in_channels * 2, kernel_size=2, stride=2, padding=0),
                                                                 label_dim=256))
            in_channels *= 2
            input_size //= 2
        
        # flatten the output and pass it through a linear layer to get the fourier coefficients
        self.fc_xy = nn.Linear(in_channels * input_size * input_size, num_fourier_terms * 2)
        self.fc_yx = nn.Linear(in_channels * input_size * input_size, num_fourier_terms * 2)
        self.fc_bias = nn.Linear(in_channels * input_size * input_size, 1)
        self.flatten = nn.Flatten()
    
    def forward(self, x, label):
        label = self.label_emb(label)
        for layer in self.res_block:
            if isinstance(layer, res_block_with_label_condition):
                x = layer(x, label)
            else:
                x = layer(x)
        x = self.flatten(x)
        x1 = self.fc_xy(x)
        x2 = self.fc_yx(x)
        x3 = self.fc_bias(x)
        Axy, Bxy = x1.chunk(2, dim=1)
        Ayx, Byx = x2.chunk(2, dim=1)
        return {'Axy': Axy, 'Bxy': Bxy, 'Ayx': Ayx, 'Byx': Byx, 'A0': x3}


class FN_coefficients_loss(nn.Module):
    '''
    This loss function is used to separate the fourier coefficients of the input image with different labels.
    Make fourier coefficients of the same label close to each other and those of different labels far from each other.
    Use cosine similarity to calculate the loss to reach the above objective.
    '''
    def __init__(self, num_classes: int = 1000):
        super(FN_coefficients_loss, self).__init__()
        self.num_classes = num_classes
        self.cos = nn.CosineSimilarity(dim=1)
        
    def forward(self, fn_coefficients, labels):
        '''
        fn_coefficients: a dict of tensors of shape (batch_size, num_fourier_terms)
            dict keys: Axy, Bxy, Ayx, Byx, A0
        labels: tensor of shape (batch_size)
        '''
        Axy, Bxy, Ayx, Byx, _ = fn_coefficients['Axy'], fn_coefficients['Bxy'], fn_coefficients['Ayx'], fn_coefficients['Byx'], fn_coefficients['A0']
        loss = 0
        label_set = torch.unique(labels)
        for l in label_set:
            Axy_l, Axy_not_l = Axy[labels == l], Axy[labels != l]
            Bxy_l, Bxy_not_l = Bxy[labels == l], Bxy[labels != l]
            Ayx_l, Ayx_not_l = Ayx[labels == l], Ayx[labels != l]
            Byx_l, Byx_not_l = Byx[labels == l], Byx[labels != l]
            if sum(labels == l) > 1:
                # closer distance between the same label for each fourier coefficient
                loss += 4 - self.cos(Axy_l.unsqueeze(1), Axy_l.unsqueeze(0)).mean() - self.cos(Bxy_l.unsqueeze(1), Bxy_l.unsqueeze(0)).mean() \
                        - self.cos(Ayx_l.unsqueeze(1), Ayx_l.unsqueeze(0)).mean() - self.cos(Byx_l.unsqueeze(1), Byx_l.unsqueeze(0)).mean()
            # farther distance between different labels for each fourier coefficient
            loss += self.cos(Axy_l.unsqueeze(1), Axy_not_l.unsqueeze(0)).mean() + self.cos(Bxy_l.unsqueeze(1), Bxy_not_l.unsqueeze(0)).mean() \
                    + self.cos(Ayx_l.unsqueeze(1), Ayx_not_l.unsqueeze(0)).mean() + self.cos(Byx_l.unsqueeze(1), Byx_not_l.unsqueeze(0)).mean()
        return loss / len(label_set)


def FourierSeries_Reconstruction(A0, An, Bn, img_size):
    '''
    Reconstruct the image from the Fourier series coefficients.
    An contains {An_x, An_y, An_xy}, Bn contains {Bn_x, Bn_y, Bn_xy}
    An is a dict {"An_x": An_x, "An_y": An_y, "An_xy": An_xy}, Bn is a dict {"Bn_x": Bn_x, "Bn_y": Bn_y, "Bn_xy": Bn_xy}
    where An_x, An_y, An_xy, Bn_x, Bn_y, Bn_xy are tensors of shape (Batch, C, N), N is the number of Fourier series terms, C is the number of channels in the image
    A0 is a tensor of shape (Batch, C), representing the DC component of the Fourier series
    
    img(i, j) = A0 + sum(An_yx * cos(2*pi*n*i/img_size) * sin(2*pi*n*j/img_size)) + sum(Bn_yx * sin(2*pi*n*i/img_size) * cos(2*pi*n*j/img_size))
                   + sum(An_xy * cos(2*pi*n*i/img_size) * cos(2*pi*n*j/img_size)) + sum(Bn_xy * sin(2*pi*n*i/img_size) * sin(2*pi*n*j/img_size))
    '''
    device = A0.device
    pi = torch.tensor(3.1415927, device=device)
    batch_size, C, N = An["An_xy"].shape
    
    # Precompute cosine and sine terms
    i_indices = torch.arange(img_size, device=device).float()   # (img_size,)
    j_indices = torch.arange(img_size, device=device).float()   # (img_size,)
    
    cos_i = torch.cos(2 * pi * i_indices[:, None] * torch.arange(N, device=device) / img_size)  # (img_size, N)
    sin_i = torch.sin(2 * pi * i_indices[:, None] * torch.arange(N, device=device) / img_size)  # (img_size, N)
    cos_j = torch.cos(2 * pi * j_indices[:, None] * torch.arange(N, device=device) / img_size)  # (img_size, N)
    sin_j = torch.sin(2 * pi * j_indices[:, None] * torch.arange(N, device=device) / img_size)  # (img_size, N)
    
    # Compute components for x, y, and xy
    # img_x = (An["An_x"][:, :, None, :] * cos_i[None, None, :, :] + Bn["Bn_x"][:, :, None, :] * sin_i[None, None, :, :]).sum(dim=-1)  # (Batch, C, img_size)
    # img_y = (An["An_y"][:, :, None, :] * cos_j[None, None, :, :] + Bn["Bn_y"][:, :, None, :] * sin_j[None, None, :, :]).sum(dim=-1)  # (Batch, C, img_size)
    img_xy = (An["An_xy"][:, :, None, None, :] * cos_i[None, None, :, None, :] * cos_j[None, None, None, :, :] + 
              Bn["Bn_xy"][:, :, None, None, :] * sin_i[None, None, :, None, :] * sin_j[None, None, None, :, :]).sum(dim=-1)  # (Batch, C, img_size, img_size)
    
    img_yx = (An["An_yx"][:, :, None, None, :] * cos_i[None, None, :, None, :] * sin_j[None, None, None, :, :] + 
              Bn["Bn_yx"][:, :, None, None, :] * sin_i[None, None, :, None, :] * cos_j[None, None, None, :, :]).sum(dim=-1)  # (Batch, C, img_size, img_size)
    
    # Combine all components
    img = img_xy + img_yx
    img += A0[:, :, None, None]  # Add the DC component
    
    return img