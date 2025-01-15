import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Attention
import numpy as np
from .FN_block import FN_coefficients_loss, FourierSeries_Reconstruction


class total_loss(nn.Module):
    def __init__(self, lambda_reconstruction: float = .0, lambda_fourier: float = 1.0, lambda__decoder: float = 1.0):
        super(total_loss, self).__init__()
        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_fourier = lambda_fourier
        self.lambda_decoder = lambda__decoder
        self.reconstruction = FourierSeries_Reconstruction
        self.fourier_loss = FN_coefficients_loss()
    
    def forward(self, encoded_image, fourier_coefficients, label, decoded_image, img):
        # Reconstruction loss, detach the fourier coefficients to avoid backpropagating through the Fourier coefficients model
        fourier_coefficients_detach_An = {'Axy': fourier_coefficients['Axy'].detach().clone(), 'Ayx': fourier_coefficients['Ayx'].detach().clone()}
        fourier_coefficients_detach_Bn = {'Bxy': fourier_coefficients['Bxy'].detach().clone(), 'Byx': fourier_coefficients['Byx'].detach().clone()}
        reconstruct_img = self.reconstruction(fourier_coefficients_detach_An, fourier_coefficients_detach_Bn,
                                              encoded_image.shape[1], encoded_image.shape[2])
        reconstruction_loss = F.mse_loss(encoded_image,
                                         reconstruct_img)
        
        # Fourier loss
        fourier_loss = self.fourier_loss(fourier_coefficients, label)
        
        # Decoder loss
        decoded_image_loss = F.mse_loss(decoded_image, img.detach())
        
        return self.lambda_reconstruction * reconstruction_loss + self.lambda_fourier * fourier_loss + self.lambda_decoder * decoded_image_loss, \
                reconstruction_loss, fourier_loss, decoded_image_loss