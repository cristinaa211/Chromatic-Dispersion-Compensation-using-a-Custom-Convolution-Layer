import torch
from torch import nn
from chain_layers.Processors import Processor
import pytorch_lightning as pl



class IQ_imbalance(Processor):
    ''' Introduce some IQ Imbalance '''
    def __init__(self, alpha_db = 1, theta_deg = 2 ,name="IQ_Imbalance"):
        super(IQ_imbalance, self).__init__()
        self.alpha_db1 = torch.tensor(alpha_db)
        self.theta_deg1 = torch.tensor(theta_deg)
        self.name = name
        self.alpha1 = 10 ** (self.alpha_db1/10) - 1
        self.theta_rad1 = torch.deg2rad(self.theta_deg1)

    def forward(self,input_data):
        self.mu = torch.cos(self.theta_rad1/2) + 1j*self.alpha1*torch.sin(self.theta_rad1/2)
        self.nu = self.alpha1*torch.cos(self.theta_rad1/2) - 1j*torch.sin(self.theta_rad1/2)
        output_data = self.mu * input_data + self.nu * torch.conj(input_data)
        return output_data

class IQ_imbalance_compensation(pl.LightningModule,nn.Module):

    def  __init__(self, alpha = 1, theta = 2 ):
        super(IQ_imbalance_compensation, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.theta = torch.tensor(theta)
        self.alpha = 10 ** (self.alpha/10) - 1
        self.theta = torch.deg2rad(self.theta)
        self.alpha_p = nn.Parameter(self.alpha,requires_grad = True)
        self.theta_p = nn.Parameter(self.theta,requires_grad = True)

    def param(self):
        self.mu =  torch.cos(self.theta_p/2) + 1j*self.alpha_p*torch.sin(self.theta_p/2)
        self.nu = self.alpha_p*torch.cos(self.theta_p/2) - 1j*torch.sin(self.theta_p/2)
        term = torch.abs(self.mu)**2 - torch.abs(self.nu)**2
        self.beta1 = torch.conj(self.mu) / term
        self.beta2 = -self.nu/term
        return self.beta1, self.beta2

    def forward(self, input_data):
        output = self.beta1 * input_data + self.beta2 * torch.conj(input_data)
        return output

