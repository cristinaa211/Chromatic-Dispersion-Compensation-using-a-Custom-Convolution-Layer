import torch
from torch import nn
from chain_layers.Processors import Processor
import pytorch_lightning as pl

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ChromaticDispersion(nn.Module, Processor):
    
    def __init__(self,D,L,Lambda,fs, name = 'CD impairement'):
        super(ChromaticDispersion, self).__init__()
        self.D = torch.tensor(D,dtype = torch.float16)
        self.L = torch.tensor(L)
        self.Lambda = torch.tensor(Lambda)
        self.fs = torch.tensor(fs)
        self.name = name
        self.c = 3e8

    def H(self,N=None):
        f = torch.fft.fftshift(torch.linspace(-self.fs/2, self.fs/2,N))
        w = 2*torch.pi*f*(1/self.fs)
        self.K = (self.D * self.L* self.Lambda ** 2 ) / (4 * torch.pi * self.c*(1/self.fs)**2)
        self.C = torch.exp(-1j*self.K*w**2)
        return self.C

    def forward(self,input_data):
        # get channel response
        input_data = torch.flatten(input_data)
        N = len(input_data)
        H = self.H(N).to(device)
        # Frequency domain
        fft_input = torch.fft.fft(input_data).to(device)
        fft_output = H * fft_input
        # Time domain
        output_data = torch.fft.ifft(fft_output)
        output_data = output_data.unsqueeze(dim = 0)
        return output_data
