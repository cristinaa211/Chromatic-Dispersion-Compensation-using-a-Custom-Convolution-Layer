from chain_layers import Processors
import torch 
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class AWGN(Processors.Processor):
    type = "channel"

    def __init__(self,M, SNR_db = 30, os = 1, is_complex = True,name = "AWGN"):
        self.M = M
        self.SNR_db = SNR_db
        self.os = os
        self.is_complex = is_complex
        self.name = name

    def get_noise(self, N, power):
        term = torch.log2(torch.tensor(self.M))
        self.sigma2 = (power * 10 ** (-self.SNR_db/10)) * self.os/term
        if self.is_complex == True:
            sigma2_r = torch.sqrt(self.sigma2/2)
            noise = torch.normal(mean = 0.0, std=float(sigma2_r),size=(1,N) )+ 1j*torch.normal(mean =0.0, std=float(sigma2_r),size=(1,N) )
        else:
            sigma2_r = torch.sqrt(self.sigma2)
            noise = torch.normal(mean = 0.0, std=np.float(sigma2_r),size=(1,N) )
        return noise
 
    def forward(self,input_data):
        N = len(input_data)
        signal_power = torch.mean(torch.abs(input_data)**2)
        self.noise = self.get_noise(N,signal_power)
        output_data = input_data.to(device) + self.noise.to(device)
        return output_data



