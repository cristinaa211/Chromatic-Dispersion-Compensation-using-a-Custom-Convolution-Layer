import torch
from chain_layers.Processors import Processor
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class LaserPhaseNoise(nn.Module,Processor):

    def __init__(self,fs, delta_f=0, K = 10 , name = 'Phase Noise' ):
        super(LaserPhaseNoise,self).__init__()
        self.fs = fs # the sampling frequency
        # the linewidth of the laser
        # ideal delta_f = 0 Hz
        self.delta_f = torch.tensor(delta_f)
        self.K = K
        self.name = name

    def laser_phase_noise(self, N):
        variance = 2*torch.pi*self.delta_f/self.fs
        std_pn = torch.sqrt(variance)
        fi_s = torch.normal(0.0, std=std_pn, size= (1,int(N/self.K)))
        theta = torch.cumsum(fi_s, dim = 0)
        theta = torch.flatten(theta)
        diag = torch.exp(theta).type(torch.float16)
        matrix = torch.diag(diag).to(device)
        I = torch.eye(int(self.K)).to(device)
        phi_theta = torch.kron(matrix,I)
        return phi_theta

    def forward(self,input_data):
        N = input_data.size(1)  
        torch.random.seed()
        pn = self.laser_phase_noise(N)
        output_data = pn * input_data.to(device)
        # # output_data = torch.matmul(pn, input_data)
        return output_data
