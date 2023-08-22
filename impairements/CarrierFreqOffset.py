import torch
from chain_layers.Processors import Processor
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'device'

class CarrierFrequencyOffset(nn.Module,Processor):
    '''Introduce Carrier Frequency Offset'''
    def __init__(self,Fs, delta, name="Carrier_Frequency_Offset"):
        super(CarrierFrequencyOffset,self).__init__()
        self.Fs = Fs
        self.w_delta = torch.tensor(delta,requires_grad=True)
        self.name = name
        self.parameteres = nn.Parameter(self.w_delta)

    def forward(self,input_data):
        N = input_data.shape[1]
        n_vect = torch.arange(0,N,1)
        phi = torch.tensor([torch.exp(1j*2*torch.pi*n*self.w_delta/self.Fs) for n in n_vect]).to(device)
        phi_matrix = torch.diag(phi).to(device)
        output_data = torch.matmul(input_data,phi_matrix).to(device)
        return output_data

