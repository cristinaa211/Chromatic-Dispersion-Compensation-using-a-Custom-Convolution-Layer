import torch 
from chain_layers.Processors import Processor
import pytorch_lightning as pl
import torch.nn as nn 
import pandas as pd 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ParametricConvLayer(pl.LightningDataModule, nn.Module,Processor):

    def __init__(self, parameter_version):
        super(ParametricConvLayer, self).__init__()
        self.version = parameter_version
        self.param = self.load_savory_parameters()
        self.parameter = nn.Parameter(self.param, requires_grad = True)
        self.M = int(len(self.parameter) - 1)

    def load_savory_parameters(self):
        df = pd.read_csv(f"./filters_tools/savory_parameters{self.version}.csv")
        params = torch.flatten(torch.tensor(df.to_numpy(dtype=complex)))
        return params

    def apply_filter(self, data, h_imp, bias=None):
        data = torch.reshape(data, (1, 1, len(data)))
        h = torch.reshape(h_imp, (1,1, len(h_imp))).to(device)
        x_r, x_i = torch.real(data).float(), torch.imag(data).float()
        w_r, w_i = torch.real(h).float(), torch.imag(h).float()
        b_r, b_i = (None, None) if bias is None else (bias.real, bias.imag)
        y_rr = torch.nn.functional.conv1d(x_r, w_r, b_r, padding=self.M)
        y_ir = torch.nn.functional.conv1d(x_i, w_r, b_r, padding=self.M)
        y_ri = torch.nn.functional.conv1d(x_r, w_i, b_i, padding=self.M)
        y_ii = torch.nn.functional.conv1d(x_i, w_i, b_i, padding=self.M)
        y1 = (y_rr - y_ii).float()
        y2 = (y_ir + y_ri).float()
        out = y1 + 1j * y2
        return out

    def forward(self, input_data):
        output_data = self.apply_filter(torch.flatten(input_data), self.parameter)
        output_data = torch.squeeze(output_data, dim=0 )
        return output_data
