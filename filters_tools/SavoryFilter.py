import torch 
from chain_layers.Processors import Processor
import pytorch_lightning as pl
import torch.nn as nn 
import pandas as pd 
from filters_tools.Filters import apply_filter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ParametricConvLayer(pl.LightningDataModule, nn.Module,Processor):

    def __init__(self, parameter_version, name = "Optimized Filter"):
        super(ParametricConvLayer, self).__init__()
        self.version = parameter_version
        self.param = self.load_savory_parameters()
        self.parameter = nn.Parameter(self.param, requires_grad = True)


    def load_savory_parameters(self):
        df = pd.read_csv(f"./filters_tools/optimized_filter_{self.version}.csv")
        params = torch.flatten(torch.tensor(df.to_numpy(dtype=complex)))
        return params

    def forward(self, input_data):
        output_data = apply_filter(input_data, self.parameter)
        return output_data


