
import pytorch_lightning as pl
import torch 
from chain_layers.Processors import Processor
import torch.nn as nn 
import pandas as pd 

device = 'cuda' if torch.cuda.is_available() else "cpu"

def ComplexLoss(lossfunction, out, y):
    out_= torch.stack((torch.real(out), torch.imag(out)), dim=0)
    y_ = torch.stack((torch.real(y), torch.imag(y)), dim=0)
    loss = lossfunction(out_, y_)
    return loss


class OptimizedFilterModel(pl.LightningModule, Processor):
    def __init__(self, lr = 1e-5,  len_cut = 1000, name = 'TrainedModel') -> None:
        super(OptimizedFilterModel, self).__init__()
        self.lr = lr
        self.name = name
        self.conv_layer = ParametricConvLayer()
        self.downsampler = DownsamplerRemove( len = len_cut)
        # self.detection_layer = SymbolDetection()
        self.lossfunction = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.downsampler(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = torch.unsqueeze(x, dim = 1)
        out = self.forward(x)
        out_= torch.stack((torch.real(out), torch.imag(out)), dim=0)
        y_ = torch.stack((torch.real(y), torch.imag(y)), dim=0)
        loss = self.lossfunction(out_, y_)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = torch.unsqueeze(x, dim = 1)
        out = self.forward(x)
        out_= torch.stack((torch.real(out), torch.imag(out)), dim=0)
        y_ = torch.stack((torch.real(y), torch.imag(y)), dim=0)
        loss = self.lossfunction(out_, y_)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = torch.unsqueeze(x, dim = 1)
        out = self.forward(x)
        out_= torch.stack((torch.real(out), torch.imag(out)), dim=0)
        y_ = torch.stack((torch.real(y), torch.imag(y)), dim=0)
        loss = self.lossfunction(out_, y_)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.conv_layer.parameters(), lr = self.lr)
        return optimizer

class DownsamplerRemove(pl.LightningModule):
    """Downsamples by a downsampling factor "os" and removes the filter delays in the data to a length of "len" """

    def __init__(self,  len = '', os=2, name="Downsampler"):
        super(DownsamplerRemove,self).__init__()
        self.os = os
        self.len = len
        self.name = name

    def remover(self, input_data):
        """Removes the filter's delays from the input data"""
        input_data = torch.flatten(input_data)
        if self.len != len(input_data):
            l = int(abs(2*self.len - len(input_data)) / 2)
            output_data = input_data[l:-l]
        return output_data

    def downsampler(self, input_data):
        """Downsamples the input data by a factor of self.os"""
        return input_data[::self.os]

    def forward(self, input_data):
        output_data = []
        for sample in input_data:
            sample = self.remover(sample)
            sample = self.downsampler(sample)
            output_data.append(sample)
        return torch.stack(output_data, axis=0)
    

class ParametricConvLayer(pl.LightningModule, nn.Module, Processor):

    def __init__(self, filename = None, original = True, parameter_version = 1, name = "Optimized Filter"):
        super(ParametricConvLayer, self).__init__()
        self.version = parameter_version
        self.filename = filename
        self.original  = original
        self.name = name
        self.param = self.load_parameters().to(device)
        self.parameter = nn.Parameter(self.param, requires_grad = True, )

    def load_parameters(self):
        """Loads the parameters of the optimized filter"""
        if self.original == True:
            filename = rf"./models/csv/optimized_filter_{self.version}.csv"
        else:
            filename = self.filename
        df = pd.read_csv(filename)
        params = torch.flatten(torch.tensor(df.to_numpy(dtype=complex)))
        return params
    
    def conv(self, in_data, h_coeffs, bias = None):
        """Convolutes a filter with the input data and provides the filtered signal"""
        padding = int(len(h_coeffs) - 1)
        if in_data.dim == 2:
            in_data = torch.unsqueeze(in_data, dim=1)
        h_coeffs = torch.reshape(h_coeffs, (1, 1, len(h_coeffs))).to(device)
        x_r, x_i = torch.real(in_data).to(device), torch.imag(in_data).to(device)
        if torch.is_complex(h_coeffs):
            w_r, w_i = torch.real(h_coeffs).type_as(x_r), torch.imag(h_coeffs).type_as(x_i)
            b_r, b_i = (None, None) if bias is None else (bias.real, bias.imag)
            y_rr = torch.nn.functional.conv1d(x_r, w_r, b_r, padding=padding)
            y_ir = torch.nn.functional.conv1d(x_i, w_r, b_r, padding=padding)
            y_ri = torch.nn.functional.conv1d(x_r, w_i, b_i, padding=padding)
            y_ii = torch.nn.functional.conv1d(x_i, w_i, b_i, padding=padding)
            y1 = (y_rr - y_ii).float()
            y2 = (y_ir + y_ri).float()
        else:
            h_coeffs = h_coeffs.type_as(x_r)
            y1 = torch.nn.functional.conv1d(x_r, h_coeffs, padding = padding)
            y2 = torch.nn.functional.conv1d(x_i, h_coeffs, padding= padding)
        out = y1 + 1j * y2
        return out


    def forward(self, x):
        """Convolutes the input data with the parameters of the filter"""
        output_data = self.conv(x, self.parameter)
        return output_data

