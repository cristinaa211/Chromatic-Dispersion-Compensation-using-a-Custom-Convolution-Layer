import pytorch_lightning as pl
import torch 
from chain_layers.Processors import Processor
import torch.nn as nn 
import pandas as pd 
import torchmetrics 

device = 'cuda' if torch.cuda.is_available() else "cpu"

class OptimizedFilterModel(pl.LightningModule):
    def __init__(self, lr,  len_cut = 1000) -> None:
        super(OptimizedFilterModel, self).__init__()
        self.conv_layer = ParametricConvLayer()
        self.downsampler = DownsamplerRemove( len = len_cut)
        self.lossfunction = nn.NLLLoss()
        self.loss_vect = []
        self.lr = lr
        self.train_acc_tensor, self.val_acc_tensor, self.test_acc_tensor = [], [], []
        self.accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes = 16)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = torch.unsqueeze(x, dim = 1)
        out = self.forward(x)
        loss_real = self.lossfunction(torch.real(out), torch.real(y))
        loss_imag = self.lossfunction(torch.imag(out), torch.imag(y))
        self.log("train_loss_epoch_real",  loss_real, on_epoch=True, prog_bar=True)
        self.log("train_loss_epoch_imag",  loss_imag, on_epoch=True, prog_bar=True)
        loss = torch.abs(torch.complex(loss_real, loss_imag))
        self.log("train_loss", loss, on_epoch=True)
        acc_real = self.accuracy_metric(torch.real(out), torch.real(y))
        acc_imag = self.accuracy_metric(torch.imag(out), torch.imag(y))
        self.log("train_acc_epoch_real",  acc_real, on_epoch=True, prog_bar=True)
        self.log("train_acc_epoch_imag",  acc_imag, on_epoch=True, prog_bar=True)
        self.train_acc_tensor.append(torch.abs(torch.complex(acc_real, acc_imag)))
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = torch.unsqueeze(x, dim = 1)
        out = self.forward(x)
        loss_real = self.lossfunction(torch.real(out), torch.real(y))
        loss_imag = self.lossfunction(torch.imag(out), torch.imag(y))
        loss = torch.abs(torch.complex(loss_real, loss_imag))
        self.log("val_loss", loss, on_epoch=True)
        acc_real = self.accuracy_metric(torch.real(out), torch.real(y))
        acc_imag = self.accuracy_metric(torch.imag(out), torch.imag(y))
        self.log("val_acc_epoch_real",  acc_real, on_epoch=True, prog_bar=True)
        self.log("val_acc_epoch_imag",  acc_imag, on_epoch=True, prog_bar=True)
        abs_val = torch.abs(torch.complex(acc_real, acc_imag))
        self.val_acc_tensor.append(abs_val)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = torch.unsqueeze(x, dim = 1)
        out = self.forward(x)
        loss_real = self.lossfunction(torch.real(out), torch.real(y))
        loss_imag = self.lossfunction(torch.imag(out), torch.imag(y))
        loss = torch.abs(torch.complex(loss_real, loss_imag))
        acc_real = self.accuracy_metric(torch.real(out), torch.real(y))
        acc_imag = self.accuracy_metric(torch.imag(out), torch.imag(y))
        self.test_acc_tensor.append(torch.abs(torch.complex(acc_real, acc_imag)))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        return optimizer
    
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.downsampler(x)
        return x


class DownsamplerRemove(pl.LightningModule):

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
    

class ParametricConvLayer( nn.Module, Processor):

    def __init__(self, parameter_version = 1, name = "Optimized Filter"):
        super(ParametricConvLayer, self).__init__()
        self.version = parameter_version
        self.param = self.load_parameters()
        self.parameter = nn.Parameter(self.param, requires_grad = True)

    def load_parameters(self):
        """Loads the parameters of the optimized filter"""
        filename = rf"./models/optimized_filter_{self.version}.csv"
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


    def forward(self, input_data):
        """Convolutes the input data with the parameters of the filter"""
        output_data = self.conv(input_data, self.parameter)
        return output_data
