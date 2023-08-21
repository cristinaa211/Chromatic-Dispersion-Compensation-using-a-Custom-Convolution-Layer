import torch 
from chain_layers.Processors import Processor

class Filter(Processor):

    def __init__(self, impulse_response, name=None):
        Processor.__init__(self)
        self.impulse_response = impulse_response
        self.N = len(self.impulse_response)
        self.name = name

    def apply_filter(self, data, h_imp, bias=None):
        h = torch.reshape(h_imp, (1, 1, len(h_imp)))
        N = self.N
        in_data = torch.flatten(data)
        in_data = torch.reshape(in_data, (1, 1, len(in_data)))
        x_r, x_i = torch.real(in_data), torch.imag(in_data)
        if torch.is_complex(h):
            w_r, w_i = torch.real(h), torch.imag(h)
            b_r, b_i = (None, None) if bias is None else (bias.real, bias.imag)
            M = int((N - 1))
            # padding = M 
            y_rr = torch.nn.functional.conv1d(x_r, w_r, b_r)
            y_ir = torch.nn.functional.conv1d(x_i, w_r, b_r)
            y_ri = torch.nn.functional.conv1d(x_r, w_i, b_i)
            y_ii = torch.nn.functional.conv1d(x_i, w_i, b_i)
            y1 = (y_rr - y_ii).float()
            y2 = (y_ir + y_ri).float()
            out = out = y1 + 1j * y2
        else:
            # padding = N-1
            y1 = torch.nn.functional.conv1d(x_r, h)
            y2 = torch.nn.functional.conv1d(x_i, h)
            out = y1 + 1j * y2
        return out

    def forward(self, input_data):
        output_data = self.apply_filter(input_data, self.impulse_response)
        output_data = torch.squeeze(output_data, dim=1)
        return output_data
