import torch 
from chain_layers.Processors import Processor

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def apply_filter(data, h_coeffs, bias = None):
    """Convolutes a filter with the input data and provides the filtered signal"""
    padding = int(len(h_coeffs) - 1)
    h_coeffs = torch.reshape(h_coeffs, (1, 1, len(h_coeffs))).to(device)
    in_data = torch.flatten(data)
    x_r, x_i = torch.real(in_data).to(device), torch.imag(in_data).to(device)
    x_r = torch.reshape(x_r, (1,  1, len(x_r)))
    x_i = torch.reshape(x_i, (1,  1, len(x_i)))
    if torch.is_complex(h_coeffs):
        w_r, w_i = torch.real(h_coeffs), torch.imag(h_coeffs)
        b_r, b_i = (None, None) if bias is None else (bias.real, bias.imag)
        y_rr = torch.nn.functional.conv1d(x_r, w_r, b_r, padding=padding)
        y_ir = torch.nn.functional.conv1d(x_i, w_r, b_r, padding=padding)
        y_ri = torch.nn.functional.conv1d(x_r, w_i, b_i, padding=padding)
        y_ii = torch.nn.functional.conv1d(x_i, w_i, b_i, padding=padding)
        y1 = (y_rr - y_ii).float()
        y2 = (y_ir + y_ri).float()
    else:
        y1 = torch.nn.functional.conv1d(x_r, h_coeffs, padding = padding)
        y2 = torch.nn.functional.conv1d(x_i, h_coeffs, padding= padding)
    out = y1 + 1j * y2
    out = torch.squeeze(out, dim = 1)
    return out

class FilterComposed(Processor):

    def __init__(self, impulse_response1, impulse_response2):
        Processor.__init__(self)
        self.impulse_response1 = impulse_response1
        self.impulse_response2 = impulse_response2
        
    def conv_himp(self):
        himp = apply_filter(self.impulse_response1, self.impulse_response2)
        # df = pd.DataFrame(himp)
        # df.to_csv('savory_2.csv', index=False)
        himp = torch.flatten(himp)
        return himp.to(device)

    def forward(self, input_data):
        if self.impulse_response2 == None:
            output_data = apply_filter(input_data, self.impulse_response1)
        else:
            h_coeffs = self.conv_himp()
            output_data = apply_filter(input_data,  h_coeffs)
        return output_data.to(device)

class Filter(Processor):

    def __init__(self, impulse_response, name=None):
        Processor.__init__(self)
        self.impulse_response = impulse_response
        self.name = name

    def forward(self, input_data):
        output_data = apply_filter(input_data, self.impulse_response)
        output_data = torch.squeeze(output_data, dim=1)
        return output_data
