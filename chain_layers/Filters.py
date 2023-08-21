import torch 
from chain_layers.Processors import Processor



class Filter_composed(Processor):

    def __init__(self, impulse_response1, impulse_response2):
        self.impulse_response1 = impulse_response1
        self.impulse_response2 = impulse_response2
        Processor.__init__(self)

    def apply_filter(self, data, h_imp, bias=None):
        N = len(h_imp)
        h = torch.reshape(h_imp, (1, 1, len(h_imp)))
        in_data = torch.flatten(data)
        in_data = torch.reshape(in_data, (1, 1, len(in_data)))
        if torch.is_complex(data):
            x_r, x_i = torch.real(in_data), torch.imag(in_data)
        else:
            x_r = data
            x_i = torch.zeros(len(data))
        if torch.is_complex(h):
            w_r, w_i = torch.real(h), torch.imag(h)
            b_r, b_i = (None, None) if bias is None else (bias.real, bias.imag)
            M = int(N - 1)
            y_rr = torch.nn.functional.conv1d(x_r, w_r, b_r, padding=M)
            y_ir = torch.nn.functional.conv1d(x_i, w_r, b_r, padding=M)
            y_ri = torch.nn.functional.conv1d(x_r, w_i, b_i, padding=M)
            y_ii = torch.nn.functional.conv1d(x_i, w_i, b_i, padding=M)
            y1 = (y_rr - y_ii).float()
            y2 = (y_ir + y_ri).float()

        else:

            y1 = torch.nn.functional.conv1d(x_r, h, padding=N - 1)
            y2 = torch.nn.functional.conv1d(x_i, h, padding=N - 1)

        out = y1 + 1j * y2
        return out

    def conv_himp(self):
        himp = self.apply_filter(self.impulse_response1, self.impulse_response2)
        # df = pd.DataFrame(himp)
        # df.to_csv('savory_2.csv', index=False)
        himp = torch.flatten(himp)
        return himp

    def forward(self, input_data):
        if self.impulse_response2 == None:
            output_data = self.apply_filter(input_data, self.impulse_response1)
        else:
            output_data = self.apply_filter(input_data, self.conv_himp())
        output_data = torch.squeeze(output_data, dim=1)

        return output_data

class Filter(Processor):

    def __init__(self, impulse_response, name=None):
        Processor.__init__(self)
        self.impulse_response = impulse_response
        self.name = name

    def apply_filter(self, data, h_imp, bias=None):
        h = torch.reshape(h_imp, (1, 1, len(h_imp)))
        in_data = torch.reshape(torch.flatten(data), (1, 1, len(torch.flatten(data))))
        x_r, x_i = torch.real(in_data), torch.imag(in_data)
        padding = len(self.impulse_response)- 1
        if torch.is_complex(h):
            w_r, w_i = torch.real(h), torch.imag(h)
            b_r, b_i = (None, None) if bias is None else (bias.real, bias.imag)
            y_rr = torch.nn.functional.conv1d(x_r, w_r, b_r, padding=padding)
            y_ir = torch.nn.functional.conv1d(x_i, w_r, b_r, padding=padding)
            y_ri = torch.nn.functional.conv1d(x_r, w_i, b_i, padding=padding)
            y_ii = torch.nn.functional.conv1d(x_i, w_i, b_i, padding=padding)
            y1 = (y_rr - y_ii).float()
            y2 = (y_ir + y_ri).float()
            out = out = y1 + 1j * y2
        else:
            y1 = torch.nn.functional.conv1d(x_r, h, padding=padding)
            y2 = torch.nn.functional.conv1d(x_i, h, padding=padding)
            out = y1 + 1j * y2
        return out

    def forward(self, input_data):
        output_data = self.apply_filter(input_data, self.impulse_response)
        output_data = torch.squeeze(output_data, dim=1)
        return output_data
