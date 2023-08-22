from chain_layers.Processors import Processor

class Transient_remover(Processor):
    def __init__(self,N, name = 'Transient_remover'):
        super(Transient_remover,self).__init__()
        self.N = int(N)
        self.name = name

    def forward(self,input_data):
        if input_data.ndim == 2:
            out = input_data[:,int(self.N/2):-int(self.N/2 )]
        elif input_data.ndim == 1:
            out = input_data[int(self.N/2):-int(self.N/2 )]
        return out




