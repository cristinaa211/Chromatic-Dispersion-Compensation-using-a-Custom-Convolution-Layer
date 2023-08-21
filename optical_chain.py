import torch 
import numpy as np
import matplotlib
matplotlib.get_backend()
from chain_layers import Processors, Generator, Modulators, Sampling, Display_tools, Noise, Filters, Transient_remover
from filter_operations import get_impulse_response, get_trunc

device = 'cuda' if torch.cuda.is_available() else 'cpu'




class OpticalChain:
    def __init__(self, Nb, M, mod_type, BW, ovs_factor, wavelength, Fs, SNR, plot):
        self.Nb = Nb 
        self.M = M
        self.mod_type = mod_type
        self.BW = BW
        self.ovs_factor = ovs_factor
        self.wavelength = wavelength
        self.Fs = Fs
        self.SNR = SNR
        self.plot = plot
        seed = np.random.choice(42)
        self.optical_chain = Processors.Chain()
        
    def emitter_side(self):
        self.optical_chain.add_processor(Generator.Symbols_Generator(M=self.M, N_row=1, N_col=self.Nb, name='generator'))
        self.optical_chain.add_processor(Processors.Recorder(name="input"))
        self.optical_chain.add_processor(Modulators.Modulator(M=self.M, normalized=True)) 
        self.optical_chain.add_processor(Processors.Recorder(name="symbols_in"))  # Record input symbols
        self.optical_chain.add_processor(Sampling.Upsampler_zero(os=self.ovs_factor))  # Oversampling

    def receiver_side(self):
        self.optical_chain.add_processor(Sampling.Downsampler_zero(os=self.ovs_factor)) #Downsampling
        self.optical_chain.add_processor(Processors.Recorder(name='symbols_out')) # Record output symbols
        if self.plot == True:
            self.optical_chain.add_processor(Display_tools.Scope(type='scatter'))
        self.optical_chain.add_processor(Modulators.Demodulator(M=self.M,  normalized=True))
        self.optical_chain.add_processor(Processors.Recorder(name="output"))
        
    def add_noise(self):
        self.optical_chain.add_processor(Noise.AWGN(M = self.M, SNR_db=self.SNR,os = self.ovs_factor))
    
    def add_filters(self, filter_choice, name):
        impulse_response = get_impulse_response(choice = filter_choice)[0]
        truncate_length =  get_trunc(filter_choice)
        self.optical_chain.add_processor(Filters.Filter(impulse_response, name = name))
        # self.optical_chain.add_processor(Transient_remover.Transient_remover(N = truncate_length, name='Transient_remover'))

    def forward(self, filter_choice):
        self.emitter_side()
        self.add_filters('srrc', name = 'SSRC Filter Tx')
        self.add_noise()
        if filter_choice:
            self.add_filters(filter_choice, name = 'Rx')
        self.add_filters('srrc', name='SSRC Filter Rx')
        self.receiver_side()
        self.optical_chain.forward()
        


if __name__ == "__main__":
    parameters = {'Nb' : 600 , 'type' : 'QAM', 'M' : 16,
                  'BW' : 10e9, 'ovs_factor' : 2,
                  'Fs' : 21.4e9, 'wavelength' : 1553e-9,
                  'SNR' : 20,
                  'plot' : True
                  }
    chain = OpticalChain(
                        Nb = parameters['Nb'], mod_type=parameters['type'], M = parameters['M'],
                        BW = parameters['BW'], ovs_factor=parameters['ovs_factor'],
                        wavelength= parameters['wavelength'], Fs=parameters['Fs'], SNR=parameters['SNR'],
                        plot=parameters['plot'] )
    chain.forward('fir')