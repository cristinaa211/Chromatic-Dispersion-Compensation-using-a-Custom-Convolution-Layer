import torch 
import numpy as np
import matplotlib
matplotlib.get_backend()
from chain_layers import Processors, Generator, Modulators, Sampling, Display_tools, Noise, Filters, Transient_remover
from chain_layers import IQ_imbalance , Phase_noise
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
        np.random.choice(42)
        self.optical_chain = Processors.Chain()
        
    def emitter_side(self):
        """Constructs the emitter side of the optical chain which contains:
        a Symbol Generator, which generates random bits
        a Recorder for monitoring the generated data
        a Modulator which will modulate the data in a M-QAM constellation 
        a Recorder for monitoring the modulated symbols
        a Upsampler which will upsample the data with an oversampling factor of ovs_factor
        """
        self.optical_chain.add_processor(Generator.Symbols_Generator(M=self.M, N_row=1, N_col=self.Nb, name='generator'))
        self.optical_chain.add_processor(Processors.Recorder(name="input"))
        self.optical_chain.add_processor(Modulators.Modulator(M=self.M, normalized=True)) 
        self.optical_chain.add_processor(Processors.Recorder(name="symbols_in"))  # Record input symbols
        self.optical_chain.add_processor(Sampling.Upsampler_zero(os=self.ovs_factor))  # Oversampling

    def receiver_side(self):
        """Constructs the receiver side of the optical chain, which contains:
        a Downsampler by a downsampling factor of ovs_factor
        a Recorder for monitoring the output symbols
        a Scope for displaying the received constellations 
        a Demodulator for demodulating the received symbols, from the same M-QAM constellation
        a Recorder for monitoring the received data
        """
        self.optical_chain.add_processor(Sampling.Downsampler_zero(os=self.ovs_factor)) #Downsampling
        self.optical_chain.add_processor(Processors.Recorder(name='symbols_out')) # Record output symbols
        if self.plot == True:
            self.optical_chain.add_processor(Display_tools.Scope(type='scatter'))
        self.optical_chain.add_processor(Modulators.Demodulator(M=self.M,  normalized=True))
        self.optical_chain.add_processor(Processors.Recorder(name="output"))
        
    def add_noise(self):
        """Adds Gaussian Noise"""
        self.optical_chain.add_processor(Noise.AWGN(M = self.M, SNR_db=self.SNR,os = self.ovs_factor))
    
    def add_filters(self, filter_choice, name):
        """Adds a filter in the optical chain
        filter_choise = 'srrc' means it adds a Squared Raised Root Cosine filter"""
        N = get_trunc(filter_choice)
        impulse_response = get_impulse_response(choice = filter_choice)[0]
        self.optical_chain.add_processor(Filters.Filter(impulse_response, name = name))
        self.optical_chain.add_processor(Transient_remover.Transient_remover(N))

    def add_savory_filter(self):
        """Composes a filter based on the Savory paper"""
        fir_impulse_response = get_impulse_response('fir')[0]
        srrc_impulse_response = get_impulse_response('srrc')[0]
        N = get_trunc('fir+srrc')
        self.optical_chain.add_processor(Filters.Filter_composed(fir_impulse_response, srrc_impulse_response))
        # self.optical_chain.add_processor(Transient_remover.Transient_remover(N))

    def add_iq_imbalance(self, alpha_db, theta_deg):
        """Adds IQ imbalance in the optical chain"""
        self.optical_chain.add_processor(IQ_imbalance.IQ_imbalance(alpha_db, theta_deg))

    def add_phase_noise(self, df = 1e5, name = "Phase Noise" ):
        """Adds Laser Phase Noise in the optical chain"""
        self.optical_chain.add_processor(Phase_noise.LaserPhaseNoise(fs = self.Fs, delta_f = df, name = name))

    def forward(self):
        self.emitter_side()
        self.add_filters('srrc', name = 'SSRC Filter Tx')
        self.add_phase_noise(name = "Phase Noise Tx")
        self.add_iq_imbalance(alpha_db=2, theta_deg=20)
        self.add_noise()
        self.add_phase_noise(name="Phase Noise Rx")
        self.add_iq_imbalance(alpha_db=1, theta_deg=10)
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
    chain.forward()