from chain_layers import Processors, Generator, Modulators, Sampling, Display_tools
from filters_tools import Filters, Transient_remover
from impairements import CD_impairement, IQ_imbalance, Noise, Phase_noise, CarrierFreqOffset
from filters_tools.filter_operations import get_impulse_response, get_trunc
import torch 
from models.model_layers import ParametricConvLayer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class OpticalChain:
    def __init__(self, Nb, M, mod_type, ovs_factor, wavelength, Fs, SNR, fiber_length, plot):
        self.Nb = Nb 
        self.M = M
        self.mod_type = mod_type
        self.ovs_factor = ovs_factor
        self.wavelength = wavelength
        self.Fs = Fs
        self.SNR = SNR
        self.fiber_length = fiber_length
        self.plot = plot
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
        self.optical_chain.add_processor(Processors.Recorder(name="targets"))  
        self.optical_chain.add_processor(Sampling.Upsampler_zero(os=self.ovs_factor))  

    def receiver_side(self):
        """Constructs the receiver side of the optical chain, which contains:
        a Downsampler by a downsampling factor of ovs_factor
        a Recorder for monitoring the output symbols
        a Scope for displaying the received constellations 
        a Demodulator for demodulating the received symbols, from the same M-QAM constellation
        a Recorder for monitoring the received data
        """
        self.optical_chain.add_processor(Sampling.Downsampler_zero(os=self.ovs_factor)) 
        self.optical_chain.add_processor(Processors.Recorder(name='symbols_out')) 
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
        impulse_response = get_impulse_response(choice = filter_choice)[0]
        self.optical_chain.add_processor(Filters.Filter(impulse_response, name = name))

    def add_savory_filter(self):
        """Composes a filter based on the Savory paper"""
        fir_impulse_response = get_impulse_response('fir')[0]
        srrc_impulse_response = get_impulse_response('srrc')[0]
        self.optical_chain.add_processor(Filters.FilterComposed(fir_impulse_response, srrc_impulse_response))

    def add_parametric_layer(self, version):
        self.optical_chain.add_processor(ParametricConvLayer(version))
        

    def add_iq_imbalance(self, alpha_db, theta_deg):
        """Adds IQ imbalance in the optical chain
        Args:
            alpha_db  : IQ imbalance amplitude mismatch coefficient
            theta_deg : IQ imbalance phase mismatch coefficient 
            """
        self.optical_chain.add_processor(IQ_imbalance.IQ_imbalance(alpha_db, theta_deg))

    def add_carrier_frequency_offset(self, delta = 2e9):
        """Adds Carrier Frequency Offset Impairement
        Args:
            delta (int): Carrier Frequency Offset impact, default to 2e9
        """
        self.optical_chain.add_processor(CarrierFreqOffset.CarrierFrequencyOffset(delta=delta, Fs= self.Fs))

    def add_phase_noise(self, df = 1e5, name = "Phase Noise" ):
        """Adds Laser Phase Noise in the optical chain
        Args:
            df : Laser phase noise - laser linewidth"""
        self.optical_chain.add_processor(Phase_noise.LaserPhaseNoise(fs = self.Fs, delta_f = df, name = name))

    def add_chromatic_dispersion(self, D):
        """Adds Chromatic Dispersion
        Args:
            D  : fiber's Chromatic dispersion parameter, default = 17e-3"""
        self.optical_chain.add_processor(CD_impairement.ChromaticDispersion(D, self.fiber_length , self.wavelength, self.Fs, name = "ChromaticDispersion" ))

    def add_transient_remover(self, choice):
        N = get_trunc(choice)
        self.optical_chain.add_processor(Transient_remover.Transient_remover(N))


    def get_input_output(self):
        input_data = getattr(self.optical_chain.input, 'data').cpu().numpy()
        input_net =  getattr(self.optical_chain.input_net, 'data').cpu().numpy()
        targets = getattr(self.optical_chain.targets, 'data').cpu().numpy()
        return input_data, input_net, targets
    
    def simulate_chain(self):
        #EMITTER SIDE
        #--------------------------------------------------
        self.emitter_side()
        self.add_filters('srrc', name = 'SSRC Filter Tx')
        # self.add_phase_noise(name = "Phase Noise Tx")
        # self.add_iq_imbalance(alpha_db=2, theta_deg=20)
        #CHANNEL
        #--------------------------------------------------
        self.add_chromatic_dispersion(D = 17e-3)
        self.add_noise()
        # self.add_carrier_frequency_offset()
        #RECEIVER SIDE
        #--------------------------------------------------
        # self.add_savory_filter()
        self.optical_chain.add_processor(Processors.Recorder(name="input_net"))
        self.add_parametric_layer(version=1)
        # self.add_phase_noise(name="Phase Noise Rx")
        # self.add_iq_imbalance(alpha_db=1, theta_deg=10)
        # self.add_filters('srrc', name='SSRC Filter Rx')
        self.add_transient_remover("param")
        self.receiver_side()
        self.optical_chain.forward().to(device)


def simulate_chain_get_data(parameters):
    chain = OpticalChain(
                        Nb = parameters['Nb'],
                        mod_type=parameters['type'], M = parameters['M'],
                        fiber_length=parameters['fiber_length'], ovs_factor=parameters['ovs_factor'], 
                        SNR=parameters['SNR'], wavelength= parameters['wavelength'], 
                        Fs=parameters['Fs'], plot=parameters['plot'] )
    chain.simulate_chain()
    input_data, input_net, targets = chain.get_input_output()
    return  input_data, input_net, targets  
