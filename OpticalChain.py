from chain_layers import Processors, Generator, Modulators, Sampling, Display_tools
from filters_tools import Filters, Transient_remover
from impairements import CD_impairement, IQ_imbalance, Noise, Phase_noise, CarrierFreqOffset
from filters_tools.filter_operations import get_impulse_response, get_trunc
import torch 
from models.model_layers import ParametricConvLayer, OptimizedFilterModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class OpticalChain:
    def __init__(self,order = ['cd', 'srrc'], Nb = 1000, mod_type = 'QAM', M = 16, SNR = 20, ovs_factor = 2, wavelength=1553e-9, Fs = 21.4e9, fiber_length = 4000, plot= False):
        """
        Represents an Optical Chain
        Args:
            order           (list) : the layers present in the optical chain, default ['cd', 'srrc']
            Nb               (int) : the length of the input generated data , default 1000
            mod_type      (string) : the type of the modulation, default 'QAM'
            M                (int) : the order of the modulation scheme (M distinct symbols in the constellation diagram), default 16
            ovs_factor       (int) : the oversampling factor, default 2
            fiber_length     (int) : the length of the fiber, default 4000
            Fs             (float) : the sampling rate, default 21.4e9
            wavelength     (float) : the wavelength of the signal, default 1553e-9
            SNR              (int) : the signal to noise ratio expressed in dBm , default 20
            plot         (boolean) : if True, then display the received constellation, default False
        """
        self.order = order
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
        """
        Constructs the emitter side of the optical chain which contains:
        a Symbol Generator, which generates random bits
        a Recorder for monitoring the generated data
        a Modulator which will modulate the data in a M-QAM constellation 
        a Recorder for monitoring the modulated symbols
        a Upsampler which will upsample the data with an oversampling factor of ovs_factor
        """
        self.optical_chain.add_processor(Generator.Symbols_Generator(M=self.M, N_row=1, N_col=self.Nb, name='generator'))
        self.optical_chain.add_processor(Processors.Recorder(name="input_data"))
        self.optical_chain.add_processor(Modulators.Modulator(M=self.M, normalized=True)) 
        self.optical_chain.add_processor(Processors.Recorder(name="targets"))  
        self.optical_chain.add_processor(Sampling.Upsampler_zero(os=self.ovs_factor))  

    def receiver_side(self):
        """
        Constructs the receiver side of the optical chain, which contains:
        a Recorder for monitoring the output symbols
        a Scope for displaying the received constellations 
        a Demodulator for demodulating the received symbols, from the same M-QAM constellation
        a Recorder for monitoring the received data
        """
        self.optical_chain.add_processor(Processors.Recorder(name='symbols_out')) 
        if self.plot == True:
            self.optical_chain.add_processor(Display_tools.Scope(type='scatter'))
        self.optical_chain.add_processor(Modulators.Demodulator(M=self.M,  normalized=True))
        self.optical_chain.add_processor(Processors.Recorder(name="output_data"))
        
    def add_noise(self):
        """Adds Gaussian Noise"""
        self.optical_chain.add_processor(Noise.AWGN(M = self.M, SNR_db=self.SNR,os = self.ovs_factor))
    
    def add_filters(self, filter_choice, name):
        """Adds a filter in the optical chain
        filter_choise = 'srrc' means it adds a Squared Raised Root Cosine filter"""
        impulse_response = get_impulse_response(choice = filter_choice)[0]
        self.optical_chain.add_processor(Filters.Filter(impulse_response, name = name))

    def add_savory_filter(self):
        """Composes a filter which will be an optimized filter for CD compensation"""
        fir_impulse_response = get_impulse_response('fir')[0]
        srrc_impulse_response = get_impulse_response('srrc')[0]
        self.optical_chain.add_processor(Filters.FilterComposed(fir_impulse_response, srrc_impulse_response))

    def add_parametric_layer(self,filename = None, original = True, version = 1):
        """Adds a Parametric Convolutional Layer, which represents an optimized filter for chromatic dispersion compensation"""
        self.optical_chain.add_processor(ParametricConvLayer(filename=filename, original=original, parameter_version=version))
    
    def add_trained_model(self, modelpath):
        model = OptimizedFilterModel(name = 'TrainedModel')
        model.load_from_checkpoint(modelpath)
        model.eval()
        with torch.no_grad():
            self.optical_chain.add_processor(model)

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
        """Removes the fitler's delays from the received data"""
        N = get_trunc(choice)
        self.optical_chain.add_processor(Transient_remover.Transient_remover(N))

    def get_input_output(self):
        """
        Returns:
        input_data         (list of floats) :  generated data, data before modulation
        input_bet  (list of complex floats) :  data after modulation 
        targets    (list of complex floats) :  data before the Parametric Layer 
        output_data        (list of floats) :  received data, data after demodulation 
        """
        input_data = getattr(self.optical_chain.input_data, 'data').cpu()
        input_net =  getattr(self.optical_chain.input_net, 'data').cpu()
        targets = getattr(self.optical_chain.targets, 'data').cpu()
        output_data = getattr(self.optical_chain.output_data, "data").cpu()
        return input_data, input_net, targets, output_data
    
    def simulate_chain(self, *args):
        #EMITTER SIDE
        #--------------------------------------------------
        self.emitter_side()
        self.add_filters('srrc', name = 'SSRC Filter Tx')
        if 'pn_tx' in self.order:
            self.add_phase_noise(name = "Phase Noise Tx")
        if 'iq_tx' in self.order:
            self.add_iq_imbalance(alpha_db=2, theta_deg=20)
        #CHANNEL
        #--------------------------------------------------
        if 'cd' in self.order : 
            self.add_chromatic_dispersion(D = 17e-3)
        self.add_noise()
        if 'cfo' in self.order : 
            self.add_carrier_frequency_offset()
        #RECEIVER SIDE
        #--------------------------------------------------
        if 'composed_filter' in self.order:
            self.add_savory_filter()
        self.optical_chain.add_processor(Processors.Recorder(name="input_net"))
        if 'eval' in self.order:
            self.add_trained_model(args[0])
        if "optimized_filter" in self.order:
            self.add_parametric_layer(version=1)
        if 'pn_rx' in self.order:
            self.add_phase_noise(name="Phase Noise Rx")
        if "iq_rx" in self.order:
            self.add_iq_imbalance(alpha_db=1, theta_deg=10)
        if 'srrc' in self.order:
            self.add_filters('srrc', name='SSRC Filter Rx')
        if "eval" not in self.order:
            self.add_transient_remover("param")
            self.optical_chain.add_processor(Sampling.Downsampler_zero(os=self.ovs_factor)) 
        self.receiver_side()
        self.optical_chain.forward().to(device)


def simulate_chain_get_data(parameters, modelpath = None):
    """Simulates the data transmission in the optical chain.
    Args:
        parameters (dict) : the parameters of the Optical Chain
                            Example:  parameters = {'oder': ['cd', 'optimized_filter'], 'Nb' : 1000 , 'type' : 'QAM', 'M' : 16, 'ovs_factor' : 2, 'fiber_length' : 4000,
                                                    'Fs' : 21.4e9, 'wavelength' : 1553e-9, 'SNR' : 20, 'plot' : False }
                            Where:
                            order           (list) : the layers present in the optical chain
                            Nb               (int) : the length of the input generated data 
                            type          (string) : the type of the modulation 
                            M                (int) : the order of the modulation scheme (M distinct symbols in the constellation diagram)
                            ovs_factor       (int) : the oversampling factor
                            fiber_length     (int) : the length of the fiber
                            Fs             (float) : the sampling rate
                            wavelength     (float) : the wavelength of the signal
                            SNR              (int) : the signal to noise ratio expressed in dBm 
                            plot         (boolean) : if True, then display the received constellation
        modelpath   (path) : the path for the trained parameters of the optimized filter, default to None
    Returns:
        input_data         (list of floats) :  generated data, data before modulation
        input_bet  (list of complex floats) :  data after modulation 
        targets    (list of complex floats) :  data before the Parametric Layer 
        output_data        (list of floats) :  received data, data after demodulation 
    """
    chain = OpticalChain(order = parameters['order'], Nb = parameters['Nb'],
                        mod_type=parameters['type'], M = parameters['M'],
                        fiber_length=parameters['fiber_length'], ovs_factor=parameters['ovs_factor'], 
                        SNR=parameters['SNR'], wavelength= parameters['wavelength'], 
                        Fs=parameters['Fs'], plot=parameters['plot'] )
    chain.simulate_chain(modelpath)
    input_data, input_net, targets, output_data = chain.get_input_output()
    return  input_data, input_net, targets , output_data


