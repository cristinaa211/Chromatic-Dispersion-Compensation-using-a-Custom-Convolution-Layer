import torch 
import matplotlib
from OpticalChain import  simulate_chain_get_data


matplotlib.get_backend()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


        


if __name__ == "__main__":
    parameters = {'Nb' : 1000 , 'type' : 'QAM', 'M' : 16,
                  'ovs_factor' : 2, 'fiber_length' : 4000,
                  'Fs' : 21.4e9, 'wavelength' : 1553e-9,
                  'SNR' : 20,
                  'plot' : True
                  }
    
    parameters_2 = {'Nb' : 1000 , 'type' : 'QAM', 'M' : 64,
                  'ovs_factor' : 2, 'fiber_length' : 1000,
                  'Fs' : 80e9, 'wavelength' : 1553e-9,
                  'SNR' : 20,
                  'plot' : True
                  }
    inp, out, symb_in, symb_ou = simulate_chain_get_data(parameters)