import torch 
import matplotlib
from OpticalChain import  simulate_chain_get_data
import pandas as pd 
import numpy as np
import json
from create_database.postgresql_operations import import_data_to_postgresql

matplotlib.get_backend()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == "__main__":
    # parameters = {'Nb' : 1000 , 'type' : 'QAM', 'M' : 16,
    #               'ovs_factor' : 2, 'fiber_length' : 4000,
    #               'Fs' : 21.4e9, 'wavelength' : 1553e-9,
    #               'SNR' : 20,
    #               'plot' : True
    #               }
    
    parameters_2 = {'Nb' : 1000 , 'type' : 'QAM', 'M' : 64,
                  'ovs_factor' : 2, 'fiber_length' : 1000,
                  'Fs' : 80e9, 'wavelength' : 1553e-9,
                  'SNR' : 20,
                  'plot' : True
                  }
    snr_range = [0, 5, 10, 15, 20]
    dataset = []
    for snr in snr_range:
        parameters = {
                'Nb' : 1000 , 'type' : 'QAM', 'M' : 16,
              'ovs_factor' : 2, 'fiber_length' : 4000,
              'Fs' : 21.4e9, 'wavelength' : 1553e-9,
              'SNR' : snr,
              'plot' : False}
        for _ in range(10):
            input_data, input_net, targets= simulate_chain_get_data(parameters)
            input_net_real = np.real(input_net)
            input_net_imag = np.imag(input_net)
            targets_real = np.real(targets)
            targets_imag = np.imag(targets)
            dataset.append({'M' : parameters['M'], "snr" : snr,
                            "input_data" : input_data.flatten().tolist(), 
                            "input_net_real" : input_net_real.flatten().tolist(),
                            "input_net_imag" : input_net_imag.flatten().tolist(), 
                            "targets_real" : targets_real.flatten().tolist(),
                            "targets_imag" : targets_imag.flatten().tolist()})
    database_config = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'OpticalData',
    'user': '',
    'password': ''
    }
    import_data_to_postgresql(table_name="QAM16_CD_Optimized_Filter", data=dataset, database_config=database_config)
