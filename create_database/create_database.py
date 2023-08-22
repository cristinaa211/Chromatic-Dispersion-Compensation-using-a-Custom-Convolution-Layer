from OpticalChain import  simulate_chain_get_data
import numpy as np
from postgresql_operations import import_data_to_postgresql


def create_database(snr_range, parameters, no_trials, database_config):
    """Generates data in the optical chain and loads it in the database"""
    dataset = []
    for snr in snr_range:
        parameters['SNR'] = snr
        for _ in range(no_trials):
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
    import_data_to_postgresql(table_name=f"QAM{parameters['M']}_CD_Optimized_Filter", data=dataset, database_config=database_config)
