from OpticalChain import  simulate_chain_get_data
import numpy as np
from .postgresql_operations import import_data_to_postgresql


def create_dataset(snr_range, parameters, no_trials, database_config):
    """Generates data in the optical chain and loads it in the database"""
    dataset = []
    for snr in snr_range:
        parameters['SNR'] = snr
        for _ in range(no_trials):
            input_data, input_net, targets, output_data = simulate_chain_get_data(parameters = parameters)
            input_net_real = np.real(input_net.numpy())
            input_net_imag = np.imag(input_net.numpy())
            targets_real = np.real(targets.numpy())
            targets_imag = np.imag(targets.numpy())
            dataset.append({'M' : parameters['M'], "snr" : snr,
                            "input_data" : input_data.flatten().tolist(), 
                            "input_net_real" : input_net_real.flatten().tolist(),
                            "input_net_imag" : input_net_imag.flatten().tolist(), 
                            "targets_real" : targets_real.flatten().tolist(),
                            "targets_imag" : targets_imag.flatten().tolist()})
    import_data_to_postgresql(table_name=f"QAM{parameters['M']}_CD_Optimized_Filter", data=dataset, database_config=database_config)


def count_unique_numbers(list_of_lists):
    # Flatten the list of lists into a single list
    flat_list = [num for sublist in list_of_lists for num in sublist]
    unique_numbers = set(flat_list)
    unique_count = len(unique_numbers)
    return unique_numbers, unique_count
    


        	
