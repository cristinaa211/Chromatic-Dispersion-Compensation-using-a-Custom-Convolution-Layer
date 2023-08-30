from OpticalChain import simulate_chain_get_data
from create_database.create_database import create_dataset


if __name__ == "__main__":
    parameters = {
		'order' : ['cd', 'optimized_filter'], 'Nb' : 1000 , 'type' : 'QAM', 'M' : 16,
		'ovs_factor' : 2, 'fiber_length' : 4000,
		'Fs' : 21.4e9, 'wavelength' : 1553e-9, 'SNR' : 20,
		'plot' : False }
    database_config = {
		        'host': 'localhost',
		        'port': 5432,
		        'dbname': 'OpticalData',
		        'user': '',
		        'password': ''
		        }
    snr_range = range(0, 21)
    create_dataset(snr_range, parameters, 500, database_config)
