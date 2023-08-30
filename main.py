import torch 
from models.model_tools import train_model_, evaluate_model, prepare_dataset, plot_results
device = 'cuda' if torch.cuda.is_available() else 'cpu'

parameters_2 = {'order' : ['cd', 'optimized_filter'],'Nb' : 1000 , 'type' : 'QAM', 'M' : 64,
                'ovs_factor' : 2, 'fiber_length' : 1000,
                'Fs' : 80e9, 'wavelength' : 1553e-9,
                'SNR' : 20,
                'plot' : True
                }

if __name__ == "__main__":
    database_config = {
                        'host': 'localhost',
                        'port': 5432,
                        'dbname': 'OpticalData',
                        'user': '',
                        'password': ''
                        }
    parameters = {
                'order' : ['cd', 'eval'], 'Nb' : 1000 , 'type' : 'QAM', 'M' : 16,
                'ovs_factor' : 2, 'fiber_length' : 4000,
                'Fs' : 21.4e9, 'wavelength' : 1553e-9, 'SNR' : 15,
                'plot' : True }
    
    batch_size, lr = 2, 1e-5
    min_epochs , max_epochs = 30, 120
    model_name, version = "optimizedFilter" ,  1.1
    input_data, targets  = prepare_dataset(database_config)
    train_model_(input_data, targets, model_name, version, batch_size, min_epochs, max_epochs , lr) 
    evaluate_model(model_name, version, parameters, n_trials = 3000)



    
