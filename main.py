import torch 
import os
from pytorch_lightning.loggers import TensorBoardLogger
import subprocess
from models.model_layers import OptimizedFilterModel
from models.model_tools import CustomDataLoader, TrainModel
from create_database import postgresql_operations
import pandas as pd

# matplotlib.get_backend()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


parameters = {'Nb' : 1000 , 'type' : 'QAM', 'M' : 16,
                'ovs_factor' : 2, 'fiber_length' : 4000,
                'Fs' : 21.4e9, 'wavelength' : 1553e-9,
                'SNR' : 20,
                'plot' : False
                }

parameters_2 = {'Nb' : 1000 , 'type' : 'QAM', 'M' : 64,
                'ovs_factor' : 2, 'fiber_length' : 1000,
                'Fs' : 80e9, 'wavelength' : 1553e-9,
                'SNR' : 20,
                'plot' : True
                }
def count_unique_complex_numbers(list_of_lists):
    # Flatten the list of lists into a single list
    flat_list = [complex_num for sublist in list_of_lists for complex_num in sublist]

    # Convert the list into a set to get unique values
    unique_complex_numbers = set(flat_list)

    # Count the number of unique complex numbers
    unique_count = len(unique_complex_numbers)

    return unique_count

def train_model_(model_name, version, batch_size, min_epochs, max_epochs, lr, logger, database_config):
    header, data = postgresql_operations.read_table_postgresql(columns="input_net_real, input_net_imag, targets_real, targets_imag",
                                                               limit = 552, table_name="qam16_cd_optimized_filter", database_config=database_config)
    data_df = pd.DataFrame(data, columns=header)
    input_data = []
    targets = []
    for i, j in zip(data_df['input_net_real'].values,data_df['input_net_imag'].values):
        input_data.append([complex(real, imag) for real, imag in zip(i, j)] )
    for i, j in zip(data_df['targets_real'].values,data_df['targets_imag'].values):
        targets.append([complex(real, imag) for real, imag in zip(i, j)] )
    targets_labels = count_unique_complex_numbers(targets)
    input_data = torch.tensor(input_data).type(torch.complex32)
    targets = torch.tensor(targets).type(torch.complex32)
    len_cut = targets.shape[1]
    dataset = CustomDataLoader(input_data, targets, batch_size)
    model = OptimizedFilterModel(lr = lr, len_cut = len_cut)
    train = TrainModel(model, dataset)
    train.train_model(min_epochs, max_epochs, debug = False, logger = logger)

    train.compare_accuracies()
    train.save_weights(model_name, version)

if __name__ == "__main__":
    database_config = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'OpticalData',
    'user': '',
    'password': ''
    }
    batch_size = 4
    min_epochs , max_epochs = 5, 1000
    lr = 1e-6
    model_name, version = "optimized_model" ,  1
    log_dir = f"./binaries_models/{model_name}_v{version}/"
    try: os.mkdir(log_dir)
    except:pass
    logger = TensorBoardLogger(log_dir, name = model_name)
    tb_process = subprocess.Popen(['tensorboard', '--logdir',log_dir, '--port', '6007', '--load_fast' ,'false']) 
    train_model_(model_name, version, batch_size, min_epochs, max_epochs , lr, logger, database_config) 
