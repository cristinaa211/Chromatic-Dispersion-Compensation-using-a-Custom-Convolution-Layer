import torch 
import os
from pytorch_lightning.loggers import TensorBoardLogger
import subprocess
from models.model_layers import OptimizedFilterModel
from models.model_tools import CustomDataLoader, TrainModel
from create_database import create_database, postgresql_operations
import pandas as pd
from chain_layers.Modulators import Demodulator
from OpticalChain import simulate_chain_get_data
from sklearn.preprocessing import OneHotEncoder

# matplotlib.get_backend()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


parameters = {'order' : ['cd', 'optimized_filter'], 'Nb' : 1000 , 'type' : 'QAM', 'M' : 16,
                'ovs_factor' : 2, 'fiber_length' : 4000,
                'Fs' : 21.4e9, 'wavelength' : 1553e-9,
                'SNR' : 20,
                'plot' : False
                }

parameters_2 = {'order' : ['cd', 'optimized_filter'],'Nb' : 1000 , 'type' : 'QAM', 'M' : 64,
                'ovs_factor' : 2, 'fiber_length' : 1000,
                'Fs' : 80e9, 'wavelength' : 1553e-9,
                'SNR' : 20,
                'plot' : True
                }

def demodulate_data(data):
    demodulator = Demodulator(M = 16)
    output_data  =  []
    for sample in data:
        output_data.append(demodulator.forward(sample))
    output_data = torch.stack(output_data)
    return output_data

def encode_targets(df): 
    targets_real = df['targets_real'].values
    targets_imag = df['targets_imag'].values
    targets_encodings, no_real_targ = create_database.count_unique_numbers(targets_real)
    encode_dict = {x : idx for idx, x in enumerate(targets_encodings)}
    real_targets_encoded = []
    imag_targets_encoded = []
    for sample in targets_real:
        real_targets_encoded.append([encode_dict[targ] for targ in sample])
    for sample in targets_imag:
        imag_targets_encoded.append([encode_dict[targ] for targ in sample])
    return real_targets_encoded, imag_targets_encoded, no_real_targ


def prepare_dataset(database_config):
    header, data = postgresql_operations.read_table_postgresql(columns="input_net_real, input_net_imag, targets_real, targets_imag",
                                                               limit = 552, table_name="qam16_cd_optimized_filter", database_config=database_config)
    data_df = pd.DataFrame(data, columns=header)
    input_data = []
    targets = []
    real_targets_encoded, imag_targets_encoded, no_classes = encode_targets(data_df)
    for i, j in zip(data_df['input_net_real'].values,data_df['input_net_imag'].values):
        input_data.append([complex(real, imag) for real, imag in zip(i, j)] )
    for i, j in zip(real_targets_encoded, imag_targets_encoded):
        targets.append([complex(real, imag) for real, imag in zip(i, j)] )
    input_data = torch.tensor(input_data)
    targets = torch.tensor(targets).type(torch.complex32)
    return input_data, targets, no_classes
    
    
def train_model_(input_data, targets,no_classes, model_name, version, batch_size, min_epochs, max_epochs, lr, logger ):
    len_cut = targets.shape[1]
    dataset = CustomDataLoader(input_data, targets, batch_size)
    model = OptimizedFilterModel(lr = lr, num_classes=no_classes,batch_size=batch_size, len_cut = len_cut)
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
    lr = 1e-5
    model_name, version = "optimized_model" ,  1
    log_dir = f"./binaries_models/{model_name}_v{version}/"
    try: os.mkdir(log_dir)
    except:pass
    logger = TensorBoardLogger(log_dir, name = model_name)
    tb_process = subprocess.Popen(['tensorboard', '--logdir',log_dir, '--port', '6007', '--load_fast' ,'false']) 
    input_data, targets , no_classes = prepare_dataset(database_config)
    train_model_(input_data, targets,no_classes, model_name, version, batch_size, min_epochs, max_epochs , lr, logger) 
    # simulate_optical_chain(parameters)
