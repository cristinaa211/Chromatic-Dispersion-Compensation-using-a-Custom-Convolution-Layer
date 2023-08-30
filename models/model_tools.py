from torch.utils.data import DataLoader, random_split, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch
import pandas as pd
import torch 
import os
from pytorch_lightning.loggers import TensorBoardLogger
import subprocess
from .model_layers import OptimizedFilterModel
from create_database import create_database, postgresql_operations
import pandas as pd
from chain_layers.Modulators import Demodulator
from performance.Performance import Performance
import matplotlib.pyplot as plt 

class CustomDataLoader(pl.LightningDataModule):
    """Creates a custom DataLoader"""
    def __init__(self, input_data, targets, batch_size):
        super(CustomDataLoader,self).__init__()
        self.input_data = input_data
        self.targets = targets 
        self.batch_size = batch_size
        self.dataset = self.create_TensorDataset()
        self.train_data, self.val_data, self.test_data = self.split_dataset()
    
    def create_TensorDataset(self):
        return TensorDataset(self.input_data, self.targets)

    def split_dataset(self):
        train_size = int(0.7 * len(self.dataset))
        val_size = int(0.5 * (len(self.dataset) - train_size))
        sizes = (train_size, len(self.dataset) - train_size)
        sizes_val = (len(self.dataset) - train_size - val_size, val_size)
        train_data, test_data = random_split(self.dataset, lengths=sizes)
        val_data, test_data = random_split(test_data, lengths=sizes_val)
        return train_data, val_data, test_data

    def train_dataloader(self):
        return DataLoader(dataset=self.train_data, batch_size = self.batch_size, num_workers=8, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(dataset = self.val_data, batch_size = self.batch_size,  num_workers=8)

    def test_dataloader(self):
        return DataLoader(dataset = self.test_data, num_workers=8)


class TrainModel():
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def train_model(self, min_epochs, max_epochs, debug, logger):
        """
        Trains the model on a minimum number of epochs of min_epochs, and a maximum number of  epochs of max_epochs. Evaluates the model using a test dataset.
        Args:
            min_epochs           (int) : minimum number of epochs
            max_epochs           (int) : maximum number of epochs
            debug            (boolean) : if True, activate the debug mode 
            logger (TensorBoardLogger) : TensorBoardLogger 
        Returns:
            model.parameters()         : the trained parameters of the model
        """
        self.trainer = pl.Trainer( devices="auto", accelerator="auto", min_epochs = min_epochs, max_epochs = max_epochs, 
                                   log_every_n_steps=10, logger = logger,
                                   callbacks=[EarlyStopping(monitor = "train_loss_epoch")], fast_dev_run=debug)
        self.trainer.fit(self.model, self.dataset.train_dataloader())
        self.trainer.validate(self.model, self.dataset.val_dataloader(), verbose=True)
        self.trainer.test(self.model, self.dataset.test_dataloader())
        print(pl.utilities.model_summary.summarize(self.model, max_depth=1))
        return self.model.parameters()
    
    def save_weights(self, model_name, version):
        """Saves the model's parameters"""
        dir_path = f"./binaries_models/{model_name}_v{version}"
        checkpoint_path =  f"./{dir_path}/{model_name}_v{version}.pkl"
        csv_file = f"./{dir_path}/{model_name}_v{version}.csv"
        parameters = self.model.state_dict()['conv_layer.parameter']
        df = pd.DataFrame(parameters)
        df.to_csv(csv_file, index=False)
        try : os.mkdir(dir_path)
        except: pass
        self.trainer.save_checkpoint(filepath=checkpoint_path)


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


def prepare_dataset_detect(database_config):
    header, data = postgresql_operations.read_table_postgresql(columns="input_net_real, input_net_imag, input_data",
                                                               limit = None, table_name="qam16_cd_optimized_filter", database_config=database_config)
    data_df = pd.DataFrame(data, columns=header)
    input_data = []
    targets = []
    for i, j in zip(data_df['input_net_real'].values, data_df['input_net_imag'].values):
        input_data.append([complex(real, imag) for real, imag in zip(i, j)] )
    input_data = torch.tensor(input_data)
    targets = torch.tensor(data_df['input_data'])
    return input_data, targets


def prepare_dataset(database_config):
    header, data = postgresql_operations.read_table_postgresql(columns="input_net_real, input_net_imag, targets_real, targets_imag",
                                                               limit = None, table_name="qam16_cd_optimized_filter", database_config=database_config)
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
    return input_data, targets
    
    
def train_model_(input_data, targets, model_name, version, batch_size, min_epochs, max_epochs, lr ):
    log_dir = f"./binaries_models/{model_name}_v{version}/"
    try: os.mkdir(log_dir)
    except:pass
    logger = TensorBoardLogger(log_dir, name = model_name)
    subprocess.Popen(['tensorboard', '--logdir',log_dir, '--port', '6006', '--load_fast' ,'false']) 
    len_cut = targets.shape[1]
    dataset = CustomDataLoader(input_data, targets, batch_size)
    model = OptimizedFilterModel(lr = lr, len_cut = len_cut)
    trainer = TrainModel(model, dataset)
    trainer.train_model(min_epochs, max_epochs, debug = False, logger = logger)
    trainer.save_weights(model_name, version)
    return trainer

def evaluate_model(model_name, version, parameters, M = 16, osnr = [0, 20], n_trials = 1000):
    try: modelpath = f"./binaries_models/{model_name}_v{version}/{model_name}_v{version}.pkl"
    except: pass
    reference_ber = f"./binaries_models/original_v1.0/evaluation_ber/ber_original_v1.0.csv"
    ref_ber_df = pd.read_csv(reference_ber)
    perf = Performance(M = M, osnr = osnr)
    ber_df = perf.run_Monte_Carlo_simulations(parameters = parameters, modelpath = modelpath,  N_trials = n_trials)
    perf.save_results(model_name, version)
    plt.figure()
    plt.semilogy(ber_df['snr'], ber_df['ber'], '-r')
    plt.semilogy(ref_ber_df['snr'], ref_ber_df['ber'], '-g')
    plt.legend(['TrainedModel', 'OptimizedFilterRef'])
    plt.xlabel("SNR")
    plt.ylabel("BER")
    plt.ylim([1e-6, 0.0])
    plt.xlim([0.0, 20])
    plt.title(f"{model_name}_v{version}")
    plt.savefig(f'./binaries_models/{model_name}_v{version}/evaluation_ber/{model_name}_v{version}_{n_trials}trials.png')
    plt.show()


def plot_results(fileame):
    reference_ber = f"./binaries_models/original_v1.0/evaluation_ber/ber_original_v1.0.csv"
    ref_ber_df = pd.read_csv(reference_ber)
    ber_df = pd.read_csv(fileame)
    plt.figure()
    plt.semilogy(ber_df['snr'], ber_df['ber'], '-r')
    plt.semilogy(ref_ber_df['snr'], ref_ber_df['ber'], '-g')
    plt.legend(['TrainedModel', 'OptimizedFilterRef'])
    plt.xlabel("SNR")
    plt.ylabel("BER")
    plt.ylim([1e-6, 0.0])
    plt.xlim([0.0, 20])
    plt.show()
    #plt.savefig(f'./binaries_models/{model_name}_v{version}/evaluation_ber/{model_name}_v{version}.jpg')


