from torch.utils.data import DataLoader, random_split, TensorDataset
import pytorch_lightning as pl
import os 
from pytorch_lightning.callbacks import EarlyStopping
import torch
import pandas as pd

class CustomDataLoader(pl.LightningDataModule):
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
        return DataLoader(dataset=self.train_data, batch_size = self.batch_size, num_workers=8)
    
    def val_dataloader(self):
        return DataLoader(dataset = self.val_data, batch_size = self.batch_size,  num_workers=8)

    def test_dataloader(self):
        return DataLoader(dataset = self.test_data, num_workers=8)


class TrainModel():
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def train_model(self, min_epochs, max_epochs, debug, logger):
        self.trainer = pl.Trainer( devices="auto", accelerator="auto", min_epochs = min_epochs, max_epochs = max_epochs, 
                                   log_every_n_steps=10, logger = logger,
                                   callbacks=[EarlyStopping(monitor = "train_acc_epoch")], fast_dev_run=debug)
        self.trainer.fit(self.model, self.dataset.train_dataloader())
        self.trainer.validate(self.model, self.dataset.val_dataloader(), verbose=True)
        self.trainer.test(self.model, self.dataset.test_dataloader())
        print(pl.utilities.model_summary.summarize(self.model, max_depth=1))
        return self.model.parameters()
    
    def save_weights(self, model_name, version):
        dir_path = f"./binaries_models/{model_name}_v{version}"
        checkpoint_path =  f"./{dir_path}/{model_name}_v{version}.pkl"
        csv_file = f"./{dir_path}/{model_name}_v{version}.csv"
        df = pd.DataFrame(list(self.model.parameters()))
        df.to_csv(csv_file)
        try : os.mkdir(dir_path)
        except: pass
        self.trainer.save_checkpoint(filepath=checkpoint_path)

    def compare_accuracies(self):
        train_accuracy = getattr(self.model, 'train_acc_tensor')
        train_accuracy = torch.mean(torch.stack(train_accuracy), dim=0)
        val_accuracy = getattr(self.model, 'val_acc_tensor' )
        val_accuracy = torch.mean(torch.stack(val_accuracy), dim=0)        
        test_acc = getattr(self.model, 'test_acc_tensor')
        test_acc = torch.mean(torch.stack(test_acc), dim=0)   
        print("Training accuracy = {}, validation accuracy = {}, test accurary = {}".format(train_accuracy, val_accuracy, test_acc))
        return train_accuracy, val_accuracy, test_acc


