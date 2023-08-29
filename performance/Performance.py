import torch 
from OpticalChain import simulate_chain_get_data
import pandas as pd 
import matplotlib.pyplot as plt
import os 

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Performance():
    def __init__(self, M =16, Nb = 1000, osnr=[0,20]):
        self.M = M
        self.osnr = osnr
        self.Nb = Nb

    def run_Monte_Carlo_simulations(self, parameters, modelpath, N_trials):
        snr_range = range(self.osnr[0], self.osnr[1]+1)
        ber = []
        for snr in snr_range:
            parameters['SNR'] = snr
            ber_trial = []
            print('I am making simulations for SNR == {}'.format(snr))
            for _ in range(N_trials):
                input_data, input_net, targets, output_data = simulate_chain_get_data(parameters, modelpath)
                binary = BitErrorRate(M=self.M, input=input_data, output=output_data)
                ber_ind = binary.compute_ber()
                ber_trial.append(ber_ind)
            ber_trial_snr = torch.mean(torch.tensor(ber_trial))
            ber.append(ber_trial_snr.item())
            print('For SNR final == {} the BER is {}'.format(snr, ber_trial_snr))
        self.ber = ber
        self.snr_range = snr_range
        self.ber_df = pd.DataFrame(data=zip(self.snr_range, self.ber), columns=['snr', 'ber'])
        return self.ber_df 

    def save_results(self, model_name, version):
        try:
            os.mkdir(f'./binaries_models/{model_name}_v{version}/evaluation_ber/')
        except: pass
        self.ber_df.to_csv(f'./binaries_models/{model_name}_v{version}/evaluation_ber/ber_{model_name}_v{version}.csv', index=False)



class BitErrorRate:
    def __init__(self, M = 16, input=None, output=None, name="BER"):
        self.M = M
        self.input_data = torch.flatten(input).tolist()
        self.output_data = torch.flatten(output).tolist()
        self.name = name

    def bin_data(self,data):
        self.data = data
        self.bit = ''
        if self.M == 4:
            for i in self.data:
                self.bit += '{0:02b}'.format(i)
        elif self.M == 8:
            for i in self.data:
                self.bit += '{0:03b}'.format(i)
        elif self.M == 16:
            for i in self.data:
                self.bit += '{0:04b}'.format(i)
        elif self.M == 32:
            for i in self.data:
                self.bit += '{0:05b}'.format(i)
        elif self.M == 64:
            for i in self.data:
                self.bit += '{0:06b}'.format(i)
        elif self.M == 128:
            for i in self.data:
                self.bit += '{0:07b}'.format(i)
        elif self.M == 256:
            for i in self.data:
                self.bit += '{0:08b}'.format(i)
        elif self.M == 2:
            for i in self.data:
                self.bit += '{0:01b}'.format(i)
        return self.bit

    def compute_ber(self):
        """Computes the bir error rate between input data and output data in the optical chain"""
        in_bin = self.bin_data(self.input_data)
        out_bin = self.bin_data(self.output_data)
        BER = 0
        if len(in_bin) == len(out_bin):
            for i in range(len(in_bin)):
                if in_bin[i] != out_bin[i]:
                    BER += 1
            BER /= len(in_bin)
        return BER