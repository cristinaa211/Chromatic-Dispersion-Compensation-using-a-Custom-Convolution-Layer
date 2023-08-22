import torch 
from OpticalChain import simulate_chain_get_data

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Performance():

    def __init__(self, M, Nb, osnr):
        self.M = M
        self.osnr = osnr
        self.Nb = Nb
        self.parameters = []

    def Monte_Carlo(self, parameters, N_trials):
        snr_range = range(self.osnr[0], self.osnr[1]+1)
        ber = []
        for SNR in snr_range:
            ber_trial = []
            print('I am making simulations for SNR == {}'.format(SNR))
            for _ in range(N_trials):
                input_data, output_data, symb_input, symb_output = simulate_chain_get_data(parameters)
                binary = Binary(self.M, input_data, output_data, False, 'BER')
                BER = binary.compute_ber()
                ber_trial.append(BER)
            ber_trial_snr = torch.mean(torch.tensor(ber_trial))
            ber.append(ber_trial_snr)
            print('For SNR final == {} the BER is {}'.format(SNR, ber_trial_snr))
        return snr_range, ber

class Binary:
    def __init__(self, M, input=None, output=None, name="BER"):
        self.M = M
        self.input_data = input
        self.output_data = output
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