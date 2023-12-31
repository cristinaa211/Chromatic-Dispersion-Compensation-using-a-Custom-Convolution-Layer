import torch
import scipy
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from filters_tools.Filters import apply_filter

def fir_cd_comp_imp_resp( Fs = 21.4e9,D = 17e-3,Lambda=1553e-9, L=4000, 
                         omega_1 =-torch.pi , omega_2 = torch.pi, 
                         zeta = 1e-14, Nc = None, plot = True ):
    """Creates a FIR filter 
    Args:
        Fs (float, optional): the sampling frequency. Defaults to 21.4e9.
        D (_type_, optional): _description_. Defaults to 17e-3.
        Lambda (float, optional): the wavelength. Defaults to 1553e-9.
        L (int, optional): the length of fiber. Defaults to 4000.
        omega_1 (_type_, optional): _description_. Defaults to -torch.pi.
        omega_2 (_type_, optional): _description_. Defaults to torch.pi.
        zeta (_type_, optional): _description_. Defaults to 1e-14.
        Nc (int, optional): the order of the filter. Defaults to None.
        plot (bool, optional): plot the filter's impulse response if True. Defaults to True.
    Returns:
        h_hat: the impulse response of the filter 
    """
    c = torch.tensor(3e8)           # the speed of light
    Ts = torch.tensor(1 / Fs)       # the sampling period
    Fs = torch.tensor(Fs)           # the sampling frequency 
    D = torch.tensor(D)             # 
    Lambda = torch.tensor(Lambda)   # the wavelength
    L = torch.tensor(L)     
    K = (D * Lambda**2 * L) / ( 4 *torch.pi * c *Ts**2)# The K term
    N = 2* torch.trunc(2 * K * torch.pi) + torch.ones(1)# The maximum number of taps of FIR filter
    N = int(N)
    #print('The length of the FIR filter is == ', N)
    if Nc == None:  Nc = N
    ## Computing of h_n_FIR
    ## h_n_FIR = inverse(Q) * D_matr
    # Q - Hermitian Toplitz matrix of NcxNc elements
    Q = torch.zeros((Nc, Nc), dtype=torch.complex64)
    FC = torch.zeros((Nc), dtype=torch.complex64) # first colon
    FR = torch.zeros((Nc), dtype=torch.complex64) # first row
    FC[0] = (omega_2 - omega_1) / (2 * torch.pi)
    FR[0] = (omega_2 - omega_1) / (2 * torch.pi)
    for m in range(Nc):
        n = 0
        if m != 0:
            term = -n + m
            first_term = torch.exp(torch.tensor(-1j * (term * omega_1))) - torch.exp(torch.tensor(-1j * (term * omega_2)))
            FC[m] = first_term * (1 / torch.tensor(2j * torch.pi * term))
    Q = torch.tensor(scipy.linalg.toeplitz(FC, FR))
    Q = Q + zeta * torch.eye(Nc)
    ## D_matr
    N_vect = np.arange(np.trunc(-Nc / 2), np.trunc(Nc / 2) + 1, 1)
    D_matrix = torch.zeros((Nc, 1), dtype=torch.complex64)
    term1 = 1 / (4 * torch.sqrt(torch.pi * K))
    term2 = torch.exp(torch.tensor(1j * 3 * torch.pi / 4)) / (2 * torch.sqrt(K))
    D_matrix = [term1 * (torch.exp((-1j * (n ** 2 / (4 * K) + 3 * torch.pi / 4))) *
                         (scipy.special.erf(term2 * (2 * K * torch.pi - n)) +
                          scipy.special.erf(term2 * (2 * K * torch.pi + n)))) for n in N_vect]
    D_matrix = torch.transpose(torch.tensor(D_matrix), -1, 0)
    # h_hat
    h_hat = torch.matmul(torch.linalg.inv(Q), D_matrix)
    if plot == True:
        fig, axes = plt.subplots(3)
        axes[0].stem(N_vect, torch.real(h_hat))
        axes[0].set(xlabel='n_th tap', ylabel='Magnitude')
        axes[0].set_title('Real parts of h_hat')
        axes[1].stem(N_vect, torch.imag(h_hat))
        axes[1].set(xlabel='n_th tap', ylabel='Magnitude')
        axes[1].set_title('Imaginary parts of h_hat')
        axes[2].stem(N_vect, torch.abs(h_hat))
        axes[2].set(xlabel='n_th tap', ylabel='Magnitude')
        axes[2].set_title('Absolute values of h_hat')
        plt.show()
    return h_hat

def srrc_impulse_response( Fs=21.4e9, delay=50, os=2, roll_off=0.25,  plot = True):
    """Creates a Square Root Raised Cosine Filter 
    Args:
        Fs (float, optional): the sampling frequency. Defaults to 21.4e9.
        delay (int, optional): the delay of the filter. Defaults to 40.
        os (int, optional): oversampling factor. Defaults to 2.
        roll_off (float, optional): _description_. Defaults to 0.25.
        plot (bool, optional): _description_. Defaults to True.

    Returns:
        h_rrc: the filter's impulse response
    """
    Ts = 1 / Fs # the sampling rate
    N = delay * os 
    t_vect = torch.arange(-N, N + 1, 1, dtype=torch.int32) * Ts / os # the total length of the filter
    h_rrc = torch.zeros(t_vect.shape)
    s = 0
    for t in t_vect:
        if t == 0:
            h_rrc[s] = (1 / Ts) * (1 + roll_off * (4 / torch.pi - 1))
        elif t == Ts / (4 * roll_off):
            h_rrc[s] = (roll_off / (Ts * torch.sqrt(torch.tensor(2)))) * \
                       ((1 + 2 / torch.pi) * torch.sin(torch.tensor(torch.pi / (4 * roll_off))) +
                        (1 - 2 / torch.pi) * torch.cos(torch.tensor(torch.pi / (4 * roll_off))))
        elif t == - Ts / (4 * roll_off):
            h_rrc[s] = (roll_off / (Ts * torch.sqrt(torch.tensor(2)))) * \
                       ((1 + 2 / torch.pi) * torch.sin(torch.tensor(torch.pi / (4 * roll_off))) +
                        (1 - 2 / torch.pi) * torch.cos(torch.tensor(torch.pi / (4 * roll_off))))
        else:
            h_rrc[s] = (1 / Ts) * (torch.sin(
                (torch.pi * t / Ts) * (1 - roll_off)) + 4 * (roll_off * t / Ts) * torch.cos(
                (torch.pi * t / Ts) * (1 + roll_off))) / \
                       ((torch.pi * t / Ts) * (1 - (4 * roll_off * t / Ts) ** 2))
        s += 1
    h_rrc = h_rrc / torch.sqrt(torch.sum(h_rrc ** 2))
    #print('The length of the SRRC filter is:', len(h_rrc))
    if plot == True:
        plt.figure()
        plt.plot(torch.arange(len(h_rrc)), h_rrc)
        plt.title('Square Root Raised Cosine Filter h(t)')
        plt.xlabel('nth tap')
        plt.ylabel('Magnitude')
        plt.show()
    return h_rrc



def get_impulse_response(choice = 'srrc', delay = 50, plot = False) :
    # omega = torch.pi*((1+self.rolloff)/self.os)
    # zeta = 1e-14
    if choice == 'srrc' :
        coeff_filter = srrc_impulse_response (delay = delay, plot=plot )
        order = len ( coeff_filter )

    elif choice == 'fir' :
        coeff_filter = fir_cd_comp_imp_resp (plot=plot )
        order = len ( coeff_filter )
    
    elif choice == 'param':
        coeff_filter = export_optmized_filter()
        order = len(coeff_filter)
    return coeff_filter, order

def get_trunc(choice):
    fir_filter_order = get_impulse_response( 'fir' ) [ 1 ] -1
    srrc_filter_ofer = get_impulse_response( 'srrc' ) [ 1 ] -1 
    optimized_filter = get_impulse_response('param')[1] - 1
    if choice == 'fir+srrc':
        cut = fir_filter_order + 2*srrc_filter_ofer 
    elif choice == 'fir' : 
        cut = fir_filter_order
    elif choice =='srrc':
        cut = 2*srrc_filter_ofer
    elif choice == 'param':
        cut = optimized_filter + srrc_filter_ofer
    return cut

def export_optmized_filter(filename = None, plot = False):
    impulse_response1 = fir_cd_comp_imp_resp(Fs = 21.4e9,D = 17e-3,Lambda=1553e-9, L=4000, 
                            omega_1 =-torch.pi , omega_2 = torch.pi, zeta = 1e-14, Nc = None, plot=False)
    impulse_response2 = srrc_impulse_response(Fs=21.4e9, delay=10, os=2, roll_off=0.25, plot = False)
    himp = apply_filter(impulse_response1, impulse_response2)
    if plot == True:
        real_h = torch.flatten(torch.real(himp).cpu())
        imag_h = torch.flatten(torch.imag(himp).cpu())
        abs_h = torch.flatten(torch.abs(himp).cpu())
        fig, axes = plt.subplots(3)
        axes[0].plot(real_h)
        axes[0].set(xlabel='n_th tap', ylabel='Magnitude')
        axes[0].set_title('Real parts of h_hat')
        axes[1].plot(imag_h)
        axes[1].set(xlabel='n_th tap', ylabel='Magnitude')
        axes[1].set_title('Imaginary parts of h_hat')
        axes[2].plot(abs_h)
        axes[2].set(xlabel='n_th tap', ylabel='Magnitude')
        axes[2].set_title('Absolute values of h_hat')
        plt.show()
        df = pd.DataFrame(himp.cpu())
        df.to_csv(f'./filters_tools/{filename}.csv', index=False)
    himp = torch.flatten(himp).cpu()
    return himp


def plot_filter_coeffs(himp):
    real_h = himp.real
    imag_h = himp.imag
    abs_h = abs(himp)
    fig, axes = plt.subplots(3)
    axes[0].plot(real_h)
    axes[0].set(xlabel='n_th tap', ylabel='Magnitude')
    axes[0].set_title('Real parts of h_hat')
    axes[1].plot(imag_h)
    axes[1].set(xlabel='n_th tap', ylabel='Magnitude')
    axes[1].set_title('Imaginary parts of h_hat')
    axes[2].plot(abs_h)
    axes[2].set(xlabel='n_th tap', ylabel='Magnitude')
    axes[2].set_title('Absolute values of h_hat')
    plt.show()





if __name__ == "__main__":
    fir_cd_comp_imp_resp(Fs = 21.4e9,D = 17e-3,Lambda=1553e-9, L=4000, 
                            omega_1 =-torch.pi , omega_2 = torch.pi, zeta = 1e-14, Nc = None, plot=True)
    srrc_impulse_response(Fs=21.4e9, delay=10, os=2, roll_off=0.25, plot = True)
