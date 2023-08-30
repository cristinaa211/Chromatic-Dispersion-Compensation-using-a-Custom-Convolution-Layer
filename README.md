# Chromatic Dispersion Compensation using a Custom Convolution Layer

The purpose of this repository is to present a method for chromatic dispersion compensation in coherent optical communications systems, by using a custom convolution layer based on the filter proposed in the paper "Optimal Least-Squares FIR Digital Filters for Compensation of Chromatic Dispersion in Digital Coherent Optical Receivers" -  http://dx.doi.org/10.1109/JLT.2014.2307916.

**The optical communication chain composition**

The emitter side of the chain contains a data generator, a data modulator, an upsampler and a Squared Root Raised Cosine Filter (SRRC). The channels adds noise and chromatic dispersion to the transmitted signal. In the receiver side there is the filter proposed by the paper, a transient remover for removing the filter's delays, a downsampler, and a data demodulator. The output data will be compared to the input data so that Bit Error Rate is computed for performance monitoring. The parameters are the following: sample frequency = 21.4GHz, over sampling factor = 2, fiber's chromatic dispersion parameter = 17-e3, the wavelength = 1553 nm, the fiber's length = 4km. The signal to noise ratio is 20dBm.  

![chain1](https://github.com/cristinaa211/Chromatic-Dispersion-Compensation-using-a-Custom-Convolution-Layer/assets/61435903/4a7e7f9a-7295-416e-9244-0f8a62c96346)

**The data is transmitted using a M-QAM modulation type. The received constellation in an ideal environment is given below** 

![image-3](https://github.com/cristinaa211/Chromatic-Dispersion-Compensation-using-a-Custom-Convolution-Layer/assets/61435903/30f89fe3-b053-436f-8f8f-efaac9e583ce)

**The effect of the Chromatic Dispersion on the received constellation**
![image-2](https://github.com/cristinaa211/Chromatic-Dispersion-Compensation-using-a-Custom-Convolution-Layer/assets/61435903/ae7dee59-5e12-46d8-aff7-4132f6fedf3a)

**The optimized filter for compensating the Chromatic Dispersion**
 
 The Filter proposed in the paper :
![fir_coef](https://github.com/cristinaa211/Chromatic-Dispersion-Compensation-using-a-Custom-Convolution-Layer/assets/61435903/57bbe591-d542-4f8f-892e-086c9d58514a)

which will be convoluted with a SRRC filter. 

![image-4](https://github.com/cristinaa211/Chromatic-Dispersion-Compensation-using-a-Custom-Convolution-Layer/assets/61435903/b47f48e4-c129-45dc-86b7-73097e242665)

**Dataset**

The purpose is to train a Deep Neural Network to compensate the chromatic dispersion. The parameters that will be updated are the filter's coefficients. The layer responsible for this will be ParametricConvolutionLayer in the network. The dataset is created by generating random data in the optical chain. The network's targets will be the data recovered just after the modulation process, in the emitter side, and the data fed to the optimized filter, as input data. All our data represent complex numbers.

The parameters of the optical chain are the following:                         
parameters = {'order' : ['cd', 'eval'], 'Nb' : 1000 , 'type' : 'QAM', 'M' : 16, 'ovs_factor' : 2, 'fiber_length' : 4000, 'Fs' : 21.4e9, 'wavelength' : 1553e-9, 'SNR' : 15, 'plot' : False } where: 

order           (list) : the layers present in the optical chain, default ['cd', 'srrc']
Nb               (int) : the length of the input generated data , default 1000
mod_type      (string) : the type of the modulation, default 'QAM'
M                (int) : the order of the modulation scheme (M distinct symbols in the constellation diagram), default 16
ovs_factor       (int) : the oversampling factor, default 2
fiber_length     (int) : the length of the fiber, default 4000
Fs             (float) : the sampling rate, default 21.4e9
wavelength     (float) : the wavelength of the signal, default 1553e-9
SNR              (int) : the signal to noise ratio expressed in dBm , default 20
plot         (boolean) : if True, then display the received constellation, default False

 The dataset is stored in a PostreSQL database. 


**Model training**

The targets will be further split in real and imaginary parts, encoded to values in [0,1, 2, 3, 4], representing the class indices. The dataset is split in training, validation and test sets. 
The final set of hyperparameters used is: {learning rate : 1e-5, batch size : 2, min epochs : 30}

**Model architecture**

The model's architecture is the following: 

- ParametricConvolutionLayer, whose coefficients are complex numbers, composed of 4 convolution layers

- DownsamplerRemove layer, having the role of downsampling by a factor of 2 and to remove the filter's delays

- Cross-Entropy Loss Function

- Adam optimizer

![Screenshot from 2023-08-29 19-13-31](https://github.com/cristinaa211/Chromatic-Dispersion-Compensation-using-a-Custom-Convolution-Layer/assets/61435903/02a08d66-5043-4a26-81f7-6f8002519b46)

**Results** 

The final metric that we want to evaluate our model on is the Bit Error Rate, which will be the number of error bits divided by the number of transmitted bits. Data is generated and passed in the optical chain, having as Chromatic Dispersion compensation layer the trained model, where the signal to noise ratio takes values between 0 and 2. 1000 Monte Carlo simulations are done for each SNR value. For each SNR value, there are 1000 computed BER values, so the final BER for each SNR value will the mean of the computed BERs. The results are compared to the ber computed when the initial optimized filter is used in the optical chain, for the same parameters. 
![optimizedFilter_v1 2](https://github.com/cristinaa211/Chromatic-Dispersion-Compensation-using-a-Custom-Convolution-Layer/assets/61435903/c0bf47ce-04d8-4307-b7fb-6e0bbead0497)
