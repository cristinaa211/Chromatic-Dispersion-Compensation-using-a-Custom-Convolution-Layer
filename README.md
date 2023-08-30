# Chromatic Dispersion Compensation using a Custom Convolution Layer

The purpose of this repository is to present a method for chromatic dispersion compensation in coherent optical communications systems, by using a custom convolution layer based on the filter proposed in the paper "Optimal Least-Squares FIR Digital Filters for Compensation of Chromatic Dispersion in Digital Coherent Optical Receivers" -  http://dx.doi.org/10.1109/JLT.2014.2307916.

**The optical communication chain composition**

THe emitter side of the chain contains a data generator, a data modulator, an upsampler and a Squared Root Raised Cosine Filter (SRRC). The channels adds noise and chromatic dispersion to the transmitted signal. In the receiver side there is the filter proposed by the paper, a transient remover for removing the filter's delays, a downsampler, and a data demodulator. The output data will be compared to the input data so that Bit Error Rate is computed for performance monitoring. The parameters are the following: sample frequency = 21.4GHz, over sampling factor = 2, fiber's chromatic dispersion parameter = 17-e3, the wavelength = 1553 nm, the fiber's length = 4km. The signal to noise ratio is 20dBm.  
![image](https://github.com/cristinaa211/Chromatic-Dispersion-Compensation-using-a-Custom-Convolution-Layer/assets/61435903/8edd28a5-35c6-4ee4-ac7e-566841c27b25)


**The data is transmitted using a M-QAM modulation type. The received constellation in an ideal environment is given below** 
![image-3](https://github.com/cristinaa211/Chromatic-Dispersion-Compensation-using-a-Custom-Convolution-Layer/assets/61435903/268fb7be-8853-4f93-ad44-ce6c2dacaa9e)


**The effect of the Chromatic Dispersion on the received constellation**
![image-2](https://github.com/cristinaa211/Chromatic-Dispersion-Compensation-using-a-Custom-Convolution-Layer/assets/61435903/a8e1baf1-a84e-4e28-ad75-e7e1beb3dcba)

**The optimized filter for compensating the Chromatic Dispersion**
 
 The Filter proposed in the paper :
![fir_coef](https://github.com/cristinaa211/Chromatic-Dispersion-Compensation-using-a-Custom-Convolution-Layer/assets/61435903/7b087809-48ed-4be7-a688-a3c7cd5818eb)


which will be convoluted with a SRRC filter:
![image-4](https://github.com/cristinaa211/Chromatic-Dispersion-Compensation-using-a-Custom-Convolution-Layer/assets/61435903/396ec4ba-b808-4d84-81f0-912c497d24fd)

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

![someimage](https://github.com/cristinaa211/Chromatic-Dispersion-Compensation-using-a-Custom-Convolution-Layer/assets/61435903/dea30325-436a-440c-81b0-daf4ff6bf097


The model's architecture is the following: 

a ParametricConvolutionLayer, whose coefficients are complex numbers, composed of 4 convolution layers

a DownsamplerRemove layer, having the role of downsampling by a factor of 2 and to remove the filter's delays

The loss function is Cross-Entropy Loss

 The optimizer is Adam optimizer

Given the filter's cofficients that we want to update, which are complex numbers, the ParametricConvolutionLayer is composed of 4 1d convolution layers. It's paramteres are initialized using the optimized filter's coefficients values. 

![Screenshot from 2023-08-29 19-13-31](https://github.com/cristinaa211/Chromatic-Dispersion-Compensation-using-a-Custom-Convolution-Layer/assets/61435903/e35c8ec1-1679-42f2-b31f-bde2a59a4792)

**Results** 

The final metric that we want to evaluate our model is Bit Error Rate, which will be the number of error bits divided by the number of transmitted bits. Thus Monte Carlo simulations are done, where data are generated and passed in the optical chain, having as Chromatic Dispersion compensation layer the trained model. 


![optimizedFilter_v1 2](https://github.com/cristinaa211/Chromatic-Dispersion-Compensation-using-a-Custom-Convolution-Layer/assets/61435903/988f21ab-0d04-4cf1-9403-dac11e04211b)
