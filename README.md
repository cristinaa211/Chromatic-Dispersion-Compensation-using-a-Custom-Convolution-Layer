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

The purpose is to train a Deep Neural Network to compensate the chromatic dispersion. The parameters that will be updated are the optimized filter's coefficients. The layer responsible for this will be ParametricConvolutionLayer in the network. 

