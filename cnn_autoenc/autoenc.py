### Sequence convolutional autoencoder

'''
    A lot of examples I found online seemed to focus on images for autoencoders.
    Or if they were doing sequence, they prefer to use LSTM (which is really
    cool for implementing feedback and memory when past state is relevant, ie
    word prediction in audio synthesis). I think CNN-AE would be interesting to
    use for dimensionality reduction of sequences to learn salient features by
    leveraging temporal correlations learned from the convolutional filters,
    which can then be used in other models for other tasks (not naming anything
    because I have ideas. ;)). Also, this provides a nice architecture for signal
    denoising, which is cool. :)

    Something curious is that this model seems to perform pretty well even
    without any form of regularization like dropout or p-norms. I would
    imagine the MaxPooling takes care of that already, but it's still
    interesting to note
'''

from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, UpSampling1D, Dropout, Reshape
from keras.models import Model
from keras import regularizers
from keras.optimizers import SGD
from keras.utils import plot_model
import numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt

# Size of the input layer. window length by number of windows
input_sig = Input(shape=(60,5,))

# encoded is the compressed, lossy representation
# rectified linear units are nice because of the nonvanishing gradient
encoded = Conv1D(30,20, activation='relu', padding='same')(input_sig)
encoded = MaxPooling1D(2)(encoded)
encoded = Conv1D(30, 6, activation='relu', padding='same')(encoded)
encoded = MaxPooling1D(2)(encoded)
# Save shape for use later in reshaping
sh = encoded.shape
# Flatten to pass into dense layer
encoded = Flatten()(encoded)
encoded = Dense(100, activation='relu')(encoded)

# decoded is the reconstruction from the encoded output
decoded = Dense(int(sh[1])*int(sh[2]), activation='relu')(encoded)
# Reshape for conv layers
decoded = Reshape((int(sh[1]),int(sh[2])))(decoded)
decoded = UpSampling1D(2)(decoded)
decoded = Conv1D(30,6,activation='relu', padding='same')(decoded)
decoded = UpSampling1D(2)(decoded)
# linear activation because we don't want to constrain the output on regression.
decoded = Conv1D(5,20, activation='linear', padding='same')(decoded)

# Mapping input to its reconstruction
autoencoder = Model(input_sig, decoded)

# Mapping input to its encoded representation
encoder = Model(input_sig, encoded)

# Creating a stochastic gradient descent with scaling learning rate
sgd = SGD(lr=0.01, momentum=.9, decay=0, nesterov=False)
# Loss is MSE because it's essentially a regression model
autoencoder.compile(optimizer=sgd, loss='mse')

# Synthesizing data. Should look similar to a PPG with some noise.
# Sampling rate 40Hz. SNR max= 7, min= 1.4
signal_list = []
signal_test = []
for j in range(5000):
    amplitude = (5-1)*np.random.rand() +1
    freq = ((200 - 40) * np.random.rand() + 40)/60
    phase = 2*np.pi*np.random.rand()*freq
    signal_list += [amplitude*(np.sin([2*np.pi*freq*25/1000*i + phase for i in
        range(60*5)]) + .6*np.sin([4*np.pi*freq*25/1000*i + 2*phase for i in
            range(60*5)]) + .2*np.sin([6*np.pi*freq*25/1000*i + 4*phase for i in
            range(60*5)])) + 5/7*np.random.rand(60*5) - (5/7)/2]

# Comment here is for normalization...
#signal_list = (np.array(signal_list).reshape(1000,60,5) - np.min(signal_list))/(np.max(signal_list) - np.min(signal_list))

# Shape is batch size, window length, number of windows
signal_list = np.array(signal_list).reshape(5000,60,5)

# Synthesizing test data
for j in range(49):
    amplitude = (5-1)*np.random.rand() + 1
    phase = 2*np.pi*np.random.rand()*freq
    freq = ((200 - 40) * np.random.rand() + 40)/60
    signal_test += [amplitude*(np.sin([2*np.pi*freq*25/1000*i + phase for i in
        range(60*5)]) + .6*np.sin([4*np.pi*freq*25/1000*i + 2*phase for i in
            range(60*5)]) + .2*np.sin([6*np.pi*freq*25/1000*i + 4*phase for i in
            range(60*5)])) + 5/7*np.random.rand(60*5) - (5/7)/2]

#signal_test = (np.array(signal_test).reshape(49,60,5) - np.min(signal_test))/(np.max(signal_test) - np.min(signal_test))
signal_test = np.array(signal_test).reshape(49,60,5)

# Saving for later use
np.save("signals.npy", [signal_test, signal_list])

# Model visualization
plot_model(autoencoder, to_file='model.png', show_shapes=True, show_layer_names=False)

# Plot for fun
for i in range(1000):
    plt.plot(signal_list[i,:,:].ravel())
plt.show()

# Training
autoencoder.fit(signal_list, signal_list,
        epochs=100,
        batch_size=256,
        shuffle=True,
        validation_data=(signal_test, signal_test))
# Save models
autoencoder.save("autoencoder_sin.h5")
encoder.save("encoder_sin.h5")

# Let's see how good it looks
prediction = autoencoder.predict(signal_test)
fig, ax = plt.subplots(7,7)
fig2, ax2 = plt.subplots(7,7)
axes = ax.ravel()
axes2 = ax2.ravel()
for i in range(len(prediction)):
    axes[i].plot(prediction[i].ravel())
    axes[i].plot(signal_test[i,:,:].ravel(), alpha=0.6)
    # FFT
    axes2[i].plot(np.fft.fftfreq(len(prediction[i].ravel()), d=25/1000), abs(np.fft.fft(prediction[i].ravel())))
    axes2[i].plot(np.fft.fftfreq(len(prediction[i].ravel()), d=25/1000), abs(np.fft.fft(signal_test[i,:,:].ravel())), alpha=0.6)
axes2[-1].legend(["denoised decoded", "Test input"])
axes[-1].legend(["denoised decoded", "Test input"])
plt.show()
