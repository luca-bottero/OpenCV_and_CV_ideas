#%%
import keras
from keras import layers
import matplotlib.pyplot as plt
import visualkeras
import numpy as np

#%%

class AE_model_TF():
    def __init__(self, img_shape = (28,28,3), lstm = False, batch_len = 4):
        self.img_shape = img_shape
        self.batch_len = batch_len
        self.lstm      = lstm
        
        if self.lstm:
            self.Create_LSTM_AE()
        else:
            self.Create_AE()
        
    def Create_AE(self):
        input_img = keras.Input(shape = self.img_shape)

        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

        # at this point the representation is (4, 4, 8) i.e. 128-dimensional
  
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.UpSampling2D((2, 2))(x)
        decoded = layers.Conv2D(self.img_shape[-1], (3, 3), activation='tanh', padding='same')(x)

        self.model = keras.Model(input_img, decoded)

        #self.model.compile(optimizer=keras.optimizers.Adam(1e-2), loss = 'mse')
        self.model.compile(optimizer=keras.optimizers.SGD(1e-1, momentum=3e-2), loss = 'mse')


        return self.model

    def Create_LSTM_AE(self):

        inp_shape = tuple([self.batch_len]) + self.img_shape

        input_img = keras.Input(shape = inp_shape)

        x = layers.ConvLSTM2D(64, (3, 3), return_sequences = True, activation='relu', padding='same')(input_img)
        x = layers.MaxPooling3D((2, 2, 2), padding='same')(x)
        x = layers.ConvLSTM2D(32, (3, 3), return_sequences = True, activation='relu', padding='same')(x)
        x = layers.MaxPooling3D((2, 2, 2), padding='same')(x)
        x = layers.ConvLSTM2D(16, (3, 3), return_sequences = True, activation='relu', padding='same')(x)
        encoded = layers.MaxPooling3D((2, 2, 2), padding='same')(x)

        # at this point the representation is (4, 4, 8) i.e. 128-dimensional

        x = layers.ConvLSTM2D(16, (3, 3), return_sequences = True, activation='relu', padding='same')(encoded)
        x = layers.UpSampling3D((2, 2, 2))(x)
        x = layers.ConvLSTM2D(32, (3, 3), return_sequences = True, activation='relu', padding='same')(x)
        x = layers.UpSampling3D((2, 2, 2))(x)
        x = layers.ConvLSTM2D(64, (3, 3), return_sequences = True, activation='relu')(x)
        x = layers.UpSampling3D((2, 2, 2))(x)
        decoded = layers.ConvLSTM2D(self.img_shape[-1], (3, 3), return_sequences = False, activation='tanh', padding='same')(x)

        self.model = keras.Model(input_img, decoded)

        #self.model.compile(optimizer=keras.optimizers.Adam(1e-2), loss = 'mse')
        self.model.compile(optimizer=keras.optimizers.SGD(1e-1, momentum=3e-2), loss = 'mse')

        return self.model

    def prepare_data(self, input):
        self.input = np.array([input.astype('float32')])/255

    def train_predict(self, input):

        self.prepare_data(input)

        if self.lstm:
            #print(self.input.shape)
            self.model.fit(self.input, self.input[:,-1,:,:], epochs = 1)
        else:
            self.model.fit(self.input, self.input, epochs = 1)
        
        self.output = self.model.predict(self.input)

        #self.output
        #self.output = np.abs(self.output)
        #self.output *= np.round((255.0/self.output.max()))
        #self.output /= self.output.max()
        #self.output *= 255
        #self.output = np.round(self.output)
        #self.output = np.squeeze(self.output, axis = 0)
        self.output = self.output[0]

        return self.input, self.output

    def plot_layered_view(self):
        visualkeras.layered_view(self.model, legend = True)




# %%
'''
from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

# %%
from keras.callbacks import TensorBoard

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))

# %%

pred = autoencoder.predict(x_train)


# %%

for i in range(5):
    plt.imshow(x_train[i])
    plt.show()
    plt.imshow(pred[i].reshape(28,28))
    plt.show()'''


# %%
