#%%
import keras
from keras import layers
import matplotlib.pyplot as plt
import visualkeras
import numpy as np

#%%

class AE_model():
    def __init__(self):
        self.Create_AE()
        
    def Create_AE(self):
        input_img = keras.Input(shape=(28, 28, 1))

        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

        # at this point the representation is (4, 4, 8) i.e. 128-dimensional

        x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(16, (3, 3), activation='relu')(x)
        x = layers.UpSampling2D((2, 2))(x)
        decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        self.model = keras.Model(input_img, decoded)
        self.model.compile(optimizer='adam', loss='binary_crossentropy')

        return self.model

    def prepare_data(self, input):
        self.input = np.reshape(input.astype('float32'), (len(input.astype('float32')), 28, 28, 1))

    def train_predict(self, input):
        self.prepare_data(input)

        self.model.fit(self.input, self.input, epochs = 1)
        self.output = self.model.predict(self.input)

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
