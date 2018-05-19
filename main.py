import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from keras import initializers
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.utils import to_categorical
np.random.seed(0)

random_input_dim = 100

class CatGAN():
    def __init__(self, lr, beta, summaries):
        self._lr = lr
        self._beta = beta
        self._model_summaries = summaries
        
    def load_generator(self):
        G = Sequential()
        G.add(Dense(256, input_dim=random_input_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        G.add(LeakyReLU(0.3))
        G.add(Dense(512))
        G.add(LeakyReLU(0.3))
        G.add(Dense(1024))
        G.add(LeakyReLU(0.3))
        G.add(Dense(2048))
        G.add(LeakyReLU(0.3))
        G.add(Dense(784, activation='tanh'))

        G.compile(loss='binary_crossentropy', optimizer=Adam(lr=self._lr, beta_1=self._beta))
        if self._model_summaries:
            G.summary()

        return G

    def load_discriminator(self):
        D = Sequential()
        D.add(Dense(1024, input_dim=784, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        D.add(LeakyReLU(0.2))
        D.add(Dropout(0.3))
        D.add(Dense(512))
        D.add(LeakyReLU(0.2))
        D.add(Dropout(0.3))
        D.add(Dense(256))
        D.add(LeakyReLU(0.2))
        D.add(Dropout(0.3))
        D.add(Dense(1, activation='sigmoid'))

        D.compile(loss='binary_crossentropy', optimizer=Adam(lr=self._lr, beta_1=self._beta))
        if self._model_summaries:
            D.summary()
        
        return D

    def load_GAN(self, discriminator, generator, random_input_dim):
        discriminator.trainable = False
        catgan_input = Input(shape=[random_input_dim,])
        x = generator(catgan_input)
        catgan_output = discriminator(x)
        catgan = Model(inputs=catgan_input, outputs=catgan_output)
        catgan.compile(loss='binary_crossentropy', optimizer=Adam(lr=self._lr, beta_1=self._beta))
        catgan.summary()

        return catgan

    def save_generated_images(self, epoch, generator, show_images=False):
        num_examples = 100
        random_noise = np.random.normal(0, 1, size=[num_examples, random_input_dim])
        generated_images = generator.predict(random_noise)
        generated_images = generated_images.reshape(examples, 28, 28)

        plt.figure(figsize=(10, 10))
        for i in range(generated_images.shape[0]):
            plt.subplot(10, 10, i+1)
            plt.imshow(generated_images[i], cmap='gray')
            plt.axis('off')

        plt.tight_layout()
        plt.savefig('generated_image_epoch_{}.png'.format(epoch))
        if show_images:
            plt.show()

    def train_gan(self, x_train, num_epochs, batch_size=256):
        batch_count = int(x_train.shape[0] // batch_size)

        generator = self.load_generator()
        discriminator = self.load_discriminator()
        catgan = self.load_GAN(discriminator, generator, random_input_dim)

        for epoch in range(1, num_epochs + 1):
            print ('------------Epoch {}------------'.format(epoch))
            for i in (range(batch_count)):
                
                random_noise = np.random.normal(0, 1, size=[batch_size, random_input_dim])
                image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

                generated_images = generator.predict(random_noise)
                X_discriminator = np.concatenate([image_batch, generated_images])

                y_discriminator = np.zeros(2 * batch_size)
                y_discriminator[:batch_size] = 0.95

                discriminator.trainable = True
                discriminator.train_on_batch(X_discriminator, y_discriminator)

                random_noise = np.random.normal(0, 1, size=[batch_size, random_input_dim])
                y_generator = np.ones(batch_size)
                discriminator.trainable = False
                catgan.train_on_batch(random_noise, y_generator)

            if num_epochs == 1 or epoch % 20 == 0:
                self.save_generated_images(epoch, generator)        

def load_cats():
    cats = np.load("./cats.npy")
    Y = []
    for i in range(cats.shape[0]):
        Y.append([1,0])
    Y = np.array(Y)

    (x_train, y_train, x_test, y_test) = train_test_split(cats, Y)
    x_train = (x_train.astype(np.float32)) / 255
    x_train = x_train.reshape(x_train.shape[0], 784)

    return (x_train, y_train, x_test, y_test)    

catgan = CatGAN(lr=0.0002, beta=0.5, summaries=False)
(x_train, y_train, x_test, y_test) = load_cats()
catgan.train_gan(x_train, 200, 256)

