import tensorflow as tf
import numpy as np

class NNValueFunction(object):
    """ NN-based state-value function """
    def __init__(self, obs_dim, hid1_mult, ep_v, bs_v, lr_v):

        self.obs_dim = obs_dim
        self.hid1_mult = hid1_mult
        self.epochs = ep_v
        self.lr = lr_v
        self.hid3_size = 10
        self.batch_size = bs_v

        self.model1 = tf.keras.Sequential()
        hid1_size = self.obs_dim * self.hid1_mult
        hid3_size = self.hid3_size  # chosen empirically
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        self.model1.add(tf.keras.layers.Dense(hid1_size, input_shape=(self.obs_dim,), activation='relu'))
        self.model1.add(tf.keras.layers.Dense(hid2_size, activation='relu'))
        self.model1.add(tf.keras.layers.Dense(hid3_size, activation='relu'))
        self.model1.add(tf.keras.layers.Dense(1))
        self.model1.compile(optimizer=tf.keras.optimizers.Adam(self.lr), loss=tf.keras.losses.MSE)


    def fit(self, x, y, logger):
        """ Fit model to current data batch + previous data batch
        Args:
            x: features
            y: target
            logger: logger to save training loss and % explained variance
        """
        history = self.model1.fit(x=x, y=y, batch_size=self.batch_size, epochs=self.epochs)
        logger.log({'ValFuncLoss': history.history['loss'][-1]})


    def predict(self, x):
        """ Predict method """
        y_hat = self.model1.predict(x)
        return y_hat