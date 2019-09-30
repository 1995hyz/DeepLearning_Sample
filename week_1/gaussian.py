#!/bin/python3.6

import numpy as np
import tensorflow as tf

from tqdm import trange

import math
import matplotlib.pyplot as plt

NUM_FEATURES = 11
NUM_SAMP = 50
BATCH_SIZE = 32
NUM_BATCHES = 1000
LEARNING_RATE = 0.1


class Data(object):
    def __init__(self, num_features=NUM_FEATURES, num_samp=NUM_SAMP):
        """
        Draw random weights and bias. Project vectors in R^NUM_FEATURES
        onto R with said weights and bias.
        """
        num_samp = NUM_SAMP
        sigma = 0.1
        np.random.seed(31415)

        # We're going to learn these paramters
        self.index = np.arange(num_samp)
        self.x = np.sort(np.random.uniform(size=(1, num_samp)))
        self.y = np.sin(2*np.pi*self.x) + sigma * np.random.normal(size=(1, num_samp))

    def get_batch(self, batch_size=BATCH_SIZE):
        """
        Select random subset of examples for training batch
        """
        choices = np.random.choice(self.index, size=batch_size)

        return self.x[0, choices], self.y[0, choices].flatten()


class Model(tf.Module):
    def __init__(self, num_features=NUM_FEATURES):
        """
        A Gaussian regression model with a bias term
        """
        self.w = tf.Variable(tf.random.normal(shape=[num_features, 1]))
        self.u = tf.Variable(tf.random.normal(shape=[num_features, 1]))
        self.sigma = tf.Variable(tf.random.normal(shape=[num_features, 1]))
        self.b = tf.Variable(tf.zeros(shape=[]))

    def __call__(self, x):
        x_minus_u = tf.subtract(x, self.u)
        x_minus_u_sq = tf.square(x_minus_u)
        sigma_squared = tf.square(self.sigma)
        phi = tf.exp(-1*tf.divide(x_minus_u_sq, sigma_squared))
        gaussian = tf.matmul(tf.transpose(self.w), phi)
        return tf.squeeze(tf.add(gaussian, self.b))


if __name__ == "__main__":
    data = Data()
    model = Model()
    optimizer = tf.optimizers.SGD(learning_rate=LEARNING_RATE)

    # Training
    bar = trange(NUM_BATCHES)
    for i in bar:
        with tf.GradientTape() as tape:
            x, y = data.get_batch()
            y_hat = model(x)
            loss = tf.reduce_mean(0.5 * (y_hat - y) ** 2)

        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(zip(grads, model.variables))

        bar.set_description(f"Loss @ {i} => {loss.numpy():0.6f}")
        bar.refresh()

    w_hat = np.squeeze(model.w.numpy())

    # Plot fit graph
    x = np.squeeze(data.x)
    y = np.squeeze(data.y)

    u_hat = np.squeeze(model.u.numpy())
    sigma_hat = np.squeeze(model.sigma.numpy())
    b_hat = np.squeeze(model.b.numpy())
    y_hat_val = []
    for x_single in x:
        gaussian = np.exp(-1*np.divide(np.square(x_single-u_hat), np.square(sigma_hat)))
        y_hat_val.append(np.add(np.matmul(np.transpose(w_hat), gaussian), b_hat))
       
    x_fine = np.linspace(0, 1, num=1000)
    plt.title("Fit")
    plt.ylabel("y")
    plt.xlabel("x")
    plt.plot(x, y, 'ro')
    plt.plot(x_fine, np.sin(2*np.pi*x_fine))
    plt.plot(x, y_hat_val, color='green', linestyle='dotted')
    plt.savefig('fit.png')

    # Plot the base functions
    plt.figure(0)
    plt.title("Bases for Fit")
    plt.ylabel("y")
    plt.xlabel("x")
    for i in range(NUM_FEATURES):
        phi = np.exp(-1*np.square(x_fine-np.repeat(u_hat[i], 1000))/np.square(np.repeat(sigma_hat[i], 1000)))
        plt.plot(x_fine, phi)
    plt.savefig('base.png')

