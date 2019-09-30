#!/bin/python3.6

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys

from tqdm import trange

NUM_SAMP = 2000  # This is the number of samples per spiral so really there are 2*NUM_SAMP samples
BATCH_SIZE = 100
NUM_BATCHES = 30000
LEARNING_RATE = 0.00168
NUM_SPIRALS = 1.5
L2_lambda = 0.01


class Data(object):
    def __init__(self, num_samp=NUM_SAMP, num_spiral=NUM_SPIRALS + .25):
        num_samp = NUM_SAMP
        sigma = .1
        np.random.seed(31416)

        theta1 = np.linspace(np.pi / 2, 2 * np.pi * num_spiral, num_samp)
        theta2 = np.linspace(np.pi / 2, 2 * np.pi * num_spiral, num_samp)
        w = 1
        r1 = w * theta1
        r2 = -w * theta2
        self.x1 = np.float32(r1 * np.cos(theta1)) + sigma * np.random.normal(
            size=(1, num_samp))  # Defaults to low of zero, high of 1
        self.y1 = np.float32(r1 * np.sin(theta1)) + sigma * np.random.normal(size=(1, num_samp))

        self.x2 = np.float32(r2 * np.cos(theta2)) + sigma * np.random.normal(
            size=(1, num_samp))  # Defaults to low of zero, high of 1
        self.y2 = np.float32(r2 * np.sin(theta2)) + sigma * np.random.normal(size=(1, num_samp))

    def get_batch(self, batch_size=BATCH_SIZE, num_samp=NUM_SAMP):
        index1 = np.random.randint(0, high=num_samp, size=int(batch_size/2))
        index2 = np.random.randint(0, high=num_samp, size=int(batch_size/2))
        out_matrix = np.array([np.concatenate((self.x1[0, index1], self.x2[0, index2])),# Spiral 1 gets a 1, spiral2 gets a 0
            np.concatenate((self.y1[0, index1], self.y2[0, index2])),
            np.concatenate((np.ones(int(batch_size/2)), np.zeros(int(batch_size/2))))]).T  # Last column is correct answer
        np.random.shuffle(out_matrix)
        return out_matrix


class Model(tf.Module):
    def __init__(self, batch_size=BATCH_SIZE):
        dim = np.array([17,17,17,17])#8x4
        self.dlen = len(dim)
        dlen = self.dlen

        self.A = [0] * 16  # This should initialize A as array of 7 0s
        self.b = [0] * 16  # This should initialize b

        self.A[0] = tf.Variable(tf.random.truncated_normal(shape=[dim[0], 2]), name='A_0') #/ dim[0]
        self.b[0] = tf.Variable(tf.random.truncated_normal(shape=[dim[0], 1]), name='b_0')# / dim[0]

        for i in range(1, dlen-1):
            self.A[i] = tf.Variable(tf.random.truncated_normal(shape=[dim[i], dim[i - 1]]), name='A_'+str(i))
            self.b[i] = tf.Variable(tf.random.truncated_normal(shape=[dim[i], 1]), name='b_'+str(i))

        self.A[dlen-1] = tf.Variable(tf.random.truncated_normal(shape=[1, dim[dlen-2]]), name='A_final')
        self.b[dlen-1] = tf.Variable(tf.random.truncated_normal(shape=[1, 1]), name='b_final')

    def __call__(self, D):
        # print("D=",str(D.shape))
        self.layer = [0] * self.dlen  # This should initialize the layers
        self.layer[0] = D
        for j in range(self.dlen-1):
            self.layer[j + 1] = tf.nn.leaky_relu(self.A[j] @ self.layer[j] + self.b[j], alpha=.2)
        self.final_layer = tf.sigmoid(self.A[self.dlen-1] @ self.layer[self.dlen-1] + self.b[self.dlen-1])
        return self.final_layer


if __name__ == "__main__":
    # Plotting spiral
    data = Data()
    plt.plot(data.x1, data.y1, 'ro')
    plt.plot(data.x2, data.y2, 'bo')
    plt.title("Full Data-set")
    plt.show()

    model = Model()
    optimizer = tf.optimizers.SGD(learning_rate=LEARNING_RATE)
    Loss_array = np.zeros((NUM_BATCHES,1))
    bar = trange(NUM_BATCHES)
    for i in bar:
        with tf.GradientTape() as tape:
            D = data.get_batch()
            p = D[:, 2]
            p_hat = model(D[:, 0:2].T)  # Extracts first 2 columns and feeds it into the model
            summ = tf.constant(0.0)
            for k in range(model.dlen):
                summ += tf.reduce_sum(model.A[k] ** 2)
            L2_loss = L2_lambda*summ
            Loss1 = -p.T * tf.math.log(tf.clip_by_value(p_hat,1e-5,1))#This clipping helps with numerical instability of log function
            Loss2 = -(1 - p.T) * tf.math.log(tf.clip_by_value(1 - p_hat,1e-5,1))
            loss = tf.reduce_mean(Loss1 + Loss2) + L2_loss
            Loss_array[i] = loss
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(zip(grads, model.variables))

        bar.set_description(f"Loss @ {i} => {loss.numpy():0.6f}")
        bar.refresh()

    # Plotting stuff
    D = data.get_batch(batch_size=BATCH_SIZE)
    p = D[:, 2].T
    p_hat = model(D[:, 0:2].T)
    answer = np.zeros((BATCH_SIZE,1))
    for i in range(BATCH_SIZE):
        if (tf.cast(p_hat[0, i], tf.float32) >= .5):  # If in spiral1
            plt.plot(D[i, 0], D[i, 1], 'ro')
            answer[i] = 1
        else:
            plt.plot(D[i, 0], D[i, 1], 'bo')
            answer[i] = 0
    #Now I add contour lines to the plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[10, 10])

    # set up grid of points to make the contour plots
    x = y = np.linspace(-10, 10, 1000).astype(np.float32)
    xx, yy = np.meshgrid(x, y)
    grid = np.array(list(zip(xx.flatten(), yy.flatten())))
    cont = np.zeros(1000**2)
    for i in range(1000):
        cont[i*100:((i+1)*100)-1] = model(grid[i*100:((i+1)*100)-1, :].T)
    plot_data = np.reshape(cont, (1000, 1000))

    # plot contour from prediction
    plt.contourf(xx, yy, plot_data, [0, 0.5, 1])
    plt.title("Results of trained model")
    plt.show()

    plt.plot(Loss_array)
    plt.title("Loss Curve")
    plt.show()
    Error = np.abs(np.round(p_hat) - p)
    Error_count = np.sum(Error)
