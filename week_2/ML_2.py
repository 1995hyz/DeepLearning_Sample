import numpy as np
import tensorflow as tf

from tqdm import trange

import matplotlib.pyplot as plt

NUM_FEATURES = 11
NUM_SAMP = 2000
BATCH_SIZE = 64
NUM_BATCHES = 10000
LEARNING_RATE = 0.01
NODE_NUM = 256
LAMBDA = 0.01


class Data(object):
    def __init__(self, num_features=NUM_FEATURES, num_samp=NUM_SAMP):
        """
        Draw random weights and bias. Project vectors in R^NUM_FEATURES
        onto R with said weights and bias.
        """
        num_samp = NUM_SAMP
        noise = 0.05
        a = 0.5
        b = 0.20
        np.random.seed(31415)

        # We're going to learn these paramters
        self.index = np.arange(num_samp*2)
        th = np.sort(np.random.rand(1, num_samp) * 9)
        self.x1 = np.float32((a + np.random.rand(1, num_samp) * noise) * np.exp(b * th) * np.cos(th)).flatten()
        self.y1 = np.float32((a + np.random.rand(1, num_samp) * noise) * np.exp(b * th) * np.sin(th)).flatten()

        self.x2 = np.float32(-(a + np.random.rand(1, num_samp) * noise) * np.exp(b * th) * np.cos(th)).flatten()
        self.y2 = np.float32(-(a + np.random.rand(1, num_samp) * noise) * np.exp(b * th) * np.sin(th)).flatten()

        self.x = np.concatenate((self.x1, self.x2))
        self.y = np.concatenate((self.y1, self.y2))
        self.label = np.concatenate((np.zeros(num_samp), np.ones(num_samp)))
        plt.plot(self.x1, self.y1, 'ro')
        plt.plot(self.x2, self.y2, 'go')
        plt.show()

    def get_batch(self, num_samp=NUM_SAMP, batch_size=BATCH_SIZE):
        """
        Select random subset of examples for training batch
        """
        choices = np.random.choice(self.index, size=batch_size)
        return self.x[choices], self.y[choices], self.label[choices]


class Model(tf.Module):
    def __init__(self, num_features=NUM_FEATURES):
        """
        multi-level perceptron network
        """
        self.weight_layer1 = tf.Variable(tf.random.normal(shape=[2, NODE_NUM]))
        self.bias_layer1 = tf.Variable(tf.zeros(shape=[1, NODE_NUM]))
        self.weight_layer2 = tf.Variable(tf.random.normal(shape=[NODE_NUM, NODE_NUM]))
        self.bias_layer2 = tf.Variable(tf.zeros(shape=[1, NODE_NUM]))
        self.weight_output = tf.Variable(tf.random.normal(shape=[NODE_NUM, 1]))
        self.bias_output = tf.Variable(tf.zeros(shape=[1, 1]))

    def __call__(self, x):
        relu_layer1 = tf.nn.relu(tf.matmul(x, self.weight_layer1) + self.bias_layer1)
        relu_layer2 = tf.nn.relu(tf.matmul(relu_layer1, self.weight_layer2) + self.bias_layer2)
        output_layer = tf.matmul(relu_layer2, self.weight_output) + self.bias_output
        return tf.squeeze(output_layer)


if __name__ == "__main__":
    data = Data()
    model = Model()
    optimizer = tf.optimizers.SGD(learning_rate=LEARNING_RATE)

    loss_track = []

    # Training
    bar = trange(NUM_BATCHES)
    for i in bar:
        with tf.GradientTape() as tape:
            x, y, label = data.get_batch()
            input_matrix = np.array([x, y])
            p = model(tf.transpose(input_matrix))
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(label, tf.float32), logits=tf.cast(p, tf.float32))) +\
                   LAMBDA * tf.reduce_sum((tf.nn.l2_loss(model.weight_layer1) + tf.nn.l2_loss(model.weight_layer2) +
                                           tf.nn.l2_loss(model.weight_output) + tf.nn.l2_loss(model.bias_layer1) +
                                           tf.nn.l2_loss(model.bias_layer2) + tf.nn.l2_loss(model.bias_output)))
            loss_track.append(loss)

        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(zip(grads, model.variables))

        bar.set_description(f"Loss @ {i} => {loss.numpy():0.6f}")
        bar.refresh()

    plt.plot(loss_track)
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[10, 10])

    # set up grid of points to make the contour plots
    xv, yv = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
    x_train = data.x.flatten()
    y_train = data.y.flatten()
    pairs = np.float32(np.vstack((xv.flatten(), yv.flatten())))
    label_new = np.float32(tf.math.sigmoid(model(tf.transpose(pairs)))).flatten()
    plt.subplot(3, 1, 2)
    contour = plt.contourf(xv, yv, np.reshape(label_new, [500, 500]), [0, 0.5, 1], extend='both')
    plt.plot(x_train[:NUM_SAMP], y_train[:NUM_SAMP], 'ro')
    plt.plot(x_train[NUM_SAMP:], y_train[NUM_SAMP:], 'yo')
    plt.show()
