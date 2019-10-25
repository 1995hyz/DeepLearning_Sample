import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import trange
import matplotlib.pyplot as plt

NUM_CLASSES = 10
BATCH_SIZE = 2048
NUM_BATCHES = 4096
LEARNING_RATE = 10**-2
MOMENTUM = .97


class Data(object):
    def __init__(self):
        np.random.seed(31415)
        self.index = 60000

        # Load MNIST dataset
        (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.mnist.load_data()
        self.x_train = (self.x_train.astype("float32") / 255).reshape(60000, 784, 1)    # 60k is number of train images
        self.x_test = (self.x_test.astype("float32") / 255).reshape(10000, 784, 1)  # 10k is number of test images
        self.y_train = np.eye(NUM_CLASSES, dtype=np.single)[self.y_train]
        self.y_test = np.eye(NUM_CLASSES, dtype=np.single)[self.y_test]

    def get_batch(self, batch_size=BATCH_SIZE):
        """
        Select random subset of examples for training batch
        """
        choices = np.random.choice(self.index, size=batch_size)
        return self.x_train[choices], self.y_train[choices]

    def get_test(self, batch_size=BATCH_SIZE):
        test_index = np.arrange(start=0, stop=batch_size)
        val_index = np.arrange(start=batch_size, stop=batch_size*2)
        test_and_validate = (self.x_test[test_index], self.y_test[test_index], self.x_test[val_index], self.y_test[val_index])
        return test_and_validate


def matrix_norm(matrix_in):
    n_columns = np.shape(matrix_in)[1]#Extracts number of columns
    matrix_out = np.zeros(np.shape(matrix_in))
    for i in range(n_columns):
        matrix_out[:, i] = matrix_in[:, i]/np.linalg.norm(matrix_in[:, i])
    return matrix_out

class Model(tf.Module):
    def __init__(self, dimensionality):
        self.theta_0_D = tf.random.normal(shape=[199210, 1])    # 199210 is total number of trainable parameters
        self.P = matrix_norm(tf.random.normal(shape=[199210, dimensionality]))   # Projects lower dimensional adaptive wight into higher dimensional parameter-space
        # We may need to add orthogonalization to self.P
        self.theta_d = tf.Variable(tf.zeros(shape=[dimensionality, 1]))     # This lower dimensional weight vector is the only learned parameter

    def __call__(self, image):
        # image is 784 x 1
        Theta_D = self.theta_0_D + self.P @ self.theta_d    # Projects into parameter space
        # We must now reshape Theta_D into the right weight and bias shapes. Order doesn't matter as
        # it'll just learn to the right places as long as order is consistent
        w1 = tf.reshape(Theta_D[0:156800], shape=[200, 784])     # Takes first 200*784 parameters and shapes it into weight
        b1 = tf.reshape(Theta_D[156800:157000], shape=[200, 1])    # Takes next 200*1 parameters and shapes it into bias

        w2 = tf.reshape(Theta_D[157000:197000], shape=[200, 200])  # Takes next 200*200 parameters and shapes it into weight
        b2 = tf.reshape(Theta_D[197000:197200], shape=[200, 1])  # Takes next 200*1 parameters and shapes it into bias

        w3 = tf.reshape(Theta_D[197200:199200], shape=[10, 200])  # Takes next 10*200 parameters and shapes it into weight
        b3 = tf.reshape(Theta_D[199200:], shape=[10, 1])  # Takes next 10*1 parameters and shapes it into bias
        # The paper does not specify the non-linearities used. For now lets stick with relu
        self.layer1 = tf.nn.relu(w1 @ image + b1)
        self.layer2 = tf.nn.relu(w2 @ self.layer1 + b2)
        self.output_layer = w3 @ self.layer2 + b3    # Maybe add a non-linearity here?
        return tf.squeeze(self.output_layer)


if __name__ == "__main__":
    data = Data()
    model = Model(dimensionality=1500)
    optimizer = tf.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM)

    loss_track = np.full((NUM_BATCHES, 1), np.nan)
    accuracy_track = np.full((NUM_BATCHES, 1), np.nan)

    # Training
    bar = trange(NUM_BATCHES)
    for i in bar:
        with tf.GradientTape() as tape:
            x, y = data.get_batch()
            p = model(x)

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.squeeze(y), logits=p))
            choice = np.argmax(p, axis=1)
            correct_choice = np.argmax(y, axis=1)
            accuracy = sum(choice == correct_choice)/BATCH_SIZE
            loss_track[i] = loss
            accuracy_track[i] = accuracy

        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(zip(grads, model.variables))

        bar.set_description(f"Loss @ {i} => {loss.numpy():0.6f}; Accuracy => {accuracy:0.6f}")
        bar.refresh()

    plt.plot(loss_track)
    plt.title("Loss")
    plt.show()
    plt.plot(accuracy_track)
    plt.title("Accuracy")
    plt.show()
