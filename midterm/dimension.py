import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tqdm import trange
import matplotlib.pyplot as plt


NUM_CLASSES = 10
BATCH_SIZE = 8192
NUM_BATCHES = 10000#previously 30000
#LEARNING_RATE = 10**-2#2.5*10**-3#previously 10**-2
#MOMENTUM = .97
f = open("log.txt", "w+")
f.write("Log file initialized\r\n")
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

    def get_test(self):
        choices = np.random.choice(10000, size=10000)
        test_index = choices[:5000]
        val_index = choices[5000:]
        return self.x_test[test_index], self.y_test[test_index], self.x_test[val_index], self.y_test[val_index]


def matrix_norm(matrix_in):
    n_columns = np.shape(matrix_in)[1]  # Extracts number of columns
    matrix_out = np.zeros(np.shape(matrix_in))
    for i in range(n_columns):
        matrix_out[:, i] = matrix_in[:, i]/np.linalg.norm(matrix_in[:, i])
    return matrix_out


class Model(tf.Module):
    def __init__(self, dimensionality):
        #It's extremely important that we keep the seeds around if we want to save checkpoints
        self.theta_0_D = tf.random.normal(shape=[199210, 1], seed=31415)    # 199210 is total number of trainable parameters
        self.P = matrix_norm(tf.random.normal(shape=[199210, dimensionality], seed=31415))   # Projects lower dimensional adaptive wight into higher dimensional parameter-space
        # We may need to add orthogonalization to self.P
        self.theta_d = tf.Variable(tf.zeros(shape=[dimensionality, 1]))     # This lower dimensional weight vector is the only learned parameter
        self.Theta_D = 0

    def __call__(self, image, theta_flag=1):
        # image is 784 x 1
        if theta_flag:#This if statement allows us to save a lot of time when running the model for test
            self.Theta_D = self.theta_0_D + self.P @ self.theta_d    # Projects into parameter space
        # We must now reshape Theta_D into the right weight and bias shapes. Order doesn't matter as
        # it'll just learn to the right places as long as order is consistent
        w1 = tf.reshape(self.Theta_D[0:156800], shape=[200, 784])     # Takes first 200*784 parameters and shapes it into weight
        b1 = tf.reshape(self.Theta_D[156800:157000], shape=[200, 1])    # Takes next 200*1 parameters and shapes it into bias

        w2 = tf.reshape(self.Theta_D[157000:197000], shape=[200, 200])  # Takes next 200*200 parameters and shapes it into weight
        b2 = tf.reshape(self.Theta_D[197000:197200], shape=[200, 1])  # Takes next 200*1 parameters and shapes it into bias

        w3 = tf.reshape(self.Theta_D[197200:199200], shape=[10, 200])  # Takes next 10*200 parameters and shapes it into weight
        b3 = tf.reshape(self.Theta_D[199200:], shape=[10, 1])  # Takes next 10*1 parameters and shapes it into bias
        # The paper does not specify the non-linearities used. For now lets stick with relu
        self.layer1 = tf.nn.relu(w1 @ image + b1)
        self.layer2 = tf.nn.relu(w2 @ self.layer1 + b2)
        self.output_layer = w3 @ self.layer2 + b3    # Maybe add a non-linearity here?
        return tf.squeeze(self.output_layer)


if __name__ == "__main__":
    data = Data()
    dim_list = [1400,100,200,300,400,600,800,1000,1200]#10,
    check_point_dict = {100:'ckpt_dim=100_it=9375-1', 200:'ckpt_dim=200_it=5000-1', 300:'ckpt_dim=300_it=8151-1', 400:'ckpt_dim=400_it=9591-1', 600:'ckpt_dim=600_it=9500-1', 800:'ckpt_dim=800_it=9500-1', 1000:'ckpt_dim=1000_it=9500-1', 1200:'ckpt_dim=1200_it=9500-1', 1400:'ckpt_dim=1400_it=9503-1'}
    for dim in dim_list:
        model = Model(dimensionality=dim)
        if dim <= 400:
            learn_rate = 10**-2
        else:
            learn_rate = 10**-3
        optimizer = tf.optimizers.Adam(learning_rate=learn_rate)

        loss_track = np.full((NUM_BATCHES, 1), np.nan)
        accuracy_track = np.full((NUM_BATCHES, 1), np.nan)
        loss_track_test = np.full((NUM_BATCHES, 1), np.nan)
        accuracy_track_test = np.full((NUM_BATCHES, 1), np.nan)

        x_test, y_test, x_val, y_val = data.get_test()
        checkpoint_directory = "training_checkpoints"

        # Training
        bar = trange(NUM_BATCHES)
        previous_loss_test = np.Inf
        save_counter = 0

        # Try to restore the checkpoint....not working

        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        status = checkpoint.restore("training_checkpoints\\" + check_point_dict[dim])
        status.assert_existing_objects_matched()
        """
        manager = tf.train.CheckpointManager(checkpoint, './training_checkpoints/ckpt_dim=100_it=9375-1', max_to_keep=3)
        checkpoint.restore(manager.latest_checkpoint)
        print("Restored from {}".format(manager.latest_checkpoint))
        """
        #optimizer.minimize()
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

            p_test = model(x_test, 0)
            loss_test = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.squeeze(y_test), logits=p_test))
            choice_test = np.argmax(p_test, axis=1)
            correct_choice_test = np.argmax(y_test, axis=1)
            accuracy_test = sum(choice_test == correct_choice_test) / 5000
            loss_track_test[i] = loss_test
            accuracy_track_test[i] = accuracy_test

            bar.set_description(f"dim={dim}: Loss @ {i} => {loss.numpy():0.6f}, Accuracy => {accuracy:0.6f};   "
                                f""
                                f"Test Loss => {loss_test.numpy():0.6f}, Test Accuracy => {accuracy_test:0.6f}")
            bar.refresh()
            if (loss_test < previous_loss_test) and (save_counter >= NUM_BATCHES/20):#max of 20 saves NUM_BATCHES/20
                save_counter = 0
                previous_loss_test = loss_test
                checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
                checkpoint_prefix = os.path.join(checkpoint_directory, f"ckpt_dim2={dim}_it={i}")
                checkpoint.save(file_prefix=checkpoint_prefix)
            save_counter = save_counter + 1

        plt.plot(loss_track)
        plt.plot(loss_track_test)
        plt.title(f"Loss for dim={dim}")
        plt.show()

        plt.plot(accuracy_track)
        plt.plot(accuracy_track_test)
        plt.title(f"Accuracy for dim={dim}")
        plt.show()
        f.write(f"dim={dim}: Loss => {loss.numpy():0.6f}, Accuracy => {accuracy:0.6f}; Test Loss => {loss_test.numpy():0.6f}, Test Accuracy => {accuracy_test:0.6f}\r\n")
    f.close()