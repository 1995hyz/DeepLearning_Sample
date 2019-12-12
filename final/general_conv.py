import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tqdm import trange
import matplotlib.pyplot as plt
import math


NUM_CLASSES = 10
BATCH_SIZE = 2048   # Lower is easier on memory
NUM_BATCHES = 4000  # previously 30000
# LEARNING_RATE = 10**-2#2.5*10**-3#previously 10**-2
# MOMENTUM = .97
f = open("log.txt", "w+")
f.write("Log file initialized\r\n")

cifar100_x_train = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
cifar100_y_train = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
cifar100_x_test = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
cifar100_y_test = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}


class Data(object):
    def __init__(self, dataset_train, dataset_test):
        np.random.seed(31415)
        self.index = 500
        self.x_train = dataset_train[0]
        self.y_train = dataset_train[1]
        self.x_test = dataset_test[0]
        self.y_test = dataset_test[1]
        self.x_train = (self.x_train.astype("float16") / 255).reshape(500, 784, 3)    # 60k is number of train images
        self.x_test = (self.x_test.astype("float16") / 255).reshape(100, 784, 3)  # 10k is number of test images

    def get_batch(self, batch_size=BATCH_SIZE):
        """
        Select random subset of examples for training batch
        """
        choices = np.random.choice(self.index, size=batch_size)
        return self.x_train[choices], self.y_train[choices]

    def get_test(self):
        choices = np.random.choice(200, size=200)
        test_index = choices[:100]
        val_index = choices[100:]
        return self.x_test[test_index], self.y_test[test_index], self.x_test[val_index], self.y_test[val_index]


def matrix_norm(matrix_in):
    n_columns = np.shape(matrix_in)[1]  # Extracts number of columns
    matrix_out = np.zeros(np.shape(matrix_in))
    for i in range(n_columns):
        matrix_out[:, i] = matrix_in[:, i]/np.linalg.norm(matrix_in[:, i])
    return matrix_out


def conv_layer(x, shape, W_parameters, b_parameters):
    W = tf.reshape(W_parameters, shape)
    b = tf.reshape(b_parameters, [shape[3]])
    return tf.nn.relu(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') + b)


class Model(tf.Module):
    # I think we'll have to change all our datatypes to float16 because casting will fuck up the gradient
    def __init__(self, dimensionality, load_flag=0):
        # It's extremely important that we keep the seeds around if we want to save checkpoints
        # This allows us to not overwrite our saved weights
        if not load_flag:#To speed up training time, for higher dimensionalities we overwrite theta_0_D to the final weight vector
            #from lower dimensions. This should not affect the final results of the dimension matmul
            self.theta_0_D = tf.random.normal(shape=[431040, 1], seed=31415, dtype="float16")    # 199210 is total number of trainable parameters
        self.P = matrix_norm(tf.random.normal(shape=[431040, dimensionality], seed=31415, dtype="float16"))   # Projects lower dimensional adaptive wight into higher dimensional parameter-space
        self.theta_d = tf.Variable(tf.zeros(shape=[dimensionality, 1], dtype="float16"))# This lower dimensional weight vector is the only learned parameter
        self.Theta_D = 0
        self.dense_w = list()
        self.dense_b = list()
        for i in range(10):
            # I'm assuming we're using 10 datasets with 10 classes each
            self.dense_w.append(tf.Variable(tf.initializers.GlorotNormal()(shape=[10, 6144], dtype="float16")))
            self.dense_b.append(tf.Variable(tf.initializers.GlorotNormal()(shape=[10, 6144], dtype="float16")))

    def __call__(self, image, theta_flag=1, dense_model=None):
        # This if statement allows us to save a lot of time when running the model for test
        if theta_flag:
            self.Theta_D = self.theta_0_D + self.P @ self.theta_d    # Projects into parameter space
        # We must now reshape Theta_D into the right weight and bias shapes. Order doesn't matter as
        # it'll just learn to the right places as long as order is consistent
        W = list()
        W.append(self.Theta_D[0:863])
        W.append(self.Theta_D[864:19295])
        W.append(self.Theta_D[19296:93023])
        W.append(self.Theta_D[93024:387935])

        b = list()
        b.append(self.Theta_D[387936:388031])
        b.append(self.Theta_D[388032:390079])
        b.append(self.Theta_D[390080:398271])
        b.append(self.Theta_D[398272:431039])

        self.layer1 = conv_layer(image, shape=[3, 3, 3, 32], W_parameters=W[0], b_parameters=b[0])

        self.layer2 = conv_layer(self.layer1, shape=[3, 3, 32, 64], W_parameters=W[1], b_parameters=b[1])
        self.pool2 = tf.nn.max_pool(self.layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        self.layer3 = conv_layer(self.pool2, shape=[3, 3, 64, 128], W_parameters=W[2], b_parameters=b[2])

        self.layer4 = conv_layer(self.layer3, shape=[3, 3, 128, 256], W_parameters=W[3], b_parameters=b[3])
        self.pool4 = tf.nn.max_pool(self.pool4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # I put a max on each channel here. Andrey said it was a good idea
        # I reduce along the channels and then reshape into earlier channels by later channels
        self.reduced = tf.reshape(tf.reduce_max(self.pool4, axis=3), shape=[6144, 32768])

        # I looked online and it seems like the easiest way to tell tensorflow not to do gradient descent on certain
        # variables is to specify in the optimizer
        # See: https://stackoverflow.com/questions/33727935/how-to-use-stop-gradient-in-tensorflow
        self.final = self.dense_w[dense_model]@self.reduced + self.dense_b[dense_model]

        return self.final

    #def save_weights(self):
    #    self.theta_0_D = self.Theta_D #saves last combined weight vector


def data_splitter():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data(label_mode='fine')
    for i in range(len(x_train)):
        class_group = math.floor(y_train[i] / 10)
        class_pos = y_train[i] % 10
        cifar100_x_train[class_pos].append(x_train[i])
        cifar100_y_train[class_pos].append(np.eye(10, dtype=np.single)[y_train[i] - class_group * 10])
    for i in range(len(x_test)):
        class_group = math.floor(y_test[i] / 10)
        class_pos = y_train[i] % 10
        cifar100_x_train[class_pos].append(x_test[i])
        cifar100_y_train[class_pos].append(np.eye(10, dtype=np.single)[y_test[i] - class_group * 10])


def flip(image: tf.Tensor) -> tf.Tensor:
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return image


def color(image: tf.Tensor) -> tf.Tensor:
    image = tf.image.random_hue(image, 0.08)
    image = tf.image.random_saturation(image, 0.6, 1.6)
    image = tf.image.random_brightness(image, 0.05)
    image = tf.image.random_contrast(image, 0.7, 1.3)
    return image


def rotate(image: tf.Tensor) -> tf.Tensor:
    return tf.image.rot90(image, tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))


def zoom(image: tf.Tensor) -> tf.Tensor:
    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    def random_crop(img):
        # Create different crops for an image
        crops = tf.image.crop_and_resize([img], boxes=boxes, box_ind=np.zeros(len(scales)), crop_size=(32, 32))
        # Return a random crop
        return crops[tf.random_uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]

    choice = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    # Only apply cropping 50% of the time
    return tf.cond(choice < 0.5, lambda: image, lambda: random_crop(image))


if __name__ == "__main__":
    data_splitter()
    dim_list = [400, 600, 800, 1000, 1200, 1400]
    for dim in dim_list:
        learn_rate = dim**-1 # This seems to be an ok learning rate calculation
        load = 0
        if dim > 400:
            load = 1
        model = Model(dimensionality=dim, load_flag=load)
        if dim > 400:
            # This line allows us to speed up training by starting higher dimensionality calculations
            # from the lower dimensionality result
            model.theta_0_D = old_weight
        optimizer = tf.optimizers.Adam(learning_rate=learn_rate)

        # 9 because we initially train on only 9 datasets
        loss_track = np.full((NUM_BATCHES, 9), np.nan)
        accuracy_track = np.full((NUM_BATCHES, 9), np.nan)

        loss_track_test = np.full((NUM_BATCHES, 9), np.nan)
        accuracy_track_test = np.full((NUM_BATCHES, 9), np.nan)

        checkpoint_directory = "training_checkpoints"

        # Training
        bar = trange(NUM_BATCHES)
        previous_loss = np.Inf
        save_counter = 0

        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        # status = checkpoint.restore("training_checkpoints\\" + check_point_dict[dim])
        # status.assert_existing_objects_matched()
        """
        manager = tf.train.CheckpointManager(checkpoint, './training_checkpoints/ckpt_dim=100_it=9375-1', max_to_keep=3)
        checkpoint.restore(manager.latest_checkpoint)
        print("Restored from {}".format(manager.latest_checkpoint))
        """
        for j_epoch in bar:
            # This for loop switches between dataset. We only do full training on 9 datasets
            # and leave the tenth as control
            for i_dataset in range(9):
                data = Data((cifar100_x_train[i_dataset], cifar100_y_train[i_dataset]),
                            (cifar100_x_test[i_dataset], cifar100_y_test[i_dataset]))
                with tf.GradientTape() as tape:
                    x, y = data.get_batch()
                    dataset = tf.data.Dataset.from_tensor_slices(x)
                    ## Data augmentation stuff
                    augmentations = [flip, color, zoom, rotate]
                    for f in augmentations:
                        dataset = dataset.map(
                            lambda m: tf.cond(tf.random_uniform([], 0, 1) > 0.75, lambda: f(m), lambda: m),
                            num_parallel_calls=4)
                    dataset = dataset.map(lambda m: tf.clip_by_value(m, 0, 1))
                    p = model(dataset, dense_model=i_dataset)     # Chooses which dense layer we're using


                    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.squeeze(y), logits=p))
                    choice = np.argmax(p, axis=1)
                    correct_choice = np.argmax(y, axis=1)
                    accuracy = sum(choice == correct_choice)/BATCH_SIZE
                    loss_track[j_epoch][i_dataset] = loss
                    accuracy_track[j_epoch][i_dataset] = accuracy

                grads = tape.gradient(loss, model.variables)#This line should only access the dims that matter but its ok. There will only be a gradient for the datasets we train on
                optimizer.apply_gradients(zip(grads, model.variables))

                x_test, y_test = data.get_batch()
                p_test = model(x_test, 0, i_dataset)
                loss_test = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.squeeze(y_test), logits=p_test))
                choice_test = np.argmax(p_test, axis=1)
                correct_choice_test = np.argmax(y_test, axis=1)
                accuracy_test = sum(choice_test == correct_choice_test) / 1000 #1000 is size of test set
                loss_track_test[j_epoch][i_dataset] = loss_test
                accuracy_track_test[j_epoch][i_dataset] = accuracy_test

                bar.set_description(f"dim={dim}: Loss for dataset {i_dataset} @ {j_epoch} => {loss.numpy():0.6f}, Accuracy => {accuracy:0.6f}"
                                    f""
                                    f"Test Loss => {loss_test.numpy():0.6f}, Test Accuracy => {accuracy_test:0.6f}")
                bar.refresh()
                if (loss.numpy < previous_loss) and (save_counter >= (NUM_BATCHES*9)/20):# max of 20 saves. This ensures only 20 saves
                    save_counter = 0
                    previous_loss = loss.numpy
                    checkpoint_prefix = os.path.join(checkpoint_directory, f"ckpt_dim2={dim}_it={j_epoch}")
                    checkpoint.save(file_prefix=checkpoint_prefix)
                save_counter = save_counter + 1

        checkpoint_prefix = os.path.join(checkpoint_directory, f"ckpt_dim2={dim}_it=end")
        checkpoint.save(file_prefix=checkpoint_prefix)
        old_weight = model.Theta_D

        plt.plot(loss_track)
        plt.plot(loss_track_test)
        plt.title(f"Loss for dim={dim}")
        plt.show()

        plt.plot(accuracy_track)
        plt.plot(accuracy_track_test)
        plt.title(f"Accuracy for dim={dim}")

        ## Training on final dataset happens here
        model = Model(dimensionality=dim, load_flag=load)
        if dim > 400:
            model.theta_0_D = old_weight  # This line allows us to speed up training by starting higher dimensionality calculations from the lower dimensionality result
        optimizer = tf.optimizers.Adam(learning_rate=learn_rate)

        loss_track = np.full((NUM_BATCHES, 9), np.nan)  # 9 because we initially train on only 9 datasets
        accuracy_track = np.full((NUM_BATCHES, 9), np.nan)

        # Did you partition test data as well?
        loss_track_test = np.full((NUM_BATCHES, 1), np.nan)
        accuracy_track_test = np.full((NUM_BATCHES, 1), np.nan)

        x_test, y_test, x_val, y_val = data.get_test()
        checkpoint_directory = "training_checkpoints"

        # Training
        bar = trange(NUM_BATCHES)
        previous_loss = np.Inf
        save_counter = 0

        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        for j_epoch in bar:
            for i_dataset in range(9):#This for loop switches between datasets. We only do full training on 9 datasets and leave the tenth as control
                with tf.GradientTape() as tape:
                    x, y = data.get_batch()#We need to change this to work with the new dataset
                    p = model(x, dense_model=i_dataset)#Chooses which dense layer we're using

                    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.squeeze(y), logits=p))
                    choice = np.argmax(p, axis=1)
                    correct_choice = np.argmax(y, axis=1)
                    accuracy = sum(choice == correct_choice)/BATCH_SIZE
                    loss_track[j_epoch][i_dataset] = loss
                    accuracy_track[j_epoch][i_dataset] = accuracy

                grads = tape.gradient(loss, model.variables)#This line should only access the dims that matter but its ok. There will only be a gradient for the datasets we train on
                optimizer.apply_gradients(zip(grads, model.variables))
                '''
                p_test = model(x_test, 0)
                loss_test = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.squeeze(y_test), logits=p_test))
                choice_test = np.argmax(p_test, axis=1)
                correct_choice_test = np.argmax(y_test, axis=1)
                accuracy_test = sum(choice_test == correct_choice_test) / 5000
                loss_track_test[j_epoch] = loss_test
                accuracy_track_test[j_epoch] = accuracy_test
                '''
                bar.set_description(f"dim={dim}: Loss for dataset {i_dataset} @ {j_epoch} => {loss.numpy():0.6f}, Accuracy => {accuracy:0.6f}")
                                    #f""
                                    #f"Test Loss => {loss_test.numpy():0.6f}, Test Accuracy => {accuracy_test:0.6f}")
                bar.refresh()
                if (loss.numpy < previous_loss) and (save_counter >= (NUM_BATCHES*9)/20):# max of 20 saves. This ensures only 20 saves
                    save_counter = 0
                    previous_loss = loss.numpy
                    checkpoint_prefix = os.path.join(checkpoint_directory, f"ckpt_dim2={dim}_it={j_epoch}")
                    checkpoint.save(file_prefix=checkpoint_prefix)
                save_counter = save_counter + 1
    f.close()
