import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist


NUM_CLASSES = 10
BATCH_SIZE = 1000
LEARNING_RATE = 0.001
l2_rate = 1e-5


class Data(object):
    def reshape(self, x):
        return np.reshape(x, [-1, 28, 28, 1])

    def __init__(self):
        np.random.seed(31416)
        images_path = "C:/Users/1995h/PycharmProjects/ML_Sample/week_3/train-images.idx3-ubyte"
        labels_path = "C:/Users/1995h/PycharmProjects/ML_Sample/week_3/train-labels.idx1-ubyte"
        test_images_path = "C:/Users/1995h/PycharmProjects/ML_Sample/week_3/t10k-images.idx3-ubyte"
        test_lables_path = "C:/Users/1995h/PycharmProjects/ML_Sample/week_3/t10k-labels.idx1-ubyte"

        self.choices = np.random.choice(60000, 60000)

        train_images, train_labels = loadlocal_mnist(images_path, labels_path)
        test_images, test_lables = loadlocal_mnist(test_images_path, test_lables_path)
        self.training = (train_images[self.choices[:50000]] / 255).reshape(-1, 28, 28, 1) # Normalize Value
        self.validation = (train_images[self.choices[50000:]] / 255).reshape(-1, 28, 28, 1)
        self.test = (test_images / 255).reshape(-1, 28, 28, 1)
        self.labels_train = np.eye(NUM_CLASSES)[train_labels[self.choices[:50000]]]
        self.labels_validate = np.eye(NUM_CLASSES)[train_labels[self.choices[50000:]]]
        self.labels_test = np.eye(NUM_CLASSES)[test_lables]


if __name__ == "__main__":
    data = Data()
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters=5, kernel_size=(3, 3), strides=2, padding='same', activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(l2_rate)))
    model.add(keras.layers.Dropout(0.025))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(filters=8, kernel_size=(3, 3), strides=2, padding='same', activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(l2_rate)))
    model.add(keras.layers.Dropout(0.10))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(NUM_CLASSES, activation="softmax"))
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(data.training, data.labels_train, epochs=100, batch_size=BATCH_SIZE,
              validation_data=(data.validation, data.labels_validate), verbose=2)

    # Plot training accuracy
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Model Accuracy")
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # Plot history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    model.evaluate(data.test, data.labels_test)
    model.summary()
