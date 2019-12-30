"""
Module holding test code to load a tensorflow model
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def initial_function():
    """
    Initial function to load in a model
    :return: None
    """
    mnist = tf.keras.datasets.mnist

    (_, _), (xtest, _) = mnist.load_data()
    x_test = tf.keras.utils.normalize(xtest, axis=1)

    index_number = 22
    plt.imshow(x_test[index_number], cmap=plt.cm.binary)  # pylint: disable=no-member

    new_model = tf.keras.models.load_model('epic_num_reader.model')

    predictions = new_model.predict(x_test)
    print(np.argmax(predictions[index_number]))
    plt.show()


if __name__ == "__main__":
    initial_function()
