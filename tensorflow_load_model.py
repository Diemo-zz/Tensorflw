import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def initial_function():
    mnist = tf.keras.datasets.mnist

    (xtrain, ytrain), (xtest, ytest) = mnist.load_data()
    x_train = tf.keras.utils.normalize(xtrain, axis=1)
    x_test = tf.keras.utils.normalize(xtest, axis=1)

    n = 22
    plt.imshow(x_test[n], cmap=plt.cm.binary)

    new_model = tf.keras.models.load_model('epic_num_reader.model')

    predictions = new_model.predict(x_test)
    print(np.argmax(predictions[n]))
    plt.show()

if __name__ == "__main__":
    initial_function()
