"""
Module holding the initial tensorflow stuff
"""
import tensorflow as tf


def test_function(length):
    """

    :return: array of ones
    """
    if not isinstance(int, length):
        print("Not a proper lenght")
        return []
    return [1]*10


def initial_function():
    """
    Initial setup function
    :return: None
    """
    mnist = tf.keras.datasets.mnist
    (xtrain, ytrain), (_, _) = mnist.load_data()
    x_train=tf.keras.utils.normalize(xtrain, axis=1)
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, ytrain, epochs=10)

    model.save('epic_num_reader.model')

if __name__ == "__main__":
    initial_function()
