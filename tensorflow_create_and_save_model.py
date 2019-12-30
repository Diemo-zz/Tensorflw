import tensorflow as tf
import functools


def test_function(length):
    """

    :return: array of ones
    """
    if not isinstance(int, length):
        print("Not a proper lenght")
        return []
    return [p for p in range(length)]
#  asdf Break pep8

mnist = tf.keras.datasets.mnist
(xtrain, ytrain), (xtest, ytest) = mnist.load_data()
x_train = tf.keras.utils.normalize(xtrain, axis=1)
x_test = tf.keras.utils.normalize(xtest, axis=1)
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
