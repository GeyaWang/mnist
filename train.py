import numpy as np
import mnist
import os

# nn library https://github.com/GeyaWang/py-nn.git
from nn.models import Sequential
from nn.layers import Dense, Conv2D, Flatten, Activation, Dropout
from nn.losses import CrossEntropy
from nn.activations import ReLU, SoftMax
from nn.optimisers import Adam

FILEPATH = 'training.ptd'


def main():
    if os.path.isfile(FILEPATH):
        model = Sequential.load(FILEPATH)
    else:
        model = Sequential()
        model.add(Conv2D(64, 3, input_shape=(28, 28, 1)))
        model.add(Activation(ReLU()))
        model.add(Conv2D(32, 3))
        model.add(Activation(ReLU()))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(10))
        model.add(Activation(SoftMax()))

    model.compile(Adam(), CrossEntropy())

    x, y, _, _ = mnist.mnist('MNIST')
    x_train = np.reshape(x, (*x.shape, 1))
    y_train = np.identity(10)[y]

    model.fit(x_train, y_train, batch_size=32, running_mean_err=100, graph=2, epochs=3, save_filepath=FILEPATH)


if __name__ == '__main__':
    main()
