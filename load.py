# nn library https://github.com/GeyaWang/py-nn.git
from nn.models import Sequential

import mnist
import random
import numpy as np

SAMPLE_SIZE = 10000


def main():
    model = Sequential.load('training.ptd')
    model.summary()

    _, _, x, y = mnist.mnist('MNIST')

    # shuffle data
    zipped_data = list(zip(x, y))
    random.shuffle(zipped_data)
    x, y = zip(*zipped_data)
    x = np.array(x)[:SAMPLE_SIZE]
    y = np.array(y)[:SAMPLE_SIZE]

    x_test = np.reshape(x, (*x.shape, 1))

    y_pred = model.predict(x_test)
    y_pred_argmax = np.argmax(y_pred, axis=1)

    n_correct = np.sum(y_pred_argmax == y)

    print()
    print(f'accuracy: {n_correct}/{SAMPLE_SIZE} ({n_correct / SAMPLE_SIZE * 100}%)')
    # accuracy: 9852/10000 (98.52 %)


if __name__ == '__main__':
    main()
