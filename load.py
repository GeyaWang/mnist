# nn library https://github.com/GeyaWang/py-nn.git
from nn.models import Sequential

import mnist
import random
import numpy as np


def main():
    model = Sequential.load('mnist_training.ptd')
    model.summary()


if __name__ == '__main__':
    main()
