#!/usr/bin/env python
# coding: utf-8
"""Demonstrate the simple perceptron algorithm."""
from random import random
import numpy as np

# define the input shape
input_shape = 3
# define your inputs, these inputs are same for all the 2-variable logic gates
inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]


def augment_inputs(inputs):
    """Augment the inputs, add +1 as the 1st dimension."""
    return np.array([(1, x[0], x[1]) for x in inputs])


print(augment_inputs(inputs))


def random_initialize_weights(input_shape):
    """Randomly initialize the weights."""
    return np.array([random() for i in range(input_shape)])


def zero_initialize_weights(input_shape):
    """Zero initialize the weights."""
    return np.zeros(input_shape)


# define your output classes or labels; this is a simple case of AND gate
classes = [0, 0, 0, 1]


def simple_perceptron(X, W, classes, itr=1):
    """Simple perceptron algorithm"""
    print(X, W)
    for i in range(itr):
        sat = list()
        for ind, x in enumerate(X):
            x = np.array(x)
            if W.dot(x) <= 0 and classes[ind] == 1:
                W += x
                sat.append(False)
            elif W.dot(x) > 0 and classes[ind] == 0:
                W -= x
                sat.append(False)
            else:
                sat.append(True)
            print(W.dot(x), classes[ind], ind)
        if np.all(sat):
            print('i =', i)
            return W
    return W


# run the simple perceptron algorithm with zero initialization
augmented_inputs = augment_inputs(inputs)
W = zero_initialize_weights(input_shape)
final_W = simple_perceptron(augmented_inputs, W, classes, 10)

# run the simple perceptron algorithm with random initialization
augmented_inputs = augment_inputs(inputs)
W = random_initialize_weights(input_shape)
classes = [0, 1, 1, 1]
final_W = simple_perceptron(augmented_inputs, W, classes, 10)
print(final_W)


# this below outputs are for the OR gate
classes = [0, 1, 1, 1]
W = random_initialize_weights(input_shape)
final_W = simple_perceptron(augmented_inputs, W, classes, 10)
print(final_W)
