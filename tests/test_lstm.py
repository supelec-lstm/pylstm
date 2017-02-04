import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../..')))

import numpy as np
import pytest
from lstm import *

@pytest.fixture
def cell1():
    dim_s = 2
    dim_x = 1
    weights = Weights(dim_s, dim_x)
    weights.wg = np.ones((dim_s, dim_x + dim_s))
    weights.wi = np.ones((dim_s, dim_x + dim_s))
    weights.wo = np.ones((dim_s, dim_x + dim_s))

    return LstmCell(weights)

def test_propagation1(cell1):
    cell = cell1

    previous_s = np.array([[1], [2]])
    previous_h = np.array([[3], [4]])
    x = np.array([[5]])
    cell.propagate(previous_s, previous_h, x)

    g = np.array([sigmoid(12)] * 2)
    assert np.allclose(cell.g, g)
    i = np.array([np.tanh(12)] * 2)
    assert np.allclose(cell.i, i)
    s = previous_s + g*i
    assert np.allclose(cell.s, s)
    o = np.array([sigmoid(12)] * 2)
    assert np.allclose(cell.o, o)
    h = o*np.tanh(s)
    assert np.allclose(cell.h, h)

def test_backpropagation1(cell1):
    cell = cell1

    previous_s = np.array([[1], [2]])
    previous_h = np.array([[3], [4]])
    x = np.array([[5]])
    cell.propagate(previous_s, previous_h, x)

    ds = np.array([[2], [1]])
    dh = np.array([[4], [3]])
    y = np.array([[0], [6]])
    cell.backpropagate(ds, dh, y)

    dh = dh + (cell.h - y)
    ds = ds + dh*cell.o*(1-cell.l**2)
    assert np.allclose(cell.ds, ds)

    dh = np.dot(cell.weights.wg.T, ds*cell.i*(1-cell.g)*cell.g) + \
        np.dot(cell.weights.wi.T, ds*cell.g*(1-cell.i**2)) + \
        np.dot(cell.weights.wo.T, dh*cell.l*(1-cell.o)*cell.o)
    assert np.allclose(cell.dh, dh[:2,:])