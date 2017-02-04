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

def test_propagation(cell1):
    cell = cell1

    previous_s = np.array([1, 2])
    previous_h = np.array([3, 4])
    x = np.array([5])
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


 