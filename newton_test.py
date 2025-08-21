import pytest
import numpy as np
import math

import newton

def test_basic_function():
    result = newton.optimize(2.95, np.cos)
    print(f"Result: {result}")
    assert np.isclose(result, math.pi)