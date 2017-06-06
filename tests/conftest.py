from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).parent


@pytest.fixture()
def evi_timeseries_1():
    csv = str(HERE.joinpath('data', 'evi_timeseries_1.csv'))
    return pd.read_csv(csv)


@pytest.fixture()
def spline_timeseries_1():
    n = 1000
    x = np.arange(n).astype(float)
    y = 100 + 50 * np.sin(x) + np.random.normal(0, 5, n)
    w = np.ones_like(x)
    return pd.DataFrame({
        'x': x,
        'y': y,
        'w': w
    })
