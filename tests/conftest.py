from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    import rpy2  # noqa
    from rpy2.robjects.packages import importr
    from rpy2.robjects import numpy2ri, pandas2ri

    print('rpy2')
    numpy2ri.activate()
    pandas2ri.activate()

    Rstats = importr('stats')
except Exception:
    HAS_RPY2 = False
else:
    HAS_RPY2 = True

HERE = Path(__file__).parent

SEED = 12345
SPAR_VALUES = (0.1, 0.2, 0.5, 0.75, 1.0, )


@pytest.fixture(scope='module')
def evi_timeseries_1():
    """ pd.DataFrame with columns x, y, w and smoothed answers in "spl_{spar}"
    """
    csv = str(HERE.joinpath('data', 'evi_timeseries_1.csv'))

    if HAS_RPY2:
        # Calculate and overwrite, helping to track for changes in how R
        # calculates the values (i.e., because git would show modified)
        df = pd.read_csv(csv)
        df_ans = _rpy2_smooth_spline(df, SPAR_VALUES)
        df_ans.to_csv(csv, index=False)
        return df_ans
    else:
        return pd.read_csv(csv)


@pytest.fixture(scope='module')
def sine_timeseries_1():
    csv = str(HERE.joinpath('data', 'sinwave_with_noise_seed12345.csv'))

    if HAS_RPY2:
        np.random.seed(SEED)
        n = 1000
        x = np.arange(n).astype(float)
        y = 100 + 50 * np.sin(x) + np.random.normal(0, 5, n)
        w = np.ones_like(x)
        df = pd.DataFrame({
            'x': x,
            'y': y,
            'w': w
        })
        df_ans = _rpy2_smooth_spline(df, SPAR_VALUES)
        df_ans.to_csv(csv)
        return df_ans
    else:
        return pd.read_csv(csv)

    return df_ans


def _rpy2_smooth_spline(df, spars):
    for spar in spars:
        spl = Rstats.smooth_spline(df['x'].values,
                                   df['y'].values,
                                   df['w'].values,
                                   spar=spar)
        cname = 'spl_{spar:02f}'.format(spar=spar)
        df[cname] = Rstats.predict_smooth_spline(spl, df['x'].values)[1]

    return df
