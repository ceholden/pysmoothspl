import numpy as np
import pytest

from pysmoothspl import sbart


@pytest.mark.parametrize('df', ('evi_timeseries_1',
                                'spline_timeseries_1'))
def test_success_timeseries(df, request):
    df = request.getfixturevalue(df)
    yhat = sbart(
        df['x'].values,
        df['y'].values,
        df['w'].values,
        0.2
    )
    # Check output is _something_ for now...
    assert np.isfinite(yhat).all()
