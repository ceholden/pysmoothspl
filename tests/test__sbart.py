import numpy as np
import pytest

from pysmoothspl import sbart


@pytest.mark.parametrize('df', ('evi_timeseries_1',
                                'spline_timeseries_1'))
def test_success_timeseries(df, request):
    df = request.getfixturevalue(df)
    knots, coef, yhat, x_min, x_range = sbart(
        df['x'].values,
        df['y'].values,
        df['w'].values,
        0.2
    )
    # Check output is _something_ for now...
    # TODO: check all outputs against R, either known values
    #       or by running rpy2
    assert np.isfinite(yhat).all()
