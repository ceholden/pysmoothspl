import numpy as np
import pytest

from pysmoothspl import sbart


@pytest.mark.parametrize('df', ('evi_timeseries_1',
                                'sine_timeseries_1'))
def test_success_timeseries(df, request):
    df = request.getfixturevalue(df)

    # df has answers in it named "spl_{spar}"
    cnames = [c for c in df.columns if 'spl_' in c]
    for cname in cnames:
        yhat_ans = df[cname]
        spar = float(cname.split('_')[1])

        knots, coef, yhat, x_min, x_range = sbart(
            df['x'].values,
            df['y'].values,
            df['w'].values,
            spar
        )

        assert np.isfinite(yhat).all()
        assert np.allclose(yhat, yhat_ans)
