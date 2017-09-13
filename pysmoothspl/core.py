""" A `scikit-learn`-esque interface to `smooth.spline`
"""
import numpy as np
try:
    import sklearn
except ImportError:
    HAS_SKLEARN = False
    Base = object
else:
    HAS_SKLEARN = True
    BASES = (sklearn.base.BaseEstimator, sklearn.base.RegressorMixin, )
    Base = type('Base', BASES, {})
from ._sbart import _sbart, _bvalues


class SmoothSpline(Base):
    """ An smoothing spline estimator that acts like `scikit-learn`

    This smoothing spline estimator wraps the C-code that underlies
    R's `smooth.spline` function.

    Attributes:
        knots_ (np.ndarray): Fitted knots (`nk_` values)
        coefs_ (np.ndarray): Spline coefficients (`nk_` values)
        X_min_ (float): Minimum X value used to fit the spline,
            (needed for predictions)
        X_range_ (float): Range of X values used to fit the spline
            (needed for predictions)

    Args:
        spar (float): Smoothing spline parameter, usually in (0, 1]

    """
    def __init__(self, spar):
        self.spar = spar
        assert isinstance(self.spar, float), (
            'Smoothing control parameter `spar` needs to be a '
            'float (usually (0, 1])')

    def fit(self, X, y, sample_weight=None, check=True):
        """ Fit a smoothing spline
        """
        # TODO: let y=None and just use X against index
        assert X.ndim == 1, "1D only for spline fit"

        X = np.asarray(X, dtype=np.float)
        y = np.asarray(y, dtype=np.float)

        if sample_weight is None:
            # Default to same weights
            sample_weight = np.ones_like(y)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float)

        assert X.shape[0] == y.shape[0] == sample_weight.shape[0]

        result = _sbart(
            X, y, sample_weight, self.spar, check=check
        )
        self.knots_ = result[0]
        self.coefs_ = result[1]
        self.X_min_ = result[3]
        self.X_range_ = result[4]

        return self

    def predict(self, X):
        assert X.ndim == 1, "1D only for spline fit"

        X = np.asarray(X, dtype=np.float)

        yhat = _bvalues(
            self.knots_,
            self.coefs_,
            X,
            self.X_min_,
            self.X_range_,
            0  # not the derivative
        )
        return yhat
