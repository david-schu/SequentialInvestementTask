import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel


def market_return(market):
    m_return = np.full(len(market)-1, np.nan)
    for t in range(len(market)):
        if t > 0:
            m_return[t-1] = (market[t] - market[t-1])/ market[t-1]
    return m_return


def pred_market_delta(markets, kernel_params={}):
    # initialize arrays to store prediction results
    mkt_pred = np.full(len(markets), np.nan)
    mkt_sig = np.full(len(markets), np.nan)

    for i, market in enumerate(markets):
        market_returns = market_return(market)
        kernel = 1.0 * Matern(**kernel_params) + WhiteKernel(noise_level=kernel_params['length_scale']*np.std(market_returns))
        X = np.atleast_2d(np.arange(1, len(market_returns) + 1)).T
        y = market_returns

        # GP regressor without hyper-parameter optimization
        gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=True)

        # Fit to data using Maximum Likelihood Estimation of the parameters
        gp.fit(X, y)

        # # Make the prediction
        y_pred, sigma = gp.predict(np.atleast_2d(len(market_returns)+5).T, return_std=True)

        mkt_pred[i] = y_pred
        mkt_sig[i] = sigma

    return mkt_pred, mkt_sig


def pred_market_delta_whole(market, kernel_params={}):
    # initialize arrays to store prediction results
    market_returns = market_return(market)
    kernel = 1.0 * Matern(**kernel_params) + WhiteKernel(noise_level=kernel_params['length_scale']*np.std(market_returns))
    X = np.atleast_2d(np.arange(1, len(market_returns) + 1)).T
    y = market_returns

    # GP regressor without hyper-parameter optimization
    gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=True)

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(X, y)

    # # Make the prediction
    y_pred, sigma = gp.predict(np.atleast_2d(np.arange(len(market_returns)+5, 122, 4)).T, return_std=True)

    mkt_pred = y_pred
    mkt_sig = sigma

    return mkt_pred, mkt_sig