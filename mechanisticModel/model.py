import numpy as np
from scipy.optimize import minimize
from utils import sigmoid, softmax


def fit(data, x0, opt_params={}):
    """
    Fit parameters of predictive behavioral model with scipy optimize routine.
    :param data: a pandas Dataframe containing trials as rows
    :param x0: initial guesses for optimization parameters
    :param opt_params: parameter names to be optimized
    :return: fitted parameters, negative log-likelihood and BIC of fit
    """

    # define function to be optimized
    fun = lambda x: nll(data, x, opt_params) + .01 * abs(x).sum()

    # fit parameters
    res = minimize(x0=x0, fun=fun)
    x_fit = res.x

    # get nll and bic of fit
    nll_ = res.fun
    bic = len(x0) * np.log((~np.isnan(data['bet'])).sum()) + 2 * nll_
    _, true_ps, all_ps = nll(data, x_fit, opt_params, return_ps=True)

    return x_fit, nll_, bic, true_ps, all_ps


def nll(data, x, opt_params, fixed_params=[], return_ps=False):
    """
    Calculate and return the negative log-likelihood of the model given a set of parameters.
    :param data: a pandas Dataframe containing trials as rows
    :param x: parameter values for nll evaluation
    :param opt_params: names of parameters contained in x
    :param fixed_params: dictionary of fixed model parameters
    :return: nll of modle with parameters x
    """

    # initialize optimization parameters
    params = {
        'w_mu': 0,
        'w_sigma': 0,
        'w_rej': 0,
        'w_crej': 0,
        'w_reg': 0,
        'w_creg': 0,
        'w_up': 0,
        'w_down': 0,
        'w_prev': 0,
        'a_mu': 0,
        'a_sigma': 0,
        'a_rej': 0,
        'a_crej': 0,
        'a_reg': 0,
        'a_creg': 0,
        'a_up': 0,
        'a_down': 0,
        'alpha': -np.inf,
    }

    for i, param in enumerate(opt_params):
        params[param] = x[i]

    for fixed_param in fixed_params:
        params[fixed_param] = fixed_params[fixed_param]

    params['alpha'] = sigmoid(params['alpha'])

    # initialize action space and array for choice probs
    action_space = np.around(np.linspace(-1, 1, 21), 1)
    action_space = action_space/np.std(action_space)
    choice_probs = np.zeros(len(data) - np.isnan(data.d_bet).sum())
    all_ps = np.zeros((len(data) - np.isnan(data.d_bet).sum(), 21))

    # initialize arrays for increasing/decreasing bet size
    up = (action_space + abs(action_space)) / 2
    down = (abs(action_space) - action_space) / 2

    if params['alpha']:
        back = 5
    else:
        back = 0

    for j, (_, trial) in enumerate(data.iterrows()):
        bet = int(trial.bet / 10)

        if not np.isnan(trial.d_bet):
            if back == 0:
                trials = trial
                back_weighing = 1
            else:
                trials = data.iloc[np.maximum(0, j - back):(j + 1)]
                back_weighing = params['alpha'] ** np.arange(len(trials) - 1, -1, -1)
            # calculate linear combination of factors
            lin_coef = (params['w_mu'] + params['a_mu'] * trial.prev) * trial.d_market_belief + \
                       (params['w_sigma'] + params['a_sigma'] * trial.prev) * trial.d_market_uncertainty + \
                       (params['w_rej'] + params['a_rej'] * trial.prev) * (trials.rejoice * back_weighing).sum()/np.sum(back_weighing) + \
                       (params['w_reg'] + params['a_reg'] * trial.prev) * (trials.regret * back_weighing).sum()/np.sum(back_weighing) + \
                       (params['w_crej'] + params['a_crej'] * trial.prev) * (trials.counterfactual_rejoice * back_weighing).sum()/np.sum(back_weighing) + \
                       (params['w_creg'] + params['a_creg'] * trial.prev) * (trials.counterfactual_regret * back_weighing).sum()/np.sum(back_weighing) + \
                       params['w_prev'] * trial.prev

            # get action values from linear factor and quadratic change
            vals = lin_coef * action_space + \
                   (params['w_up'] + params['a_up'] * trial.prev) * up ** 2 + \
                   (params['w_down'] + params['a_down'] * trial.prev) * down ** 2

            # select valid action range
            low = 0 + (10 - bet)
            high = 21 - bet
            vals = vals[low:high]
            # Q = np.log(Q - min(Q) + 1)

            # get action probability
            p_q = softmax(vals, 1)
            choice_probs[j] = p_q[int(data.iloc[j + 1].bet / 10)]
            all_ps[j, low:high] = p_q
    # compute nll
    nll_ = -np.log(choice_probs + 1e-9).sum()

    if return_ps:
        return nll_, choice_probs, all_ps

    return nll_
