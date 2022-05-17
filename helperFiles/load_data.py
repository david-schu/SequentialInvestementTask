import sys

sys.path.insert(0, './..')
import numpy as np
import os
import pandas as pd
import scipy.io
import utils
import GaussianProcess as GP

kernel_params = {
    'length_scale': 20,
    'nu': 1.5,
}

# Patient data readout
pids = np.load('../data/pids.npy')
behav_data = []
info_data = []
market_data = np.load('../data/markets.npy')

for pid in pids:
    mat = scipy.io.loadmat('../data/Bubble_patients_raw/Behaviour/bubble_P' + str(pid) + '.mat')
    mdata = mat['data']
    ndata = {n: mdata[n][0, 0][0] for n in mdata.dtype.names}

    p_data = pd.DataFrame.from_dict(ndata)
    p_data = p_data.dropna()
    p_data['marketRound'] = p_data['marketRound'] - 11
    behav_data.append(p_data)



for pid, p in zip(pids,behav_data):
    marketVals = []
    for i, trial in p.iterrows():
        marketVals.append(market_data[int(trial.marketId), :41 + int(trial.marketRound+1) * 4])

    mkt_pred, mkt_sig = GP.pred_market_delta(marketVals, kernel_params)

    portfolio = 100
    s_data = []
    threshold = np.nanmean(p['bet'])  # np.nanmedian(bets)

    for i, trial in p.iterrows():
        market_vals = marketVals[i]
        d_market = (market_vals[-1] - market_vals[-5]) / market_vals[-5]
        d_portfolio = d_market * trial.bet / 100 * portfolio
        actual = d_portfolio
        counterfactual = d_market * portfolio - actual
        d_bet = trial.bet
        rejoice = utils.get_rejoice(d_market, d_portfolio, trial.bet, threshold)
        regret = utils.get_regret(d_market, d_portfolio, trial.bet, threshold)
        counterfactual_rejoice = utils.get_counterfactual_rejoice(d_market, d_portfolio, trial.bet, portfolio,
                                                                  threshold)
        counterfactual_regret = utils.get_counterfactual_regret(d_market, d_portfolio, trial.bet, portfolio, threshold)
        portfolio += d_portfolio

        try:
            d_bet = int(p.loc[i + 1].bet - trial.bet)
        except:
            d_bet = np.nan

        if trial.marketRound == 0:
            d_portfolios = []
            rpe = d_portfolio
        else:
            norm = np.std(d_portfolios) if np.std(d_portfolios) != 0 else 1
            rpe = d_portfolio - np.mean(d_portfolios) / norm

        d_portfolios.append(d_portfolio)

        s_data.append({'bet': trial.bet,
                       'portfolio': portfolio,
                       'actual': actual,
                       'counterfactual': counterfactual,
                       'd_bet': d_bet,
                       'rpe': rpe,
                       'd_market': d_market,
                       'd_market_belief': mkt_pred[i],
                       'd_market_uncertainty': mkt_sig[i],
                       'regret': regret,
                       'rejoice': rejoice,
                       'counterfactual_regret': counterfactual_regret,
                       'counterfactual_rejoice': counterfactual_rejoice
                       })
    s_data = pd.DataFrame(s_data)
    s_data.to_pickle('../data/Bubble_Patients/S' + str(pid) + '.pkl')


# Healthy subjects data readout
subject = 0
for filename in os.listdir('../data/Bubble_2007_raw/RawBehavioral_Bubble_2021/'):

    with open(os.path.join('../data/Bubble_2007_raw/RawBehavioral_Bubble_2021/', filename), 'r') as f:
        lines = f.readlines()

        actions = []
        markets = []
        get_line = False
        for line in lines:
            if (len(markets) + 1) % 21 == 0:
                split = line.split('Portfolio:value=')[1]
                split = split.split('graph:prices=')
                split = split[1].split('] G-and-L:value=')
                market = split[0].replace('[', '').replace(']', '')
                market = np.array([float(s) for s in market.split(',')])
                markets.append(market)

            if get_line:
                split = line.split('Portfolio:value=')[1]
                split = split.split('graph:prices=')
                split = split[1].split('] G-and-L:value=')
                market = split[0].replace('[', '').replace(']', '')
                market = np.array([float(s) for s in market.split(',')])
                markets.append(market)
                get_line = False
                continue

            split = line.split('submit ')
            if len(split) > 1:
                try:
                    actions.append(int(float(split[1])))
                    get_line = True
                except:
                    pass

    del markets[0::21]

    mkt_pred, mkt_sig = GP.pred_market_delta(markets, kernel_params)

    portfolio = 100
    s_data = []
    threshold = np.nanmean(actions)

    for i, market_vals in enumerate(markets):
        d_market = (market_vals[-1] - market_vals[-5]) / market_vals[-5]
        d_portfolio = d_market * actions[i] / 100 * portfolio
        actual = d_portfolio
        counterfactual = d_market * portfolio - actual
        d_bet = actions[i]
        rejoice = utils.get_rejoice(d_market, d_portfolio, actions[i], threshold)
        regret = utils.get_regret(d_market, d_portfolio, actions[i], threshold)
        counterfactual_rejoice = utils.get_counterfactual_rejoice(d_market, d_portfolio, actions[i], portfolio,
                                                                  threshold)
        counterfactual_regret = utils.get_counterfactual_regret(d_market, d_portfolio, actions[i], portfolio, threshold)
        portfolio += d_portfolio

        try:
            d_bet = int(actions[i+1] - actions[i])
        except:
            d_bet = np.nan

        if (i % 20) == 0:
            d_portfolios = []
            rpe = d_portfolio
        else:
            norm = np.std(d_portfolios) if np.std(d_portfolios) != 0 else 1
            rpe = d_portfolio - np.mean(d_portfolios) / norm

        d_portfolios.append(d_portfolio)

        s_data.append({'bet': actions[i],
                       'portfolio': portfolio,
                       'actual': actual,
                       'counterfactual': counterfactual,
                       'd_bet': d_bet,
                       'rpe': rpe,
                       'd_market': d_market,
                       'd_market_belief': mkt_pred[i],
                       'd_market_uncertainty': mkt_sig[i],
                       'regret': regret,
                       'rejoice': rejoice,
                       'counterfactual_regret': counterfactual_regret,
                       'counterfactual_rejoice': counterfactual_rejoice
                       })
    s_data = pd.DataFrame(s_data)
    s_data.to_pickle('../data/Bubble_Controls/S' + str(subject) + '.pkl')

    subject += 1
print('Done')
