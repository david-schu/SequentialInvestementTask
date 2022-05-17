from utils import load_data_patient as load_data
from model import fit, nll
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt

subject = load_data(6)
subject['prev'] = subject['bet']
for column in subject.columns:
    if not (column in ['d_bet_norm', 'bet', 'd_bet']) and abs(subject[column]).sum() > 0:
        subject[column] = zscore(subject[column], nan_policy='omit')

opt_params = ['w_mu',
              'w_sigma',
              'w_rej',
              'w_crej',
              'w_reg',
              'w_creg',
              'w_prev',
              'w_up',
              'w_down',
              # 'a_mu',
              # 'a_sigma',
              # # 'a_rej',
              # # 'a_crej',
              # # 'a_reg',
              # # 'a_creg',
              # 'a_up',
              # 'a_down',
              'alpha'
]

x0 = np.random.normal(0, .1, len(opt_params))
# x0 = np.zeros(len(opt_params))
x_fi, ll, bic = fit(subject, x0, opt_params)

fit = dict(zip(opt_params, x_fit))

# fit = []
# for i in range(n-1):
#     fit.append(dict(zip(opt_params, x_fit[i*len(opt_params):(i+1)*len(opt_params)])))

nll(subject, x_fit, opt_params)

print('done')
