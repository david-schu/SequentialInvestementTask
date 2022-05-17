import numpy as np
import scipy.io
import pandas as pd
import os


def load_data_controls(subj_id):
    """
    helper file to load data
    :param subj_id: either integer of patient id or 'all', meaning all patients are added to list
    :return: either a file containing patient data or a list of all patient data
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if subj_id == 'all':
        data = []
        files = [file for file in os.listdir(dir_path + '/data/Bubble_Controls/') if file.endswith('.pkl')]
        sort = [files[i] for i in np.argsort([int(x.split('S')[1].split('.')[0]) for x in files])]
        for filename in sort:
            p_data = pd.read_pickle(dir_path + '/data/Bubble_Controls/' + filename)
            data.append(p_data)
    elif os.path.exists(dir_path + '/data/Bubble_Controls/S' + str(subj_id) + '.pkl'):
        data = pd.read_pickle(dir_path + '/data/Bubble_Controls/S' + str(subj_id) + '.pkl')
    else:
        raise ValueError('Invalid Patient ID')
    return data


def load_data_patient(patient_id):
    """
    helper file to load data
    :param patient_id: either interger of patient id or 'all', meaning all patients are added to list
    :return: either a file containing patient data or a list of all patient data
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = []
    files = [file for file in os.listdir(dir_path + '/data/Bubble_Patients/') if file.endswith('.pkl')]
    sort = [files[i] for i in np.argsort([int(x.split('S')[1].split('.')[0]) for x in files])]
    for filename in sort:
        p_data = pd.read_pickle(dir_path + '/data/Bubble_Patients/' + filename)
        data.append(p_data)
    if not patient_id == 'all':
        data = data[patient_id]
    return data


def load_data_neural(patient_id, neurotransmitter, event):
    """
    helper file to load data
    :param event: event of data recording out of ('start', 'submit','reveal')
    :param neurotransmitter: neurotransmitter to load. Has to be ('da', 'se', 'ne')
    :param patient_id: either interger of patient id or 'all', meaning all patients are added to list
    :return: either a file containing patient data or a list of all patient data
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = []
    files = [file for file in os.listdir(dir_path + '/data/Bubble_patients_raw/Neurochemistry/')
             if (('_' + neurotransmitter + '_') in file and event in file.lower())]

    sort = [files[i] for i in np.argsort([int(x.split('P')[1].split('.')[0]) for x in files])]
    for filename in sort:
        p_data = scipy.io.loadmat(dir_path + '/data/Bubble_patients_raw/Neurochemistry/' + filename)['timeSeries']
        data.append(p_data)
    if not patient_id=='all':
        data = data[patient_id]
    return data

def softmax(x, beta=1):
    """
    :param x: data array
    :param beta: inverse temperature parameter
    :return: return softmax of given array
    """
    max_exp = np.max(beta * x)
    p_sum = (np.exp(beta * x - max_exp)).sum()
    p = np.exp(beta * x - max_exp) / p_sum
    return p


def sigmoid(x, beta=1):
    """
    :param x: data point
    :param beta: inverse temperature parameter
    :return: return sigmoid of x -> map from range (-inf,inf) to (0,1)
    """
    s = 1 / (1 + np.exp(-beta*x))
    return s


def inv_sigmoid(x, beta=1):
    """
    :param x: data point
    :param beta: inverse temperature parameter
    :return: return reverse sigmoid of x -> map from range (0,1) to (-inf,inf)
    """
    s = np.log(x/(1-x)) / beta
    return s


def map_to(x, tmin, tmax, rmin=0, rmax=1):
    """
    Map an array to new range
    Parameters:
        x [array] array to be mapped
        tmin [double] the target lower bound
        tmax [double] the target upper bound
        rmin [double] the original lower bound
        rmax [double] the original upper bound
    Outputs:
        x_t [array] the mapped array
    """
    if tmin == tmax:
        x_t = x*0+tmin
    else:
        x_t = (x-rmin)*(tmax-tmin)/(rmax-rmin)+tmin
    return x_t


def get_rejoice(d_market, d_portfolio, prev_bet, t=50):
    rejoice = 0
    if d_market > 0 and prev_bet > t:
        rejoice = d_portfolio
    return rejoice


def get_regret(d_market, d_portfolio, prev_bet, t=50):
    regret = 0
    if d_market < 0 and prev_bet > t:
        regret = -d_portfolio
    return regret


def get_counterfactual_rejoice(d_market, d_portfolio, prev_bet, portfolio, t=50):
    counterfactual_rejoice = 0
    if d_market < 0 and prev_bet <= t:
        counterfactual_rejoice = d_portfolio - d_market * portfolio
    return counterfactual_rejoice


def get_counterfactual_regret(d_market, d_portfolio, prev_bet, portfolio, t=50):
    counterfactual_regret = 0
    if d_market > 0 and prev_bet <= t:
        counterfactual_regret = d_market * portfolio - d_portfolio
    return counterfactual_regret
