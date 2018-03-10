import multiprocessing
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import preprocessing
from MultiColumnEncoder import *
from multiprocessing import Pool, cpu_count

ewma = pd.Series.ewm
titles = ['v_l', 'sum_b', 'percent']
from datetime import datetime
from tqdm import tqdm, tqdm_pandas


# types = {'time', 'date': np.datetime, 'v_l', 'q', 'n_tr', 'sum_b', 'code_azs', 'id', 'first_prch', 'location', 'region', 'code', 'code1', 'percent', 'type'}

def train_group_id(df):
    df[1].sort_values(by='date', inplace=True)
    d = df[1].groupby(['year', 'month'], as_index=False).agg(
        {'v_l': np.mean, 'sum_b': np.sum, 'percent': np.sum, 'type': lambda x: stats.mode(x)[0]})
    d['target'] = d['month'].shift(-1) - d['month']
    d['target'] = d['target'].apply(lambda x: 1 if x in [1, -11] else 0)
    d.drop(['year', 'month'], axis=1, inplace=True)
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    d[titles] = pd.DataFrame(scaler.fit_transform(d[titles]),
                             index=d.index, columns=titles)
    return d

def test_group_id(df):
    df[1].sort_values(by='date', inplace=True)
    d = df[1].groupby(['year', 'month'], as_index=False).agg(
        {'v_l': np.mean, 'sum_b': np.sum, 'percent': np.sum, 'type': lambda x: stats.mode(x)[0]})
    d['target'] = d['month'].shift(-1) - d['month']
    d['target'] = d['target'].apply(lambda x: 1 if x in [1, -11] else 0)
    d.drop(['year', 'month', 'target'], axis=1, inplace=True)
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    d[titles] = pd.DataFrame(scaler.fit_transform(d[titles]),
                             index=d.index, columns=titles)
    cur_row = d.tail(1)
    cur_row['id'] = df[0]
    return cur_row

def applyParallel(dfGrouped, func):
    with Pool(cpu_count()) as p:
        ret_list = p.map(func, [group for group in dfGrouped])
    return pd.concat(ret_list)


def loading(path='/Users/lex/Dev/GitHub/AI.Hack/data/train_data.csv'):
    """
    Simple uploading data function
    :param data: raw data in
    :return:
    """
    # raw_data = pd.read_csv('ex.csv', index_col=0)
    raw_data = pd.read_csv(path, parse_dates=['date'], index_col=0, low_memory=False)
    raw_data.fillna(value={'sum_b': 0, 'v_l': 0, 'percent': 0}, inplace=True)
    raw_data['time'].fillna('04:00:00', inplace=True)
    raw_data['month'] = raw_data['date'].apply(lambda x: x.month)
    raw_data['year'] = raw_data['date'].apply(lambda x: x.year)

    # raw_data['date'] = raw_data[['date', 'time']].apply(
    #     lambda x: pd.to_datetime(x[0] + ' ' + x[1], errors='coerce'), axis=1)
    groups = raw_data.groupby('id')
    data = pd.DataFrame(columns=['v_l', 'sum_b', 'percent', 'type', 'target'])
    data = applyParallel(groups, train_group_id)
    #     group[1].sort_values(by='date', inplace=True)
    #     d = group[1].groupby(['year', 'month'], as_index=False).agg(
    #         {'v_l': np.mean, 'sum_b': np.sum, 'percent': np.sum, 'type': lambda x: stats.mode(x)[0]})
    #     d['target'] = d['month'].shift(-1) - d['month']
    #     d['target'] = d['target'].apply(lambda x: 1 if x in [1, -11] else 0)
    #     d.drop(['year', 'month'], axis=1, inplace=True)
    #     scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    #     d[titles] = pd.DataFrame(scaler.fit_transform(d[titles]),
    #                              index=d.index, columns=titles)
    #     data = data.append(d)
    #     # le = MultiColumnEncoder.EncodeCategorical(columns=['type']).fit(d)
    #     # data = le.transform(d)
    #     # fwd = ewma(sum, com=0.3).mean()  # take EWMA in fwd direction
    #     # bwd = ewma(sum[::-1], com=0.3).mean()  # take EWMA in bwd direction
    #     # filtered = np.vstack((fwd, bwd[::-1]))  # lump fwd and bwd together
    #     # filtered = np.mean(filtered, axis=0)
    #     # plt.plot(group[1]['date'], group[1]['sum_b'])
    #     # plt.show()
    #     print('t')
    t_t = list(data)
    t_t.remove('target')
    X = data[t_t]
    # Y = data[list(data_target)[1:2]]
    Y = data['target']
    # data_stat(X)
    return X, Y


def loading_test(path='/Users/lex/Dev/GitHub/AI.Hack/data/test_data.csv'):
    """
    Simple uploading data function
    :param data: raw data in
    :return:
    """
    # raw_data = pd.read_csv('ex.csv', index_col=0)
    raw_data = pd.read_csv(path, parse_dates=['date'], index_col=0, low_memory=False)
    raw_data.fillna(value={'sum_b': 0, 'v_l': 0, 'percent': 0}, inplace=True)
    raw_data['time'].fillna('04:00:00', inplace=True)
    raw_data['month'] = raw_data['date'].apply(lambda x: x.month)
    raw_data['year'] = raw_data['date'].apply(lambda x: x.year)

    # raw_data['date'] = raw_data[['date', 'time']].apply(
    #     lambda x: pd.to_datetime(x[0] + ' ' + x[1], errors='coerce'), axis=1)
    groups = raw_data.groupby('id')
    data = pd.DataFrame(columns=['id', 'v_l', 'sum_b', 'percent', 'type'])
    data = applyParallel(groups, test_group_id)
    # for group in groups:
    #     group[1].sort_values(by='date', inplace=True)
    #     d = group[1].groupby(['year', 'month'], as_index=False).agg(
    #         {'v_l': np.mean, 'sum_b': np.sum, 'percent': np.sum, 'type': lambda x: stats.mode(x)[0]})
    #     d['target'] = d['month'].shift(-1) - d['month']
    #     d['target'] = d['target'].apply(lambda x: 1 if x in [1, -11] else 0)
    #     d.drop(['year', 'month', 'target'], axis=1, inplace=True)
    #     scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    #     d[titles] = pd.DataFrame(scaler.fit_transform(d[titles]),
    #                              index=d.index, columns=titles)
    #     cur_row = d.tail(1)
    #     cur_row['id'] = group[0]
    #     data = data.append(cur_row)
    #     print('t')
    return data
