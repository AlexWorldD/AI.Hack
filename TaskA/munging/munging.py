import multiprocessing
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import preprocessing
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from MultiColumnEncoder import *
from multiprocessing import Pool, cpu_count
from sklearn.linear_model import LogisticRegression

ewma = pd.Series.ewm
titles = ['v_l', 'sum_b', 'percent', 'q']
from datetime import datetime
from tqdm import tqdm, tqdm_pandas


# types = {'time', 'date': np.datetime, 'v_l', 'q', 'n_tr', 'sum_b', 'code_azs', 'id', 'first_prch', 'location', 'region', 'code', 'code1', 'percent', 'type'}

def RF(data, scoring='roc_auc', parallel=[4, 2]):
    """Testing linear method for train"""
    train, test = split(data)
    if test.shape[0] > 0:
        try:
            train_X, train_Y = split_dataset(train)
            test_X, _ = split_dataset(test)

            rf = RandomForestClassifier(n_estimators=100, random_state=241, max_depth=8, n_jobs=-1)
            rf.fit(train_X, train_Y)
            lr = LogisticRegression(C=10, solver='saga', n_jobs=-1)
            # nn = MLPClassifier(random_state=241, verbose=1, hidden_layer_sizes=(10, 100, 20))

            # ---- EPIC HERE ----
            lr.fit(train_X, train_Y)
            # nn.fit(train_X, train_Y)

            if 1 in rf.classes_:
                res1 = (rf.predict_proba(test_X)[0][-1] + lr.predict_proba(test_X)[0][-1]) / 2
                if test['series'].values[0] != 0:
                    res = res1 / (1 + test['series'].values[0] * 2)
                else:
                    res = res1 if res1 > 0.0001 else 0.000000001
            else:
                res = 0.0
        except:
            if 1 in rf.classes_:
                res1 = rf.predict_proba(test_X)[0][-1]
                if test['series'].values[0] != 0:
                    res = res1 / np.log(test['series'].values[0]) * 2
                else:
                    res = res1 if res1 > 0.0001 else 0.000000001
            else:
                res = 0.0
    else:
        res = 0.5
    return res


def split_dataset(d):
    t_t = list(d)
    t_t.remove('target')
    X = d[t_t]
    # Y = data[list(data_target)[1:2]]
    Y = d['target']
    return X, Y


def split(data):
    return data[:-1], data[-1:]


def train_group_id(df):
    df[1].sort_values(by='date', inplace=True)
    d = df[1].groupby(['year', 'month'], as_index=False).agg(
        {'v_l': np.mean, 'sum_b': np.sum, 'percent': np.sum, 'type': lambda x: stats.mode(x)[0], 'q': np.sum})
    d['target'] = d['month'].shift(-1) - d['month']
    d['target'] = d['target'].apply(lambda x: 1 if x in [1, -11] else 0)
    d.drop(['year', 'month'], axis=1, inplace=True)
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    d[titles] = pd.DataFrame(scaler.fit_transform(d[titles]),
                             index=d.index, columns=titles)
    return d


def to_part_day(hour):
    if hour < 9:
        if hour > 5:
            return 'morning'
        else:
            return 'night'
    elif hour < 14:
        return 'noun'
    elif hour < 18:
        return 'day'
    elif hour < 24:
        return 'evening'


def add_features(df):
    fuel_year = {key: group[group['q'] == 0]['sum_b'].sum() / max(group[group['q'] == 0]['v_l'].sum(), 1) for key, group
                 in
                 df.groupby('year')}
    df['fuel_type'] = df['year'].apply(lambda x: fuel_year[x])
    products_year = {key: group[group['v_l'] == 0]['sum_b'].sum() / max(group[group['v_l'] == 0]['q'].sum(), 1) for
                     key, group
                     in
                     df.groupby('year')}
    df['products_type'] = df['year'].apply(lambda x: products_year[x])
    df['week_day'] = df['date'].apply(lambda x: x.weekday())
    df['day_part'] = df['time'].apply(lambda x: to_part_day(pd.to_datetime(x).hour))


def test_group_id(df, encode='label', scale=True):
    features = ['avg_money', 'avg_percent', 'avg_fuel', 'avg_q', 'series']
    df[1].sort_values(by='date', inplace=True)
    # df[1].to_csv('one_group.csv', index=False)
    add_features(df[1])
    # Calculate total/mean statistics per month
    # Fuel volume
    df[1]['fuel_m'] = df[1].groupby(['year', 'month'])['v_l'].transform(lambda x: np.sum(x))
    df[1]['fuel_total'] = df[1].groupby(['year'])['v_l'].transform(lambda x: max(np.sum(x), 1))
    # Purchase counts
    df[1]['q_m'] = df[1].groupby(['year', 'month'])['q'].transform(lambda x: np.sum(x))
    df[1]['q_total'] = df[1].groupby(['year'])['q'].transform(lambda x: max(np.sum(x), 1))
    # Bonus usage
    df[1]['percent_m'] = df[1].groupby(['year', 'month'])['percent'].transform(lambda x: np.sum(x))
    df[1]['percent_total'] = df[1].groupby(['year'])['percent'].transform(lambda x: max(np.sum(x), 1))
    # Shut up and take ny money!
    df[1]['money_m'] = df[1].groupby(['year', 'month'])['sum_b'].transform(lambda x: np.sum(x))
    df[1]['money_total'] = df[1].groupby(['year'])['sum_b'].transform(lambda x: max(np.sum(x), 1))
    # Calculate average statistics per month accordingly yearly consumption
    df[1]['avg_money'] = df[1].apply(lambda row: row['sum_b'] / row['money_total'], axis=1)
    df[1]['avg_fuel'] = df[1].apply(lambda row: row['fuel_m'] / row['fuel_total'], axis=1)
    df[1]['avg_q'] = df[1].apply(lambda row: row['q_m'] / row['q_total'], axis=1)
    df[1]['avg_percent'] = df[1].apply(lambda row: row['percent_m'] / row['percent_total'], axis=1)

    d = df[1].groupby(['year', 'month'], as_index=False).agg(
        {'avg_money': np.mean, 'avg_fuel': np.mean, 'avg_q': np.mean, 'type': lambda x: stats.mode(x)[0],
         'avg_percent': np.mean, 'day_part': lambda x: stats.mode(x)[0], 'week_day': lambda x: stats.mode(x)[0]})
    if encode == 'onehot':
        d = pd.get_dummies(d, columns=['day_part', 'week_day'])
    else:
        le = MultiColumnEncoder.EncodeCategorical(columns=['day_part', 'week_day']).fit(d)
        d = le.transform(d)
    d['target'] = d['month'].shift(-1) - d['month']
    d['target'] = d['target'].apply(lambda x: 0 if x in [1, -11] else 1)
    d['series'] = pd.Series([max(x) for x in pd.DataFrame(
        [(d['month'] - d['month'].shift(s)).apply(lambda x: s if x in [s, s - 12] else 1) for s in
         range(1, 12)]).transpose().values])
    d.drop(['year', 'month'], axis=1, inplace=True)
    if scale:
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        d[features] = pd.DataFrame(scaler.fit_transform(d[features]),
                                   index=d.index, columns=features)
    res = RF(d)
    print(res)
    return pd.DataFrame(data={'id': [df[0]], 'proba': [res]})


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


def loading_group(path='../one_group.csv'):
    """
    Simple uploading data function
    :param data: raw data in
    :return:
    """
    # raw_data = pd.read_csv('ex.csv', index_col=0)
    raw_data = pd.read_csv(path, parse_dates=['date'], low_memory=False)
    raw_data.fillna(value={'sum_b': 0, 'v_l': 0, 'percent': 0, 'q': 0}, inplace=True)
    raw_data['time'].fillna('04:00:00', inplace=True)

    # raw_data['date'] = raw_data[['date', 'time']].apply(
    #     lambda x: pd.to_datetime(x[0] + ' ' + x[1], errors='coerce'), axis=1)
    groups = raw_data.groupby('id')
    data = pd.DataFrame(columns=['id', 'v_l', 'sum_b', 'percent', 'type', 'q'])
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


def loading_test(path='../data/test_data.csv'):
    """
    Simple uploading data function
    :param data: raw data in
    :return:
    """
    # raw_data = pd.read_csv('ex.csv', index_col=0)
    raw_data = pd.read_csv(path, parse_dates=['date'], index_col=0, low_memory=False)
    raw_data.fillna(value={'sum_b': 0, 'v_l': 0, 'percent': 0, 'q': 0}, inplace=True)
    raw_data['time'].fillna('04:00:00', inplace=True)
    raw_data['month'] = raw_data['date'].apply(lambda x: x.month)
    raw_data['year'] = raw_data['date'].apply(lambda x: x.year)
    groups = raw_data.groupby('id')
    data = pd.DataFrame(columns=['id', 'v_l', 'sum_b', 'percent', 'type', 'q'])
    data = applyParallel(groups, test_group_id)
    return data
