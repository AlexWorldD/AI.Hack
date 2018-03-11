#!/usr/bin/env python
# -*- coding: utf-8 -*-
# from __future__ import absolute_import
import warnings
warnings.filterwarnings("ignore")
import os, re, json
import numpy as np
import pandas as pd
from time import time
from munging import *
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, SelectFromModel

def find_bestRF(drop='', fea='', scoring='roc_auc', title='', selectK='', fillna=True, f='FeaturesBIN3',
                t='Targets2016', cut=False, parallel=[4, 2], required='', phone='Operator'):
    """Testing linear method for train"""
    train_X, train_Y = loading()
    print('DONE with train data!')

    N = [20, 50, 100, 200, 500, 1000]
    N_v2 = [500]
    grid = [
        {'criterion': ('gini', 'entropy'),
         'n_estimators': N,
         'max_features': [0.5, 'log2', 'auto', None],
         'max_depth': [None, 10]
         }
    ]
    grid_light = [
        {'n_estimators': N_v2,
         'max_features': [0.5],
         'max_depth': [3, 5]
         }
    ]
    # KFold for splitting
    cv = KFold(n_splits=2,
               shuffle=True,
               random_state=241)
    rf = RandomForestClassifier(random_state=241, n_jobs=-1)
    lr = LogisticRegression(C=10, n_jobs=-1)

    test_X = loading_test()
    print('DONE with test data!')
    clf = GridSearchCV(estimator=rf,
                       param_grid=grid_light,
                       scoring=scoring,
                       cv=cv,
                       n_jobs=parallel[0],
                       pre_dispatch=parallel[1],
                       verbose=2)
    # ---- EPIC HERE ----
    clf.fit(train_X, train_Y)
    result = pd.DataFrame(columns=['id', 'proba'])
    result['id'] = test_X['id']
    result['proba'] = pd.DataFrame(clf.best_estimator_.fit(train_X, train_Y).predict_proba(test_X))
    result.to_csv('result.csv')


if __name__ == '__main__':
    # r1 = pd.read_csv('../results/resultV2.csv', index_col=0)
    # r2 = pd.read_csv('../result.csv', index_col=0)
    # data = loading_test()
    data = loading_group()
    # res = pd.read_csv('/Users/lex/Dev/GitHub/AI.Hack/result.csv')
    # data.to_csv('resultV2.csv', index=False)
    print('Great!')