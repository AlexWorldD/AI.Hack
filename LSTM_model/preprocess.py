import pandas as pd
import numpy as np
from datetime import datetime
from multiprocessing import Pool
from preprocess_sub import *
import os

x = pd.read_csv('test_data.csv')
x['y+w'] = x.date.apply(lambda a: a.split('-')[0] + '-' + str(datetime.strptime(a, '%Y-%m-%d').isocalendar()[1]).zfill(2))
x['y+m'] = x.date.apply(lambda a: a.split('-')[0] + '-'  + a.split('-')[1].zfill(2))
#x.to_csv('new_test.csv')
w17 = ['2017-' + str(a).zfill(2) for a in range(1, 53)]
w16 = ['2016-' + str(a).zfill(2) for a in range(1, 53)]
w = w16 + w17

m17 = ['2017-' + str(a).zfill(2) for a in range(1, 13)]
m16 = ['2016-' + str(a).zfill(2) for a in range(1, 13)]
m = m16 + m17

tims = pd.DataFrame(m, columns=['y+m'])
wims = pd.DataFrame(w, columns=['y+w'])
ids = x.id.unique().tolist()
dataset = pd.DataFrame()
temp = []
i = 0

m = Pool(8)

users = [u for u in np.array_split(ids, 8)]
m.map(f, users)
pd.DataFrame(temp).to_csv('wop2.csv')

d = pd.DataFrame()
for i in os.listdir('./data/'):
    tmp = pd.read_csv('./data/' + i)
    tmp.head()
    d = d.append(tmp)
d.to_csv('out.csv')