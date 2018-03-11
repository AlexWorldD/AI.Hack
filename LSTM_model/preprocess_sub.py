import numpy as np
import pandas as pd
from datetime import datetime


def f(id_s):
    x = pd.read_csv('train_data.csv')
    x['y+w'] = x.date.apply(lambda a: a.split('-')[0] + '-' + str(datetime.strptime(a, '%Y-%m-%d').isocalendar()[1]).zfill(2))
    x['y+m'] = x.date.apply(lambda a: a.split('-')[0] + '-'  + a.split('-')[1].zfill(2))
    w17 = ['2017-' + str(a).zfill(2) for a in range(1, 53)]
    w16 = ['2016-' + str(a).zfill(2) for a in range(1, 53)]
    w = w16 + w17

    m17 = ['2017-' + str(a).zfill(2) for a in range(1, 13)]
    m16 = ['2016-' + str(a).zfill(2) for a in range(1, 13)]
    m = m16 + m17

    tims = pd.DataFrame(m, columns=['y+m'])
    wims = pd.DataFrame(w, columns=['y+w'])
    temp = []
    
    
    for i in id_s.tolist():
        name = str(i)
        print(name)
        tmp = wims.set_index('y+w').join(x[x.id == i].set_index('y+w'))
        tmp['y+w'] = tmp.index
        tmp.fillna(0)
        tmp['events'] = tmp.groupby('y+w').date.count()
        tmp = tmp.drop_duplicates('y+w')
        tmp.sort_values('y+w')
        temp.append({'id': str(i), 'events': tmp.events.values.tolist(), 'fuel': tmp.v_l.values.tolist(),
                     'staff': tmp.q.values.tolist()})
    pd.DataFrame(temp).to_csv(name + 'wop2.csv')