from keras.models import load_model
from pandas import read_excel, read_csv
import numpy as np
import re

sequence_length = 2080
encoded_legth = 26

dataset = read_csv('./data/out3.csv')

def generate_test(df):
    tmp = df.events.tolist()
    for num, i in enumerate(tmp):
        a = np.array([int(j) for j in re.split(r'\D+', i[1:-1])])
        feat2 = np.array([float(j) for j in df.fuel[num][1:-1].replace('nan', '0.0').split(', ')])
        feat2 = feat2 / max(max(feat2), 1)
        start = np.where(a > 0)[0][0]
        end = np.where(a > 0)[0][-1]
        if end - start >= 24:
            tmp3 = feat2[end - 25:end]
            tmp2 = a[end - 24:end]
        else:
            b = np.zeros(24 - (end - start))
            tmp3 = np.append(b, feat2[start:end])[end - 25: end]
            tmp2 = np.append(b, a[start:end])[end - 24: end]
        if len(tmp2) < 24:
            tmp2 = np.zeros(24)
            tmp3 = np.zeros(24)
        yield np.array([[[tmp2[k], tmp3[k]] for k in range(24)]]), df.id[num]
    

model = load_model('model2.h5')
res3 = []
for i in generate_test(dataset):
    pred = model.predict(i[0])[0]
    res3.append((i[1], pred[0]))
    
result = pd.DataFrame(res, columns=['id', 'pred'])
qqq = sub.set_index('id').join(result.set_index('id')).fillna(0)
qqq['id'] = qqq.index
qqq[['id', 'pred']].to_csv('SUB.csv', index=False)