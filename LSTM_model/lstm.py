from keras.layers import Input, LSTM, RepeatVector, Dense, Dropout
from keras.models import Sequential, Model
from pandas import read_excel, read_csv
from keras.preprocessing import sequence
from keras.utils import to_categorical
import numpy as np
from keras.preprocessing.text import Tokenizer
from itertools import combinations
import re

sequence_length = 2080
encoded_legth = 26

dataset = read_csv('./data/out.csv')

def generate_samp(df):
    tmp = df.events.tolist()
    for num, i in enumerate(tmp):
        a = np.array([int(j) for j in re.split(r'\D+', i[1:-1])])
        feat2 = np.array([float(j) for j in df.fuel[num][1:-1].replace('nan', '0.0').split(', ')])
        feat2 = feat2 / max(max(feat2), 1)
        start = np.where(a > 0)[0][0]
        end = np.where(a > 0)[0][-1]
        if end - start >= 25:
            tmp3 = feat2[end - 25:end]
            tmp2 = a[end - 25:end]
        else:
            b = np.zeros(26 - (end - start))
            tmp2 = np.append(b, a[start:end])[end - 25: end]
            tmp3 = np.append(b, feat2[start:end])[end - 25: end]
        if len(tmp2) != 25:
            tmp2 = np.zeros(25)
            tmp3 = np.zeros(25)
        y = lambda z: 1 if z else 0
        yield np.array([[[tmp2[k], tmp3[k]] for k in range(24)]]), to_categorical(y(tmp2[24]), 2)


def generate_test(df):
    tmp = df.events.tolist()
    for i in tmp:
        a = np.array([int(j) for j in re.split(r'\D+', i[1:-1])])
        start = np.where(a > 0)[0][0]
        end = np.where(a > 0)[0][-1]
        if end - start >= 24:
            tmp2 = a[end - 24:end]
        else:
            b = np.zeros(24 - (end - start))
            tmp2 = np.append(b, a[start:end])[end - 24: end]
        yield np.array([[[k] for k in tmp2]])
    
timesteps, input_dim = 24, 2
inputs = Input(shape=(timesteps, input_dim))
encoded = LSTM(8)(inputs)
drop = Dropout(0.5)(encoded)
conf = Dense(2, activation='softmax')(drop)
model = Model(inputs, conf)

model.compile(loss='binary_crossentropy',
                        optimizer='RMSprop',
                        metrics=['accuracy'])
model.fit_generator(generate_samp(dataset), steps_per_epoch=1500, epochs=20)

model.save('model3.h5')
