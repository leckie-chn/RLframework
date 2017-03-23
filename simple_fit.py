import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Activation, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam

mode = 'test'
scale = 50.00
base = 10.0
num_data = 10000
data = np.random.random((num_data, 1)) * scale + base
model = Sequential()
model.add(Dense(32, input_dim=1, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error', metrics=['mean_squared_error'])

if mode == 'train':
    num_round = 1000
    y = 1.0 / data
    his = model.fit(data, y, nb_epoch=num_round, batch_size=32, verbose=0)

    plt.figure()
    plt.plot(np.log10(his.history['mean_squared_error']), 'b-')
    plt.savefig('logs/simple_loss.png')

    model.save_weights('saved_networks/simple_fit.h5')


elif mode == 'test':
    model.load_weights('saved_networks/simple_fit.h5')

    data = np.sort(data, axis=0)
    pred = model.predict_on_batch(data)[:, 0]
    plt.figure()
    plt.plot(np.arange(base, scale, 1e-4), 1.0 / np.arange(base, scale, 1e-4), 'b-')
    plt.plot(data, pred, 'ro')
    plt.show()
    plt.close()
