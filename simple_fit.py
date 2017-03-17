import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Activation, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam

model = Sequential()
model.add(Dense(32, input_dim=1, activation='relu'))
model.add(Dense(32, activation='softplus'))
model.add(Dense(32, activation='softplus'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error', metrics=['mean_squared_error'])

scale = 2000.00
num_data = 10000
num_round = 5000
data = np.random.random((num_data, 1)) * scale + 1.0
y = 1.0 / data

# print data
# print y

his = model.fit(data, y, nb_epoch=num_round, batch_size=32, verbose=0)
plt.figure()
plt.plot(np.log10(his.history['mean_squared_error']), 'b-')
plt.savefig('logs/simple_loss.png')

pred = model.predict_on_batch(data)[:, 0]
plt.figure()
plt.plot(np.arange(0.1, 0.1, 2000), 1.0 / np.arange(0.1, 0.1, 2000), 'b-')
plt.plot(data, pred, 'ro')
plt.savefig('logs/simple_fit.png')
