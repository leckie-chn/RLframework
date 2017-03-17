import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Activation, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam

model = Sequential()
model.add(Dense(32, input_dim=2, activation='relu'))
model.add(Dense(32, activation='softplus'))
model.add(Dense(32, activation='softplus'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error', metrics=['mean_squared_error'])

eps = 1e-8  # prevent from division by 0
num_data = 10000
num_round = 5000
data = np.random.random((num_data, 2)) + eps
y = (data[:, 0] / data[:, 1]).reshape(num_data)

# print data
# print y

his = model.fit(data, y, nb_epoch=num_round, batch_size=32, verbose=0)
plt.figure()
plt.plot(np.log10(his.history['mean_squared_error']), 'b-')
plt.show()
plt.close()
