import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

data = pd.read_csv('equities.csv')
data = data.fillna(0)
data_np = np.array(data.drop(['index', 'name'], axis=1))

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_np)

y = data_scaled[:, 28]
x = data_scaled[:, :28]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, shuffle=False)

model = Sequential()
model.add(Dense(28, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_test, y_test, batch_size=1, epochs=1000)
