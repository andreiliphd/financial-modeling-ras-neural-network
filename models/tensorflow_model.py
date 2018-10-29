import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

data = pd.read_csv('equities.csv')
data = data.fillna(0)
data_np = np.array(data.drop(['index', 'name'], axis = 1))

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_np)
y = data_scaled[:, 28]
x = data_scaled[:, :28]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, shuffle=False)

X_train_placeholder = tf.placeholder(shape=[None, 28],dtype=tf.float32)
y_train_placeholder = tf.placeholder(shape=[None, None],dtype=tf.float32)

nn = tf.layers.dense(X_train_placeholder, 28, activation=tf.nn.relu)
nn = tf.layers.dense(nn, 100, activation=tf.nn.relu)
nn = tf.layers.dense(nn, 200, activation=tf.nn.relu)
nn = tf.layers.dense(nn, 100, activation=tf.nn.relu)
nn = tf.layers.dense(nn, 200, activation=tf.nn.relu)
nn = tf.layers.dense(nn, 100, activation=tf.nn.relu)
nn = tf.layers.dense(nn, 50, activation=tf.nn.relu)
nn = tf.layers.dense(nn, 1)

y_train = y_train.reshape(15,1)

cost = tf.reduce_mean((nn - y_train_placeholder)**2)
optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(5000):
        _, val = sess.run([optimizer, cost],feed_dict={X_train_placeholder: X_train, y_train_placeholder: y_train})
        if step % 5 == 0:
            print("step: {}, value: {}".format(step, val))
