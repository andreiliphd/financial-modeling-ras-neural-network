import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch

data = pd.read_csv('equities.csv')
data = data.fillna(0)
data_np = np.array(data.drop(['index', 'name'], axis = 1))

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_np)
y = data_scaled[:, 28]
x = data_scaled[:, :28]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, shuffle=False)

X_train_torch = torch.from_numpy(X_train).float()
X_test_torch = torch.from_numpy(X_test).float()
y_train_torch = torch.from_numpy(y_train).float()
y_test_torch = torch.from_numpy(y_test).float()

model = torch.nn.Sequential(
        torch.nn.Linear(28, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 200),
        torch.nn.ReLU(),
        torch.nn.Linear(200, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 200),
        torch.nn.ReLU(),
        torch.nn.Linear(200,100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 1)
        )
loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(10000):
    y_pred = model(X_train_torch)
    loss = loss_fn(y_pred, torch.reshape(y_train_torch, (15,1)))
    print(t, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
