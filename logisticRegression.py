# import libaries
import torch as tch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Prepare Data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1234)

# Scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = tch.from_numpy(X_train.astype(np.float32))
X_test = tch.from_numpy(X_test.astype(np.float32))
y_train = tch.from_numpy(y_train.astype(np.float32))
y_test = tch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0],1)
y_test = y_test.view(y_test.shape[0],1)

# Create model
class LogisticRegrassion(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegrassion, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_predicted = tch.sigmoid(self.linear(x))
        return y_predicted

model = LogisticRegrassion(n_features)

# Loss and optimizer
learning_rate = 0.05
criterion = nn.BCELoss()
optimizer = tch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass & loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)
    # Backward pass
    loss.backward()
    # Update weights
    optimizer.step()
    # Zero gradients
    optimizer.zero_grad()

    if (epoch + 1) % 100 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

with tch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {acc:.4f}')